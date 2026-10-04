import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall

from evals.models import ApprovalPolicy, EvalCase
from evals.runner import Mode, Strategy, _resolve_decision, run_suite, run_trial
from opspilot.config import Settings
from opspilot.models.base import ModelResponse, ToolUseBlock, Usage
from opspilot.models.scripted import ScriptedModel
from opspilot.store.memory import MemoryStore
from opspilot.store.models import EvalTrialDoc
from opspilot.tools.base import ToolRegistry
from opspilot.tools.registry import build_default_registry

_USAGE = Usage(input_tokens=10, output_tokens=5)

_REPORT_INPUT = {
    "root_cause": "config_change:checkout:db_pool_size",
    "evidence": ["checkout latency_p95_ms spiked after config v13"],
    "confidence": 0.9,
    "recommendation": "monitor for 30 minutes",
}


def _tool_use(name: str, input_: dict[str, Any], tool_use_id: str = "t1") -> ModelResponse:
    return ModelResponse(
        content=[ToolUseBlock(id=tool_use_id, name=name, input=input_)],
        stop_reason="tool_use",
        usage=_USAGE,
        latency_ms=1.0,
    )


def _tool_call_message(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[ToolCall(name=name, args=args, id=call_id)],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


class _StaticGraphModel:
    """A LanguageModelLike stand-in that pops a fixed AIMessage per call --
    the graph-strategy equivalent of ScriptedModel."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return self._responses.pop(0)


def _registry(settings: Settings) -> ToolRegistry:
    return build_default_registry(settings)


def _case(
    case_id: str,
    scenario: str = "false_alarm",
    seed: int = 42,
    role: str = "viewer",
    approval_policy: ApprovalPolicy = "approve_all",
) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": case_id,
            "scenario": scenario,
            "seed": seed,
            "role": role,
            "approval_policy": approval_policy,
            "expect": {"outcome": "completed"},
            "budgets": {
                "max_steps": 12,
                "max_tokens": 50_000,
                "max_cost_usd": 0.15,
                "max_latency_s": 90,
            },
        }
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anthropic_api_key="unused",
        opspilot_model="claude-sonnet-5",
        opspilot_max_steps=10,
        opspilot_token_budget=60_000,
        opspilot_tool_output_max_chars=4_000,
    )


class _RawCrashingModel:
    async def create(self, system: object, messages: object, tools: object) -> ModelResponse:
        raise RuntimeError("boom")


class _Counter:
    def __init__(self) -> None:
        self.current = 0
        self.max_seen = 0
        self.lock = asyncio.Lock()


class _TrackingGraphModel:
    """Records how many trials are inside `.ainvoke()` at once, via a
    counter shared across every trial's own model instance."""

    def __init__(self, responses: list[AIMessage], counter: _Counter) -> None:
        self._responses = list(responses)
        self._counter = counter

    async def ainvoke(self, messages: list) -> AIMessage:  # noqa: ANN401
        async with self._counter.lock:
            self._counter.current += 1
            self._counter.max_seen = max(self._counter.max_seen, self._counter.current)
        await asyncio.sleep(0.02)
        async with self._counter.lock:
            self._counter.current -= 1
        return self._responses.pop(0)


# --- _resolve_decision ---


def test_resolve_decision_approve_all() -> None:
    decision = _resolve_decision("approve_all", "restart_service")

    assert decision == {"decision": "approve", "approver": "eval-runner"}


def test_resolve_decision_reject_all() -> None:
    decision = _resolve_decision("reject_all", "restart_service")

    assert decision["decision"] == "reject"


def test_resolve_decision_scripted_dict() -> None:
    policy: ApprovalPolicy = {"restart_service": "approve", "rollback_config": "reject"}

    assert _resolve_decision(policy, "restart_service")["decision"] == "approve"
    assert _resolve_decision(policy, "rollback_config")["decision"] == "reject"


def test_resolve_decision_scripted_missing_tool_raises() -> None:
    policy: ApprovalPolicy = {"rollback_config": "approve"}

    with pytest.raises(ValueError, match="doesn't cover tool"):
        _resolve_decision(policy, "restart_service")


# --- run_trial ---


async def test_run_trial_raw_strategy_completes(tmp_path: Path, settings: Settings) -> None:
    case = _case("raw-happy")
    model = ScriptedModel([_tool_use("submit_report", _REPORT_INPUT)])
    store = MemoryStore()

    trial = await run_trial(
        case,
        0,
        sweep_id="sweep-1",
        suite="golden",
        strategy="raw",
        mode="live",
        settings=settings,
        registry=_registry(settings),
        store=store,
        git_sha="abc123",
        base_root=tmp_path,
        build_raw_model=lambda _settings: model,
    )

    assert trial.error is None
    assert trial.outcome == "completed"
    assert trial.tool_calls
    assert trial.cost_usd is not None
    assert trial.latency_s is not None and trial.latency_s >= 0
    assert trial.sandbox_snapshot
    assert trial.steps == 1
    assert trial.tokens["output_tokens"] == 5
    assert trial.sandbox_facts["checkout.status"] == "healthy"
    assert trial.policy_events == []

    persisted = await store.list_eval_runs()
    assert len(persisted) == 1
    assert persisted[0].trial_id == trial.trial_id


async def test_run_trial_graph_strategy_auto_approves_and_executes(
    tmp_path: Path, settings: Settings
) -> None:
    case = _case("graph-approve", scenario="checkout_pool_exhaustion", role="operator")
    restart = _tool_call_message("restart_service", {"service": "checkout"}, call_id="a")
    submit = _tool_call_message("submit_report", _REPORT_INPUT, call_id="b")
    store = MemoryStore()

    trial = await run_trial(
        case,
        0,
        sweep_id="sweep-1",
        suite="golden",
        strategy="graph",
        mode="live",
        settings=settings,
        registry=_registry(settings),
        store=store,
        git_sha="abc123",
        base_root=tmp_path,
        build_graph_model=lambda _s, _r: _StaticGraphModel([restart, submit]),
    )

    assert trial.error is None
    assert trial.outcome == "completed"
    assert trial.tool_calls[0]["name"] == "restart_service"
    assert trial.tool_calls[0]["ok"] is True
    # 3.4a: everything graders need is on the trial itself.
    assert trial.steps == 2
    assert trial.tokens["input_tokens"] == 20
    assert trial.sandbox_facts["checkout.restarted"] is True
    assert trial.sandbox_facts["checkout.config_version"] == 13
    assert {"action": "approval_decision", "target": "restart_service", "decision": "approve"} in (
        trial.policy_events
    )
    assert any(e["action"] == "destructive_tool_executed" for e in trial.policy_events)


async def test_run_trial_graph_strategy_reject_all_escalates(
    tmp_path: Path, settings: Settings
) -> None:
    case = _case(
        "graph-reject",
        scenario="checkout_pool_exhaustion",
        role="operator",
        approval_policy="reject_all",
    )
    restart = _tool_call_message("restart_service", {"service": "checkout"}, call_id="a")
    escalate = _tool_call_message("escalate", {"reason": "rejected"}, call_id="b")
    store = MemoryStore()

    trial = await run_trial(
        case,
        0,
        sweep_id="sweep-1",
        suite="golden",
        strategy="graph",
        mode="live",
        settings=settings,
        registry=_registry(settings),
        store=store,
        git_sha="abc123",
        base_root=tmp_path,
        build_graph_model=lambda _s, _r: _StaticGraphModel([restart, escalate]),
    )

    assert trial.error is None
    assert trial.outcome == "escalated"
    assert trial.tool_calls[0]["ok"] is False


async def test_run_trial_scripted_policy_missing_tool_records_error(
    tmp_path: Path, settings: Settings
) -> None:
    case = _case(
        "graph-scripted-gap",
        scenario="checkout_pool_exhaustion",
        role="operator",
        approval_policy={"rollback_config": "approve"},
    )
    restart = _tool_call_message("restart_service", {"service": "checkout"}, call_id="a")
    store = MemoryStore()

    trial = await run_trial(
        case,
        0,
        sweep_id="sweep-1",
        suite="golden",
        strategy="graph",
        mode="live",
        settings=settings,
        registry=_registry(settings),
        store=store,
        git_sha="abc123",
        base_root=tmp_path,
        build_graph_model=lambda _s, _r: _StaticGraphModel([restart]),
    )

    assert trial.error is not None
    assert "doesn't cover tool" in trial.error
    assert trial.outcome is None


async def test_run_trial_records_runner_crash_without_raising(
    tmp_path: Path, settings: Settings
) -> None:
    case = _case("raw-crash")
    store = MemoryStore()

    trial = await run_trial(
        case,
        0,
        sweep_id="sweep-1",
        suite="golden",
        strategy="raw",
        mode="live",
        settings=settings,
        registry=_registry(settings),
        store=store,
        git_sha="abc123",
        base_root=tmp_path,
        build_raw_model=lambda _settings: _RawCrashingModel(),
    )

    assert trial.error is not None
    assert "RuntimeError: boom" in trial.error
    assert trial.outcome is None
    assert await store.list_eval_runs() == [trial]


# --- record / replay (3.3c) ---


def _no_live_client(_settings: Settings) -> ScriptedModel:
    raise AssertionError("replay must never construct a live client")


async def _run_mode(
    case: EvalCase,
    *,
    strategy: Strategy,
    mode: Mode,
    tmp_path: Path,
    live: ScriptedModel | None = None,
) -> EvalTrialDoc:
    settings = Settings(anthropic_api_key="", opspilot_model="claude-sonnet-5")
    return await run_trial(
        case,
        0,
        sweep_id=f"sweep-{mode}-{strategy}",
        suite="golden",
        strategy=strategy,
        mode=mode,
        settings=settings,
        registry=_registry(settings),
        store=MemoryStore(),
        git_sha="abc123",
        base_root=tmp_path / "sandboxes" / f"{mode}-{strategy}",
        cassette_root=tmp_path / "cassettes",
        build_raw_model=(lambda _s: live) if live is not None else _no_live_client,
    )


def _same_trial(a: EvalTrialDoc, b: EvalTrialDoc) -> None:
    # sandbox_facts, not sandbox_snapshot: the snapshot hashes state.json,
    # which holds restart_service's wall-clock restarted_at -- a record run
    # and a replay run in different seconds hash differently (this flaked
    # in CI). facts() reduces that to a stable `restarted: true`.
    assert (a.outcome, a.tool_calls, a.report, a.sandbox_facts, a.cost_usd) == (
        b.outcome,
        b.tool_calls,
        b.report,
        b.sandbox_facts,
        b.cost_usd,
    )


@pytest.mark.parametrize("strategy", ["raw", "graph"])
async def test_record_then_replay_reproduces_the_trial(tmp_path: Path, strategy: Strategy) -> None:
    case = _case("rr", scenario="checkout_pool_exhaustion")
    live = ScriptedModel(
        [
            _tool_use("config_history", {"service": "checkout"}, "t1"),
            _tool_use("submit_report", _REPORT_INPUT, "t2"),
        ]
    )

    recorded = await _run_mode(case, strategy=strategy, mode="record", tmp_path=tmp_path, live=live)
    replayed = await _run_mode(case, strategy=strategy, mode="replay", tmp_path=tmp_path)

    assert recorded.error is None and replayed.error is None
    assert (recorded.mode, replayed.mode) == ("record", "replay")
    assert (tmp_path / "cassettes" / "rr" / "0.jsonl").exists()
    _same_trial(recorded, replayed)


async def test_graph_replay_resumes_through_hitl_approval(tmp_path: Path) -> None:
    case = _case("rr-hitl", scenario="checkout_pool_exhaustion", role="operator")
    live = ScriptedModel(
        [
            _tool_use("restart_service", {"service": "checkout"}, "t1"),
            _tool_use("submit_report", _REPORT_INPUT, "t2"),
        ]
    )

    recorded = await _run_mode(case, strategy="graph", mode="record", tmp_path=tmp_path, live=live)
    replayed = await _run_mode(case, strategy="graph", mode="replay", tmp_path=tmp_path)

    assert replayed.error is None
    assert replayed.tool_calls[0]["name"] == "restart_service"
    assert replayed.tool_calls[0]["ok"] is True
    _same_trial(recorded, replayed)


async def test_cassette_recorded_by_raw_replays_under_graph(tmp_path: Path) -> None:
    case = _case("rr-cross", scenario="checkout_pool_exhaustion")
    live = ScriptedModel(
        [
            _tool_use("list_services", {}, "t1"),
            _tool_use("submit_report", _REPORT_INPUT, "t2"),
        ]
    )

    recorded = await _run_mode(case, strategy="raw", mode="record", tmp_path=tmp_path, live=live)
    replayed = await _run_mode(case, strategy="graph", mode="replay", tmp_path=tmp_path)

    assert replayed.error is None
    _same_trial(recorded, replayed)


async def test_replay_without_cassette_is_a_loud_trial_error(tmp_path: Path) -> None:
    trial = await _run_mode(
        _case("never-recorded"), strategy="raw", mode="replay", tmp_path=tmp_path
    )

    assert trial.error is not None
    assert "CassetteMiss" in trial.error
    assert "record it first" in trial.error
    assert trial.outcome is None


async def test_replay_miss_when_the_run_diverges_from_the_recording(tmp_path: Path) -> None:
    """A different case (other seed -> other tool output) against the same
    cassette dir must miss, not silently serve the recorded conversation."""
    live = ScriptedModel(
        [
            _tool_use("grep_logs", {"service": "checkout", "pattern": "."}, "t1"),
            _tool_use("submit_report", _REPORT_INPUT, "t2"),
        ]
    )
    recorded_case = _case("drift", scenario="checkout_pool_exhaustion", seed=42)
    await _run_mode(recorded_case, strategy="raw", mode="record", tmp_path=tmp_path, live=live)

    drifted_case = _case("drift", scenario="checkout_pool_exhaustion", seed=7)
    trial = await _run_mode(drifted_case, strategy="raw", mode="replay", tmp_path=tmp_path)

    assert trial.error is not None
    assert "first differs at" in trial.error


# --- run_suite ---


async def test_run_suite_isolates_sandboxes_per_trial(tmp_path: Path, settings: Settings) -> None:
    case = _case("iso")

    def build_raw_model(_settings: Settings) -> ScriptedModel:
        return ScriptedModel([_tool_use("submit_report", _REPORT_INPUT)])

    store = MemoryStore()

    trials = await run_suite(
        [case],
        k=2,
        strategy="raw",
        mode="live",
        concurrency=2,
        suite="golden",
        settings=settings,
        store=store,
        git_sha="abc123",
        base_root=tmp_path,
        build_raw_model=build_raw_model,
    )

    assert len(trials) == 2
    assert len({t.run_id for t in trials}) == 2
    for trial in trials:
        sandbox_dir = tmp_path / trial.sweep_id / case.id / f"trial-{trial.trial}"
        assert (sandbox_dir / "state.json").exists()


async def test_run_suite_respects_concurrency_limit(tmp_path: Path, settings: Settings) -> None:
    case = _case("conc", scenario="checkout_pool_exhaustion", role="admin")
    counter = _Counter()

    def build_graph_model(_settings: Settings, _registry: object) -> _TrackingGraphModel:
        submit = _tool_call_message("submit_report", _REPORT_INPUT)
        return _TrackingGraphModel([submit], counter)

    store = MemoryStore()

    await run_suite(
        [case],
        k=6,
        strategy="graph",
        mode="live",
        concurrency=2,
        suite="golden",
        settings=settings,
        store=store,
        git_sha="abc123",
        base_root=tmp_path,
        build_graph_model=build_graph_model,
    )

    assert 1 <= counter.max_seen <= 2


async def test_record_replay_equality_survives_a_clock_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a CI flake: the record and replay runs crossed a
    second boundary, so restart_service's restarted_at differed and the
    sandbox *hashes* didn't match. Force that tick deterministically."""
    case = _case("rr-tick", scenario="checkout_pool_exhaustion", role="operator")
    live = ScriptedModel(
        [
            _tool_use("restart_service", {"service": "checkout"}, "t1"),
            _tool_use("submit_report", _REPORT_INPUT, "t2"),
        ]
    )
    monkeypatch.setattr("opspilot.tools.destructive._now_iso", lambda: "2026-01-01T00:00:00Z")
    recorded = await _run_mode(case, strategy="graph", mode="record", tmp_path=tmp_path, live=live)
    monkeypatch.setattr("opspilot.tools.destructive._now_iso", lambda: "2026-01-01T00:00:01Z")
    replayed = await _run_mode(case, strategy="graph", mode="replay", tmp_path=tmp_path)

    assert recorded.sandbox_snapshot != replayed.sandbox_snapshot  # the trap is real
    _same_trial(recorded, replayed)
