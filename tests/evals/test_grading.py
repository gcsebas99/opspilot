from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evals.graders.base import Grade
from evals.graders.judge import CRITERIA
from evals.grading import pass_rule
from evals.models import EvalCase
from evals.runner import Mode, regrade_sweep, run_suite, run_trial
from opspilot.cli import app
from opspilot.config import Settings
from opspilot.models.base import ModelResponse, ToolUseBlock, Usage
from opspilot.models.scripted import ScriptedModel
from opspilot.store.memory import MemoryStore
from opspilot.store.models import EvalTrialDoc
from opspilot.tools.registry import build_default_registry

_SETTINGS = Settings.model_validate(
    {
        "ANTHROPIC_API_KEY": "",
        "OPSPILOT_MODEL": "claude-haiku-4-5",
        "OPSPILOT_JUDGE_MODEL": "claude-sonnet-5",
    }
)
_REPORT = {
    "root_cause": "no_incident",
    "evidence": ["all services healthy"],
    "actions_taken": [],
    "confidence": 0.9,
    "recommendation": "none",
}


def _case() -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "false-alarm-viewer",  # a real golden id, so regrade can find it
            "scenario": "false_alarm",
            "seed": 42,
            "role": "viewer",
            "expect": {"outcome": "completed", "root_cause": "no_incident"},
            "budgets": {
                "max_steps": 10,
                "max_tokens": 40000,
                "max_cost_usd": 1,
                "max_latency_s": 60,
            },
        }
    )


def _agent() -> ScriptedModel:
    return ScriptedModel(
        [
            ModelResponse(
                content=[ToolUseBlock(id="t1", name="submit_report", input=_REPORT)],
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
                latency_ms=1.0,
            )
        ]
    )


def _judge(*averages: int) -> ScriptedModel:
    return ScriptedModel(
        [
            ModelResponse(
                content=[
                    ToolUseBlock(
                        id="j",
                        name="submit_grades",
                        input={c: {"score": a, "rationale": "r"} for c in CRITERIA},
                    )
                ],
                stop_reason="tool_use",
                usage=Usage(input_tokens=100, output_tokens=50),
                latency_ms=1.0,
            )
            for a in averages
        ]
    )


def _no_live(_settings: Settings) -> ScriptedModel:
    raise AssertionError("replay must never build a live client")


async def _trial(
    tmp_path: Path, mode: Mode, *, agent: Any = _no_live, judge: Any = _no_live
) -> EvalTrialDoc:
    return await run_trial(
        _case(),
        0,
        sweep_id=f"s-{mode}",
        suite="golden",
        strategy="graph",
        mode=mode,
        settings=_SETTINGS,
        registry=build_default_registry(_SETTINGS),
        store=MemoryStore(),
        git_sha="x",
        base_root=tmp_path / "sbx" / mode,
        cassette_root=tmp_path / "cassettes",
        build_raw_model=agent,
        judge=True,
        build_judge_client=judge,
    )


# --- pass rule ---


def _g(name: str, passed: bool, score: float = 1.0) -> Grade:
    return Grade(name=name, passed=passed, score=score)


@pytest.mark.parametrize(
    ("trajectory", "outcome_score", "budget", "judge", "expected"),
    [
        (True, 1.0, True, True, True),
        (True, 0.5, True, True, True),  # partial diagnosis credit still passes
        (True, 0.0, True, True, False),
        (False, 1.0, True, True, False),  # a great report can't rescue a bad trajectory
        (True, 1.0, False, True, False),
        (True, 1.0, True, False, False),
        (True, 1.0, True, None, True),  # --no-judge: deterministic layers only
    ],
)
def test_pass_rule_requires_every_layer(
    trajectory: bool, outcome_score: float, budget: bool, judge: bool | None, expected: bool
) -> None:
    grades = {
        "trajectory": _g("trajectory", trajectory),
        "outcome": _g("outcome", outcome_score >= 0.5, outcome_score),
        "budget": _g("budget", budget),
    }
    if judge is not None:
        grades["judge"] = _g("judge", judge)
    assert pass_rule(grades) is expected


# --- grading inside run_trial ---


async def test_trial_is_graded_by_all_four_layers(tmp_path: Path) -> None:
    judge = _judge(5)
    trial = await _trial(tmp_path, "record", agent=lambda _s: _agent(), judge=lambda _s: judge)

    assert trial.error is None and trial.grading_error is None
    assert set(trial.grades or {}) == {"trajectory", "outcome", "budget", "judge"}
    assert trial.passed is True
    assert "no_incident" in judge.calls[0]["messages"][0]["content"]


async def test_judge_record_then_replay_reproduces_grades(tmp_path: Path) -> None:
    recorded = await _trial(
        tmp_path, "record", agent=lambda _s: _agent(), judge=lambda _s: _judge(4)
    )
    replayed = await _trial(tmp_path, "replay")

    assert (tmp_path / "cassettes" / "false-alarm-viewer" / "0.judge.jsonl").exists()
    assert replayed.grading_error is None
    assert replayed.grades is not None and recorded.grades is not None
    assert (
        replayed.grades["judge"]["details"]["criteria"]
        == (recorded.grades["judge"]["details"]["criteria"])
    )
    assert replayed.passed == recorded.passed


async def test_missing_judge_cassette_is_a_grading_error_not_an_agent_error(
    tmp_path: Path,
) -> None:
    await _trial(tmp_path, "record", agent=lambda _s: _agent(), judge=lambda _s: _judge(4))
    (tmp_path / "cassettes" / "false-alarm-viewer" / "0.judge.jsonl").unlink()

    trial = await _trial(tmp_path, "replay")

    assert trial.error is None  # the agent run itself replayed fine
    assert trial.outcome == "completed"
    assert trial.grading_error is not None and "CassetteMiss" in trial.grading_error
    assert trial.passed is False


# --- regrade ---


async def test_regrade_adds_judge_to_a_sweep_graded_without_it(tmp_path: Path) -> None:
    store = MemoryStore()
    trials = await run_suite(
        [_case()],
        k=1,
        strategy="graph",
        mode="record",
        concurrency=1,
        suite="golden",
        settings=_SETTINGS,
        store=store,
        git_sha="x",
        base_root=tmp_path / "sbx",
        cassette_root=tmp_path / "cassettes",
        build_raw_model=lambda _s: _agent(),
        judge=False,
    )
    assert "judge" not in (trials[0].grades or {})

    regraded = await regrade_sweep(
        trials[0].sweep_id,
        store=store,
        settings=_SETTINGS,
        mode="record",
        judge=True,
        cassette_root=tmp_path / "cassettes",
        build_judge_client=lambda _s: _judge(2),
    )

    assert regraded[0].grades is not None and "judge" in regraded[0].grades
    assert regraded[0].passed is False  # judge avg 2.0 < 3.5
    persisted = (await store.list_eval_runs(trials[0].sweep_id))[0]
    assert persisted.grades == regraded[0].grades
    assert persisted.passed is False


async def test_regrade_skips_trials_whose_agent_run_crashed(tmp_path: Path) -> None:
    store = MemoryStore()
    trials = await run_suite(
        [_case()],
        k=1,
        strategy="raw",
        mode="live",
        concurrency=1,
        suite="golden",
        settings=_SETTINGS,
        store=store,
        git_sha="x",
        base_root=tmp_path / "sbx",
        build_raw_model=lambda _s: ScriptedModel([]),  # exhausted -> the run crashes
    )
    assert trials[0].error is not None

    regraded = await regrade_sweep(
        trials[0].sweep_id, store=store, settings=_SETTINGS, mode="replay", judge=False
    )
    assert regraded[0].grades is None and regraded[0].error == trials[0].error


def test_eval_grade_requires_a_persistent_store(monkeypatch: pytest.MonkeyPatch) -> None:
    from opspilot.config import get_settings

    monkeypatch.setenv("OPSPILOT_STORE", "memory")
    get_settings.cache_clear()
    try:
        result = CliRunner().invoke(app, ["eval", "grade", "some-sweep"])
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 1
    assert "requires OPSPILOT_STORE=mongo" in result.stdout
