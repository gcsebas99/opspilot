import asyncio
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver

from evals.grading import grade_trial
from evals.loader import load_cases
from evals.models import ApprovalPolicy, EvalCase
from opspilot.config import Settings
from opspilot.context.assembler import CONTEXT_DIR, prompt_version
from opspilot.env.generator import build_sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.loops.graph import resume_react_graph, run_react_graph
from opspilot.loops.react_raw import LoopEvent, run_react_loop
from opspilot.models.base import ModelClient
from opspilot.models.factory import (
    LiveChatFactory,
    LiveClientFactory,
    ModelMode,
    build_chat_model,
    build_model_client,
    default_live_chat_model,
    default_live_client,
)
from opspilot.observability.audit import record_loop_audit
from opspilot.observability.instrumentation import record_loop_spans
from opspilot.observability.pricing import cost_usd
from opspilot.observability.tracer import Tracer
from opspilot.store.base import Store
from opspilot.store.models import EvalTrialDoc, RunDoc
from opspilot.tools.base import ToolRegistry
from opspilot.tools.registry import build_default_registry

Strategy = Literal["raw", "graph"]
Mode = ModelMode

DEFAULT_BASE_ROOT = Path("tmp/eval_runs")


DEFAULT_CASSETTE_ROOT = Path("evals/cassettes")


# [HARNESS:EVAL] Model construction is injectable, not inlined -- the
# runner's whole job is to be called many times, from tests and from every
# mode. The injected factories build only the *live* backend; the runner
# wraps it per mode (opspilot/models/factory.py), so a test can record with
# a ScriptedModel as the "live" model and then replay the resulting cassette.
def cassette_path(cassette_root: Path, case_id: str, trial: int) -> Path:
    # One file per (case, trial): k trials are k independent samples, and
    # trials never share a writer when they run concurrently.
    return cassette_root / case_id / f"{trial}.jsonl"


def judge_cassette_path(cassette_root: Path, case_id: str, trial: int) -> Path:
    # Next to the agent's cassette, separate file: re-recording the judge
    # (rubric change) must never touch the agent's recorded conversation.
    return cassette_root / case_id / f"{trial}.judge.jsonl"


def _judge_client(
    settings: Settings,
    mode: Mode,
    path: Path,
    build_judge_client: LiveClientFactory,
) -> ModelClient:
    judge_settings = settings.model_copy(update={"opspilot_model": settings.opspilot_judge_model})
    return build_model_client(judge_settings, mode, path, build_judge_client)


async def _apply_grading(
    case: EvalCase,
    trial_doc: EvalTrialDoc,
    *,
    judge: ModelClient | None,
    judge_model: str,
) -> EvalTrialDoc:
    try:
        result = await grade_trial(case, trial_doc, judge=judge, judge_model=judge_model)
    except Exception as exc:  # noqa: BLE001
        # e.g. a judge CassetteMiss in replay: loud (error -> non-zero exit),
        # but kept apart from the agent's own result on this trial.
        return trial_doc.model_copy(
            update={
                "grades": None,
                "passed": False,
                "grading_error": f"{type(exc).__name__}: {exc}",
            }
        )
    return trial_doc.model_copy(
        update={
            "grades": {name: grade.model_dump() for name, grade in result.grades.items()},
            "passed": result.passed,
            "grading_error": None,
        }
    )


# [HARNESS:EVAL] Approval auto-resolution -- what a human's interactive
# `y`/`n` prompt (cli.py's `opspilot run`) becomes when nothing is watching.
# WHY: an unscripted tool under a scripted policy is a *dataset* bug (the
# case author forgot to cover a tool this run can actually hit), not a
# runtime condition to paper over with a default -- guessing "reject" or
# "approve" would silently mask exactly the case authoring mistake this
# should surface.
def _resolve_decision(policy: ApprovalPolicy, tool: str) -> dict[str, Any]:
    if policy == "approve_all":
        return {"decision": "approve", "approver": "eval-runner"}
    if policy == "reject_all":
        return {
            "decision": "reject",
            "approver": "eval-runner",
            "reason": "eval: reject_all policy",
        }
    if tool not in policy:
        raise ValueError(
            f"approval_policy is scripted but doesn't cover tool {tool!r} -- "
            "add it to the case's approval_policy dict"
        )
    if policy[tool] == "approve":
        return {"decision": "approve", "approver": "eval-runner"}
    return {
        "decision": "reject",
        "approver": "eval-runner",
        "reason": f"eval: scripted reject for {tool!r}",
    }


_POLICY_ACTIONS = {"approval_decision", "permission_denied", "destructive_tool_executed"}


async def _policy_events(store: Store, run_id: str) -> list[dict[str, str]]:
    return [
        {"action": entry.action, "target": entry.target, "decision": entry.decision}
        for entry in await store.list_audit(run_id)
        if entry.action in _POLICY_ACTIONS
    ]


async def run_trial(
    case: EvalCase,
    trial: int,
    *,
    sweep_id: str,
    suite: str,
    strategy: Strategy,
    mode: Mode,
    settings: Settings,
    registry: ToolRegistry,
    store: Store,
    git_sha: str,
    base_root: Path,
    cassette_root: Path = DEFAULT_CASSETTE_ROOT,
    build_raw_model: LiveClientFactory = default_live_client,
    build_graph_model: LiveChatFactory = default_live_chat_model,
    # Off by default: a library caller (or a test) must opt in to judging,
    # since a live-mode judge is a real API call. The CLI turns it on.
    judge: bool = False,
    build_judge_client: LiveClientFactory = default_live_client,
) -> EvalTrialDoc:
    """Run one (case, trial) to completion and persist the result.

    [HARNESS:ORCH] Never lets one trial's crash take down a whole sweep --
    exactly the same boundary as the tool registry (opspilot/tools/base.py):
    catch broadly here, record the failure on the trial's own document, and
    let every other trial in the sweep keep going.
    """
    run_id = f"{sweep_id}:{case.id}:{trial}"
    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    prompt_version_value = prompt_version(CONTEXT_DIR, registry.to_anthropic_schema())

    trial_doc = EvalTrialDoc(
        trial_id=run_id,
        sweep_id=sweep_id,
        suite=suite,
        case_id=case.id,
        trial=trial,
        run_id=run_id,
        git_sha=git_sha,
        prompt_version=prompt_version_value,
        model=settings.opspilot_model,
        strategy=strategy,
        mode=mode,
        started_at=started_at,
    )

    try:
        scenario = get_scenario(case.scenario)
        # Isolated per (case, trial) -- never keyed by scenario+seed alone,
        # since two different cases can legitimately share both (e.g. the
        # same scenario at different roles) and would otherwise clobber
        # each other's sandbox when trials run concurrently.
        sandbox = build_sandbox(scenario, case.seed, base_root / case.id / f"trial-{trial}")
        tracer = Tracer(store, run_id=run_id)

        await store.insert_run(
            RunDoc(
                run_id=run_id,
                created_at=started_at,
                scenario=case.scenario,
                seed=case.seed,
                role=case.role,
                model=settings.opspilot_model,
                prompt_version=prompt_version_value,
                strategy=strategy,
            )
        )

        if strategy == "raw":
            raw_model = build_model_client(
                settings, mode, cassette_path(cassette_root, case.id, trial), build_raw_model
            )
            events: list[LoopEvent] = []
            async with tracer.span(
                "run",
                "react_loop",
                scenario=case.scenario,
                seed=case.seed,
                role=case.role,
                strategy="raw",
            ) as run_span:
                result = await run_react_loop(
                    model=raw_model,
                    registry=registry,
                    sandbox=sandbox,
                    alert=scenario.alert_text,
                    settings=settings,
                    role=case.role,
                    on_event=events.append,
                )
                await record_loop_spans(
                    tracer, run_span.start, events, model=settings.opspilot_model
                )
                await record_loop_audit(
                    store,
                    run_span.start,
                    events,
                    run_id=run_id,
                    role=case.role,
                    prompt_version=prompt_version_value,
                    model=settings.opspilot_model,
                )
        else:
            bound_model = build_chat_model(
                settings,
                registry,
                mode,
                cassette_path(cassette_root, case.id, trial),
                build_raw_model,
                build_graph_model,
            )
            # A fresh in-memory checkpointer per trial -- eval trials run
            # start-to-finish in this one process, so there's no need for
            # the cross-process durability Mongo's checkpointer exists for
            # (2.5's `opspilot approve`), and it avoids writing throwaway
            # checkpoint state into a real Mongo for every trial.
            checkpointer = InMemorySaver()
            async with tracer.span(
                "run",
                "react_graph",
                scenario=case.scenario,
                seed=case.seed,
                role=case.role,
                strategy="graph",
            ):
                result = await run_react_graph(
                    model=bound_model,
                    registry=registry,
                    sandbox=sandbox,
                    alert=scenario.alert_text,
                    settings=settings,
                    tracer=tracer,
                    store=store,
                    checkpointer=checkpointer,
                    run_id=run_id,
                    prompt_version=prompt_version_value,
                    role=case.role,
                )
            while result.outcome == "awaiting_approval":
                assert result.pending_approval is not None
                decision = _resolve_decision(case.approval_policy, result.pending_approval["tool"])
                result = await resume_react_graph(
                    model=bound_model,
                    registry=registry,
                    sandbox=sandbox,
                    settings=settings,
                    tracer=tracer,
                    store=store,
                    checkpointer=checkpointer,
                    run_id=run_id,
                    decision=decision,
                    prompt_version=prompt_version_value,
                )

        latency_s = time.monotonic() - started_monotonic
        cost = cost_usd(
            settings.opspilot_model,
            input_tokens=result.tokens.input_tokens,
            output_tokens=result.tokens.output_tokens,
            cache_creation_input_tokens=result.tokens.cache_creation_input_tokens,
            cache_read_input_tokens=result.tokens.cache_read_input_tokens,
        )
        await store.update_run(
            run_id,
            {
                "outcome": result.outcome,
                "steps": result.steps,
                "tokens": result.tokens.model_dump(),
                "cost_usd": cost,
                "finished_at": datetime.now(UTC),
            },
        )
        trial_doc = trial_doc.model_copy(
            update={
                "outcome": result.outcome,
                "tool_calls": [tc.model_dump() for tc in result.tool_calls],
                "report": result.report,
                "sandbox_snapshot": result.sandbox_snapshot,
                "sandbox_facts": sandbox.facts(),
                "policy_events": await _policy_events(store, run_id),
                "steps": result.steps,
                "tokens": result.tokens.model_dump(),
                "cost_usd": cost,
                "latency_s": latency_s,
            }
        )
    except Exception as exc:  # noqa: BLE001
        trial_doc = trial_doc.model_copy(
            update={
                "error": f"{type(exc).__name__}: {exc}",
                "latency_s": time.monotonic() - started_monotonic,
            }
        )

    # A crashed run has nothing meaningful to grade -- it's already a failure.
    if trial_doc.error is None:
        judge_client = (
            _judge_client(
                settings,
                mode,
                judge_cassette_path(cassette_root, case.id, trial),
                build_judge_client,
            )
            if judge
            else None
        )
        trial_doc = await _apply_grading(
            case, trial_doc, judge=judge_client, judge_model=settings.opspilot_judge_model
        )
    else:
        trial_doc = trial_doc.model_copy(update={"passed": False})

    await store.insert_eval_run(trial_doc)
    return trial_doc


async def run_suite(
    cases: list[EvalCase],
    *,
    k: int,
    strategy: Strategy,
    mode: Mode,
    concurrency: int,
    suite: str,
    settings: Settings,
    store: Store,
    git_sha: str,
    base_root: Path = DEFAULT_BASE_ROOT,
    cassette_root: Path = DEFAULT_CASSETTE_ROOT,
    build_raw_model: LiveClientFactory = default_live_client,
    build_graph_model: LiveChatFactory = default_live_chat_model,
    # Off by default: a library caller (or a test) must opt in to judging,
    # since a live-mode judge is a real API call. The CLI turns it on.
    judge: bool = False,
    build_judge_client: LiveClientFactory = default_live_client,
) -> list[EvalTrialDoc]:
    """Run every case in `cases` for `k` trials each, `concurrency` at a
    time. One sweep = one call to this function -- `sweep_id` is what lets
    `opspilot eval compare` (3.5) later tell two sweeps of the same suite
    apart, since `git_sha` alone can't (rerunning the same commit, or an
    ablation that changes only OPSPILOT_MODEL, shares a git_sha).
    """
    sweep_id = str(uuid.uuid4())
    registry = build_default_registry(settings)
    sweep_root = base_root / sweep_id
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(case: EvalCase, trial: int) -> EvalTrialDoc:
        async with semaphore:
            return await run_trial(
                case,
                trial,
                sweep_id=sweep_id,
                suite=suite,
                strategy=strategy,
                mode=mode,
                settings=settings,
                registry=registry,
                store=store,
                git_sha=git_sha,
                base_root=sweep_root,
                cassette_root=cassette_root,
                build_raw_model=build_raw_model,
                build_graph_model=build_graph_model,
                judge=judge,
                build_judge_client=build_judge_client,
            )

    if judge and not settings.opspilot_judge_model:
        raise ValueError("judging is on but OPSPILOT_JUDGE_MODEL is empty (or pass --no-judge)")
    return await asyncio.gather(*(_bounded(case, trial) for case in cases for trial in range(k)))


async def regrade_sweep(
    sweep_id: str,
    *,
    store: Store,
    settings: Settings,
    mode: Mode,
    judge: bool,
    cassette_root: Path = DEFAULT_CASSETTE_ROOT,
    build_judge_client: LiveClientFactory = default_live_client,
) -> list[EvalTrialDoc]:
    """Re-grade a stored sweep without re-running the agent.

    [HARNESS:EVAL] Graders are decoupled from runs. WHY: fixing a grader
    or a case's expectations shouldn't cost a re-run of every agent trial --
    trials store everything graders need (3.4a), so re-grading is free for
    the deterministic layers and replayable for the judge. Uses the
    *current* golden case definitions, which is the point: dataset fixes
    apply to old runs.
    """
    cases = {case.id: case for case in load_cases("golden")}
    regraded: list[EvalTrialDoc] = []
    for trial in await store.list_eval_runs(sweep_id):
        case = cases.get(trial.case_id)
        if case is None or trial.error is not None:
            # Case since deleted, or the agent run itself crashed: nothing to grade.
            regraded.append(trial)
            continue
        judge_client = (
            _judge_client(
                settings,
                mode,
                judge_cassette_path(cassette_root, trial.case_id, trial.trial),
                build_judge_client,
            )
            if judge
            else None
        )
        updated = await _apply_grading(
            case, trial, judge=judge_client, judge_model=settings.opspilot_judge_model
        )
        await store.update_eval_run(
            trial.trial_id,
            {
                "grades": updated.grades,
                "passed": updated.passed,
                "grading_error": updated.grading_error,
            },
        )
        regraded.append(updated)
    return regraded
