import asyncio
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console

from evals.calibration import (
    CALIBRATION_CASSETTE_ROOT,
    MIN_PASS_AGREEMENT,
    MIN_WITHIN_ONE,
    load_calibration,
    run_judge_check,
)
from evals.compare import load_comparison
from evals.gates import gating_regressions, pass_rate_gate
from evals.graders.judge import CRITERIA
from evals.loader import load_cases
from evals.models import EvalCase
from evals.report import EvalReport, build_report, failed_checks, write_report
from evals.runner import regrade_sweep, run_suite
from opspilot.config import get_settings
from opspilot.env.cli import app as env_app
from opspilot.loops.graph import build_checkpointer
from opspilot.loops.react_raw import LoopEvent, RunResult
from opspilot.models.cassette import CassetteMiss
from opspilot.models.factory import (
    LiveCallRefused,
    ModelMode,
    build_model_client,
    demo_cassette_path,
    resolve_mode,
)
from opspilot.observability.audit import (
    verify_chain,
)
from opspilot.observability.metrics import build_dashboard, run_summary
from opspilot.policy.permissions import Role as PermRole
from opspilot.runs import create_run, open_run, resume_graph, run_raw, start_graph
from opspilot.store.base import Store
from opspilot.store.factory import build_store
from opspilot.store.models import EvalTrialDoc, SpanDoc

app = typer.Typer(
    name="opspilot",
    help="OpsPilot — an incident triage agent for a fake e-commerce stack (ShopStack).",
    no_args_is_help=True,
)
app.add_typer(env_app, name="env")

console = Console()


@app.callback()
def main() -> None:
    """OpsPilot CLI. Subcommands (eval) are added in later sub-tasks."""


def _print_event(event: LoopEvent) -> None:
    data: dict[str, Any] = event.data
    if event.type == "step_start":
        console.print(f"[bold cyan]-- step {data['step']} --[/bold cyan]")
    elif event.type == "model_call":
        usage = data["usage"]
        console.print(
            f"  [dim]model:[/dim] stop_reason={data['stop_reason']} "
            f"in={usage['input_tokens']} out={usage['output_tokens']} "
            f"cache_read={usage['cache_read_input_tokens']} "
            f"latency={data['latency_ms']:.0f}ms"
        )
    elif event.type == "tool_call":
        console.print(
            f"  [yellow]tool_call:[/yellow] {data['name']}({data['input']}) "
            f"ok={data['ok']} {data['duration_ms']:.0f}ms"
        )
    elif event.type == "exit":
        console.print(f"[bold]exit:[/bold] outcome={data['outcome']} after {data['steps']} step(s)")


def _print_summary(result: RunResult, run_id: str) -> None:
    console.print()
    console.print(f"[bold]run_id:[/bold] {run_id}")
    console.print(f"[bold]outcome:[/bold] {result.outcome}")
    console.print(f"[bold]steps:[/bold] {result.steps}")
    tokens = result.tokens
    console.print(
        f"[bold]tokens:[/bold] in={tokens.input_tokens} out={tokens.output_tokens} "
        f"cache_write={tokens.cache_creation_input_tokens} "
        f"cache_read={tokens.cache_read_input_tokens}"
    )
    if result.report is not None:
        console.print("[bold]report:[/bold]")
        for key, value in result.report.items():
            console.print(f"  {key}: {value}")


@app.command("run")
def run(
    scenario: str = typer.Option(
        ..., "--scenario", help="Scenario name, e.g. checkout_pool_exhaustion."
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed for the sandbox."),
    role: str = typer.Option(
        "viewer", "--role", help="Permission role: viewer (safe), operator, or admin."
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Override OPSPILOT_MAX_STEPS."),
    strategy: str = typer.Option(
        "graph", "--strategy", help="Loop implementation: raw (Day 1) or graph (LangGraph, 2.3)."
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="replay (default, $0, needs a recorded cassette), record (API on cassette "
        "miss), or live (API every call). Default comes from OPSPILOT_MODEL_MODE.",
    ),
    cassette: Path | None = typer.Option(
        None,
        "--cassette",
        help="Cassette file for record/replay (default: evals/cassettes/demo/"
        "<scenario>-s<seed>-<role>.jsonl).",
    ),
) -> None:
    """Run the ReAct agent against a scenario's alert."""
    if strategy not in ("raw", "graph"):
        console.print(f"[red]--strategy must be 'raw' or 'graph', got {strategy!r}[/red]")
        raise typer.Exit(code=1)
    if role not in ("viewer", "operator", "admin"):
        console.print(f"[red]--role must be 'viewer', 'operator', or 'admin', got {role!r}[/red]")
        raise typer.Exit(code=1)
    model_mode = _resolve_mode_or_exit(mode)
    if cassette is None and model_mode != "live":
        cassette = demo_cassette_path(scenario, seed, role)
    with _cassette_miss_exits(model_mode):
        asyncio.run(
            _run_async(scenario, seed, role, max_steps, strategy, model_mode, cassette)  # type: ignore[arg-type]
        )


def _resolve_mode_or_exit(requested: str | None) -> ModelMode:
    try:
        return resolve_mode(requested, get_settings())
    except (ValueError, LiveCallRefused) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@contextmanager
def _cassette_miss_exits(mode: ModelMode) -> Iterator[None]:
    try:
        yield
    except CassetteMiss as exc:
        console.print(f"\n[red]cassette miss ({mode}):[/red] {exc}")
        console.print(
            "[dim]This path wasn't recorded (e.g. a different approval decision, or the "
            "prompt changed). Re-run with --mode record to add it (calls the API).[/dim]"
        )
        raise typer.Exit(code=1) from exc


def _print_mode_banner(mode: ModelMode, cassette: Path | None) -> None:
    if mode == "live":
        console.print("[bold red]mode: live[/bold red] -- every model call hits the API ($)")
    elif mode == "record":
        console.print(
            f"[bold yellow]mode: record[/bold yellow] -- API only on cassette miss ($), "
            f"saving to {cassette}"
        )
    else:
        console.print(f"[bold green]mode: replay[/bold green] -- $0, no network, from {cassette}")


async def _run_async(
    scenario_name: str,
    seed: int,
    role: PermRole,
    max_steps: int | None,
    strategy: Literal["raw", "graph"],
    mode: ModelMode,
    cassette: Path | None,
) -> None:
    settings = get_settings()
    store = build_store(settings)
    await store.ensure_indexes()

    try:
        handle = await create_run(
            store,
            settings,
            scenario=scenario_name,
            seed=seed,
            role=role,
            strategy=strategy,
            mode=mode,
            cassette=cassette,
            sandbox_dir=Path("tmp/runs") / f"{scenario_name}-{seed}",
        )
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except NotImplementedError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]alert:[/bold] {handle.alert}")
    console.print(f"[dim]strategy:[/dim] {strategy}")
    _print_mode_banner(mode, cassette)
    console.print(f"[dim]sandbox:[/dim] {handle.sandbox.root}\n")

    if strategy == "raw":
        result = await run_raw(handle, store, on_event=_print_event, max_steps=max_steps)
    else:
        checkpointer, mongo_client = build_checkpointer(settings)
        console.print(
            "[dim](graph strategy: live per-step output lands in `opspilot trace`)[/dim]\n"
        )
        try:
            result = await start_graph(handle, store, checkpointer, max_steps=max_steps)
            # [HARNESS:HITL] The interactive half of the approval loop: a blocking y/n.
            # WHY: the graph only pauses and resumes; this is where a human decides.
            # `opspilot approve` and the web UI are the out-of-process equivalents.
            while result.outcome == "awaiting_approval":
                pending = result.pending_approval
                assert pending is not None
                console.print(
                    f"\n[bold yellow]approval required:[/bold yellow] "
                    f"{pending['tool']}({pending['args']})"
                )
                console.print(f"[dim]reason:[/dim] {pending['reason']}")
                console.print(f"[dim]approval_id:[/dim] {pending['approval_id']}")
                decision: dict[str, Any]
                if typer.confirm("Approve this action?"):
                    decision = {"decision": "approve", "approver": role}
                else:
                    reject_reason = typer.prompt("Rejection reason", default="")
                    decision = {"decision": "reject", "approver": role, "reason": reject_reason}
                result = await resume_graph(
                    handle, store, checkpointer, decision, max_steps=max_steps
                )
        finally:
            if mongo_client is not None:
                mongo_client.close()

    _print_summary(result, handle.run.run_id)


@app.command("web")
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (0.0.0.0 in a container)."),
    port: int = typer.Option(8000, "--port", help="Port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)."),
) -> None:
    """Serve the web UI (FastAPI + HTMX). Replay mode by default -- $0."""
    import uvicorn  # imported here: only this command needs a web server

    uvicorn.run("opspilot.web.app:create_app", factory=True, host=host, port=port, reload=reload)


@app.command("trace")
def trace(run_id: str = typer.Argument(..., help="Run ID to show the waterfall for.")) -> None:
    """Print a waterfall tree of spans for one run."""
    asyncio.run(_trace_async(run_id))


async def _trace_async(run_id: str) -> None:
    settings = get_settings()
    store = build_store(settings)

    summary = await run_summary(store, run_id)
    if summary is None:
        console.print(f"[red]no run found with id {run_id!r}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]run:[/bold] {summary.run_id}  scenario={summary.scenario}  "
        f"outcome={summary.outcome}"
    )
    console.print(
        f"[bold]steps:[/bold] {summary.steps}  cost=${summary.cost_usd or 0:.4f}  "
        f"duration={summary.duration_s or 0:.1f}s"
    )
    console.print(f"[bold]tokens:[/bold] {summary.tokens}\n")

    spans = await store.list_spans(run_id)
    _print_waterfall(spans)


def _print_waterfall(spans: list[SpanDoc]) -> None:
    by_parent: dict[str | None, list[SpanDoc]] = {}
    for span in spans:
        by_parent.setdefault(span.parent_id, []).append(span)
    for children in by_parent.values():
        children.sort(key=lambda s: s.start)

    def _print_node(span: SpanDoc, depth: int) -> None:
        indent = "  " * depth
        duration = f"{span.duration_ms:.0f}ms" if span.duration_ms is not None else "?"
        status_color = "red" if span.status == "error" else "green"
        console.print(
            f"{indent}[{status_color}]{span.kind}[/{status_color}] {span.name} ({duration})"
        )
        for child in by_parent.get(span.span_id, []):
            _print_node(child, depth + 1)

    for root in by_parent.get(None, []):
        _print_node(root, 0)


@app.command("metrics")
def metrics() -> None:
    """Print the cross-run metrics dashboard."""
    asyncio.run(_metrics_async())


async def _metrics_async() -> None:
    settings = get_settings()
    store: Store = build_store(settings)

    dashboard = await build_dashboard(store)

    console.print("[bold]Model call latency (ms)[/bold]")
    p50 = dashboard.model_call_latency_percentiles.get("p50", 0.0)
    p95 = dashboard.model_call_latency_percentiles.get("p95", 0.0)
    console.print(f"  p50={p50:.0f}  p95={p95:.0f}\n")

    console.print("[bold]Tool error rate[/bold]")
    for tool, rate in sorted(dashboard.tool_error_rates.items()):
        console.print(f"  {tool}: {rate:.1%}")
    console.print()

    console.print("[bold]Outcomes[/bold]")
    for outcome, count in sorted(dashboard.outcomes_distribution.items()):
        console.print(f"  {outcome}: {count}")
    console.print()

    console.print("[bold]Avg cost by scenario[/bold]")
    for scenario, avg in sorted(dashboard.avg_cost_by_scenario.items()):
        console.print(f"  {scenario}: ${avg:.4f}")


approvals_app = typer.Typer(help="Inspect pending HITL approvals.")
app.add_typer(approvals_app, name="approvals")


@approvals_app.command("list")
def approvals_list() -> None:
    """List all pending approvals across runs."""
    asyncio.run(_approvals_list_async())


async def _approvals_list_async() -> None:
    settings = get_settings()
    store = build_store(settings)
    pending = await store.list_pending_approvals()
    if not pending:
        console.print("[dim]no pending approvals[/dim]")
        return
    for approval in pending:
        console.print(
            f"[bold]{approval.approval_id}[/bold]  run={approval.run_id}  "
            f"tool={approval.tool}({approval.args})"
        )
        console.print(f"  [dim]reason:[/dim] {approval.reason}")
        console.print(f"  [dim]requested_at:[/dim] {approval.requested_at.isoformat()}")


@app.command("approve")
def approve(
    approval_id: str = typer.Argument(
        ..., help="Approval ID, e.g. from `opspilot approvals list`."
    ),
    reject: bool = typer.Option(False, "--reject", help="Reject instead of approve."),
    reason: str = typer.Option("", "--reason", help="Reason (used when --reject)."),
    approver: str = typer.Option("cli", "--approver", help="Name recorded as the approver."),
) -> None:
    """Resume a paused run by approving or rejecting its pending tool call."""
    with _cassette_miss_exits("replay"):
        asyncio.run(_approve_async(approval_id, reject, reason, approver))


async def _approve_async(approval_id: str, reject: bool, reason: str, approver: str) -> None:
    settings = get_settings()
    # [HARNESS:HITL] Cross-process resume needs a shared checkpointer.
    # WHY: InMemorySaver lives in the paused process's memory; only MongoDBSaver
    # lets a separate `opspilot approve` see the paused thread.
    if settings.opspilot_store != "mongo":
        console.print(
            "[red]opspilot approve requires OPSPILOT_STORE=mongo -- the paused run's "
            "checkpoint only survives across processes in Mongo, never in memory.[/red]"
        )
        raise typer.Exit(code=1)

    store = build_store(settings)
    approval = await store.get_approval(approval_id)
    if approval is None:
        console.print(f"[red]no approval found with id {approval_id!r}[/red]")
        raise typer.Exit(code=1)

    try:
        handle = await open_run(store, settings, approval.run_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    run = handle.run
    checkpointer, mongo_client = build_checkpointer(settings)

    decision: dict[str, Any] = (
        {"decision": "reject", "approver": approver, "reason": reason or "no reason given"}
        if reject
        else {"decision": "approve", "approver": approver}
    )

    try:
        result = await resume_graph(handle, store, checkpointer, decision)
    except LiveCallRefused as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        if mongo_client is not None:
            mongo_client.close()

    _print_summary(result, run.run_id)
    if result.outcome == "awaiting_approval":
        pending = result.pending_approval
        assert pending is not None
        console.print(
            f"\n[bold yellow]another approval is pending:[/bold yellow] "
            f"{pending['approval_id']} -- run `opspilot approve {pending['approval_id']}` "
            "to continue."
        )


audit_app = typer.Typer(help="Inspect and verify the tamper-evident audit log.")
app.add_typer(audit_app, name="audit")


@audit_app.command("verify")
def audit_verify() -> None:
    """Recompute the audit log's hash chain and report the first broken link, if any."""
    asyncio.run(_audit_verify_async())


async def _audit_verify_async() -> None:
    settings = get_settings()
    store = build_store(settings)
    entries = await store.list_audit()
    result = verify_chain(entries)

    if result.ok:
        console.print(
            f"[green]audit log verified:[/green] {result.total_entries} entries, chain intact"
        )
        return

    console.print(
        f"[red]audit log TAMPERED[/red]: chain breaks at entry "
        f"{result.broken_at_index} of {result.total_entries} ({result.reason})"
    )
    raise typer.Exit(code=1)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


eval_app = typer.Typer(help="Run, grade, and inspect eval sweeps.")
app.add_typer(eval_app, name="eval")


@eval_app.callback(invoke_without_command=True)
def eval_command(
    ctx: typer.Context,
    suite: str = typer.Option("golden", "--suite", help="golden (all cases) or a tag filter."),
    k: int = typer.Option(1, "--k", help="Trials per case."),
    strategy: str = typer.Option("graph", "--strategy", help="Loop implementation: raw or graph."),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="replay (cassettes only, no network, $0), record (API on cassette miss, "
        "saves responses), or live (real API). Default comes from OPSPILOT_MODEL_MODE.",
    ),
    concurrency: int = typer.Option(4, "--concurrency", help="Max trials running at once."),
    judge: bool = typer.Option(
        True, "--judge/--no-judge", help="Run the LLM judge (same mode as the agent)."
    ),
    min_pass_rate: float | None = typer.Option(
        None, "--min-pass-rate", help="CI gate: exit 1 if pass@1 is below this (0-1)."
    ),
) -> None:
    """Run a suite of golden eval cases, grade every trial, and persist it."""
    if ctx.invoked_subcommand is not None:
        return
    if strategy not in ("raw", "graph"):
        console.print(f"[red]--strategy must be 'raw' or 'graph', got {strategy!r}[/red]")
        raise typer.Exit(code=1)
    model_mode = _resolve_mode_or_exit(mode)
    asyncio.run(
        _eval_async(suite, k, strategy, model_mode, concurrency, judge, min_pass_rate)  # type: ignore[arg-type]
    )


def _grade_summary(trial: EvalTrialDoc) -> str:
    grades = trial.grades or {}
    parts = []
    if "trajectory" in grades:
        parts.append(f"traj={grades['trajectory']['score']:.2f}")
    if "outcome" in grades:
        parts.append(f"outcome={grades['outcome']['score']:.1f}")
    if "budget" in grades:
        parts.append(f"budget={'ok' if grades['budget']['passed'] else 'OVER'}")
    if "judge" in grades:
        average = grades["judge"]["details"].get("average")
        parts.append(f"judge={average:.1f}" if average is not None else "judge=ERR")
    return " ".join(parts)


def _print_trials(trials: list[EvalTrialDoc], mode: str, judge: bool) -> int:
    """Print one line per trial (+ why it failed); return how many errored."""
    errored = 0
    judge_cost = 0.0
    for trial in sorted(trials, key=lambda t: (t.case_id, t.trial)):
        # Rich's `console.print` treats "[...]" as a (possibly unknown)
        # style tag and silently drops it -- a case_id in brackets would
        # vanish from the output rather than error, which is exactly what
        # happened here until this was caught by an actual live run.
        label = f"{trial.case_id}#{trial.trial}"
        if trial.error is not None or trial.grading_error is not None:
            errored += 1
            what = "ERROR" if trial.error is not None else "GRADING ERROR"
            console.print(f"[red]{label} {what}:[/red] {trial.error or trial.grading_error}")
            continue
        verdict = "[green]PASS[/green]" if trial.passed else "[red]FAIL[/red]"
        console.print(
            f"{verdict} {label}  {_grade_summary(trial)}  "
            f"cost=${trial.cost_usd or 0:.4f} latency={trial.latency_s or 0:.1f}s"
        )
        for line in failed_checks(trial.grades):
            console.print(f"     [dim]- {line}[/dim]")
        judge_grade = (trial.grades or {}).get("judge")
        if judge_grade is not None:
            judge_cost += judge_grade["details"].get("cost_usd") or 0.0

    passed = sum(1 for t in trials if t.passed)
    console.print(
        f"\n[bold]passed:[/bold] {passed}/{len(trials)}" + ("" if judge else " (no judge)")
    )
    if judge:
        console.print(f"[dim]judge cost: ${judge_cost:.4f}[/dim]")
    if mode == "replay":
        console.print(
            "[dim]replay: costs shown are what the recorded calls cost; spend was $0.[/dim]"
        )
    return errored


def _write_and_print_report(
    trials: list[EvalTrialDoc], cases: dict[str, EvalCase], *, suite: str, judge: bool
) -> EvalReport:
    report = build_report(trials, cases, suite=suite, judged=judge)
    json_path, md_path = write_report(report)
    console.print(f"[bold]report:[/bold] {md_path}  [dim](+ {json_path.name})[/dim]")
    return report


async def _eval_async(
    suite: str,
    k: int,
    strategy: Literal["raw", "graph"],
    mode: Literal["live", "record", "replay"],
    concurrency: int,
    judge: bool,
    min_pass_rate: float | None = None,
) -> None:
    settings = get_settings()
    store = build_store(settings)
    await store.ensure_indexes()

    cases = load_cases(suite)  # type: ignore[arg-type]
    if not cases:
        console.print(f"[yellow]no cases found for suite {suite!r}[/yellow]")
        raise typer.Exit(code=1)

    git_sha = _git_sha()
    console.print(
        f"[bold]suite:[/bold] {suite}  [bold]cases:[/bold] {len(cases)}  [bold]k:[/bold] {k}  "
        f"[bold]trials:[/bold] {len(cases) * k}"
    )
    console.print(
        f"[dim]strategy={strategy} mode={mode} concurrency={concurrency} git_sha={git_sha} "
        f"judge={settings.opspilot_judge_model if judge else 'off'}[/dim]\n"
    )

    try:
        trials = await run_suite(
            cases,
            k=k,
            strategy=strategy,
            mode=mode,
            concurrency=concurrency,
            suite=suite,
            settings=settings,
            store=store,
            git_sha=git_sha,
            judge=judge,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    errored = _print_trials(trials, mode, judge)
    console.print(f"[bold]sweep_id:[/bold] {trials[0].sweep_id}")
    report = _write_and_print_report(trials, {c.id: c for c in cases}, suite=suite, judge=judge)
    # Non-zero exit on any runner or grading error -- in replay that includes
    # every agent *or judge* cassette miss, which is what lets CI fail on
    # prompt/rubric drift. Plain FAILs only fail the run through the explicit
    # --min-pass-rate gate: the threshold is the caller's (CI's) decision.
    if errored:
        raise typer.Exit(code=1)
    if min_pass_rate is not None:
        failure = pass_rate_gate(report.metrics, min_pass_rate)
        if failure is not None:
            console.print(f"[bold red]gate failed:[/bold red] {failure}")
            raise typer.Exit(code=1)
        console.print(f"[green]gate passed:[/green] pass@1 >= {min_pass_rate:.0%}")


@eval_app.command("grade")
def eval_grade(
    sweep_id: str = typer.Argument(..., help="Sweep to re-grade (printed by `opspilot eval`)."),
    mode: str | None = typer.Option(
        None, "--mode", help="Mode for the judge: replay (default), record, or live."
    ),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Re-run the LLM judge too."),
) -> None:
    """Re-grade a stored sweep with the current graders and golden cases -- no agent re-run."""
    model_mode = _resolve_mode_or_exit(mode)
    asyncio.run(_eval_grade_async(sweep_id, model_mode, judge))


async def _eval_grade_async(sweep_id: str, mode: ModelMode, judge: bool) -> None:
    settings = get_settings()
    # Same reasoning as `opspilot approve`: an in-memory store died with the
    # process that ran the sweep, so there's nothing left to re-grade.
    if settings.opspilot_store != "mongo":
        console.print(
            "[red]opspilot eval grade requires OPSPILOT_STORE=mongo -- trials from an "
            "in-memory store don't outlive the `opspilot eval` process.[/red]"
        )
        raise typer.Exit(code=1)
    store = build_store(settings)
    trials = await regrade_sweep(sweep_id, store=store, settings=settings, mode=mode, judge=judge)
    if not trials:
        console.print(f"[red]no trials found for sweep {sweep_id!r}[/red]")
        raise typer.Exit(code=1)
    errored = _print_trials(trials, mode, judge)
    golden = {c.id: c for c in load_cases("golden")}
    _write_and_print_report(trials, golden, suite=trials[0].suite, judge=judge)
    if errored:
        raise typer.Exit(code=1)


@eval_app.command("compare")
def eval_compare(
    a: str = typer.Argument(..., help="Baseline: report .json, sweep id prefix, or `previous`."),
    b: str = typer.Argument(..., help="Candidate: report .json, sweep id prefix, or `latest`."),
    fail_on_tag: list[str] = typer.Option(
        [],
        "--fail-on-tag",
        help="Only regressions on cases with this tag fail (exit 1). Repeatable. Default: any.",
    ),
) -> None:
    """Diff two eval reports: regressions, fixes, metric deltas. Exits 1 on gating regressions."""
    try:
        comparison = load_comparison(a, b)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    for label, meta in (("A", comparison.a), ("B", comparison.b)):
        console.print(
            f"[bold]{label}:[/bold] {meta.suite} {meta.sweep_id[:8]}  model={meta.model} "
            f"strategy={meta.strategy} mode={meta.mode} k={meta.k} git={meta.git_sha}"
        )
    for warning in comparison.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    for note in comparison.notes:
        console.print(f"[dim]note: {note}[/dim]")

    console.print()
    for delta in comparison.deltas:
        before, after = delta.fmt(delta.before), delta.fmt(delta.after)
        change = "" if delta.change is None else f"  ({delta.fmt(delta.change, signed=True)})"
        console.print(f"  {delta.name:18} {before:>10} -> {after:<10}{change}")

    sections = (
        ("regression", "red", "REGRESSIONS"),
        ("fix", "green", "fixes"),
        ("changed", "yellow", "changed"),
        ("added", "cyan", "added"),
        ("removed", "cyan", "removed"),
    )
    for kind, color, title in sections:
        diffs = comparison.of_kind(kind)  # type: ignore[arg-type]
        if not diffs:
            continue
        console.print(f"\n[bold {color}]{title} ({len(diffs)})[/bold {color}]")
        for diff in diffs:
            rates = " -> ".join(
                "--" if r is None else f"{r:.0%}" for r in (diff.before_rate, diff.after_rate)
            )
            console.print(
                f"  {diff.case_id}: {diff.before or '--'} -> {diff.after or '--'} ({rates})"
            )

    unchanged = len(comparison.of_kind("unchanged"))
    console.print(f"\n[dim]{unchanged} case(s) unchanged[/dim]")
    gating = gating_regressions(comparison, fail_on_tag)
    if gating:
        scope = f" on tag(s) {', '.join(fail_on_tag)}" if fail_on_tag else ""
        console.print(f"[bold red]gate failed:[/bold red] {len(gating)} regression(s){scope}")
        raise typer.Exit(code=1)
    if comparison.of_kind("regression"):
        console.print(
            f"[yellow]regressions outside {', '.join(fail_on_tag)} don't fail this gate[/yellow]"
        )


@eval_app.command("judge-check")
def eval_judge_check(
    mode: str | None = typer.Option(
        None, "--mode", help="replay (default), record, or live -- for the judge's calls."
    ),
    reveal: bool = typer.Option(
        False,
        "--reveal",
        help="Show the judge's scores even if some items lack human scores (un-blinds you).",
    ),
) -> None:
    """Check the LLM judge against human-scored reports (evals/judge_calibration.yaml)."""
    model_mode = _resolve_mode_or_exit(mode)
    with _cassette_miss_exits(model_mode):
        asyncio.run(_judge_check_async(model_mode, reveal))


async def _judge_check_async(mode: ModelMode, reveal: bool) -> None:
    settings = get_settings()
    judge_settings = settings.model_copy(update={"opspilot_model": settings.opspilot_judge_model})
    items = load_calibration()

    report = await run_judge_check(
        items,
        settings=settings,
        judge_client_for=lambda item: build_model_client(
            judge_settings, mode, CALIBRATION_CASSETTE_ROOT / f"{item.id}.jsonl"
        ),
    )
    console.print(
        f"[bold]judge:[/bold] {report.judge_model}  "
        f"[dim]prompt={report.judge_prompt_version} mode={mode}[/dim]\n"
    )
    for result in report.items:
        if result.judge_error:
            console.print(f"[red]{result.id}: judge error: {result.judge_error}[/red]")

    unscored = [item.id for item in items if not item.is_scored()]
    if unscored and not reveal:
        # Blind by design: seeing the judge's numbers first would anchor the
        # human scores that are supposed to check it.
        console.print(
            f"[yellow]{len(unscored)}/{len(items)} items have no human scores yet:[/yellow] "
            + ", ".join(unscored)
        )
        console.print(
            "Score them in evals/judge_calibration.yaml first (the judge's scores stay "
            "hidden until you do), then re-run. --reveal un-blinds."
        )
        raise typer.Exit(code=1)

    for result in report.items:
        expected = result.expected_avg
        exp_text = f"{expected:.1f}" if expected is not None else "  - "
        diffs = " ".join(
            f"{c[:4]}={result.judged.get(c, '?')}"
            + (f"/{result.expected[c]}" if result.expected.get(c) is not None else "")
            for c in CRITERIA
        )
        console.print(
            f"{result.id:34} {result.tier:9} human={exp_text} judge={result.judged_avg:.1f}  "
            f"[dim]{diffs}[/dim]"
        )
    console.print(
        f"\n[bold]within-1:[/bold] {report.within_one:.0%}  "
        f"[bold]MAE:[/bold] {report.mean_abs_error:.2f}  "
        f"[bold]pass/fail agreement:[/bold] {report.pass_agreement:.0%}"
    )
    worst = sorted(report.per_criterion_mae.items(), key=lambda kv: -kv[1])
    console.print(
        "[dim]MAE by criterion: " + ", ".join(f"{c}={m:.2f}" for c, m in worst) + "[/dim]"
    )
    if report.trusted:
        console.print("[bold green]judge TRUSTED[/bold green] on this calibration set")
    else:
        console.print(
            f"[bold red]judge NOT TRUSTED[/bold red] (needs within-1 >= {MIN_WITHIN_ONE:.0%} "
            f"and pass/fail agreement >= {MIN_PASS_AGREEMENT:.0%})"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
