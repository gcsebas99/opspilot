import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import anthropic
import typer
from langchain_anthropic import ChatAnthropic
from rich.console import Console

from opspilot.config import get_settings
from opspilot.context.assembler import CONTEXT_DIR, prompt_version
from opspilot.env.cli import app as env_app
from opspilot.env.generator import build_sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.loops.graph import build_checkpointer, run_react_graph
from opspilot.loops.react_raw import LoopEvent, RunResult, run_react_loop
from opspilot.models.anthropic_model import AnthropicModel
from opspilot.observability.instrumentation import record_loop_spans
from opspilot.observability.metrics import build_dashboard, run_summary
from opspilot.observability.pricing import cost_usd
from opspilot.observability.tracer import Tracer
from opspilot.policy.permissions import Role as PermRole
from opspilot.store.base import Store
from opspilot.store.factory import build_store
from opspilot.store.models import RunDoc, SpanDoc
from opspilot.tools.registry import build_default_registry

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
) -> None:
    """Run the ReAct agent against a scenario's alert."""
    if strategy not in ("raw", "graph"):
        console.print(f"[red]--strategy must be 'raw' or 'graph', got {strategy!r}[/red]")
        raise typer.Exit(code=1)
    if role not in ("viewer", "operator", "admin"):
        console.print(f"[red]--role must be 'viewer', 'operator', or 'admin', got {role!r}[/red]")
        raise typer.Exit(code=1)
    asyncio.run(_run_async(scenario, seed, role, max_steps, strategy))  # type: ignore[arg-type]


async def _run_async(
    scenario_name: str,
    seed: int,
    role: PermRole,
    max_steps: int | None,
    strategy: Literal["raw", "graph"],
) -> None:
    settings = get_settings()

    try:
        scn = get_scenario(scenario_name)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    out_dir = Path("tmp/runs") / f"{scenario_name}-{seed}"
    try:
        sandbox = build_sandbox(scn, seed, out_dir)
    except NotImplementedError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]alert:[/bold] {scn.alert_text}")
    console.print(f"[dim]strategy:[/dim] {strategy}")
    console.print(f"[dim]sandbox:[/dim] {sandbox.root}\n")

    registry = build_default_registry(settings)

    store = build_store(settings)
    await store.ensure_indexes()
    run_id = str(uuid.uuid4())
    tracer = Tracer(store, run_id=run_id)

    await store.insert_run(
        RunDoc(
            run_id=run_id,
            created_at=datetime.now(UTC),
            scenario=scenario_name,
            seed=seed,
            role=role,
            model=settings.opspilot_model,
            prompt_version=prompt_version(CONTEXT_DIR, registry.to_anthropic_schema()),
            strategy=strategy,
        )
    )

    if strategy == "raw":
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=0)
        raw_model = AnthropicModel(client=client, model=settings.opspilot_model)
        events: list[LoopEvent] = []

        def on_event(event: LoopEvent) -> None:
            events.append(event)
            _print_event(event)

        async with tracer.span(
            "run", "react_loop", scenario=scenario_name, seed=seed, role=role, strategy="raw"
        ) as run_span:
            result = await run_react_loop(
                model=raw_model,
                registry=registry,
                sandbox=sandbox,
                alert=scn.alert_text,
                settings=settings,
                role=role,
                max_steps=max_steps,
                on_event=on_event,
            )
            # Must stay inside the `async with` -- record_span() reads the
            # ambient parent span from a contextvar that's reset the moment
            # this block exits, so writing the nested spans after exit
            # would silently produce a flat trace (parent_id=None everywhere).
            await record_loop_spans(tracer, run_span.start, events, model=settings.opspilot_model)
    else:
        chat_model = ChatAnthropic(
            model=settings.opspilot_model, max_tokens=8192, api_key=settings.anthropic_api_key
        )
        bound_model = chat_model.bind_tools(registry.to_anthropic_schema())
        checkpointer, mongo_client = build_checkpointer(settings)
        console.print(
            "[dim](graph strategy: live per-step output lands in `opspilot trace`)[/dim]\n"
        )
        try:
            async with tracer.span(
                "run", "react_graph", scenario=scenario_name, seed=seed, role=role, strategy="graph"
            ):
                result = await run_react_graph(
                    model=bound_model,
                    registry=registry,
                    sandbox=sandbox,
                    alert=scn.alert_text,
                    settings=settings,
                    tracer=tracer,
                    checkpointer=checkpointer,
                    run_id=run_id,
                    role=role,
                    max_steps=max_steps,
                )
        finally:
            if mongo_client is not None:
                mongo_client.close()

    total_cost = cost_usd(
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
            "cost_usd": total_cost,
            "finished_at": datetime.now(UTC),
        },
    )

    _print_summary(result, run_id)


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


if __name__ == "__main__":
    app()
