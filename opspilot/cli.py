import asyncio
from pathlib import Path
from typing import Any

import anthropic
import typer
from rich.console import Console

from opspilot.config import get_settings
from opspilot.env.cli import app as env_app
from opspilot.env.generator import build_sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.loops.react_raw import LoopEvent, RunResult, run_react_loop
from opspilot.models.anthropic_model import AnthropicModel
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
        console.print(f"  [yellow]tool_call:[/yellow] {data['name']}({data['input']})")
    elif event.type == "exit":
        console.print(f"[bold]exit:[/bold] outcome={data['outcome']} after {data['steps']} step(s)")


def _print_summary(result: RunResult) -> None:
    console.print()
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
    allow_destructive: bool = typer.Option(
        False, "--allow-destructive", help="Permit destructive tools (restart/rollback)."
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Override OPSPILOT_MAX_STEPS."),
) -> None:
    """Run the hand-written ReAct loop against a scenario's alert."""
    asyncio.run(_run_async(scenario, seed, allow_destructive, max_steps))


async def _run_async(
    scenario_name: str, seed: int, allow_destructive: bool, max_steps: int | None
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
    console.print(f"[dim]sandbox:[/dim] {sandbox.root}\n")

    registry = build_default_registry(settings)
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=0)
    model = AnthropicModel(client=client, model=settings.opspilot_model)

    result = await run_react_loop(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert=scn.alert_text,
        settings=settings,
        allow_destructive=allow_destructive,
        max_steps=max_steps,
        on_event=_print_event,
    )

    _print_summary(result)


if __name__ == "__main__":
    app()
