from pathlib import Path

import typer
from rich.console import Console

from opspilot.env.generator import build_sandbox
from opspilot.env.scenarios import get_scenario

app = typer.Typer(help="Build and inspect ShopStack sandboxes by hand.")
console = Console()


@app.command("build")
def build(
    scenario: str = typer.Option(
        ..., "--scenario", help="Scenario name, e.g. checkout_pool_exhaustion."
    ),
    seed: int = typer.Option(
        42, "--seed", help="RNG seed; same (scenario, seed) reproduces byte-identical output."
    ),
    out: Path = typer.Option(
        Path("./tmp/sbx"), "--out", help="Directory to materialize the sandbox into."
    ),
) -> None:
    """Materialize a scenario's sandbox on disk for manual inspection."""
    try:
        scn = get_scenario(scenario)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        sandbox = build_sandbox(scn, seed, out)
    except NotImplementedError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Built[/green] {scn.name} (seed={seed}) at {sandbox.root}")
    console.print(f"[bold]alert:[/bold] {scn.alert_text}")
    for rel, digest in sandbox.snapshot().items():
        console.print(f"  {rel}: {digest[:12]}...")
