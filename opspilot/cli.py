import typer

from opspilot.env.cli import app as env_app

app = typer.Typer(
    name="opspilot",
    help="OpsPilot — an incident triage agent for a fake e-commerce stack (ShopStack).",
    no_args_is_help=True,
)
app.add_typer(env_app, name="env")


@app.callback()
def main() -> None:
    """OpsPilot CLI. Subcommands (run, eval) are added in later sub-tasks."""


if __name__ == "__main__":
    app()
