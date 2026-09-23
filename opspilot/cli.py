import typer

app = typer.Typer(
    name="opspilot",
    help="OpsPilot — an incident triage agent for a fake e-commerce stack (ShopStack).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """OpsPilot CLI. Subcommands (run, env, eval) are added in later sub-tasks."""


if __name__ == "__main__":
    app()
