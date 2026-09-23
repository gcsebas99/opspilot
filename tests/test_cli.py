from typer.testing import CliRunner

from opspilot.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "OpsPilot" in result.stdout
