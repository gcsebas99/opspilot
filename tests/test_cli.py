from typer.testing import CliRunner

from opspilot.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "OpsPilot" in result.stdout


def test_audit_verify_help() -> None:
    result = runner.invoke(app, ["audit", "verify", "--help"])
    assert result.exit_code == 0
    assert "chain" in result.stdout.lower()
