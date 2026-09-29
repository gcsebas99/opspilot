import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evals.models import EvalCase
from evals.runner import run_trial
from opspilot.cli import app
from opspilot.config import Settings, get_settings
from opspilot.models.base import ModelResponse, ToolUseBlock, Usage
from opspilot.models.scripted import ScriptedModel
from opspilot.store.memory import MemoryStore
from opspilot.tools.registry import build_default_registry

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "OpsPilot" in result.stdout


def test_audit_verify_help() -> None:
    result = runner.invoke(app, ["audit", "verify", "--help"])
    assert result.exit_code == 0
    assert "chain" in result.stdout.lower()


# --- model modes (3.3d) ---


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    # chdir so the developer's .env isn't read and tmp/runs lands in tmp_path.
    monkeypatch.chdir(tmp_path)
    for var in ("ANTHROPIC_API_KEY", "OPSPILOT_MODEL_MODE", "MONGODB_URI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPSPILOT_STORE", "memory")
    monkeypatch.setenv("OPSPILOT_MODEL", "claude-haiku-4-5")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_run_live_without_key_is_refused(isolated_env: Path) -> None:
    result = runner.invoke(app, ["run", "--scenario", "false_alarm", "--mode", "live"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is empty" in result.stdout


def test_run_defaults_to_replay_and_misses_loudly(isolated_env: Path) -> None:
    result = runner.invoke(app, ["run", "--scenario", "false_alarm"])
    assert result.exit_code == 1
    assert "mode: replay" in result.stdout
    assert "cassette miss" in result.stdout
    assert "--mode record" in result.stdout


@pytest.mark.parametrize("strategy", ["raw", "graph"])
def test_cassette_recorded_by_eval_replays_as_a_demo(isolated_env: Path, strategy: str) -> None:
    """Record once (here with a scripted stand-in for the API), then the demo
    command replays it for $0 with no API key -- under either strategy."""
    case = EvalCase.model_validate(
        {
            "id": "demo",
            "scenario": "false_alarm",
            "seed": 42,
            "role": "viewer",
            "approval_policy": "approve_all",
            "expect": {"outcome": "completed"},
            "budgets": {"max_steps": 5, "max_tokens": 1, "max_cost_usd": 1, "max_latency_s": 1},
        }
    )
    settings = Settings.model_validate(
        {"ANTHROPIC_API_KEY": "", "OPSPILOT_MODEL": "claude-haiku-4-5"}
    )
    live = ScriptedModel(
        [
            ModelResponse(
                content=[ToolUseBlock(id="t1", name="list_services", input={})],
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
                latency_ms=1.0,
            ),
            ModelResponse(
                content=[
                    ToolUseBlock(
                        id="t2",
                        name="submit_report",
                        input={
                            "root_cause": "no_incident",
                            "evidence": ["all services healthy"],
                            "actions_taken": [],
                            "confidence": 0.9,
                            "recommendation": "none",
                        },
                    )
                ],
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
                latency_ms=1.0,
            ),
        ]
    )
    recorded = asyncio.run(
        run_trial(
            case,
            0,
            sweep_id="s",
            suite="demo",
            strategy="raw",
            mode="record",
            settings=settings,
            registry=build_default_registry(settings),
            store=MemoryStore(),
            git_sha="x",
            base_root=isolated_env / "sbx",
            cassette_root=isolated_env / "cassettes",
            build_raw_model=lambda _s: live,
        )
    )
    assert recorded.error is None

    result = runner.invoke(
        app,
        [
            "run",
            "--scenario",
            "false_alarm",
            "--strategy",
            strategy,
            "--cassette",
            str(isolated_env / "cassettes" / "demo" / "0.jsonl"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "mode: replay" in result.stdout
    assert "outcome: completed" in result.stdout
    assert "no_incident" in result.stdout
