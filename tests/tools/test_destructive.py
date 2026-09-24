import json
from pathlib import Path

import yaml

from opspilot.env.sandbox import Sandbox
from opspilot.tools.base import ToolRegistry
from opspilot.tools.destructive import DESTRUCTIVE_TOOLS


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in DESTRUCTIVE_TOOLS:
        registry.register(tool)
    return registry


def test_restart_service_marks_healthy_and_persists(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("restart_service", {"service": "checkout"}, checkout_sandbox)

    assert result.ok is True
    assert result.data is not None
    assert result.data["status"] == "healthy"
    assert "restarted_at" in result.data

    state = json.loads((checkout_sandbox.root / "state.json").read_text())
    assert state["checkout"]["status"] == "healthy"


def test_restart_service_unknown_service(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("restart_service", {"service": "nope"}, checkout_sandbox)

    assert result.ok is False


def test_rollback_config_reverts_and_persists(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "rollback_config", {"service": "checkout", "version": 12}, checkout_sandbox
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data["db_pool_size"] == 50

    config = yaml.safe_load((checkout_sandbox.root / "config" / "checkout.yaml").read_text())
    assert config["db_pool_size"] == 50


def test_rollback_config_unknown_version(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "rollback_config", {"service": "checkout", "version": 999}, checkout_sandbox
    )

    assert result.ok is False


def test_rollback_deploy_reverts_to_previous_version(tmp_path: Path) -> None:
    root = tmp_path / "sbx"
    root.mkdir()
    sandbox = Sandbox(root=root)
    deploys = [
        {"service": "web", "version": "1.0.0", "ts": "2025-01-01T00:00:00Z", "author": "alice"},
        {"service": "web", "version": "2.0.0", "ts": "2025-01-01T00:10:00Z", "author": "bob"},
    ]
    sandbox.path("deploys.json").write_text(json.dumps(deploys))

    result = _registry().execute("rollback_deploy", {"service": "web"}, sandbox)

    assert result.ok is True
    assert result.data is not None
    assert result.data["version"] == "1.0.0"

    updated = json.loads((root / "deploys.json").read_text())
    assert len(updated) == 3
    assert updated[-1]["version"] == "1.0.0"


def test_rollback_deploy_no_earlier_deploy(payments_sandbox: Sandbox) -> None:
    result = _registry().execute("rollback_deploy", {"service": "payments"}, payments_sandbox)

    assert result.ok is False
    assert "no earlier deploy" in result.content
