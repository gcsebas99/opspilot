from pathlib import Path

from opspilot.config import Settings
from opspilot.env.generator import build_sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.tools.registry import build_default_registry

_SETTINGS = Settings(anthropic_api_key="", opspilot_model="m")


def test_facts_read_config_version_and_status(tmp_path: Path) -> None:
    sandbox = build_sandbox(get_scenario("checkout_pool_exhaustion"), 42, tmp_path / "s")
    facts = sandbox.facts()

    assert facts["checkout.config_version"] == 13
    assert facts["checkout.config.db_pool_size"] == 5
    assert facts["checkout.status"] == "healthy"
    assert "checkout.restarted" not in facts


def test_facts_reflect_destructive_tool_side_effects(tmp_path: Path) -> None:
    sandbox = build_sandbox(get_scenario("checkout_pool_exhaustion"), 42, tmp_path / "s")
    registry = build_default_registry(_SETTINGS)

    assert registry.execute("rollback_config", {"service": "checkout", "version": 12}, sandbox).ok
    assert registry.execute("restart_service", {"service": "inventory"}, sandbox).ok
    facts = sandbox.facts()

    assert facts["checkout.config_version"] == 12
    assert facts["checkout.config.db_pool_size"] == 50
    # Wall-clock restarted_at is reduced to a stable boolean fact.
    assert facts["inventory.restarted"] is True
    assert not any(str(v).startswith("20") and "T" in str(v) for v in facts.values())


def test_facts_deploy_version_follows_rollback(tmp_path: Path) -> None:
    sandbox = build_sandbox(get_scenario("payments_bad_deploy"), 42, tmp_path / "s")
    registry = build_default_registry(_SETTINGS)
    assert sandbox.facts()["payments.deploy_version"] == "2.3.1"

    assert registry.execute("rollback_deploy", {"service": "payments"}, sandbox).ok

    assert sandbox.facts()["payments.deploy_version"] == "2.3.0"
