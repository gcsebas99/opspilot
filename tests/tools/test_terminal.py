from opspilot.env.sandbox import Sandbox
from opspilot.tools.base import ToolRegistry
from opspilot.tools.terminal import TERMINAL_TOOLS


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in TERMINAL_TOOLS:
        registry.register(tool)
    return registry


def test_submit_report(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "submit_report",
        {
            "root_cause": "config_change:checkout:db_pool_size",
            "evidence": ["checkout latency_p95_ms spiked after config v13"],
            "actions_taken": ["rollback_config(checkout, 12)"],
            "confidence": 0.9,
            "recommendation": "monitor for 30 minutes",
        },
        checkout_sandbox,
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data["root_cause"] == "config_change:checkout:db_pool_size"


def test_submit_report_confidence_out_of_range_is_invalid(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "submit_report",
        {"root_cause": "x", "evidence": [], "confidence": 1.5, "recommendation": "n/a"},
        checkout_sandbox,
    )

    assert result.ok is False


def test_submit_report_empty_evidence_is_invalid(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "submit_report",
        {
            "root_cause": "config_change:checkout:db_pool_size",
            "evidence": [],
            "confidence": 0.9,
            "recommendation": "n/a",
        },
        checkout_sandbox,
    )

    assert result.ok is False


def test_submit_report_root_cause_bad_format_is_invalid(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "submit_report",
        {
            "root_cause": "the checkout pool is exhausted!",
            "evidence": ["checkout latency spiked"],
            "confidence": 0.9,
            "recommendation": "n/a",
        },
        checkout_sandbox,
    )

    assert result.ok is False


def test_submit_report_no_incident_is_valid(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "submit_report",
        {
            "root_cause": "no_incident",
            "evidence": ["metrics within normal bounds"],
            "confidence": 0.95,
            "recommendation": "none",
        },
        checkout_sandbox,
    )

    assert result.ok is True


def test_escalate(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "escalate", {"reason": "disk full, no safe tool fixes it"}, checkout_sandbox
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data["reason"] == "disk full, no safe tool fixes it"
