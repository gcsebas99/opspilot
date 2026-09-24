from opspilot.env.sandbox import Sandbox
from opspilot.tools.base import ToolRegistry
from opspilot.tools.read import READ_TOOLS


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in READ_TOOLS:
        registry.register(tool)
    return registry


def test_list_services(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("list_services", {}, checkout_sandbox)

    assert result.ok is True
    assert "checkout: healthy" in result.content
    assert result.data is not None
    assert set(result.data["state"]) == {"web", "checkout", "payments", "inventory", "db"}


def test_grep_logs_finds_fault_signal(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "grep_logs", {"service": "checkout", "pattern": "connection pool timeout"}, checkout_sandbox
    )

    assert result.ok is True
    assert "connection pool timeout" in result.content
    assert result.data is not None
    assert result.data["match_count"] > 0


def test_grep_logs_unknown_service(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("grep_logs", {"service": "nope", "pattern": "."}, checkout_sandbox)

    assert result.ok is False


def test_grep_logs_invalid_regex(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "grep_logs", {"service": "checkout", "pattern": "["}, checkout_sandbox
    )

    assert result.ok is False
    assert "invalid regex" in result.content


def test_query_metrics_shows_spike(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "query_metrics", {"service": "checkout", "metric": "latency_p95_ms"}, checkout_sandbox
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data["summary"]["max"] > 1000  # baseline is ~150, spike ramps past 3000


def test_query_metrics_unknown_metric(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute(
        "query_metrics", {"service": "checkout", "metric": "not_a_metric"}, checkout_sandbox
    )

    assert result.ok is False


def test_read_config(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("read_config", {"service": "checkout"}, checkout_sandbox)

    assert result.ok is True
    assert result.data is not None
    assert result.data["db_pool_size"] == 5


def test_read_config_unknown_service(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("read_config", {"service": "nope"}, checkout_sandbox)

    assert result.ok is False


def test_config_history_shows_pool_size_change(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("config_history", {"service": "checkout"}, checkout_sandbox)

    assert result.ok is True
    assert result.data == {"versions": [12, 13]}
    assert "-db_pool_size: 50" in result.content
    assert "+db_pool_size: 5" in result.content


def test_list_deploys_filtered_by_service(payments_sandbox: Sandbox) -> None:
    result = _registry().execute("list_deploys", {"service": "payments"}, payments_sandbox)

    assert result.ok is True
    assert result.data is not None
    assert len(result.data["deploys"]) == 1
    assert result.data["deploys"][0]["version"] == "2.3.1"


def test_list_deploys_all_services(payments_sandbox: Sandbox) -> None:
    result = _registry().execute("list_deploys", {}, payments_sandbox)

    assert result.ok is True
    assert result.data is not None
    assert len(result.data["deploys"]) >= 4  # 3 boring noise deploys + the payments one


def test_load_runbook_known_name(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("load_runbook", {"name": "db_pool_issues"}, checkout_sandbox)

    assert result.ok is True
    assert "Diagnostic steps" in result.content


def test_load_runbook_unknown_name(checkout_sandbox: Sandbox) -> None:
    result = _registry().execute("load_runbook", {"name": "not_a_runbook"}, checkout_sandbox)

    assert result.ok is False
    assert "not_a_runbook" in result.content
