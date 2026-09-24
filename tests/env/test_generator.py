import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from opspilot.env.generator import METRICS, SERVICES, WINDOW_MINUTES, build_sandbox
from opspilot.env.scenarios import get_scenario


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_same_seed_produces_identical_files(tmp_path: Path) -> None:
    scenario = get_scenario("checkout_pool_exhaustion")
    sandbox_a = build_sandbox(scenario, seed=42, root=tmp_path / "a")
    sandbox_b = build_sandbox(scenario, seed=42, root=tmp_path / "b")

    assert _hash_tree(sandbox_a.root) == _hash_tree(sandbox_b.root)


def test_different_seed_changes_noise_but_keeps_the_fault_signal(tmp_path: Path) -> None:
    scenario = get_scenario("checkout_pool_exhaustion")
    sandbox_a = build_sandbox(scenario, seed=1, root=tmp_path / "a")
    sandbox_b = build_sandbox(scenario, seed=2, root=tmp_path / "b")

    assert _hash_tree(sandbox_a.root) != _hash_tree(sandbox_b.root)

    for sandbox in (sandbox_a, sandbox_b):
        log_text = (sandbox.root / "logs" / "checkout.log").read_text()
        assert "connection pool timeout" in log_text

        config = yaml.safe_load((sandbox.root / "config" / "checkout.yaml").read_text())
        assert config["db_pool_size"] == 5

        history_v12 = yaml.safe_load(
            (sandbox.root / "config" / "history" / "checkout" / "v12.yaml").read_text()
        )
        assert history_v12["db_pool_size"] == 50


@pytest.mark.parametrize("name", ["inventory_memory_leak", "db_disk_full", "prompt_injection"])
def test_stub_scenarios_raise_not_implemented(tmp_path: Path, name: str) -> None:
    scenario = get_scenario(name)
    with pytest.raises(NotImplementedError):
        build_sandbox(scenario, seed=1, root=tmp_path / "sbx")


def test_build_creates_expected_layout(tmp_path: Path) -> None:
    scenario = get_scenario("payments_bad_deploy")
    sandbox = build_sandbox(scenario, seed=7, root=tmp_path / "sbx")

    for service in SERVICES:
        assert (sandbox.root / "logs" / f"{service}.log").exists()
        assert (sandbox.root / "config" / f"{service}.yaml").exists()
    assert (sandbox.root / "metrics.db").exists()
    assert (sandbox.root / "deploys.json").exists()
    assert (sandbox.root / "state.json").exists()

    deploys = json.loads((sandbox.root / "deploys.json").read_text())
    assert any(d["service"] == "payments" and d["version"] == "2.3.1" for d in deploys)

    state = json.loads((sandbox.root / "state.json").read_text())
    assert all(state[service]["status"] == "healthy" for service in SERVICES)


def test_metrics_db_has_full_series(tmp_path: Path) -> None:
    scenario = get_scenario("checkout_pool_exhaustion")
    sandbox = build_sandbox(scenario, seed=3, root=tmp_path / "sbx")

    conn = sqlite3.connect(sandbox.root / "metrics.db")
    try:
        count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        assert count == len(SERVICES) * len(METRICS) * WINDOW_MINUTES
    finally:
        conn.close()


def test_false_alarm_has_no_config_or_deploy_changes(tmp_path: Path) -> None:
    scenario = get_scenario("false_alarm")
    sandbox = build_sandbox(scenario, seed=5, root=tmp_path / "sbx")

    config = yaml.safe_load((sandbox.root / "config" / "checkout.yaml").read_text())
    assert config["db_pool_size"] == 50
    assert config["version"] == 1

    log_text = (sandbox.root / "logs" / "checkout.log").read_text()
    assert "ERROR" not in log_text
