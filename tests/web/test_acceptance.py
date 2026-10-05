"""Spec 4.1 acceptance, end to end in replay ($0, no key, no network):
start a run as operator -> it pauses -> approve in the browser -> the run
completes -> the trace is visible -- and every page renders along the way."""

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from opspilot.store.memory import MemoryStore
from opspilot.web.app import HTMX_STOP_POLLING, create_app
from tests.web.test_app import _REPLAY, _poll


def test_operator_run_pause_approve_complete(tmp_path: Path) -> None:
    store = MemoryStore()
    app = create_app(_REPLAY, store=store, runs_root=tmp_path / "runs")
    with TestClient(app) as browser:
        # 1. A visitor arrives and switches to an operator identity.
        assert "Replay demo." in browser.get("/").text
        browser.post("/act-as", data={"role": "operator", "name": "sebas", "next": "/"})

        # 2. Starts an operator-agent run on the pool-exhaustion incident.
        started = browser.post(
            "/runs",
            data={"path": "checkout_pool_exhaustion|42|operator", "strategy": "graph"},
            follow_redirects=False,
        )
        run_url = started.headers["location"]
        run_id = run_url.rsplit("/", 1)[-1]
        assert browser.get(run_url).status_code == 200

        # 3. The agent investigates, then pauses before its destructive fix.
        _poll(browser, run_id, until="Paused for approval")
        card = browser.get(f"/runs/{run_id}/approval")
        assert card.status_code == HTMX_STOP_POLLING
        assert "rollback_config" in card.text and "Approve as sebas (operator)" in card.text
        assert "rollback_config" in browser.get("/approvals").text

        # 4. The human approves in the browser.
        approval_id = card.text.split('action="/approvals/')[1].split('"')[0]
        decided = browser.post(
            f"/approvals/{approval_id}", data={"decision": "approve"}, follow_redirects=False
        )
        assert decided.headers["location"] == run_url

        # 5. The run completes with the right diagnosis and a visible trace.
        live = _poll(browser, run_id, until="badge completed")
        assert browser.get(f"/runs/{run_id}/live").status_code == HTMX_STOP_POLLING
        for expected in (
            "config_change:checkout:db_pool_size",  # the report
            "model_call",
            "policy_check",
            "approval_wait",  # the human's think-time
            "tool_call",
        ):
            assert expected in live, expected

        # 6. The side effect really happened in the sandbox, and was audited.
        run = asyncio.run(store.get_run(run_id))
        assert run is not None and run.sandbox_dir is not None
        config = (Path(run.sandbox_dir) / "config" / "checkout.yaml").read_text()
        assert "version: 12" in config
        actions = [e.action for e in asyncio.run(store.list_audit(run_id))]
        assert actions.index("approval_decision") < actions.index("destructive_tool_executed")

        # 7. Every page renders afterwards.
        for page in ("/", "/runs", "/approvals", "/metrics", "/evals", "/healthz"):
            assert browser.get(page).status_code == 200, page
        assert "completed" in browser.get("/metrics").text


def test_metrics_page_on_an_empty_store(tmp_path: Path) -> None:
    with TestClient(create_app(_REPLAY, store=MemoryStore(), runs_root=tmp_path)) as c:
        html = c.get("/metrics").text
    assert "No finished runs yet." in html
    assert "recorded" in html  # replay costs are labeled


def test_evals_page_falls_back_to_committed_baseline(tmp_path: Path) -> None:
    app = create_app(
        _REPLAY, store=MemoryStore(), runs_root=tmp_path, reports_dir=tmp_path / "none"
    )
    with TestClient(app) as c:
        html = c.get("/evals").text
    assert "committed CI baseline" in html
    assert "prompt-injection-operator-reject-all" in html
    assert "pass^k (reliability)" in html


def test_evals_page_prefers_newest_local_report(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    baseline = Path("evals/baselines/smoke.json")
    data = json.loads(baseline.read_text())
    data["meta"]["suite"] = "my-local-sweep"
    (reports / "20990101T000000Z-x-aaaaaaaa.json").write_text(json.dumps(data))

    app = create_app(_REPLAY, store=MemoryStore(), runs_root=tmp_path, reports_dir=reports)
    with TestClient(app) as c:
        html = c.get("/evals").text
    assert "newest local report" in html and "my-local-sweep" in html


def test_evals_page_404_without_any_report(tmp_path: Path) -> None:
    app = create_app(
        _REPLAY,
        store=MemoryStore(),
        runs_root=tmp_path,
        reports_dir=tmp_path / "none",
        baseline_path=tmp_path / "missing.json",
    )
    with TestClient(app) as c:
        assert c.get("/evals").status_code == 404
