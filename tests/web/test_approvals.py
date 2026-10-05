import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opspilot.store.memory import MemoryStore
from opspilot.web.app import HTMX_STOP_POLLING, create_app
from tests.web.test_app import _REPLAY, _poll, _start

OPERATOR_PATH = "checkout_pool_exhaustion|42|operator"


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def client(tmp_path: Path, store: MemoryStore) -> Iterator[TestClient]:
    with TestClient(create_app(_REPLAY, store=store, runs_root=tmp_path / "runs")) as c:
        yield c


def _act_as(client: TestClient, role: str, name: str = "sebas") -> None:
    response = client.post(
        "/act-as", data={"role": role, "name": name, "next": "/"}, follow_redirects=False
    )
    assert response.status_code == 303


def _paused_run(client: TestClient, store: MemoryStore) -> tuple[str, str]:
    """Start an operator run and wait for its approval; return (run_id, approval_id)."""
    run_id = _start(client, OPERATOR_PATH)
    _poll(client, run_id, until="Paused for approval")
    card = client.get(f"/runs/{run_id}/approval")
    assert card.status_code == HTMX_STOP_POLLING  # card shown -> stop polling it
    # From the store, not the page: a viewer's card has no form to scrape.
    pending = asyncio.run(store.list_pending_approvals())
    return run_id, next(a.approval_id for a in pending if a.run_id == run_id)


def test_act_as_sets_cookies_and_shows_identity(client: TestClient) -> None:
    _act_as(client, "operator", "sebas")
    html = client.get("/").text
    assert 'value="sebas"' in html
    assert '<option value="operator" selected>' in html
    assert "demo auth, not a real login" in html


def test_act_as_rejects_unknown_roles_and_offsite_redirects(client: TestClient) -> None:
    response = client.post(
        "/act-as",
        data={"role": "root", "name": "x", "next": "https://evil.example"},
        follow_redirects=False,
    )
    assert response.headers["location"] == "/"
    assert '<option value="viewer" selected>' in client.get("/").text


def test_viewer_sees_card_but_cannot_decide(client: TestClient, store: MemoryStore) -> None:
    _act_as(client, "viewer")
    run_id, approval_id = _paused_run(client, store)
    assert "who can watch but not decide" in client.get(f"/runs/{run_id}/approval").text

    # Hidden buttons aren't security -- POST directly, as an attacker would.
    response = client.post(f"/approvals/{approval_id}", data={"decision": "approve"})

    assert response.status_code == 403


async def _status(store: MemoryStore, approval_id: str) -> str:
    approval = await store.get_approval(approval_id)
    assert approval is not None
    return approval.status


def test_viewer_post_leaves_approval_pending(client: TestClient, store: MemoryStore) -> None:
    _act_as(client, "viewer")
    _, approval_id = _paused_run(client, store)
    client.post(f"/approvals/{approval_id}", data={"decision": "approve"})
    assert asyncio.run(_status(store, approval_id)) == "pending"


def test_operator_approves_run_completes_and_is_audited(
    client: TestClient, store: MemoryStore
) -> None:
    _act_as(client, "operator", "sebas")
    run_id, approval_id = _paused_run(client, store)

    response = client.post(
        f"/approvals/{approval_id}", data={"decision": "approve"}, follow_redirects=False
    )
    assert response.status_code == 303 and response.headers["location"] == f"/runs/{run_id}"

    body = _poll(client, run_id, until="badge completed")
    assert "config_change:checkout:db_pool_size" in body
    assert "approval_wait" in body  # the human's think-time is in the trace

    entries = asyncio.run(store.list_audit(run_id))
    decision = next(e for e in entries if e.action == "approval_decision")
    assert (decision.actor, decision.decision) == ("sebas (operator)", "approve")

    # A second click after the decision: 409, nothing re-runs.
    again = client.post(f"/approvals/{approval_id}", data={"decision": "approve"})
    assert again.status_code == 409
    assert "Already approved by sebas (operator)" in again.text


def test_reject_on_unrecorded_branch_ends_with_friendly_note(
    client: TestClient, store: MemoryStore
) -> None:
    _act_as(client, "admin")
    run_id, approval_id = _paused_run(client, store)

    client.post(f"/approvals/{approval_id}", data={"decision": "reject", "reason": "not now"})

    body = _poll(client, run_id, until="badge error")
    assert "wasn&#39;t recorded for the public demo" in body


def test_edit_is_refused_in_replay(client: TestClient, store: MemoryStore) -> None:
    _act_as(client, "operator")
    run_id, approval_id = _paused_run(client, store)
    assert "Approve with edits" not in client.get(f"/runs/{run_id}/approval").text

    response = client.post(
        f"/approvals/{approval_id}", data={"decision": "edit", "args": '{"version": 11}'}
    )
    assert response.status_code == 400
    assert "replay demo" in response.text


def test_unknown_approval_and_bad_decision(client: TestClient) -> None:
    _act_as(client, "operator")
    assert client.post("/approvals/nope", data={"decision": "approve"}).status_code == 404
    assert client.post("/approvals/nope", data={"decision": "yolo"}).status_code == 400


def test_approvals_page_lists_pending_and_nav_counts(
    client: TestClient, store: MemoryStore
) -> None:
    run_id, _ = _paused_run(client, store)
    html = client.get("/approvals").text
    assert "rollback_config" in html
    assert f'href="/runs/{run_id}"' in html
    assert 'badge awaiting_approval">1<' in html  # nav counter


def test_approval_region_keeps_polling_while_running_and_stops_when_done(
    client: TestClient,
) -> None:
    run_id = _start(client, "false_alarm|42|viewer")  # never pauses
    _poll(client, run_id, until="badge completed")
    region = client.get(f"/runs/{run_id}/approval")
    assert region.status_code == HTMX_STOP_POLLING and region.text == ""
