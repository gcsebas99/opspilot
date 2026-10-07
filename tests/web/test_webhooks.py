import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from opspilot.store.memory import MemoryStore
from opspilot.web.app import create_app
from opspilot.web.webhooks import (
    ALERT_ROUTES,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    AlertPayload,
    InvalidSignature,
    UnroutableAlert,
    route_alert,
    sign,
    verify,
)
from scripts.canary import main as canary_main
from scripts.canary import run_canary
from tests.web.test_app import _REPLAY, _poll

SECRET = "whsec-test"
_SETTINGS = _REPLAY.model_copy(update={"opspilot_webhook_secret": SECRET})


# --- signing ---


def test_sign_then_verify_roundtrip() -> None:
    body, ts = b'{"a":1}', "1700000000"
    verify(SECRET, ts, sign(SECRET, ts, body), body, now=1700000000)


@pytest.mark.parametrize(
    ("secret", "ts", "body", "now", "message"),
    [
        ("other-secret", "1700000000", b'{"a":1}', 1700000000, "mismatch"),
        (SECRET, "1700000000", b'{"a":2}', 1700000000, "mismatch"),  # tampered body
        (SECRET, "1700000000", b'{"a":1}', 1700000301, "window"),  # stale
        (SECRET, "1700000000", b'{"a":1}', 1699999699, "window"),  # from the future
    ],
)
def test_verify_rejects(secret: str, ts: str, body: bytes, now: int, message: str) -> None:
    signature = sign(secret, "1700000000", b'{"a":1}')
    with pytest.raises(InvalidSignature, match=message):
        verify(SECRET, ts, signature, body, now=now)


def test_timestamp_is_inside_the_mac() -> None:
    body = b"{}"
    signature = sign(SECRET, "1700000000", body)
    with pytest.raises(InvalidSignature, match="mismatch"):
        verify(SECRET, "1700000100", signature, body, now=1700000100)  # swapped timestamp


def test_verify_rejects_missing_or_garbage_headers() -> None:
    with pytest.raises(InvalidSignature, match="missing"):
        verify(SECRET, None, "sha256=x", b"{}")
    with pytest.raises(InvalidSignature, match="not an integer"):
        verify(SECRET, "yesterday", "sha256=x", b"{}")


# --- routing ---


@pytest.mark.parametrize(("key", "scenario"), list(ALERT_ROUTES.items()))
def test_routes(key: tuple[str, str], scenario: str) -> None:
    service, signal = key
    assert route_alert(AlertPayload(alert_id="a", service=service, signal=signal)) == scenario


def test_scenario_override_and_unroutable() -> None:
    injected = AlertPayload(
        alert_id="a", service="checkout", signal="latency", scenario="prompt_injection"
    )
    assert route_alert(injected) == "prompt_injection"
    with pytest.raises(UnroutableAlert, match="no route"):
        route_alert(AlertPayload(alert_id="a", service="checkout", signal="vibes"))
    with pytest.raises(UnroutableAlert, match="unknown scenario"):
        route_alert(AlertPayload(alert_id="a", service="x", signal="y", scenario="nope"))


# --- the endpoint ---


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def client(tmp_path: Path, store: MemoryStore) -> Iterator[TestClient]:
    with TestClient(create_app(_SETTINGS, store=store, runs_root=tmp_path)) as c:
        yield c


def _post(client: TestClient, payload: dict[str, Any], secret: str = SECRET) -> Any:
    body = json.dumps(payload).encode()
    ts = str(int(time.time()))
    return client.post(
        "/webhooks/alert",
        content=body,
        headers={TIMESTAMP_HEADER: ts, SIGNATURE_HEADER: sign(secret, ts, body)},
    )


FALSE_ALARM = {"alert_id": "a-1", "service": "checkout", "signal": "error_rate", "summary": "hi"}


def _wait_finished(client: TestClient, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status: dict[str, Any] = client.get(f"/api/runs/{run_id}").json()
        if status["finished"]:
            return status
        time.sleep(0.05)
    raise AssertionError("run never finished")


def test_unsigned_or_badly_signed_requests_start_nothing(
    client: TestClient, store: MemoryStore
) -> None:
    unsigned = client.post("/webhooks/alert", json=FALSE_ALARM)
    wrong = _post(client, FALSE_ALARM, secret="guess")
    assert (unsigned.status_code, wrong.status_code) == (401, 401)
    assert asyncio.run(store.list_runs()) == []


def test_signed_alert_runs_as_system_and_completes(client: TestClient, store: MemoryStore) -> None:
    response = _post(client, FALSE_ALARM)
    assert response.status_code == 202
    data = response.json()
    assert data["duplicate"] is False and data["status_url"] == f"/api/runs/{data['run_id']}"

    status = _wait_finished(client, data["run_id"])

    assert (status["outcome"], status["role"], status["trigger"]) == (
        "completed",
        "system",
        "webhook",
    )
    assert status["report"]["root_cause"] == "no_incident"
    run = asyncio.run(store.get_run(data["run_id"]))
    assert run is not None and run.alert is not None and run.alert["summary"] == "hi"


def test_same_alert_twice_is_one_run(client: TestClient, store: MemoryStore) -> None:
    first = _post(client, FALSE_ALARM)
    second = _post(client, FALSE_ALARM)

    assert (first.status_code, second.status_code) == (202, 200)
    assert second.json() == {**first.json(), "duplicate": True}
    _wait_finished(client, first.json()["run_id"])
    assert len(asyncio.run(store.list_runs())) == 1


async def test_concurrent_duplicate_deliveries_start_one_run(tmp_path: Path) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from opspilot.web.service import WebRunService

    store = MemoryStore()
    service = WebRunService(store, _SETTINGS, InMemorySaver(), tmp_path)
    alert = AlertPayload(**FALSE_ALARM)
    results = await asyncio.gather(
        *(service.start_from_alert(alert, "false_alarm") for _ in range(5))
    )
    await service.wait_idle()

    assert len({run_id for run_id, _ in results}) == 1
    assert sorted(dup for _, dup in results) == [False, True, True, True, True]
    assert len(await store.list_runs()) == 1


def test_webhook_run_pauses_for_a_human(client: TestClient) -> None:
    alert = {"alert_id": "p-1", "service": "checkout", "signal": "latency"}
    run_id = _post(client, alert).json()["run_id"]
    _poll(client, run_id, until="Paused for approval")

    client.post("/act-as", data={"role": "operator", "name": "oncall", "next": "/"})
    card = client.get(f"/runs/{run_id}/approval").text
    approval_id = card.split('action="/approvals/')[1].split('"')[0]
    client.post(f"/approvals/{approval_id}", data={"decision": "approve"})

    assert _wait_finished(client, run_id)["outcome"] == "completed"


def test_unroutable_unrecorded_and_malformed_are_422(client: TestClient) -> None:
    assert (
        _post(client, {"alert_id": "x", "service": "checkout", "signal": "vibes"}).status_code
        == 422
    )
    injected = {
        "alert_id": "y",
        "service": "checkout",
        "signal": "latency",
        "scenario": "prompt_injection",
    }
    unrecorded = _post(client, injected)  # routable, but no system cassette in replay
    assert unrecorded.status_code == 422 and "no recorded path" in unrecorded.json()["error"]
    assert _post(client, {"service": "checkout"}).status_code == 422  # no alert_id


async def test_failed_start_releases_the_key_so_a_retry_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claim the key, then fail to start: the key must be released, or the
    sender's retry would be told "duplicate" for a run that never existed."""
    from langgraph.checkpoint.memory import InMemorySaver

    import opspilot.web.service as service_module
    from opspilot.web.service import WebRunService

    store = MemoryStore()
    service = WebRunService(store, _SETTINGS, InMemorySaver(), tmp_path)
    alert = AlertPayload(**FALSE_ALARM)
    real_create_run = service_module.create_run

    async def broken_create_run(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("disk full")

    monkeypatch.setattr(service_module, "create_run", broken_create_run)
    with pytest.raises(RuntimeError, match="disk full"):
        await service.start_from_alert(alert, "false_alarm")

    monkeypatch.setattr(service_module, "create_run", real_create_run)
    run_id, duplicate = await service.start_from_alert(alert, "false_alarm")  # the retry
    await service.wait_idle()

    assert duplicate is False
    assert [r.run_id for r in await store.list_runs()] == [run_id]


def test_webhook_disabled_without_secret(tmp_path: Path) -> None:
    with TestClient(create_app(_REPLAY, store=MemoryStore(), runs_root=tmp_path)) as c:
        assert _post(c, FALSE_ALARM).status_code == 503


def test_api_runs_unknown(client: TestClient) -> None:
    assert client.get("/api/runs/nope").status_code == 404


# --- the canary ---


def test_canary_passes_against_a_healthy_app(client: TestClient) -> None:
    result = run_canary(client, SECRET, poll_s=0.02, sleep=lambda _s: time.sleep(0.02))
    assert result.ok, result.message
    assert result.run_id is not None


def test_canary_fails_loudly_on_a_bad_secret(client: TestClient) -> None:
    result = run_canary(client, "wrong", poll_s=0.02, sleep=lambda _s: time.sleep(0.02))
    assert not result.ok and "401" in result.message


def test_canary_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPSPILOT_URL", raising=False)
    monkeypatch.delenv("OPSPILOT_WEBHOOK_SECRET", raising=False)
    assert canary_main([]) == 2
    assert canary_main(["--skip-if-unconfigured"]) == 0
