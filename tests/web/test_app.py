import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opspilot.config import Settings
from opspilot.store.memory import MemoryStore
from opspilot.web.app import HTMX_STOP_POLLING, create_app

_REPLAY = Settings.model_validate(
    {"ANTHROPIC_API_KEY": "", "OPSPILOT_MODEL": "claude-haiku-4-5", "OPSPILOT_MODEL_MODE": "replay"}
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    # `with` runs the lifespan (store, checkpointer, service) and keeps the
    # app's event loop alive so background run tasks can make progress.
    app = create_app(_REPLAY, store=MemoryStore(), runs_root=tmp_path / "runs")
    with TestClient(app) as c:
        yield c


def _start(client: TestClient, path: str, strategy: str = "graph") -> str:
    response = client.post(
        "/runs", data={"path": path, "strategy": strategy}, follow_redirects=False
    )
    assert response.status_code == 303, response.text
    return response.headers["location"].rsplit("/", 1)[-1]


def _poll(client: TestClient, run_id: str, until: str, timeout: float = 10.0) -> str:
    """Poll the live partial like htmx does, until `until` appears in it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}/live").text
        if until in body:
            return body
        time.sleep(0.05)
    raise AssertionError(f"{until!r} never appeared; last body:\n{body}")


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "mode": "replay"}


def test_index_shows_banner_and_only_recorded_paths(client: TestClient) -> None:
    html = client.get("/").text
    assert "Replay demo." in html
    assert "ANTHROPIC_API_KEY" in html
    assert 'value="checkout_pool_exhaustion|42|operator"' in html
    assert "htmx.org@2.0.11" in html and 'integrity="sha384-' in html


def test_unrecorded_or_malformed_paths_are_rejected(client: TestClient) -> None:
    unrecorded = client.post("/runs", data={"path": "db_disk_full|42|admin"})
    assert unrecorded.status_code == 400
    assert "a recorded demo path" in unrecorded.text  # "isn't" is HTML-escaped

    malformed = client.post("/runs", data={"path": "nonsense"})
    assert malformed.status_code == 400

    bad_role = client.post("/runs", data={"path": "false_alarm|42|system"})
    assert bad_role.status_code == 400


def test_viewer_run_completes_and_polling_stops(client: TestClient) -> None:
    run_id = _start(client, "false_alarm|42|viewer")
    assert client.get(f"/runs/{run_id}").status_code == 200

    body = _poll(client, run_id, until="badge completed")
    final = client.get(f"/runs/{run_id}/live")

    assert final.status_code == HTMX_STOP_POLLING  # htmx stops polling a finished run
    assert "no_incident" in body  # the submitted report is shown
    assert "model_call" in body and "tool_call" in body  # waterfall rendered
    assert run_id in client.get("/runs").text


def test_operator_run_pauses_and_keeps_polling(client: TestClient) -> None:
    run_id = _start(client, "checkout_pool_exhaustion|42|operator")

    _poll(client, run_id, until="Paused for approval")
    response = client.get(f"/runs/{run_id}/live")

    assert response.status_code == 200  # still polling: a decision may come
    assert "awaiting_approval" in response.text
    assert "policy_check" in response.text


def test_raw_strategy_runs_too(client: TestClient) -> None:
    run_id = _start(client, "false_alarm|42|viewer", strategy="raw")
    body = _poll(client, run_id, until="badge completed")
    assert "no_incident" in body


def test_unknown_run(client: TestClient) -> None:
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/live").status_code == HTMX_STOP_POLLING


def test_no_banner_in_live_mode(tmp_path: Path) -> None:
    live = _REPLAY.model_copy(
        update={
            "opspilot_model_mode": "live",
            "anthropic_api_key": "sk-x",
            "opspilot_demo_live": True,
            "opspilot_live_token": "owner-secret",
        }
    )
    with TestClient(create_app(live, store=MemoryStore(), runs_root=tmp_path)) as c:
        assert "Replay demo." not in c.get("/").text
