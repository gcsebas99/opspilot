import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opspilot.store.memory import MemoryStore
from opspilot.store.models import RunDoc
from opspilot.web.app import create_app
from opspilot.web.guards import (
    GuardRejected,
    RateLimiter,
    check_live_config,
    check_live_start,
    live_run_settings,
)
from tests.web.test_app import _REPLAY

_LIVE = _REPLAY.model_copy(
    update={
        "opspilot_model_mode": "live",
        "anthropic_api_key": "sk-x",
        "opspilot_demo_live": True,
        "opspilot_live_token": "owner-secret",
        "opspilot_daily_run_cap": 2,
    }
)


# --- rate limiter ---


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_blocks_over_limit_then_window_slides() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=3, window_s=60, clock=clock)
    for _ in range(3):
        limiter.check("1.2.3.4")

    with pytest.raises(GuardRejected) as exc:
        limiter.check("1.2.3.4")
    assert exc.value.status_code == 429 and exc.value.retry_after_s == 61

    limiter.check("5.6.7.8")  # other clients are unaffected
    clock.now += 60  # oldest events fall out of the window
    limiter.check("1.2.3.4")


def test_rejected_attempts_dont_extend_the_lockout() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_s=60, clock=clock)
    limiter.check("ip")
    for _ in range(5):
        with pytest.raises(GuardRejected):
            limiter.check("ip")
    clock.now += 60
    limiter.check("ip")  # hammering while blocked didn't push the window out


def test_web_start_is_rate_limited(tmp_path: Path) -> None:
    settings = _REPLAY.model_copy(update={"opspilot_rate_limit_per_min": 2})
    with TestClient(create_app(settings, store=MemoryStore(), runs_root=tmp_path)) as client:
        for _ in range(2):
            ok = client.post(
                "/runs", data={"path": "false_alarm|42|viewer"}, follow_redirects=False
            )
            assert ok.status_code == 303
        blocked = client.post("/runs", data={"path": "false_alarm|42|viewer"})

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    assert "Too many runs" in blocked.text


# --- live mode configuration ---


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"opspilot_demo_live": False}, "OPSPILOT_DEMO_LIVE=true"),
        ({"opspilot_live_token": ""}, "requires OPSPILOT_LIVE_TOKEN"),
    ],
)
def test_live_server_refuses_to_start_without_both_switches(
    update: dict[str, object], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        check_live_config(_LIVE.model_copy(update=update))


def test_replay_needs_no_live_config_and_keeps_its_budget() -> None:
    check_live_config(_REPLAY)
    assert live_run_settings(_REPLAY) is _REPLAY


def test_live_runs_get_the_lower_token_budget() -> None:
    settings = _LIVE.model_copy(
        update={"opspilot_token_budget": 60_000, "opspilot_live_token_budget": 25_000}
    )
    assert live_run_settings(settings).opspilot_token_budget == 25_000


# --- live start: owner token + daily cap ---


def _live_run(created_at: datetime) -> RunDoc:
    return RunDoc(
        run_id=f"r-{created_at.timestamp()}",
        created_at=created_at,
        scenario="false_alarm",
        seed=42,
        role="viewer",
        model="m",
        prompt_version="v",
        strategy="graph",
        mode="live",
    )


def test_wrong_or_missing_token_is_refused() -> None:
    for token in ("", "owner-secreT", "x"):
        with pytest.raises(GuardRejected) as exc:
            asyncio.run(check_live_start(_LIVE, MemoryStore(), token))
        assert exc.value.status_code == 403


def test_daily_cap_counts_only_todays_live_runs() -> None:
    store = MemoryStore()
    now = datetime.now(UTC)
    asyncio.run(store.insert_run(_live_run(now - timedelta(days=1))))  # yesterday: ignored
    asyncio.run(store.insert_run(_live_run(now)))
    asyncio.run(check_live_start(_LIVE, store, "owner-secret"))  # 1 of 2 used

    asyncio.run(store.insert_run(_live_run(now + timedelta(seconds=1))))
    with pytest.raises(GuardRejected) as exc:
        asyncio.run(check_live_start(_LIVE, store, "owner-secret"))
    assert exc.value.status_code == 429 and "cap (2)" in str(exc.value)


def test_replay_skips_token_and_cap() -> None:
    asyncio.run(check_live_start(_REPLAY, MemoryStore(), ""))


def test_web_live_start_checks_token_then_cap(tmp_path: Path) -> None:
    store = MemoryStore()
    now = datetime.now(UTC)
    for i in range(2):  # cap already reached -- so no live run (or API call) can start
        asyncio.run(store.insert_run(_live_run(now + timedelta(seconds=i))))
    with TestClient(create_app(_LIVE, store=store, runs_root=tmp_path)) as client:
        assert 'name="live_token"' in client.get("/").text
        no_token = client.post("/runs", data={"path": "false_alarm|42|viewer"})
        capped = client.post(
            "/runs", data={"path": "false_alarm|42|viewer", "live_token": "owner-secret"}
        )

    assert no_token.status_code == 403
    assert capped.status_code == 429  # the right token got past the 403, then hit the cap
