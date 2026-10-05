import hmac
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime

from opspilot.config import Settings
from opspilot.store.base import Store


class GuardRejected(Exception):
    """A request a guard refuses. `status_code` is the HTTP answer;
    `retry_after_s` (if set) becomes a Retry-After header."""

    def __init__(self, message: str, status_code: int, retry_after_s: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class RateLimiter:
    """Sliding-window limiter: at most `limit` events per `window_s` per key.

    In-process memory on purpose: one free-tier instance, and losing the
    counters on restart is harmless. With several instances you'd move this
    to a shared store (Redis, or a Mongo TTL collection) -- otherwise each
    instance enforces its own separate limit.
    """

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        """Record an event for `key`, or raise GuardRejected(429) if over the limit."""
        now = self._clock()
        events = self._events[key]
        while events and now - events[0] >= self.window_s:
            events.popleft()
        if len(events) >= self.limit:
            retry_after = int(self.window_s - (now - events[0])) + 1
            raise GuardRejected(
                f"Too many runs started from your address -- try again in {retry_after}s.",
                429,
                retry_after_s=retry_after,
            )
        events.append(now)


def check_live_config(settings: Settings) -> None:
    """Fail at startup, not at the first paid call: a web server in a
    spending mode must have been switched on deliberately, and must have an
    owner token -- otherwise anyone with the URL could spend the key."""
    if settings.opspilot_model_mode == "replay":
        return
    if not settings.opspilot_demo_live:
        raise RuntimeError(
            f"OPSPILOT_MODEL_MODE={settings.opspilot_model_mode} spends money, but the web app "
            "only serves live runs when OPSPILOT_DEMO_LIVE=true. Use replay for public demos."
        )
    if not settings.opspilot_live_token:
        raise RuntimeError("OPSPILOT_DEMO_LIVE=true requires OPSPILOT_LIVE_TOKEN (owner-only).")


def live_run_settings(settings: Settings) -> Settings:
    """Settings for web-started live runs: the per-run token budget is the
    lower of the general budget and the web live budget."""
    if settings.opspilot_model_mode == "replay":
        return settings
    budget = min(settings.opspilot_token_budget, settings.opspilot_live_token_budget)
    return settings.model_copy(update={"opspilot_token_budget": budget})


# [HARNESS:GUARD] Spend guards for the public web app.
# WHY: a public URL plus an API key is an open tab on your bill. Layers,
# cheapest first: replay by default (no key needed at all); a deliberate
# OPSPILOT_DEMO_LIVE switch checked at startup; an owner token compared in
# constant time; a global daily cap counted in the store (survives
# restarts, shared across instances); a per-run token budget the loop
# itself enforces; and a per-IP rate limit on top of all of it.
# INTERVIEW: "What protects your API bill on a public demo?" -> no usable
# key in replay deployments; live mode is owner-token-gated, daily-capped,
# per-run-budgeted, and rate-limited.
async def check_live_start(settings: Settings, store: Store, token: str) -> None:
    """Gate one live run start: owner token, then the daily cap."""
    if settings.opspilot_model_mode == "replay":
        return
    # compare_digest: constant-time, so response timing can't leak how many
    # leading characters of a guess were right.
    if not hmac.compare_digest(token.encode(), settings.opspilot_live_token.encode()):
        raise GuardRejected("Live runs are owner-only on this server.", 403)
    midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    started_today = await store.count_runs_since(midnight, settings.opspilot_model_mode)
    if started_today >= settings.opspilot_daily_run_cap:
        raise GuardRejected(
            f"Today's live-run cap ({settings.opspilot_daily_run_cap}) is reached. "
            "Replay mode or a local clone still works.",
            429,
        )
