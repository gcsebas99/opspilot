import hashlib
import hmac
import time

from pydantic import BaseModel, Field

from opspilot.env.scenarios import SCENARIOS

SIGNATURE_HEADER = "X-Signature"
TIMESTAMP_HEADER = "X-Timestamp"
DEFAULT_TOLERANCE_S = 300


class AlertPayload(BaseModel):
    """What a monitoring system POSTs to /webhooks/alert."""

    alert_id: str = Field(min_length=1, max_length=200, description="Idempotency key.")
    service: str
    signal: str
    summary: str = Field(default="", max_length=2000)
    # Simulation knob, documented: picks the sandbox "world" directly. The
    # only way to reach prompt_injection, whose alert is identical to
    # checkout_pool_exhaustion's on purpose -- that's what makes it an attack.
    scenario: str | None = None


class InvalidSignature(Exception):
    pass


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """`sha256=<hex>` over "{timestamp}.{body}" -- the timestamp is inside the
    MAC, so it can't be swapped without breaking the signature."""
    mac = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


# [HARNESS:GUARD] Signed webhooks: only the secret's holder can wake the agent.
# WHY: HMAC over timestamp+body proves the sender and that nothing was altered;
# a stale timestamp is rejected, so a captured request can't be replayed later.
# Constant-time compare, so timing can't leak a correct prefix.
def verify(
    secret: str,
    timestamp: str | None,
    signature: str | None,
    body: bytes,
    *,
    tolerance_s: int = DEFAULT_TOLERANCE_S,
    now: float | None = None,
) -> None:
    """Raise InvalidSignature unless the request is authentic and fresh."""
    if not timestamp or not signature:
        raise InvalidSignature("missing signature or timestamp header")
    try:
        sent_at = int(timestamp)
    except ValueError as exc:
        raise InvalidSignature("timestamp is not an integer") from exc
    current = time.time() if now is None else now
    if abs(current - sent_at) > tolerance_s:
        raise InvalidSignature("timestamp outside the allowed window")
    if not hmac.compare_digest(sign(secret, timestamp, body), signature):
        raise InvalidSignature("signature mismatch")


# Which sandbox "world" an alert is about. Explicit and reviewable: in a real
# deployment this is the alert-routing config, not something the model guesses.
ALERT_ROUTES: dict[tuple[str, str], str] = {
    ("checkout", "latency"): "checkout_pool_exhaustion",
    ("checkout", "error_rate"): "false_alarm",
    ("payments", "error_rate"): "payments_bad_deploy",
    ("inventory", "memory"): "inventory_memory_leak",
    ("web", "errors"): "db_disk_full",
}


class UnroutableAlert(ValueError):
    pass


def route_alert(alert: AlertPayload) -> str:
    if alert.scenario is not None:
        if alert.scenario not in SCENARIOS:
            raise UnroutableAlert(f"unknown scenario {alert.scenario!r}")
        return alert.scenario
    scenario = ALERT_ROUTES.get((alert.service, alert.signal))
    if scenario is None:
        raise UnroutableAlert(f"no route for service={alert.service!r} signal={alert.signal!r}")
    return scenario
