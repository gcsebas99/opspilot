"""Production canary: prove the deployed app still works, end to end.

Sends a *signed* false_alarm alert to the deployed webhook and checks the run
ends `completed` with root cause `no_incident`. Runs weekly from
.github/workflows/canary.yml; also runnable by hand:

    OPSPILOT_WEBHOOK_SECRET=... uv run python scripts/canary.py --url http://127.0.0.1:8000

Against a replay deployment this costs $0 -- and still exercises everything a
real alert would: signature check, idempotency, routing, the agent run,
tools, policy, persistence, and the status API.
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from opspilot.web.webhooks import SIGNATURE_HEADER, TIMESTAMP_HEADER, sign


@dataclass
class CanaryResult:
    ok: bool
    message: str
    run_id: str | None = None


def _wake(
    client: httpx.Client, timeout_s: float, poll_s: float, sleep: Callable[[float], None]
) -> None:
    """Free-tier hosts sleep when idle; the first request may take a while."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if client.get("/healthz").status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"app didn't answer /healthz within {timeout_s:.0f}s")
        sleep(poll_s)


# [HARNESS:LOOP] Heartbeat outer loop: the canary as an online eval.
# WHY: replay evals prove the *code*; only a scheduled synthetic alert with a known
# answer proves the *deployment* (secrets, DB, routing, server awake) end to end.
def run_canary(
    client: httpx.Client,
    secret: str,
    *,
    wake_timeout_s: float = 120,
    run_timeout_s: float = 90,
    poll_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> CanaryResult:
    try:
        _wake(client, wake_timeout_s, poll_s, sleep)
    except TimeoutError as exc:
        return CanaryResult(False, str(exc))

    payload = {
        "alert_id": f"canary-{uuid.uuid4()}",  # unique: a canary must never dedupe
        "service": "checkout",
        "signal": "error_rate",  # routes to the false_alarm scenario
        "summary": "synthetic canary alert",
    }
    body = json.dumps(payload).encode()
    timestamp = str(int(time.time()))
    response = client.post(
        "/webhooks/alert",
        content=body,
        headers={
            "Content-Type": "application/json",
            TIMESTAMP_HEADER: timestamp,
            SIGNATURE_HEADER: sign(secret, timestamp, body),
        },
    )
    if response.status_code != 202:
        return CanaryResult(False, f"webhook answered {response.status_code}: {response.text}")
    run_id = str(response.json()["run_id"])

    deadline = time.monotonic() + run_timeout_s
    while True:
        status = client.get(f"/api/runs/{run_id}").json()
        if status.get("finished"):
            break
        if time.monotonic() > deadline:
            return CanaryResult(False, f"run didn't finish within {run_timeout_s:.0f}s", run_id)
        sleep(poll_s)

    root_cause = (status.get("report") or {}).get("root_cause")
    if status.get("outcome") != "completed" or root_cause != "no_incident":
        return CanaryResult(
            False,
            f"expected completed/no_incident, got {status.get('outcome')}/{root_cause} "
            f"(error: {status.get('error')})",
            run_id,
        )
    return CanaryResult(True, "completed with no_incident", run_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=os.environ.get("OPSPILOT_URL", ""))
    parser.add_argument(
        "--skip-if-unconfigured",
        action="store_true",
        help="Exit 0 with a notice when URL/secret aren't set (before the first deploy).",
    )
    args = parser.parse_args(argv)
    secret = os.environ.get("OPSPILOT_WEBHOOK_SECRET", "")
    if not args.url or not secret:
        message = "canary not configured: set OPSPILOT_URL and OPSPILOT_WEBHOOK_SECRET"
        print(f"::notice::{message}" if args.skip_if_unconfigured else message)
        return 0 if args.skip_if_unconfigured else 2

    with httpx.Client(base_url=args.url.rstrip("/"), timeout=30) as client:
        result = run_canary(client, secret)
    print(f"canary {'OK' if result.ok else 'FAILED'}: {result.message} (run {result.run_id})")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
