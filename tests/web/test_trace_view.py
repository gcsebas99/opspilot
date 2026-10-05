from datetime import UTC, datetime, timedelta

from opspilot.store.models import SpanDoc
from opspilot.web.trace_view import build_waterfall

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _span(sid: str, kind: str, start_s: float, end_s: float | None, parent: str | None, **attrs):  # type: ignore[no-untyped-def]
    return SpanDoc(
        span_id=sid,
        run_id="r",
        parent_id=parent,
        kind=kind,  # type: ignore[arg-type]
        name=sid,
        start=T0 + timedelta(seconds=start_s),
        end=None if end_s is None else T0 + timedelta(seconds=end_s),
        duration_ms=None if end_s is None else (end_s - start_s) * 1000,
        attrs=attrs,
    )


def test_nesting_order_and_timeline_positions() -> None:
    spans = [
        _span("tool", "tool_call", 6, 10, "run", ok=True, output_size=42),
        _span("run", "run", 0, 10, None),
        _span("model", "model_call", 0, 5, "run", input_tokens=100, output_tokens=7),
        _span("policy", "policy_check", 5, 6, "run", decisions={"c1": None}),
    ]
    rows = build_waterfall(spans, now=T0 + timedelta(seconds=10))

    assert [(r.name, r.depth) for r in rows] == [
        ("run", 0),
        ("model", 1),
        ("policy", 1),
        ("tool", 1),
    ]
    model = rows[1]
    assert (model.offset_pct, model.width_pct) == (0.0, 50.0)
    assert rows[3].offset_pct == 60.0
    assert model.detail == "in=100 out=7"
    assert rows[2].detail == "allowed"
    assert rows[3].detail == "ok=True 42 chars"


def test_running_span_extends_to_now_and_blocked_policy_is_shown() -> None:
    spans = [
        _span("run", "run", 0, None, None),
        _span("policy", "policy_check", 2, 3, "run", decisions={"c1": "requires approval"}),
    ]
    rows = build_waterfall(spans, now=T0 + timedelta(seconds=4))

    assert rows[0].width_pct == 100.0  # open span runs to "now"
    assert rows[1].detail == "blocked: requires approval"


def test_empty() -> None:
    assert build_waterfall([], now=T0) == []
