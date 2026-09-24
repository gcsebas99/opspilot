from datetime import UTC, datetime, timedelta
from typing import Any

from opspilot.loops.react_raw import LoopEvent
from opspilot.observability.instrumentation import record_loop_spans
from opspilot.observability.tracer import Tracer
from opspilot.store.memory import MemoryStore


def _model_call_event(
    latency_ms: float, input_tokens: int = 100, output_tokens: int = 50
) -> LoopEvent:
    return LoopEvent(
        type="model_call",
        data={
            "step": 1,
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            "latency_ms": latency_ms,
        },
    )


def _tool_call_event(
    name: str,
    duration_ms: float,
    ok: bool = True,
    output_size: int = 42,
    input_: dict[str, Any] | None = None,
) -> LoopEvent:
    return LoopEvent(
        type="tool_call",
        data={
            "step": 1,
            "name": name,
            "input": input_ if input_ is not None else {"service": "checkout"},
            "ok": ok,
            "duration_ms": duration_ms,
            "truncated": False,
            "output_size": output_size,
        },
    )


async def test_record_loop_spans_writes_model_and_tool_spans_in_order() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")
    run_start = datetime(2025, 1, 1, tzinfo=UTC)
    events = [
        LoopEvent(type="step_start", data={"step": 1}),
        _model_call_event(latency_ms=100.0),
        _tool_call_event("grep_logs", duration_ms=20.0),
        LoopEvent(type="exit", data={"outcome": "completed", "steps": 1}),
    ]

    await record_loop_spans(tracer, run_start, events, model="claude-sonnet-5")

    spans = await store.list_spans("run-1")
    assert [s.kind for s in spans] == ["model_call", "tool_call"]

    model_span, tool_span = spans
    assert model_span.start == run_start
    assert model_span.duration_ms == 100.0
    assert model_span.attrs["cost_usd"] > 0
    assert model_span.attrs["stop_reason"] == "tool_use"

    # the tool span starts exactly where the model span ended
    assert tool_span.start == run_start + timedelta(milliseconds=100.0)
    assert tool_span.duration_ms == 20.0
    assert tool_span.attrs["tool"] == "grep_logs"
    assert tool_span.attrs["output_size"] == 42


async def test_record_loop_spans_marks_error_status_for_failed_tool_calls() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")
    events = [_tool_call_event("rollback_config", duration_ms=5.0, ok=False)]

    await record_loop_spans(
        tracer, datetime(2025, 1, 1, tzinfo=UTC), events, model="claude-sonnet-5"
    )

    spans = await store.list_spans("run-1")
    assert spans[0].status == "error"


async def test_record_loop_spans_redacts_large_args() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")
    huge_input = {"evidence": ["x" * 1000]}
    event = _tool_call_event("submit_report", duration_ms=1.0, input_=huge_input)

    await record_loop_spans(
        tracer, datetime(2025, 1, 1, tzinfo=UTC), [event], model="claude-sonnet-5"
    )

    spans = await store.list_spans("run-1")
    assert isinstance(spans[0].attrs["args"], str)
    assert spans[0].attrs["args"].startswith("<redacted")


async def test_record_loop_spans_keeps_small_args_as_dict() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")
    events = [_tool_call_event("read_config", duration_ms=1.0, input_={"service": "checkout"})]

    await record_loop_spans(
        tracer, datetime(2025, 1, 1, tzinfo=UTC), events, model="claude-sonnet-5"
    )

    spans = await store.list_spans("run-1")
    assert spans[0].attrs["args"] == {"service": "checkout"}


async def test_record_loop_spans_nests_under_an_active_parent_span() -> None:
    # Regression test: record_loop_spans() must be called from *inside* the
    # `async with tracer.span("run", ...)` block. Calling it after that
    # block exits silently produces a flat trace -- the contextvar the
    # nesting relies on is already reset by then. This test wraps it
    # correctly and asserts the reconstructed spans actually nest.
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")
    events = [
        _model_call_event(latency_ms=10.0),
        _tool_call_event("list_services", duration_ms=1.0),
    ]

    async with tracer.span("run", "react_loop") as run_span:
        await record_loop_spans(tracer, run_span.start, events, model="claude-sonnet-5")

    spans = await store.list_spans("run-1")
    by_kind = {s.kind: s for s in spans}
    run_span_id = by_kind["run"].span_id
    assert by_kind["model_call"].parent_id == run_span_id
    assert by_kind["tool_call"].parent_id == run_span_id
