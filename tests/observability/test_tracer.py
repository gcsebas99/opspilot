from datetime import UTC, datetime

import pytest

from opspilot.observability.tracer import Tracer
from opspilot.store.memory import MemoryStore


async def test_span_writes_a_span_doc() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")

    async with tracer.span("tool_call", "grep_logs", tool="grep_logs") as handle:
        handle.set_attr("extra", "value")

    spans = await store.list_spans("run-1")
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "tool_call"
    assert span.name == "grep_logs"
    assert span.parent_id is None
    assert span.status == "ok"
    assert span.attrs == {"tool": "grep_logs", "extra": "value"}
    assert span.duration_ms is not None
    assert span.duration_ms >= 0


async def test_span_marks_error_status_on_exception() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")

    with pytest.raises(ValueError, match="kaboom"):
        async with tracer.span("tool_call", "boom"):
            raise ValueError("kaboom")

    spans = await store.list_spans("run-1")
    assert spans[0].status == "error"


async def test_nested_spans_set_parent_id() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")

    async with tracer.span("run", "react_loop"), tracer.span("tool_call", "grep_logs"):
        pass

    spans = await store.list_spans("run-1")
    by_name = {s.name: s for s in spans}
    assert by_name["grep_logs"].parent_id == by_name["react_loop"].span_id


async def test_record_span_writes_directly_with_given_timing() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")
    start = datetime(2025, 1, 1, tzinfo=UTC)

    await tracer.record_span(
        "model_call", "model.create", start=start, duration_ms=250.0, stop_reason="tool_use"
    )

    spans = await store.list_spans("run-1")
    assert len(spans) == 1
    span = spans[0]
    assert span.start == start
    assert span.duration_ms == 250.0
    assert span.attrs == {"stop_reason": "tool_use"}


async def test_record_span_nests_under_currently_open_span() -> None:
    store = MemoryStore()
    tracer = Tracer(store, run_id="run-1")

    async with tracer.span("run", "react_loop") as run_span:
        await tracer.record_span(
            "model_call", "model.create", start=run_span.start, duration_ms=10.0
        )

    spans = await store.list_spans("run-1")
    by_name = {s.name: s for s in spans}
    assert by_name["model.create"].parent_id == by_name["react_loop"].span_id
