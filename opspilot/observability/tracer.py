import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from opspilot.store.base import Store
from opspilot.store.models import SpanDoc

SpanKind = Literal[
    "run", "model_call", "tool_call", "policy_check", "approval_wait", "compaction", "guardrail"
]
SpanStatus = Literal["ok", "error"]

_current_span_id: ContextVar[str | None] = ContextVar("_current_span_id", default=None)


class SpanHandle:
    """Yielded by Tracer.span(). `start` lets a caller anchor further work
    (e.g. reconstructing nested spans) to this span's exact opening time
    without capturing a second, slightly-later timestamp of its own."""

    def __init__(self, start: datetime, attrs: dict[str, Any]) -> None:
        self.start = start
        self.attrs = attrs

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value


class Tracer:
    """Writes SpanDoc records to a Store, threading parent/child
    relationships through a contextvar instead of explicit parameters --
    any code running "underneath" an open span (including in a different
    coroutine awaited from inside it) sees that span as its parent.
    """

    def __init__(self, store: Store, run_id: str) -> None:
        self._store = store
        self.run_id = run_id

    @asynccontextmanager
    async def span(self, kind: SpanKind, name: str, **attrs: Any) -> AsyncIterator[SpanHandle]:
        span_id = str(uuid.uuid4())
        parent_id = _current_span_id.get()
        start = datetime.now(UTC)
        start_monotonic = time.monotonic()
        token = _current_span_id.set(span_id)
        handle = SpanHandle(start=start, attrs=dict(attrs))
        status: SpanStatus = "ok"
        try:
            yield handle
        except Exception:
            status = "error"
            raise
        finally:
            _current_span_id.reset(token)
            duration_ms = (time.monotonic() - start_monotonic) * 1000
            await self._store.insert_span(
                SpanDoc(
                    span_id=span_id,
                    run_id=self.run_id,
                    parent_id=parent_id,
                    kind=kind,
                    name=name,
                    start=start,
                    end=start + timedelta(milliseconds=duration_ms),
                    duration_ms=duration_ms,
                    status=status,
                    attrs=handle.attrs,
                )
            )

    async def record_span(
        self,
        kind: SpanKind,
        name: str,
        *,
        start: datetime,
        duration_ms: float,
        status: SpanStatus = "ok",
        **attrs: Any,
    ) -> None:
        """Write a span whose start/end already happened -- e.g. one
        reconstructed from a Day 1 loop event, which reports latency_ms
        after the fact rather than being wrapped in `async with
        tracer.span(...)`. Still nests under the currently-open span.
        """
        parent_id = _current_span_id.get()
        await self._store.insert_span(
            SpanDoc(
                span_id=str(uuid.uuid4()),
                run_id=self.run_id,
                parent_id=parent_id,
                kind=kind,
                name=name,
                start=start,
                end=start + timedelta(milliseconds=duration_ms),
                duration_ms=duration_ms,
                status=status,
                attrs=dict(attrs),
            )
        )
