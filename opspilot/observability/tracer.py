import json
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

_MAX_ARGS_CHARS = 500


def redact_if_large(args: dict[str, Any], max_chars: int = _MAX_ARGS_CHARS) -> dict[str, Any] | str:
    """Shrink a tool's args to a placeholder if too large for a span's attrs
    -- shared by the raw loop's event replay (instrumentation.py) and the
    graph's tools node, so both strategies redact identically."""
    serialized = json.dumps(args, sort_keys=True, default=str)
    if len(serialized) <= max_chars:
        return args
    return f"<redacted: {len(serialized)} chars>"


class SpanHandle:
    """Yielded by Tracer.span(). `start` lets a caller anchor further work
    (e.g. reconstructing nested spans) to this span's exact opening time
    without capturing a second, slightly-later timestamp of its own.

    `status` defaults to "ok" and is only flipped to "error" automatically
    when an exception propagates out of the block -- set it explicitly
    (`handle.status = "error"`) for a business-logic failure (e.g. a tool
    that returned ok=False without raising) that should still show up as a
    failed span.
    """

    def __init__(self, start: datetime, attrs: dict[str, Any]) -> None:
        self.start = start
        self.attrs = attrs
        self.status: SpanStatus = "ok"

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
        try:
            yield handle
        except Exception:
            handle.status = "error"
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
                    status=handle.status,
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
