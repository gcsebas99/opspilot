from datetime import datetime, timedelta

from opspilot.loops.react_raw import LoopEvent
from opspilot.observability.pricing import cost_usd
from opspilot.observability.tracer import Tracer, redact_if_large


# [HARNESS:OBS] Reconstructing spans from events instead of instrumenting the
# loop inline.
# WHY: Day 1's `on_event` callback is synchronous, and the loop's exit
# conditions never needed to know tracing exists -- making it async just to
# write spans inline would be a bigger, more invasive change than this
# sub-task calls for. Every value needed to reconstruct exact span timing
# (latency_ms, duration_ms) is already in the event payload, and the loop
# executes model/tool calls strictly sequentially (never concurrently) -- so
# a single post-hoc pass, walking the events in order and advancing a clock
# by each one's own reported duration, reproduces the real timeline exactly,
# not approximately.
# INTERVIEW: "How did you add tracing without rewriting the loop?" -> the
# loop already emitted step_start/model_call/tool_call/exit events for the
# CLI's live printer; tracing is a second consumer of that same stream,
# replayed into spans after the run completes rather than written inline.
async def record_loop_spans(
    tracer: Tracer, run_start: datetime, events: list[LoopEvent], model: str
) -> None:
    cursor = run_start
    for event in events:
        if event.type == "model_call":
            latency_ms = event.data["latency_ms"]
            usage = event.data["usage"]
            await tracer.record_span(
                "model_call",
                "model.create",
                start=cursor,
                duration_ms=latency_ms,
                status="ok",
                model=model,
                stop_reason=event.data["stop_reason"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cache_creation_input_tokens=usage["cache_creation_input_tokens"],
                cache_read_input_tokens=usage["cache_read_input_tokens"],
                cost_usd=cost_usd(
                    model,
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    cache_creation_input_tokens=usage["cache_creation_input_tokens"],
                    cache_read_input_tokens=usage["cache_read_input_tokens"],
                ),
            )
            cursor += timedelta(milliseconds=latency_ms)
        elif event.type == "tool_call":
            duration_ms = event.data["duration_ms"]
            await tracer.record_span(
                "tool_call",
                event.data["name"],
                start=cursor,
                duration_ms=duration_ms,
                status="ok" if event.data["ok"] else "error",
                tool=event.data["name"],
                args=redact_if_large(event.data["input"]),
                ok=event.data["ok"],
                truncated=event.data["truncated"],
                output_size=event.data["output_size"],
            )
            # [HARNESS:GUARD] A sibling of the tool_call span (this whole
            # reconstruction is flat, one level under "run" -- see this
            # module's WHY above), not nested under it -- the graph
            # strategy nests it instead, since its spans are written live
            # inside an open tool_call context. Same signal, two shapes.
            injection_patterns = event.data.get("injection_patterns") or []
            if injection_patterns:
                await tracer.record_span(
                    "guardrail",
                    event.data["name"],
                    start=cursor,
                    duration_ms=0.0,
                    tool=event.data["name"],
                    patterns=injection_patterns,
                )
            cursor += timedelta(milliseconds=duration_ms)
