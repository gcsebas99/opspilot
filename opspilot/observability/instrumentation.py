from datetime import datetime, timedelta

from opspilot.loops.react_raw import LoopEvent
from opspilot.observability.pricing import cost_usd
from opspilot.observability.tracer import Tracer, redact_if_large


# [HARNESS:OBS] Raw-loop spans are rebuilt from its event stream, after the run.
# WHY: the loop already emits events (for the CLI printer) with each step's own
# duration, and runs strictly sequentially -- so replaying them advances a clock
# exactly. Tracing becomes a second consumer, with no change to the loop itself.
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
            # [HARNESS:GUARD] Injection signal recorded as a guardrail span.
            # WHY: here it's a sibling of the tool_call span (this rebuilt trace is flat);
            # the graph strategy writes it live, nested under the call.
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
