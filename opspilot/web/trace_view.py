from dataclasses import dataclass
from datetime import datetime

from opspilot.store.models import SpanDoc


@dataclass(frozen=True)
class WaterfallRow:
    depth: int
    kind: str
    name: str
    status: str
    duration_ms: float | None
    # Position on the run's timeline, as % of the whole run -- what makes it
    # a waterfall instead of a list: you *see* which step ate the time.
    offset_pct: float
    width_pct: float
    detail: str


def _detail(span: SpanDoc) -> str:
    """One short line of the attrs that matter most for each span kind
    (attr names as written by loops/graph.py and observability/instrumentation.py)."""
    a = span.attrs
    if span.kind == "model_call":
        return f"in={a.get('input_tokens', '?')} out={a.get('output_tokens', '?')}"
    if span.kind == "tool_call":
        size = f" {a['output_size']} chars" if "output_size" in a else ""
        return f"ok={a.get('ok')}{size}"
    if span.kind == "policy_check":
        # call_id -> None (allowed) or the denial / rejection reason.
        reasons = [r for r in (a.get("decisions") or {}).values() if r]
        return "allowed" if not reasons else f"blocked: {reasons[0][:80]}"
    return ""


# [HARNESS:OBS] The trace as a waterfall -- the run's anatomy at a glance.
# WHY: the same spans `opspilot trace` prints, laid out on a shared timeline
# (parent/child nesting = what caused what, bar offset/width = when and how
# long). Rendered from the store on every poll, so a run in progress fills
# in live and a paused run visibly stops at its policy check.
# INTERVIEW: "How do you debug an agent run?" -> a trace of every model
# call, policy decision and tool call, nested and timed, not just logs.
def build_waterfall(spans: list[SpanDoc], now: datetime) -> list[WaterfallRow]:
    if not spans:
        return []
    t0 = min(s.start for s in spans)
    t_end = max((s.end or now) for s in spans)
    total = max((t_end - t0).total_seconds(), 1e-6)

    children: dict[str | None, list[SpanDoc]] = {}
    for span in spans:
        children.setdefault(span.parent_id, []).append(span)
    for siblings in children.values():
        siblings.sort(key=lambda s: s.start)

    known_ids = {s.span_id for s in spans}
    # Roots: no parent, or a parent that isn't in this run's spans.
    roots = [s for s in spans if s.parent_id is None or s.parent_id not in known_ids]
    roots.sort(key=lambda s: s.start)

    rows: list[WaterfallRow] = []

    def visit(span: SpanDoc, depth: int) -> None:
        end = span.end or now  # still running -> extend to "now"
        rows.append(
            WaterfallRow(
                depth=depth,
                kind=span.kind,
                name=span.name,
                status=span.status,
                duration_ms=span.duration_ms,
                offset_pct=100 * (span.start - t0).total_seconds() / total,
                width_pct=max(0.5, 100 * (end - span.start).total_seconds() / total),
                detail=_detail(span),
            )
        )
        for child in children.get(span.span_id, []):
            visit(child, depth + 1)

    for root in roots:
        visit(root, 0)
    return rows
