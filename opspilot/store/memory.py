from typing import Any

from opspilot.store.models import ApprovalDoc, AuditDoc, RunDoc, SpanDoc


class MemoryStore:
    """In-memory Store. No I/O -- keeps tests fast and lets the CLI run
    without Mongo. Data does not survive past the process.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunDoc] = {}
        self._spans: list[SpanDoc] = []
        self._audit: list[AuditDoc] = []
        self._approvals: dict[str, ApprovalDoc] = {}

    async def ensure_indexes(self) -> None:
        pass  # nothing to index on plain dicts/lists

    async def insert_run(self, run: RunDoc) -> None:
        self._runs[run.run_id] = run

    async def get_run(self, run_id: str) -> RunDoc | None:
        return self._runs.get(run_id)

    async def list_runs(self) -> list[RunDoc]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    async def update_run(self, run_id: str, updates: dict[str, Any]) -> None:
        existing = self._runs.get(run_id)
        if existing is None:
            raise KeyError(f"no run {run_id!r} to update")
        self._runs[run_id] = existing.model_copy(update=updates)

    async def insert_span(self, span: SpanDoc) -> None:
        self._spans.append(span)

    async def list_spans(self, run_id: str) -> list[SpanDoc]:
        return sorted((s for s in self._spans if s.run_id == run_id), key=lambda s: s.start)

    async def insert_audit(self, entry: AuditDoc) -> None:
        self._audit.append(entry)

    async def list_audit(self, run_id: str | None = None) -> list[AuditDoc]:
        entries = self._audit if run_id is None else [a for a in self._audit if a.run_id == run_id]
        return sorted(entries, key=lambda a: a.ts)

    async def get_last_audit(self) -> AuditDoc | None:
        # Insertion order (append) doubles as the chain order here, matching
        # MongoStore's "last inserted" semantics (sorted by _id there).
        return self._audit[-1] if self._audit else None

    async def insert_approval(self, approval: ApprovalDoc) -> None:
        self._approvals[approval.approval_id] = approval

    async def get_approval(self, approval_id: str) -> ApprovalDoc | None:
        return self._approvals.get(approval_id)

    async def update_approval(self, approval_id: str, updates: dict[str, Any]) -> None:
        existing = self._approvals.get(approval_id)
        if existing is None:
            raise KeyError(f"no approval {approval_id!r} to update")
        self._approvals[approval_id] = existing.model_copy(update=updates)

    async def list_pending_approvals(self) -> list[ApprovalDoc]:
        return [a for a in self._approvals.values() if a.status == "pending"]

    async def model_call_latency_percentiles(self) -> dict[str, float]:
        durations = sorted(
            s.duration_ms
            for s in self._spans
            if s.kind == "model_call" and s.duration_ms is not None
        )
        return {"p50": _percentile(durations, 0.5), "p95": _percentile(durations, 0.95)}

    async def tool_error_rates(self) -> dict[str, float]:
        totals: dict[str, int] = {}
        errors: dict[str, int] = {}
        for span in self._spans:
            if span.kind != "tool_call":
                continue
            totals[span.name] = totals.get(span.name, 0) + 1
            if span.status == "error":
                errors[span.name] = errors.get(span.name, 0) + 1
        return {tool: errors.get(tool, 0) / total for tool, total in totals.items()}

    async def outcomes_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for run in self._runs.values():
            if run.outcome is None:
                continue
            counts[run.outcome] = counts.get(run.outcome, 0) + 1
        return counts

    async def avg_cost_by_scenario(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for run in self._runs.values():
            if run.cost_usd is None:
                continue
            totals.setdefault(run.scenario, []).append(run.cost_usd)
        return {scenario: sum(costs) / len(costs) for scenario, costs in totals.items()}


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile. Not the same algorithm MongoStore's
    approximate $percentile uses -- each backend only needs to be
    internally consistent, not bit-identical to the other.
    """
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round(p * (len(sorted_values) - 1)))
    return sorted_values[index]
