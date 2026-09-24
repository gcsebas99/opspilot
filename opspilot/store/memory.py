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
