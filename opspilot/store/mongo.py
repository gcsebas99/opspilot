from typing import Any

import pymongo
from pymongo import AsyncMongoClient

from opspilot.store.models import ApprovalDoc, AuditDoc, RunDoc, SpanDoc


class MongoStore:
    """Real persistence via pymongo's native async client.

    Stable since PyMongo 4.13 -- Motor is no longer needed (PLAN.md's
    dependency notes call this out explicitly).
    """

    def __init__(self, uri: str, db_name: str = "opspilot") -> None:
        self._client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(uri)
        self._db = self._client[db_name]

    async def close(self) -> None:
        await self._client.close()

    # [HARNESS:OBS] Index design follows query shape, not table shape.
    # WHY: `spans` is always queried scoped to one run, in time order (to
    # render a waterfall) -- a compound (run_id, start) index makes that a
    # single index scan. `runs` is browsed globally by recency or filtered
    # by scenario, so its indexes lead with those fields instead. Same
    # reasoning for audit_log (time-ordered, and per-run lookups) and
    # approvals (filtered by status for the pending queue, and per-run).
    # INTERVIEW: "Why is spans a separate collection with its own index
    # instead of embedding it in the run doc?" -> unbounded growth (a long
    # run can produce far more spans than fit under Mongo's 16MB document
    # limit) and a different, more frequent query pattern than the run
    # document itself.
    async def ensure_indexes(self) -> None:
        await self._db.runs.create_index([("created_at", pymongo.DESCENDING)])
        await self._db.runs.create_index([("scenario", pymongo.ASCENDING)])
        await self._db.spans.create_index(
            [("run_id", pymongo.ASCENDING), ("start", pymongo.ASCENDING)]
        )
        await self._db.audit_log.create_index([("ts", pymongo.ASCENDING)])
        await self._db.audit_log.create_index([("run_id", pymongo.ASCENDING)])
        await self._db.approvals.create_index([("status", pymongo.ASCENDING)])
        await self._db.approvals.create_index([("run_id", pymongo.ASCENDING)])

    async def insert_run(self, run: RunDoc) -> None:
        await self._db.runs.insert_one(run.model_dump())

    async def get_run(self, run_id: str) -> RunDoc | None:
        doc = await self._db.runs.find_one({"run_id": run_id})
        return RunDoc.model_validate(doc) if doc else None

    async def list_runs(self) -> list[RunDoc]:
        docs = [doc async for doc in self._db.runs.find(sort=[("created_at", pymongo.DESCENDING)])]
        return [RunDoc.model_validate(doc) for doc in docs]

    async def update_run(self, run_id: str, updates: dict[str, Any]) -> None:
        result = await self._db.runs.update_one({"run_id": run_id}, {"$set": updates})
        if result.matched_count == 0:
            raise KeyError(f"no run {run_id!r} to update")

    async def insert_span(self, span: SpanDoc) -> None:
        await self._db.spans.insert_one(span.model_dump())

    async def list_spans(self, run_id: str) -> list[SpanDoc]:
        docs = [
            doc
            async for doc in self._db.spans.find(
                {"run_id": run_id}, sort=[("start", pymongo.ASCENDING)]
            )
        ]
        return [SpanDoc.model_validate(doc) for doc in docs]

    async def insert_audit(self, entry: AuditDoc) -> None:
        await self._db.audit_log.insert_one(entry.model_dump())

    async def list_audit(self, run_id: str | None = None) -> list[AuditDoc]:
        query = {} if run_id is None else {"run_id": run_id}
        docs = [
            doc async for doc in self._db.audit_log.find(query, sort=[("ts", pymongo.ASCENDING)])
        ]
        return [AuditDoc.model_validate(doc) for doc in docs]

    async def get_last_audit(self) -> AuditDoc | None:
        doc = await self._db.audit_log.find_one(sort=[("_id", pymongo.DESCENDING)])
        return AuditDoc.model_validate(doc) if doc else None

    async def insert_approval(self, approval: ApprovalDoc) -> None:
        await self._db.approvals.insert_one(approval.model_dump())

    async def get_approval(self, approval_id: str) -> ApprovalDoc | None:
        doc = await self._db.approvals.find_one({"approval_id": approval_id})
        return ApprovalDoc.model_validate(doc) if doc else None

    async def update_approval(self, approval_id: str, updates: dict[str, Any]) -> None:
        result = await self._db.approvals.update_one(
            {"approval_id": approval_id}, {"$set": updates}
        )
        if result.matched_count == 0:
            raise KeyError(f"no approval {approval_id!r} to update")

    async def list_pending_approvals(self) -> list[ApprovalDoc]:
        docs = [doc async for doc in self._db.approvals.find({"status": "pending"})]
        return [ApprovalDoc.model_validate(doc) for doc in docs]
