from datetime import datetime
from typing import Any

import pymongo
from pymongo import AsyncMongoClient

from opspilot.store.models import ApprovalDoc, AuditDoc, EvalTrialDoc, RunDoc, SpanDoc


class MongoStore:
    """Real persistence via pymongo's native async client.

    Stable since PyMongo 4.13 -- Motor is no longer needed (PLAN.md's
    dependency notes call this out explicitly).
    """

    def __init__(self, uri: str, db_name: str = "opspilot") -> None:
        # [HARNESS:HITL] tz_aware=True -- BSON has no timezone concept, so a
        # naive client hands back naive datetimes for anything written as
        # datetime.now(UTC) (every timestamp field here). That's silent
        # until code actually subtracts one against a fresh aware
        # datetime.now(UTC) -- exactly what resume_react_graph does to
        # compute the approval_wait span duration (requested_at, read back
        # from Mongo, vs. decided_at, freshly created) -- raising
        # `TypeError: can't subtract offset-naive and offset-aware
        # datetimes`. Caught by the live Mongo verification for 2.5, not by
        # any test against MemoryStore (which never round-trips through BSON
        # and so never loses tzinfo in the first place).
        self._client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(uri, tz_aware=True)
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
        await self._db.eval_runs.create_index([("sweep_id", pymongo.ASCENDING)])
        await self._db.eval_runs.create_index([("case_id", pymongo.ASCENDING)])
        # unique=True is what makes claim_idempotency_key atomic across
        # requests, workers and instances: the database refuses the second insert.
        await self._db.webhook_deliveries.create_index([("key", pymongo.ASCENDING)], unique=True)

    async def insert_run(self, run: RunDoc) -> None:
        await self._db.runs.insert_one(run.model_dump())

    async def get_run(self, run_id: str) -> RunDoc | None:
        doc = await self._db.runs.find_one({"run_id": run_id})
        return RunDoc.model_validate(doc) if doc else None

    async def list_runs(self) -> list[RunDoc]:
        docs = [doc async for doc in self._db.runs.find(sort=[("created_at", pymongo.DESCENDING)])]
        return [RunDoc.model_validate(doc) for doc in docs]

    async def count_runs_since(self, since: datetime, mode: str) -> int:
        # Server-side count on the indexed created_at field -- no documents
        # come back over the wire, just the number.
        return await self._db.runs.count_documents({"mode": mode, "created_at": {"$gte": since}})

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

    async def claim_approval(self, approval_id: str, updates: dict[str, Any]) -> bool:
        # The filter does the check: only a still-pending doc matches, and
        # Mongo applies a single-document update atomically -- two racing
        # requests can't both match.
        result = await self._db.approvals.update_one(
            {"approval_id": approval_id, "status": "pending"}, {"$set": updates}
        )
        if result.matched_count == 1:
            return True
        if await self._db.approvals.count_documents({"approval_id": approval_id}, limit=1) == 0:
            raise KeyError(f"no approval {approval_id!r} to claim")
        return False

    async def claim_idempotency_key(self, key: str, run_id: str) -> str | None:
        try:
            await self._db.webhook_deliveries.insert_one({"key": key, "run_id": run_id})
            return None
        except pymongo.errors.DuplicateKeyError:
            existing = await self._db.webhook_deliveries.find_one({"key": key})
            return str(existing["run_id"]) if existing else None

    async def release_idempotency_key(self, key: str) -> None:
        await self._db.webhook_deliveries.delete_one({"key": key})

    async def list_pending_approvals(self) -> list[ApprovalDoc]:
        docs = [doc async for doc in self._db.approvals.find({"status": "pending"})]
        return [ApprovalDoc.model_validate(doc) for doc in docs]

    async def insert_eval_run(self, trial: EvalTrialDoc) -> None:
        await self._db.eval_runs.insert_one(trial.model_dump())

    async def list_eval_runs(self, sweep_id: str | None = None) -> list[EvalTrialDoc]:
        query = {} if sweep_id is None else {"sweep_id": sweep_id}
        docs = [
            doc
            async for doc in self._db.eval_runs.find(
                query, sort=[("case_id", pymongo.ASCENDING), ("trial", pymongo.ASCENDING)]
            )
        ]
        return [EvalTrialDoc.model_validate(doc) for doc in docs]

    async def update_eval_run(self, trial_id: str, updates: dict[str, Any]) -> None:
        result = await self._db.eval_runs.update_one({"trial_id": trial_id}, {"$set": updates})
        if result.matched_count == 0:
            raise KeyError(f"no eval trial {trial_id!r} to update")

    # [HARNESS:OBS] Real Mongo aggregation pipelines, not fetch-then-compute.
    # WHY: $percentile (MongoDB 7+, confirmed against the project's own
    # mongo:7 container rather than assumed) and $group push the computation
    # to the database -- the alternative (pulling every span/run doc back
    # and reducing in Python) doesn't scale past a small dev dataset and
    # defeats the point of using an aggregation database. MemoryStore
    # mirrors the same *results* in plain Python purely so tests don't need
    # Mongo running -- it is not meant to demonstrate the technique.
    # INTERVIEW: "Why $percentile with method: approximate instead of exact
    # sort+index?" -> approximate (t-digest) percentiles are the documented,
    # performant choice for this use case; exact percentiles require sorting
    # the whole collection.
    async def model_call_latency_percentiles(self) -> dict[str, float]:
        cursor = await self._db.spans.aggregate(
            [
                {"$match": {"kind": "model_call"}},
                {
                    "$group": {
                        "_id": None,
                        "pcts": {
                            "$percentile": {
                                "input": "$duration_ms",
                                "p": [0.5, 0.95],
                                "method": "approximate",
                            }
                        },
                    }
                },
            ]
        )
        docs = [doc async for doc in cursor]
        if not docs:
            return {"p50": 0.0, "p95": 0.0}
        p50, p95 = docs[0]["pcts"]
        return {"p50": p50, "p95": p95}

    async def tool_error_rates(self) -> dict[str, float]:
        cursor = await self._db.spans.aggregate(
            [
                {"$match": {"kind": "tool_call"}},
                {
                    "$group": {
                        "_id": "$name",
                        "total": {"$sum": 1},
                        "errors": {"$sum": {"$cond": [{"$eq": ["$status", "error"]}, 1, 0]}},
                    }
                },
            ]
        )
        docs = [doc async for doc in cursor]
        return {doc["_id"]: doc["errors"] / doc["total"] for doc in docs}

    async def outcomes_distribution(self) -> dict[str, int]:
        cursor = await self._db.runs.aggregate(
            [
                {"$match": {"outcome": {"$ne": None}}},
                {"$group": {"_id": "$outcome", "count": {"$sum": 1}}},
            ]
        )
        docs = [doc async for doc in cursor]
        return {doc["_id"]: doc["count"] for doc in docs}

    async def avg_cost_by_scenario(self) -> dict[str, float]:
        cursor = await self._db.runs.aggregate(
            [
                {"$match": {"cost_usd": {"$ne": None}}},
                {"$group": {"_id": "$scenario", "avg_cost": {"$avg": "$cost_usd"}}},
            ]
        )
        docs = [doc async for doc in cursor]
        return {doc["_id"]: doc["avg_cost"] for doc in docs}
