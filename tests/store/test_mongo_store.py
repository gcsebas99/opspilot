import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from pymongo import AsyncMongoClient

from opspilot.store.models import RunDoc, SpanDoc
from opspilot.store.mongo import MongoStore

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
TEST_DB = "opspilot_test"

pytestmark = pytest.mark.integration


async def _mongo_reachable() -> bool:
    client: AsyncMongoClient[dict] = AsyncMongoClient(MONGO_URI, serverSelectionTimeoutMS=500)
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False
    finally:
        await client.close()


@pytest.fixture
async def store() -> AsyncIterator[MongoStore]:
    if not await _mongo_reachable():
        pytest.skip(f"no reachable MongoDB at {MONGO_URI}")

    mongo_store = MongoStore(MONGO_URI, db_name=TEST_DB)
    try:
        yield mongo_store
    finally:
        await mongo_store.close()
        cleanup_client: AsyncMongoClient[dict] = AsyncMongoClient(MONGO_URI)
        await cleanup_client.drop_database(TEST_DB)
        await cleanup_client.close()


async def test_mongo_store_run_roundtrip_and_indexes(store: MongoStore) -> None:
    await store.ensure_indexes()

    run = RunDoc(
        run_id="run-mongo-1",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        scenario="checkout_pool_exhaustion",
        seed=42,
        role="operator",
        model="claude-opus-5",
        prompt_version="abc123",
        strategy="raw",
    )
    await store.insert_run(run)

    fetched = await store.get_run("run-mongo-1")
    assert fetched is not None
    assert fetched.scenario == "checkout_pool_exhaustion"

    await store.update_run("run-mongo-1", {"outcome": "completed"})
    updated = await store.get_run("run-mongo-1")
    assert updated is not None
    assert updated.outcome == "completed"

    runs = await store.list_runs()
    assert any(r.run_id == "run-mongo-1" for r in runs)


async def test_mongo_store_aggregations(store: MongoStore) -> None:
    # Real Mongo aggregation pipelines ($percentile, $group) -- confirms the
    # syntax is actually valid against the running mongo:7, not just typed
    # correctly.
    await store.insert_run(
        RunDoc(
            run_id="run-agg-1",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            scenario="checkout_pool_exhaustion",
            seed=42,
            role="operator",
            model="claude-opus-5",
            prompt_version="abc123",
            strategy="raw",
            outcome="completed",
            cost_usd=0.20,
        )
    )
    await store.insert_span(
        SpanDoc(
            span_id="span-1",
            run_id="run-agg-1",
            kind="model_call",
            name="model.create",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            duration_ms=42.0,
        )
    )
    await store.insert_span(
        SpanDoc(
            span_id="span-2",
            run_id="run-agg-1",
            kind="tool_call",
            name="grep_logs",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            status="error",
        )
    )

    percentiles = await store.model_call_latency_percentiles()
    assert percentiles["p50"] > 0

    error_rates = await store.tool_error_rates()
    assert error_rates["grep_logs"] == 1.0

    outcomes = await store.outcomes_distribution()
    assert outcomes["completed"] == 1

    avg_cost = await store.avg_cost_by_scenario()
    assert avg_cost["checkout_pool_exhaustion"] == pytest.approx(0.20)
