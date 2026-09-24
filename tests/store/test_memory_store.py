from datetime import UTC, datetime

import pytest

from opspilot.store.memory import MemoryStore
from opspilot.store.models import ApprovalDoc, AuditDoc, RunDoc, SpanDoc


def _run(run_id: str = "run-1", created_at: datetime | None = None) -> RunDoc:
    return RunDoc(
        run_id=run_id,
        created_at=created_at or datetime(2025, 1, 1, tzinfo=UTC),
        scenario="checkout_pool_exhaustion",
        seed=42,
        role="operator",
        model="claude-opus-5",
        prompt_version="abc123",
        strategy="raw",
    )


def _span(span_id: str, run_id: str = "run-1", start: datetime | None = None) -> SpanDoc:
    return SpanDoc(
        span_id=span_id,
        run_id=run_id,
        kind="tool_call",
        name="grep_logs",
        start=start or datetime(2025, 1, 1, tzinfo=UTC),
    )


def _audit(run_id: str = "run-1", ts: datetime | None = None) -> AuditDoc:
    return AuditDoc(
        ts=ts or datetime(2025, 1, 1, tzinfo=UTC),
        actor="operator",
        action="rollback_config",
        target="checkout",
        decision="allow",
        run_id=run_id,
        prompt_version="abc123",
        model="claude-opus-5",
    )


def _approval(approval_id: str = "appr-1", run_id: str = "run-1") -> ApprovalDoc:
    return ApprovalDoc(
        approval_id=approval_id,
        run_id=run_id,
        tool="rollback_config",
        args={"service": "checkout", "version": 12},
        reason="destructive action requires approval",
        requested_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


async def test_ensure_indexes_is_a_noop(store: MemoryStore) -> None:
    await store.ensure_indexes()  # must not raise


async def test_insert_and_get_run(store: MemoryStore) -> None:
    await store.insert_run(_run())

    fetched = await store.get_run("run-1")

    assert fetched is not None
    assert fetched.scenario == "checkout_pool_exhaustion"


async def test_get_run_missing_returns_none(store: MemoryStore) -> None:
    assert await store.get_run("nope") is None


async def test_list_runs_sorted_newest_first(store: MemoryStore) -> None:
    await store.insert_run(_run("run-1", created_at=datetime(2025, 1, 1, tzinfo=UTC)))
    await store.insert_run(_run("run-2", created_at=datetime(2025, 1, 2, tzinfo=UTC)))

    runs = await store.list_runs()

    assert [r.run_id for r in runs] == ["run-2", "run-1"]


async def test_update_run(store: MemoryStore) -> None:
    await store.insert_run(_run())

    await store.update_run("run-1", {"outcome": "completed", "steps": 5})

    fetched = await store.get_run("run-1")
    assert fetched is not None
    assert fetched.outcome == "completed"
    assert fetched.steps == 5


async def test_update_run_missing_raises(store: MemoryStore) -> None:
    with pytest.raises(KeyError):
        await store.update_run("nope", {"outcome": "completed"})


async def test_insert_and_list_spans_scoped_to_run_ordered_by_start(store: MemoryStore) -> None:
    await store.insert_span(_span("s2", start=datetime(2025, 1, 1, 0, 1, tzinfo=UTC)))
    await store.insert_span(_span("s1", start=datetime(2025, 1, 1, 0, 0, tzinfo=UTC)))
    await store.insert_span(_span("other", run_id="run-2"))

    spans = await store.list_spans("run-1")

    assert [s.span_id for s in spans] == ["s1", "s2"]


async def test_insert_and_list_audit_all_and_scoped(store: MemoryStore) -> None:
    await store.insert_audit(_audit("run-1", ts=datetime(2025, 1, 1, 0, 0, tzinfo=UTC)))
    await store.insert_audit(_audit("run-2", ts=datetime(2025, 1, 1, 0, 1, tzinfo=UTC)))

    all_entries = await store.list_audit()
    scoped = await store.list_audit("run-1")

    assert len(all_entries) == 2
    assert len(scoped) == 1
    assert scoped[0].run_id == "run-1"


async def test_get_last_audit(store: MemoryStore) -> None:
    assert await store.get_last_audit() is None

    await store.insert_audit(_audit(ts=datetime(2025, 1, 1, tzinfo=UTC)))
    await store.insert_audit(_audit(ts=datetime(2025, 1, 2, tzinfo=UTC)))

    last = await store.get_last_audit()

    assert last is not None
    assert last.ts == datetime(2025, 1, 2, tzinfo=UTC)


async def test_insert_and_get_approval(store: MemoryStore) -> None:
    await store.insert_approval(_approval())

    fetched = await store.get_approval("appr-1")

    assert fetched is not None
    assert fetched.status == "pending"


async def test_update_approval(store: MemoryStore) -> None:
    await store.insert_approval(_approval())

    await store.update_approval("appr-1", {"status": "approved", "approver": "sebas"})

    fetched = await store.get_approval("appr-1")
    assert fetched is not None
    assert fetched.status == "approved"
    assert fetched.approver == "sebas"


async def test_update_approval_missing_raises(store: MemoryStore) -> None:
    with pytest.raises(KeyError):
        await store.update_approval("nope", {"status": "approved"})


async def test_list_pending_approvals(store: MemoryStore) -> None:
    await store.insert_approval(_approval("appr-1"))
    await store.insert_approval(_approval("appr-2"))
    await store.update_approval("appr-2", {"status": "approved"})

    pending = await store.list_pending_approvals()

    assert [a.approval_id for a in pending] == ["appr-1"]
