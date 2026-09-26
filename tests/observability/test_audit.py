from datetime import UTC, datetime

from opspilot.loops.react_raw import LoopEvent
from opspilot.observability.audit import (
    record_audit,
    record_loop_audit,
    record_prompt_version_change_if_needed,
    verify_chain,
)
from opspilot.store.memory import MemoryStore
from opspilot.store.models import RunDoc


async def _seed_chain(store: MemoryStore, n: int) -> None:
    for i in range(n):
        await record_audit(
            store,
            actor="operator",
            action="permission_denied",
            target=f"tool_{i}",
            decision="denied",
            run_id="run-1",
            prompt_version="v1",
            model="claude-sonnet-5",
            ts=datetime(2025, 1, 1, tzinfo=UTC),
        )


async def test_record_audit_chains_hashes_across_entries() -> None:
    store = MemoryStore()

    first = await record_audit(
        store,
        actor="operator",
        action="permission_denied",
        target="restart_service",
        decision="denied",
        run_id="run-1",
        prompt_version="v1",
        model="claude-sonnet-5",
        ts=datetime(2025, 1, 1, tzinfo=UTC),
    )
    second = await record_audit(
        store,
        actor="admin",
        action="destructive_tool_executed",
        target="rollback_config",
        decision="executed",
        run_id="run-1",
        prompt_version="v1",
        model="claude-sonnet-5",
        ts=datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert first.prev_hash is None
    assert first.hash is not None
    assert second.prev_hash == first.hash
    assert second.hash != first.hash


async def test_verify_chain_ok_for_untampered_entries() -> None:
    store = MemoryStore()
    await _seed_chain(store, 4)

    result = verify_chain(await store.list_audit())

    assert result.ok is True
    assert result.total_entries == 4
    assert result.broken_at_index is None


async def test_verify_chain_empty_is_ok() -> None:
    assert verify_chain([]).ok is True


async def test_verify_chain_detects_tampered_field_at_exact_index() -> None:
    # The spec's accept criterion: modify a record -> verify fails at that index.
    store = MemoryStore()
    await _seed_chain(store, 4)
    entries = await store.list_audit()

    tampered = list(entries)
    tampered[2] = tampered[2].model_copy(update={"decision": "approved (tampered)"})

    result = verify_chain(tampered)

    assert result.ok is False
    assert result.broken_at_index == 2


async def test_verify_chain_detects_reordering() -> None:
    store = MemoryStore()
    await _seed_chain(store, 3)
    entries = await store.list_audit()

    reordered = [entries[0], entries[2], entries[1]]

    result = verify_chain(reordered)

    assert result.ok is False
    assert result.broken_at_index == 1


async def test_record_loop_audit_logs_permission_denial() -> None:
    store = MemoryStore()
    events = [
        LoopEvent(
            type="tool_call",
            data={
                "name": "restart_service",
                "duration_ms": 1.0,
                "ok": False,
                "risk": "destructive",
                "policy_decision": "deny",
                "denial_reason": "viewer role cannot run 'restart_service'",
            },
        )
    ]

    await record_loop_audit(
        store,
        datetime(2025, 1, 1, tzinfo=UTC),
        events,
        run_id="run-1",
        role="viewer",
        prompt_version="v1",
        model="claude-sonnet-5",
    )

    entries = await store.list_audit("run-1")
    assert len(entries) == 1
    assert entries[0].action == "permission_denied"
    assert entries[0].target == "restart_service"
    assert entries[0].decision == "viewer role cannot run 'restart_service'"


async def test_record_loop_audit_logs_destructive_execution() -> None:
    store = MemoryStore()
    events = [
        LoopEvent(
            type="tool_call",
            data={
                "name": "rollback_config",
                "duration_ms": 1.0,
                "ok": True,
                "risk": "destructive",
                "policy_decision": "allow",
                "denial_reason": None,
            },
        )
    ]

    await record_loop_audit(
        store,
        datetime(2025, 1, 1, tzinfo=UTC),
        events,
        run_id="run-1",
        role="admin",
        prompt_version="v1",
        model="claude-sonnet-5",
    )

    entries = await store.list_audit("run-1")
    assert len(entries) == 1
    assert entries[0].action == "destructive_tool_executed"
    assert entries[0].decision == "executed"


async def test_record_loop_audit_ignores_allowed_read_calls() -> None:
    store = MemoryStore()
    events = [
        LoopEvent(
            type="tool_call",
            data={
                "name": "list_services",
                "duration_ms": 1.0,
                "ok": True,
                "risk": "read",
                "policy_decision": "allow",
                "denial_reason": None,
            },
        )
    ]

    await record_loop_audit(
        store,
        datetime(2025, 1, 1, tzinfo=UTC),
        events,
        run_id="run-1",
        role="viewer",
        prompt_version="v1",
        model="claude-sonnet-5",
    )

    assert await store.list_audit("run-1") == []


async def test_record_prompt_version_change_if_needed_logs_on_change() -> None:
    store = MemoryStore()
    await store.insert_run(
        RunDoc(
            run_id="run-0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            scenario="checkout_pool_exhaustion",
            seed=42,
            role="viewer",
            model="claude-sonnet-5",
            prompt_version="v1",
            strategy="raw",
        )
    )

    await record_prompt_version_change_if_needed(
        store,
        run_id="run-1",
        prompt_version="v2",
        model="claude-sonnet-5",
        ts=datetime(2025, 1, 2, tzinfo=UTC),
    )

    entries = await store.list_audit()
    assert len(entries) == 1
    assert entries[0].action == "prompt_version_changed"
    assert entries[0].decision == "v1 -> v2"


async def test_record_prompt_version_change_if_needed_silent_when_unchanged() -> None:
    store = MemoryStore()
    await store.insert_run(
        RunDoc(
            run_id="run-0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            scenario="checkout_pool_exhaustion",
            seed=42,
            role="viewer",
            model="claude-sonnet-5",
            prompt_version="v1",
            strategy="raw",
        )
    )

    await record_prompt_version_change_if_needed(
        store,
        run_id="run-1",
        prompt_version="v1",
        model="claude-sonnet-5",
        ts=datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert await store.list_audit() == []


async def test_record_prompt_version_change_if_needed_silent_on_first_ever_run() -> None:
    store = MemoryStore()

    await record_prompt_version_change_if_needed(
        store,
        run_id="run-1",
        prompt_version="v1",
        model="claude-sonnet-5",
        ts=datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert await store.list_audit() == []
