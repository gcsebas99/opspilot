import asyncio
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from opspilot.config import Settings
from opspilot.loops.graph import ApprovalAlreadyDecided
from opspilot.models.factory import (
    DEMO_CASSETTE_ROOT,
    LiveCallRefused,
    demo_cassette_path,
    recorded_demo_paths,
)
from opspilot.runs import RunHandle, create_run, open_run, resume_graph, run_raw, start_graph
from opspilot.store.memory import MemoryStore

# Replay against the committed demo cassettes: real recorded model output,
# no key, no network (tests/conftest.py blocks any real HTTP).
_SETTINGS = Settings.model_validate({"ANTHROPIC_API_KEY": "", "OPSPILOT_MODEL": "claude-haiku-4-5"})


async def _operator_run(tmp_path: Path, store: MemoryStore) -> RunHandle:
    return await create_run(
        store,
        _SETTINGS,
        scenario="checkout_pool_exhaustion",
        seed=42,
        role="operator",
        strategy="graph",
        mode="replay",
        cassette=demo_cassette_path("checkout_pool_exhaustion", 42, "operator"),
        sandbox_dir=tmp_path / "run",
    )


async def test_create_run_persists_rundoc_with_sandbox_dir(tmp_path: Path) -> None:
    store = MemoryStore()
    handle = await _operator_run(tmp_path, store)

    run = await store.get_run(handle.run.run_id)
    assert run is not None
    assert run.sandbox_dir == str(tmp_path / "run")
    assert run.mode == "replay" and run.cassette is not None
    assert (tmp_path / "run" / "state.json").exists()


async def test_create_run_unknown_scenario_raises_keyerror(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        await create_run(
            MemoryStore(),
            _SETTINGS,
            scenario="nope",
            seed=1,
            role="viewer",
            strategy="graph",
            mode="replay",
            cassette=None,
            sandbox_dir=tmp_path,
        )


async def test_pause_then_resume_from_a_reopened_handle(tmp_path: Path) -> None:
    """Start -> pause for approval -> rebuild the handle from the store (as
    another request/process would) -> approve -> completed, on one sandbox."""
    store = MemoryStore()
    checkpointer = InMemorySaver()
    handle = await _operator_run(tmp_path, store)

    paused = await start_graph(handle, store, checkpointer)
    assert paused.outcome == "awaiting_approval"
    assert paused.pending_approval is not None
    assert paused.pending_approval["tool"] == "rollback_config"
    stored = await store.get_run(handle.run.run_id)
    assert stored is not None
    assert stored.outcome == "awaiting_approval" and stored.finished_at is None

    reopened = await open_run(store, _SETTINGS, handle.run.run_id)
    assert reopened.sandbox.root == handle.sandbox.root
    done = await resume_graph(
        reopened, store, checkpointer, {"decision": "approve", "approver": "test"}
    )

    assert done.outcome == "completed"
    assert reopened.sandbox.facts()["checkout.config_version"] == 12  # really rolled back
    stored = await store.get_run(handle.run.run_id)
    assert stored is not None
    assert stored.finished_at is not None and stored.cost_usd is not None


async def test_raw_run_through_the_service(tmp_path: Path) -> None:
    store = MemoryStore()
    handle = await create_run(
        store,
        _SETTINGS,
        scenario="false_alarm",
        seed=42,
        role="viewer",
        strategy="raw",
        mode="replay",
        cassette=demo_cassette_path("false_alarm", 42, "viewer"),
        sandbox_dir=tmp_path / "raw",
    )
    seen: list[str] = []

    result = await run_raw(handle, store, on_event=lambda e: seen.append(e.type))

    assert result.outcome == "completed"
    assert result.report is not None and result.report["root_cause"] == "no_incident"
    assert "exit" in seen  # the caller's event hook still fires
    assert await store.list_spans(handle.run.run_id)  # trace recorded


async def test_resume_refuses_live_without_a_key(tmp_path: Path) -> None:
    """A run started live must not resume in a process with no key -- and a
    replay run must never quietly become a live one."""
    store = MemoryStore()
    handle = await _operator_run(tmp_path, store)
    await store.update_run(handle.run.run_id, {"mode": "live"})
    reopened = await open_run(store, _SETTINGS, handle.run.run_id)

    with pytest.raises(LiveCallRefused, match="ANTHROPIC_API_KEY is empty"):
        reopened.chat_model()


async def test_open_run_unknown_id() -> None:
    with pytest.raises(KeyError):
        await open_run(MemoryStore(), _SETTINGS, "missing")


# --- recorded demo catalog ---


def test_catalog_lists_committed_demo_cassettes() -> None:
    paths = {(p.scenario, p.seed, p.role) for p in recorded_demo_paths()}
    assert ("checkout_pool_exhaustion", 42, "operator") in paths
    for p in recorded_demo_paths():
        assert p.path == demo_cassette_path(p.scenario, p.seed, p.role)
        assert p.path.parent == DEMO_CASSETTE_ROOT


def test_catalog_ignores_junk_unknown_scenarios_and_roles(tmp_path: Path) -> None:
    for name in [
        "false_alarm-s7-viewer.jsonl",  # valid
        "false_alarm-s7-system.jsonl",  # valid: webhook runs (the UI filters it out)
        "false_alarm-s7-root.jsonl",  # not a role
        "nonexistent_scenario-s1-viewer.jsonl",
        "README.jsonl",
        "false_alarm-s7-viewer.judge.jsonl",  # stem has a dot -> no match
    ]:
        (tmp_path / name).write_text("")

    found = [(p.scenario, p.seed, p.role) for p in recorded_demo_paths(tmp_path)]
    assert found == [("false_alarm", 7, "system"), ("false_alarm", 7, "viewer")]


async def test_two_concurrent_resumes_only_one_wins(tmp_path: Path) -> None:
    """Double-click / two tabs / two operators: both requests reach resume at
    once. The atomic claim lets exactly one through; the other is told who
    won, and the audit log records exactly one decision."""
    store = MemoryStore()
    checkpointer = InMemorySaver()
    handle = await _operator_run(tmp_path, store)
    await start_graph(handle, store, checkpointer)

    a = await open_run(store, _SETTINGS, handle.run.run_id)
    b = await open_run(store, _SETTINGS, handle.run.run_id)
    results = await asyncio.gather(
        resume_graph(a, store, checkpointer, {"decision": "approve", "approver": "alice"}),
        resume_graph(b, store, checkpointer, {"decision": "approve", "approver": "bob"}),
        return_exceptions=True,
    )

    losers = [r for r in results if isinstance(r, ApprovalAlreadyDecided)]
    winners = [r for r in results if not isinstance(r, BaseException)]
    assert len(winners) == 1 and len(losers) == 1
    assert winners[0].outcome == "completed"
    assert losers[0].approver == "alice"  # gather starts `a` first, so alice wins
    decisions = [
        e for e in await store.list_audit(handle.run.run_id) if e.action == "approval_decision"
    ]
    assert [e.actor for e in decisions] == ["alice"]


async def test_lost_sandbox_is_rebuilt_when_nothing_destructive_ran(tmp_path: Path) -> None:
    """Render free wipes the disk on every spin-down. A run paused before its
    first destructive action can be resumed on a regenerated sandbox."""
    import shutil

    store = MemoryStore()
    checkpointer = InMemorySaver()
    handle = await _operator_run(tmp_path, store)
    assert (await start_graph(handle, store, checkpointer)).outcome == "awaiting_approval"

    shutil.rmtree(handle.sandbox.root)  # the host restarted while we waited
    reopened = await open_run(store, _SETTINGS, handle.run.run_id)
    done = await resume_graph(
        reopened, store, checkpointer, {"decision": "approve", "approver": "test"}
    )

    assert done.outcome == "completed"
    assert reopened.sandbox.facts()["checkout.config_version"] == 12


async def test_lost_sandbox_after_a_destructive_action_refuses_to_rebuild(
    tmp_path: Path,
) -> None:
    import shutil
    from datetime import UTC, datetime

    from opspilot.runs import SandboxLost
    from opspilot.store.models import AuditDoc

    store = MemoryStore()
    handle = await _operator_run(tmp_path, store)
    await store.insert_audit(
        AuditDoc(
            ts=datetime.now(UTC),
            actor="operator",
            action="destructive_tool_executed",
            target="restart_service",
            decision="ok",
            run_id=handle.run.run_id,
            prompt_version="v",
            model="m",
        )
    )
    shutil.rmtree(handle.sandbox.root)

    with pytest.raises(SandboxLost, match="already ran restart_service"):
        await open_run(store, _SETTINGS, handle.run.run_id)


async def test_resumed_run_spans_nest_under_their_own_run_span(tmp_path: Path) -> None:
    """The original run span closes at the pause; the resume opens its own, so
    nothing after an approval is left at the top level of the trace."""
    store = MemoryStore()
    checkpointer = InMemorySaver()
    handle = await _operator_run(tmp_path, store)
    await start_graph(handle, store, checkpointer)
    reopened = await open_run(store, _SETTINGS, handle.run.run_id)
    await resume_graph(reopened, store, checkpointer, {"decision": "approve", "approver": "t"})

    spans = await store.list_spans(handle.run.run_id)
    roots = [s for s in spans if s.parent_id is None]
    assert sorted(s.name for s in roots) == ["react_graph", "react_graph.resume"]
    resume = next(s for s in roots if s.name == "react_graph.resume")
    wait = next(s for s in spans if s.kind == "approval_wait")
    assert wait.parent_id == resume.span_id
    assert any(s.kind == "tool_call" and s.parent_id == resume.span_id for s in spans)
