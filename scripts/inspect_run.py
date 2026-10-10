"""Run one replayed scenario in-process and show what the harness recorded.

    uv run python scripts/inspect_run.py          # false_alarm as viewer
    uv run python scripts/inspect_run.py --scenario checkout_pool_exhaustion \
        --role operator --approve --tamper

Prints the trace (spans as a tree), the audit log with its hash chain, and the
chain verification -- and with --tamper, edits one audit entry and shows the
verification catching it. Replay mode and an in-memory store: $0, no API key,
no database. A learning tool for docs/learn/ (paths 5, 6 and 9).
"""

import argparse
import asyncio
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from opspilot.config import Settings
from opspilot.models.factory import demo_cassette_path
from opspilot.observability.audit import verify_chain
from opspilot.observability.metrics import build_dashboard
from opspilot.runs import create_run, resume_graph, start_graph
from opspilot.store.memory import MemoryStore
from opspilot.web.trace_view import build_waterfall


async def inspect(scenario: str, role: str, approve: bool, tamper: bool) -> int:
    settings = Settings.model_validate(
        {"ANTHROPIC_API_KEY": "", "OPSPILOT_MODEL": "claude-haiku-4-5", "OPSPILOT_STORE": "memory"}
    )
    cassette = demo_cassette_path(scenario, 42, role)
    if not cassette.exists():
        print(f"no recorded demo path for {scenario} / {role} ({cassette})")
        return 2
    store = MemoryStore()
    checkpointer = InMemorySaver()
    with tempfile.TemporaryDirectory() as tmp:
        handle = await create_run(
            store,
            settings,
            scenario=scenario,
            seed=42,
            role=role,  # type: ignore[arg-type]
            strategy="graph",
            mode="replay",
            cassette=cassette,
            sandbox_dir=Path(tmp) / "sandbox",
        )
        result = await start_graph(handle, store, checkpointer)
        if result.outcome == "awaiting_approval":
            print(f"paused for approval: {result.pending_approval}")
            if not approve:
                print("(pass --approve to approve it and let the run finish)")
            else:
                decision = {"decision": "approve", "approver": "inspect_run (operator)"}
                result = await resume_graph(handle, store, checkpointer, decision)

    run_id = handle.run.run_id
    print(f"\noutcome: {result.outcome}   steps: {result.steps}   tokens: {result.tokens}\n")

    print("TRACE (kind  name  duration)")
    for row in build_waterfall(await store.list_spans(run_id), now=datetime.now(UTC)):
        duration = f"{row.duration_ms:.0f}ms" if row.duration_ms is not None else "..."
        detail = f"   [{row.detail}]" if row.detail else ""
        print(f"  {'  ' * row.depth}{row.kind:<14} {row.name:<22} {duration:>8}{detail}")

    entries = await store.list_audit(run_id)
    print("\nAUDIT LOG (action -> target: decision   prev_hash -> hash)")
    for e in entries:
        prev = (e.prev_hash or "none")[:8]
        print(f"  {e.action} -> {e.target}: {e.decision[:50]}   {prev} -> {(e.hash or '')[:8]}")
    check = verify_chain(entries)
    print(f"\nchain verification: {'OK' if check.ok else 'BROKEN'} ({check.total_entries} entries)")

    if tamper and entries:
        index = len(entries) // 2
        forged = entries[index].model_copy(update={"decision": "tampered"})
        tampered = [*entries[:index], forged, *entries[index + 1 :]]
        check = verify_chain(tampered)
        print(f"\nafter editing entry {index}'s decision to 'tampered':")
        print(f"  verification: BROKEN at entry {check.broken_at_index}: {check.reason}")

    dashboard = await build_dashboard(store)
    print(
        f"\nMETRICS  outcomes={dashboard.outcomes_distribution}  "
        f"tool error rates={dashboard.tool_error_rates}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenario", default="false_alarm")
    parser.add_argument("--role", default="viewer")
    parser.add_argument("--approve", action="store_true", help="approve a pending action")
    parser.add_argument("--tamper", action="store_true", help="edit an audit entry, re-verify")
    args = parser.parse_args(argv)
    return asyncio.run(inspect(args.scenario, args.role, args.approve, args.tamper))


if __name__ == "__main__":
    sys.exit(main())
