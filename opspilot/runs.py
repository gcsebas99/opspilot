"""The run lifecycle, shared by the CLI and the web app.

create -> start (graph or raw) -> [pause for approval -> resume]* -> finished.
Both front-ends call these functions, so "what a run is" lives in one place;
only the *interaction* differs (a terminal y/n prompt vs. an approve button).
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import LanguageModelLike
from langgraph.checkpoint.base import BaseCheckpointSaver

from opspilot.config import Settings
from opspilot.context.assembler import CONTEXT_DIR, prompt_version
from opspilot.env.generator import build_sandbox
from opspilot.env.sandbox import Sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.loops.graph import resume_react_graph, run_react_graph
from opspilot.loops.react_raw import LoopEvent, RunResult, run_react_loop
from opspilot.models.base import ModelClient
from opspilot.models.factory import (
    LiveCallRefused,
    ModelMode,
    build_chat_model,
    build_model_client,
)
from opspilot.observability.audit import record_loop_audit, record_prompt_version_change_if_needed
from opspilot.observability.instrumentation import record_loop_spans
from opspilot.observability.pricing import cost_usd
from opspilot.observability.tracer import Tracer
from opspilot.policy.permissions import Role
from opspilot.store.base import Store
from opspilot.store.models import RunDoc
from opspilot.tools.base import ToolRegistry
from opspilot.tools.registry import build_default_registry

Strategy = Literal["raw", "graph"]


@dataclass
class RunHandle:
    """Everything one run needs, rebuilt identically by whichever process
    touches it next (the one that starts it, or the one that resumes it)."""

    run: RunDoc
    sandbox: Sandbox
    registry: ToolRegistry
    tracer: Tracer
    # The settings the run was *started* with, model pinned to run.model --
    # a resume must use the same model even if OPSPILOT_MODEL changed since.
    settings: Settings
    alert: str

    @property
    def cassette(self) -> Path | None:
        return Path(self.run.cassette) if self.run.cassette is not None else None

    def _check_can_call(self) -> None:
        # A replayed demo must never silently turn into a paid live call --
        # at start or, more subtly, at a resume in a different process.
        if self.run.mode != "replay" and not self.settings.anthropic_api_key:
            raise LiveCallRefused(
                f"run {self.run.run_id} is mode={self.run.mode} but ANTHROPIC_API_KEY is empty"
            )

    def chat_model(self) -> LanguageModelLike:
        self._check_can_call()
        return build_chat_model(self.settings, self.registry, self.run.mode, self.cassette)

    def raw_model(self) -> ModelClient:
        self._check_can_call()
        return build_model_client(self.settings, self.run.mode, self.cassette)


async def create_run(
    store: Store,
    settings: Settings,
    *,
    scenario: str,
    seed: int,
    role: Role,
    strategy: Strategy,
    mode: ModelMode,
    cassette: Path | None,
    sandbox_dir: Path,
    run_id: str | None = None,
    trigger: str | None = None,
    alert: dict[str, Any] | None = None,
) -> RunHandle:
    """Materialize the sandbox and persist the RunDoc. Raises KeyError for
    an unknown scenario -- the caller decides how to present that."""
    scn = get_scenario(scenario)
    sandbox = build_sandbox(scn, seed, sandbox_dir)
    registry = build_default_registry(settings)
    run_id = run_id or str(uuid.uuid4())
    current_prompt_version = prompt_version(CONTEXT_DIR, registry.to_anthropic_schema())

    # Must run before insert_run below -- it looks at the *previous* most
    # recent run to detect a change, and this run's own RunDoc would
    # otherwise already be that "previous" run by the time it checks.
    await record_prompt_version_change_if_needed(
        store,
        run_id=run_id,
        prompt_version=current_prompt_version,
        model=settings.opspilot_model,
        ts=datetime.now(UTC),
    )
    run = RunDoc(
        run_id=run_id,
        created_at=datetime.now(UTC),
        scenario=scenario,
        seed=seed,
        role=role,
        model=settings.opspilot_model,
        prompt_version=current_prompt_version,
        strategy=strategy,
        mode=mode,
        cassette=str(cassette) if cassette is not None else None,
        sandbox_dir=str(sandbox.root),
        trigger=trigger,
        alert=alert,
    )
    await store.insert_run(run)
    return RunHandle(
        run=run,
        sandbox=sandbox,
        registry=registry,
        tracer=Tracer(store, run_id=run_id),
        settings=settings,
        alert=scn.alert_text,
    )


# [HARNESS:HITL] Resume = rebuild the run from durable state, not from memory.
# WHY: an approval can arrive minutes later, in another HTTP request or
# another process (`opspilot approve`, a restarted web server). Everything a
# resume needs -- sandbox location, model, mode, cassette, prompt version --
# comes from the stored RunDoc; the conversation itself comes from the
# checkpointer. Nothing depends on the process that started the run.
# INTERVIEW: "What if the server restarts while a run awaits approval?" ->
# RunDoc + Mongo checkpointer hold everything; open_run() rebuilds it.
async def open_run(store: Store, settings: Settings, run_id: str) -> RunHandle:
    """Rebuild a handle for an existing run (e.g. to resume it after approval)."""
    run = await store.get_run(run_id)
    if run is None:
        raise KeyError(f"no run found with id {run_id!r}")
    # Point at the run's *existing*, already-mutated sandbox -- build_sandbox()
    # would regenerate the scenario and wipe whatever the run already did.
    # Runs from before sandbox_dir was stored used the CLI's fixed layout.
    sandbox_dir = run.sandbox_dir or str(Path("tmp/runs") / f"{run.scenario}-{run.seed}")
    return RunHandle(
        run=run,
        sandbox=Sandbox(root=Path(sandbox_dir)),
        registry=build_default_registry(settings),
        tracer=Tracer(store, run_id=run.run_id),
        settings=settings.model_copy(update={"opspilot_model": run.model}),
        alert=get_scenario(run.scenario).alert_text,
    )


async def start_graph(
    handle: RunHandle,
    store: Store,
    checkpointer: BaseCheckpointSaver[Any],
    max_steps: int | None = None,
) -> RunResult:
    run = handle.run
    async with handle.tracer.span(
        "run", "react_graph", scenario=run.scenario, seed=run.seed, role=run.role, strategy="graph"
    ):
        result = await run_react_graph(
            model=handle.chat_model(),
            registry=handle.registry,
            sandbox=handle.sandbox,
            alert=handle.alert,
            settings=handle.settings,
            tracer=handle.tracer,
            store=store,
            checkpointer=checkpointer,
            run_id=run.run_id,
            prompt_version=run.prompt_version,
            role=run.role,  # type: ignore[arg-type]
            max_steps=max_steps,
        )
    await record_progress(store, handle, result)
    return result


async def resume_graph(
    handle: RunHandle,
    store: Store,
    checkpointer: BaseCheckpointSaver[Any],
    decision: dict[str, Any],
    max_steps: int | None = None,
) -> RunResult:
    result = await resume_react_graph(
        model=handle.chat_model(),
        registry=handle.registry,
        sandbox=handle.sandbox,
        settings=handle.settings,
        tracer=handle.tracer,
        store=store,
        checkpointer=checkpointer,
        run_id=handle.run.run_id,
        decision=decision,
        prompt_version=handle.run.prompt_version,
        max_steps=max_steps,
    )
    await record_progress(store, handle, result)
    return result


async def run_raw(
    handle: RunHandle,
    store: Store,
    on_event: Callable[[LoopEvent], None] | None = None,
    max_steps: int | None = None,
) -> RunResult:
    run = handle.run
    events: list[LoopEvent] = []

    def collect(event: LoopEvent) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    async with handle.tracer.span(
        "run", "react_loop", scenario=run.scenario, seed=run.seed, role=run.role, strategy="raw"
    ) as run_span:
        result = await run_react_loop(
            model=handle.raw_model(),
            registry=handle.registry,
            sandbox=handle.sandbox,
            alert=handle.alert,
            settings=handle.settings,
            role=run.role,  # type: ignore[arg-type]
            max_steps=max_steps,
            on_event=collect,
        )
        # Must stay inside the `async with` -- record_span() reads the
        # ambient parent span from a contextvar that's reset the moment
        # this block exits, so writing the nested spans after exit
        # would silently produce a flat trace (parent_id=None everywhere).
        await record_loop_spans(handle.tracer, run_span.start, events, model=run.model)
        await record_loop_audit(
            store,
            run_span.start,
            events,
            run_id=run.run_id,
            role=run.role,
            prompt_version=run.prompt_version,
            model=run.model,
        )
    await record_progress(store, handle, result)
    return result


async def record_progress(store: Store, handle: RunHandle, result: RunResult) -> float:
    """Persist where the run stands. A paused run gets its outcome
    ("awaiting_approval") and cost-so-far, but no finished_at -- the web UI
    reads this to show "waiting for approval" rather than "done"."""
    cost = cost_usd(
        handle.run.model,
        input_tokens=result.tokens.input_tokens,
        output_tokens=result.tokens.output_tokens,
        cache_creation_input_tokens=result.tokens.cache_creation_input_tokens,
        cache_read_input_tokens=result.tokens.cache_read_input_tokens,
    )
    updates: dict[str, Any] = {
        "outcome": result.outcome,
        "steps": result.steps,
        "tokens": result.tokens.model_dump(),
        "cost_usd": cost,
        "report": result.report,
    }
    if result.outcome != "awaiting_approval":
        updates["finished_at"] = datetime.now(UTC)
    await store.update_run(handle.run.run_id, updates)
    return cost
