import asyncio
import logging
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver

from opspilot.config import Settings
from opspilot.loops.graph import ApprovalAlreadyDecided
from opspilot.models.cassette import CassetteMiss
from opspilot.models.factory import DEMO_CASSETTE_ROOT, demo_cassette_path, recorded_demo_paths
from opspilot.policy.permissions import Role, can_decide_approval
from opspilot.runs import RunHandle, create_run, open_run, resume_graph, run_raw, start_graph
from opspilot.store.base import Store
from opspilot.web.guards import check_live_start

log = logging.getLogger(__name__)

NOT_RECORDED = (
    "This path wasn't recorded for the public demo, so there's no model response to replay. "
    "Clone the repo and set ANTHROPIC_API_KEY to run it live."
)


class InvalidRunRequest(ValueError):
    """The requested scenario/seed/role isn't something this server can run."""


class DecisionError(Exception):
    """A decision request the server refuses; `status_code` is the HTTP answer."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Actor:
    """Who is clicking -- demo auth from a cookie, NOT a real login."""

    name: str
    role: Role

    @property
    def label(self) -> str:
        return f"{self.name} ({self.role})"


class WebRunService:
    """Starts runs in the background and keeps track of them.

    Runs are asyncio tasks on the server's own event loop -- not FastAPI's
    BackgroundTasks (tied to one request's lifecycle, meant for short jobs)
    and not a job queue (overkill for one free-tier instance). A run that
    pauses for approval simply ends its task; the approve endpoint starts a
    new task to resume it, rebuilt from the store (opspilot.runs.open_run).
    """

    def __init__(
        self,
        store: Store,
        settings: Settings,
        checkpointer: BaseCheckpointSaver[Any],
        runs_root: Path,
        demo_root: Path = DEMO_CASSETTE_ROOT,
    ) -> None:
        self.store = store
        self.settings = settings
        self.checkpointer = checkpointer
        self.runs_root = runs_root
        self.demo_root = demo_root
        # Strong references: the event loop only keeps weak ones, so an
        # unreferenced task can be garbage-collected mid-run.
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def replay(self) -> bool:
        return self.settings.opspilot_model_mode == "replay"

    def offered_paths(self) -> list[tuple[str, int, str]]:
        """What the start form may offer: in replay, exactly the recorded
        demo paths -- so a visitor can't pick a run with nothing to replay."""
        return [(p.scenario, p.seed, p.role) for p in recorded_demo_paths(self.demo_root)]

    async def start(
        self,
        scenario: str,
        seed: int,
        role: Role,
        strategy: Literal["raw", "graph"],
        live_token: str = "",
    ) -> str:
        if self.replay and (scenario, seed, role) not in self.offered_paths():
            raise InvalidRunRequest(f"{scenario} / seed {seed} / {role} isn't a recorded demo path")
        # No-op in replay; in live mode: owner token, then the daily cap.
        await check_live_start(self.settings, self.store, live_token)
        run_id = str(uuid.uuid4())
        cassette = (
            None
            if self.settings.opspilot_model_mode == "live"
            else self.demo_root / demo_cassette_path(scenario, seed, role).name
        )
        try:
            handle = await create_run(
                self.store,
                self.settings,
                scenario=scenario,
                seed=seed,
                role=role,
                strategy=strategy,
                mode=self.settings.opspilot_model_mode,
                cassette=cassette,
                sandbox_dir=self.runs_root / run_id,
                run_id=run_id,
            )
        except KeyError as exc:
            raise InvalidRunRequest(str(exc)) from exc
        self._spawn(self._guarded(handle.run.run_id, self._execute(handle)))
        return run_id

    # [HARNESS:HITL] The web half of the approval loop -- a button instead of
    # the CLI's y/n prompt, resumed from durable state by a fresh task.
    # WHY: the run that paused is long gone (its task ended at the
    # interrupt); open_run() rebuilds it from the RunDoc + checkpointer, so
    # this works the same after a server restart (with Mongo). Checks run
    # in order of cheapness: exists -> allowed to decide -> still pending.
    # The pending check here is for UX (a clean 409 for a late click); the
    # atomic claim in resume_react_graph is what guarantees exactly-once
    # when two requests truly race.
    async def decide(
        self,
        approval_id: str,
        actor: Actor,
        decision: Literal["approve", "reject", "edit"],
        reason: str = "",
        edited_args: dict[str, Any] | None = None,
    ) -> str:
        """Validate a decision and resume the run in the background.
        Returns the run id; raises DecisionError with the HTTP status to send."""
        approval = await self.store.get_approval(approval_id)
        if approval is None:
            raise DecisionError("No such approval.", 404)
        if not can_decide_approval(actor.role):
            raise DecisionError(
                f"A {actor.role} can't decide approvals -- act as an operator or admin.", 403
            )
        if approval.status != "pending":
            raise DecisionError(
                f"Already {approval.status} by {approval.approver or 'someone'}.", 409
            )
        payload: dict[str, Any] = {"decision": decision, "approver": actor.label}
        if decision == "reject":
            payload["reason"] = reason.strip() or "no reason given"
        elif decision == "edit":
            if self.replay:
                # Edited args change the tool result, so the next model call
                # was never recorded -- refuse up front instead of failing later.
                raise DecisionError("Editing arguments isn't available in the replay demo.", 400)
            if not isinstance(edited_args, dict):
                raise DecisionError("Edited arguments must be a JSON object.", 400)
            payload["args"] = edited_args

        handle = await open_run(self.store, self.settings, approval.run_id)
        self._spawn(self._guarded(handle.run.run_id, self._resume(handle, payload)))
        return handle.run.run_id

    async def _resume(self, handle: RunHandle, decision: dict[str, Any]) -> None:
        await resume_graph(handle, self.store, self.checkpointer, decision)

    async def _execute(self, handle: RunHandle) -> None:
        if handle.run.strategy == "raw":
            await run_raw(handle, self.store)
        else:
            await start_graph(handle, self.store, self.checkpointer)

    async def _guarded(self, run_id: str, work: Coroutine[Any, Any, None]) -> None:
        """Run `work`; on failure, record it on the run instead of losing it
        in a background task nobody awaits."""
        try:
            await work
        except ApprovalAlreadyDecided as exc:
            # Lost a true race to another decision -- that one is resuming
            # the run; this one must do nothing (and must not mark it failed).
            log.info("run %s: %s", run_id, exc)
        except CassetteMiss:
            await self.store.update_run(run_id, {"outcome": "error", "error": NOT_RECORDED})
        except Exception as exc:  # noqa: BLE001
            log.exception("run %s failed", run_id)
            await self.store.update_run(
                run_id, {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"}
            )

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_idle(self) -> None:
        """Wait for every in-flight run task (tests, graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
