import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from opspilot.config import Settings, get_settings
from opspilot.loops.graph import build_checkpointer
from opspilot.models.factory import DEMO_CASSETTE_ROOT
from opspilot.policy.permissions import Role, can_decide_approval
from opspilot.store.base import Store
from opspilot.store.factory import build_store
from opspilot.store.models import RunDoc
from opspilot.web.service import Actor, DecisionError, InvalidRunRequest, WebRunService
from opspilot.web.trace_view import build_waterfall

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
DEFAULT_RUNS_ROOT = Path("tmp/web_runs")
# HTMX treats this status as "stop polling" (htmx.org/docs, "Polling").
HTMX_STOP_POLLING = 286


def _service(request: Request) -> WebRunService:
    service: WebRunService = request.app.state.service
    return service


# Demo auth: who you are lives in two cookies, set by POST /act-as. Anyone
# can pick any role -- that's the point of a demo, and the UI says so. What
# matters is that the *server* reads the role on every decision.
ROLE_COOKIE, NAME_COOKIE = "opspilot_role", "opspilot_name"
HUMAN_ROLES: tuple[Role, ...] = ("viewer", "operator", "admin")


def current_actor(request: Request) -> Actor:
    role = request.cookies.get(ROLE_COOKIE, "viewer")
    name = (request.cookies.get(NAME_COOKIE) or "visitor").strip()[:40] or "visitor"
    return Actor(name=name, role=role if role in HUMAN_ROLES else "viewer")


async def _render(
    request: Request, name: str, status_code: int = 200, **context: Any
) -> HTMLResponse:
    service = _service(request)
    pending = await service.store.list_pending_approvals()
    return TEMPLATES.TemplateResponse(
        request,
        name,
        {
            "mode": service.settings.opspilot_model_mode,
            "replay": service.replay,
            "actor": current_actor(request),
            "human_roles": HUMAN_ROLES,
            "pending_count": len(pending),
            **context,
        },
        status_code=status_code,
    )


def _is_finished(run: RunDoc) -> bool:
    return run.finished_at is not None or run.outcome == "error"


def create_app(
    settings: Settings | None = None,
    *,
    store: Store | None = None,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    demo_root: Path = DEMO_CASSETTE_ROOT,
) -> FastAPI:
    """App factory -- tests pass their own settings/store/paths; `opspilot
    web` and uvicorn call it with none and get env-configured defaults."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        resolved_store = store or build_store(resolved)
        await resolved_store.ensure_indexes()
        checkpointer, mongo_client = build_checkpointer(resolved)
        service = WebRunService(resolved_store, resolved, checkpointer, runs_root, demo_root)
        app.state.service = service
        try:
            yield
        finally:
            await service.shutdown()
            if mongo_client is not None:
                mongo_client.close()

    app = FastAPI(title="OpsPilot", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, str]:
        return {"status": "ok", "mode": _service(request).settings.opspilot_model_mode}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        service = _service(request)
        recent = (await service.store.list_runs())[:5]
        return await _render(request, "index.html", paths=service.offered_paths(), recent=recent)

    @app.post("/runs")
    async def start_run(
        request: Request, path: str = Form(...), strategy: str = Form("graph")
    ) -> Response:
        try:
            scenario, seed, role = path.split("|")
            if strategy not in ("raw", "graph") or role not in ("viewer", "operator", "admin"):
                raise InvalidRunRequest(f"invalid strategy/role: {strategy!r}, {role!r}")
            run_id = await _service(request).start(scenario, int(seed), role, strategy)  # type: ignore[arg-type]
        except (InvalidRunRequest, ValueError) as exc:
            return await _render(request, "error.html", status_code=400, message=str(exc))
        # 303 See Other: after a POST, the browser must GET the run page.
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/runs", response_class=HTMLResponse)
    async def list_runs(request: Request) -> HTMLResponse:
        return await _render(request, "runs.html", runs=await _service(request).store.list_runs())

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(request: Request, run_id: str) -> HTMLResponse:
        run = await _service(request).store.get_run(run_id)
        if run is None:
            return await _render(request, "error.html", status_code=404, message="No such run.")
        return await _render(request, "run.html", run=run)

    @app.get("/runs/{run_id}/live", response_class=HTMLResponse)
    async def run_live(request: Request, run_id: str) -> HTMLResponse:
        """The polled partial: header + waterfall + final report (from the RunDoc)."""
        store = _service(request).store
        run = await store.get_run(run_id)
        if run is None:
            return HTMLResponse("No such run.", status_code=HTMX_STOP_POLLING)
        spans = await store.list_spans(run_id)
        finished = _is_finished(run)
        return await _render(
            request,
            "_run_live.html",
            # Same markup either way; the status code alone tells HTMX
            # whether to keep polling.
            status_code=HTMX_STOP_POLLING if finished else 200,
            run=run,
            finished=finished,
            rows=build_waterfall(spans, now=datetime.now(UTC)),
        )

    @app.post("/act-as")
    async def act_as(
        role: str = Form(...), name: str = Form("visitor"), next: str = Form("/")
    ) -> Response:
        # Only same-site paths -- never redirect to a URL a form field supplied.
        target = next if next.startswith("/") and not next.startswith("//") else "/"
        response = RedirectResponse(target, status_code=303)
        safe_role = role if role in HUMAN_ROLES else "viewer"
        safe_name = name.strip()[:40] or "visitor"
        for key, value in ((ROLE_COOKIE, safe_role), (NAME_COOKIE, safe_name)):
            response.set_cookie(key, value, httponly=True, samesite="lax", max_age=30 * 86400)
        return response

    @app.get("/approvals", response_class=HTMLResponse)
    async def approvals(request: Request) -> HTMLResponse:
        pending = await _service(request).store.list_pending_approvals()
        return await _render(request, "approvals.html", approvals=pending)

    @app.post("/approvals/{approval_id}")
    async def decide(
        request: Request,
        approval_id: str,
        decision: str = Form(...),
        reason: str = Form(""),
        args: str = Form(""),
    ) -> Response:
        if decision not in ("approve", "reject", "edit"):
            return await _render(request, "error.html", 400, message="Unknown decision.")
        edited: dict[str, Any] | None = None
        if decision == "edit":
            try:
                parsed = json.loads(args or "null")
            except json.JSONDecodeError:
                parsed = None
            edited = parsed if isinstance(parsed, dict) else None
        try:
            run_id = await _service(request).decide(
                approval_id,
                current_actor(request),
                decision,  # type: ignore[arg-type]
                reason=reason,
                edited_args=edited,
            )
        except DecisionError as exc:
            return await _render(request, "error.html", exc.status_code, message=str(exc))
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}/approval", response_class=HTMLResponse)
    async def run_approval(request: Request, run_id: str) -> HTMLResponse:
        """The approval card's own polled region, separate from /live so a
        half-typed reject reason is never wiped by a refresh. Polls until a
        pending approval appears, then answers 286 so the form stays put."""
        store = _service(request).store
        run = await store.get_run(run_id)
        pending = [a for a in await store.list_pending_approvals() if a.run_id == run_id]
        if pending:
            actor = current_actor(request)
            return await _render(
                request,
                "_approval_card.html",
                HTMX_STOP_POLLING,
                approval=pending[0],
                can_decide=can_decide_approval(actor.role),
            )
        done = run is None or _is_finished(run)
        return HTMLResponse("", status_code=HTMX_STOP_POLLING if done else 200)

    return app
