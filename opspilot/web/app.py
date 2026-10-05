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
from opspilot.store.base import Store
from opspilot.store.factory import build_store
from opspilot.store.models import RunDoc
from opspilot.web.service import InvalidRunRequest, WebRunService
from opspilot.web.trace_view import build_waterfall

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
DEFAULT_RUNS_ROOT = Path("tmp/web_runs")
# HTMX treats this status as "stop polling" (htmx.org/docs, "Polling").
HTMX_STOP_POLLING = 286


def _service(request: Request) -> WebRunService:
    service: WebRunService = request.app.state.service
    return service


def _render(request: Request, name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
    service = _service(request)
    return TEMPLATES.TemplateResponse(
        request,
        name,
        {"mode": service.settings.opspilot_model_mode, "replay": service.replay, **context},
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
        return _render(request, "index.html", paths=service.offered_paths(), recent=recent)

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
            return _render(request, "error.html", status_code=400, message=str(exc))
        # 303 See Other: after a POST, the browser must GET the run page.
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/runs", response_class=HTMLResponse)
    async def list_runs(request: Request) -> HTMLResponse:
        return _render(request, "runs.html", runs=await _service(request).store.list_runs())

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(request: Request, run_id: str) -> HTMLResponse:
        run = await _service(request).store.get_run(run_id)
        if run is None:
            return _render(request, "error.html", status_code=404, message="No such run.")
        return _render(request, "run.html", run=run)

    @app.get("/runs/{run_id}/live", response_class=HTMLResponse)
    async def run_live(request: Request, run_id: str) -> HTMLResponse:
        """The polled partial: header + waterfall + final report (from the RunDoc)."""
        store = _service(request).store
        run = await store.get_run(run_id)
        if run is None:
            return HTMLResponse("No such run.", status_code=HTMX_STOP_POLLING)
        spans = await store.list_spans(run_id)
        finished = _is_finished(run)
        return _render(
            request,
            "_run_live.html",
            # Same markup either way; the status code alone tells HTMX
            # whether to keep polling.
            status_code=HTMX_STOP_POLLING if finished else 200,
            run=run,
            finished=finished,
            rows=build_waterfall(spans, now=datetime.now(UTC)),
        )

    return app
