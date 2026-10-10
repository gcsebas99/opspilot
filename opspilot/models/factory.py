import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelLike
from pydantic import BaseModel

from opspilot.config import Settings
from opspilot.env.scenarios import SCENARIOS
from opspilot.models.anthropic_model import AnthropicModel
from opspilot.models.base import ModelClient
from opspilot.models.cassette import Cassette
from opspilot.models.langchain_adapter import ModelClientChatModel
from opspilot.models.recording import RecordingModel
from opspilot.models.replay import ReplayModel
from opspilot.tools.base import ToolRegistry

ModelMode = Literal["live", "record", "replay"]

DEMO_CASSETTE_ROOT = Path("evals/cassettes/demo")

LiveClientFactory = Callable[[Settings], ModelClient]
LiveChatFactory = Callable[[Settings, ToolRegistry], LanguageModelLike]


def default_live_client(settings: Settings) -> ModelClient:
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=0)
    return AnthropicModel(client=client, model=settings.opspilot_model)


def default_live_chat_model(settings: Settings, registry: ToolRegistry) -> LanguageModelLike:
    chat_model = ChatAnthropic(
        model=settings.opspilot_model, max_tokens=8192, api_key=settings.anthropic_api_key
    )
    return chat_model.bind_tools(registry.to_anthropic_schema())


class LiveCallRefused(RuntimeError):
    """A mode that spends money was requested without what it needs."""


# [HARNESS:GUARD] Cost guard: spending money is always an explicit opt-in.
# WHY: a demo or a forgotten loop must never burn API credit by accident. The
# default is replay ($0); live/record need --mode or OPSPILOT_MODEL_MODE, and fail
# fast without a key.
def resolve_mode(requested: str | None, settings: Settings) -> ModelMode:
    mode = requested or settings.opspilot_model_mode
    if mode not in ("live", "record", "replay"):
        raise ValueError(f"mode must be 'live', 'record', or 'replay', got {mode!r}")
    if mode != "replay" and not settings.anthropic_api_key:
        raise LiveCallRefused(
            f"mode {mode!r} calls the Anthropic API but ANTHROPIC_API_KEY is empty"
        )
    return cast(ModelMode, mode)


def demo_cassette_path(scenario: str, seed: int, role: str) -> Path:
    # No strategy in the name: raw and graph send identical requests
    # (tests/loops/test_strategy_parity.py), so one cassette serves both.
    return DEMO_CASSETTE_ROOT / f"{scenario}-s{seed}-{role}.jsonl"


class DemoPath(BaseModel):
    scenario: str
    seed: int
    role: str
    path: Path


_DEMO_NAME = re.compile(r"^(?P<scenario>[a-z][a-z0-9_]*)-s(?P<seed>\d+)-(?P<role>[a-z]+)$")


def recorded_demo_paths(root: Path = DEMO_CASSETTE_ROOT) -> list[DemoPath]:
    """Every scenario x seed x role a replay demo can actually serve -- the
    inverse of demo_cassette_path(). The web UI offers exactly these, so a
    visitor can never pick a run that would hit an empty cassette."""
    paths = []
    for file in sorted(root.glob("*.jsonl")):
        match = _DEMO_NAME.match(file.stem)
        if match is None or match["scenario"] not in SCENARIOS:
            continue
        if match["role"] not in ("viewer", "operator", "admin", "system"):
            continue
        paths.append(
            DemoPath(
                scenario=match["scenario"], seed=int(match["seed"]), role=match["role"], path=file
            )
        )
    return paths


# [HARNESS:EVAL] One switch decides whether a run costs money: live/record/replay.
# WHY: the mode belongs to the run, not the loop (which only sees a ModelClient).
# Live factories are passed in uncalled, so replay never builds an API client.
def build_model_client(
    settings: Settings,
    mode: ModelMode,
    cassette_path: Path | None,
    live_client: LiveClientFactory = default_live_client,
) -> ModelClient:
    if mode == "live":
        return live_client(settings)
    if cassette_path is None:
        raise ValueError(f"mode {mode!r} needs a cassette path")
    cassette = Cassette(cassette_path)
    if mode == "record":
        return RecordingModel(live_client(settings), cassette, settings.opspilot_model)
    return ReplayModel(cassette, settings.opspilot_model)


def build_chat_model(
    settings: Settings,
    registry: ToolRegistry,
    mode: ModelMode,
    cassette_path: Path | None,
    live_client: LiveClientFactory = default_live_client,
    live_chat_model: LiveChatFactory = default_live_chat_model,
) -> LanguageModelLike:
    """The graph's model. Live keeps ChatAnthropic; record/replay go through
    ModelClientChatModel so both strategies share one cassette format."""
    if mode == "live":
        return live_chat_model(settings, registry)
    client = build_model_client(settings, mode, cassette_path, live_client)
    return ModelClientChatModel(client=client).bind_tools(registry.to_anthropic_schema())
