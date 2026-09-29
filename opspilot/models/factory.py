from collections.abc import Callable
from pathlib import Path
from typing import Literal

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelLike

from opspilot.config import Settings
from opspilot.models.anthropic_model import AnthropicModel
from opspilot.models.base import ModelClient
from opspilot.models.cassette import Cassette
from opspilot.models.langchain_adapter import ModelClientChatModel
from opspilot.models.recording import RecordingModel
from opspilot.models.replay import ReplayModel
from opspilot.tools.base import ToolRegistry

ModelMode = Literal["live", "record", "replay"]

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


# [HARNESS:EVAL] One switch decides whether a run costs money.
# WHY: live/record/replay is a property of the *run*, not of the loop -- the
# loop only sees a ModelClient. Live factories are passed in uncalled, so
# replay never constructs an API client and needs no API key at all.
# INTERVIEW: "How do you run agent evals in CI for free?" -> the same
# harness in replay mode: recorded responses, zero network, loud on drift.
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
