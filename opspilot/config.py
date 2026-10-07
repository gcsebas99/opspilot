from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables / .env.

    No secrets or model IDs are hardcoded anywhere else in the codebase —
    everything reads through this object.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    opspilot_model: str = Field(default="", alias="OPSPILOT_MODEL")
    opspilot_judge_model: str = Field(default="", alias="OPSPILOT_JUDGE_MODEL")
    opspilot_max_steps: int = Field(default=15, alias="OPSPILOT_MAX_STEPS")
    opspilot_token_budget: int = Field(default=60_000, alias="OPSPILOT_TOKEN_BUDGET")
    opspilot_tool_output_max_chars: int = Field(
        default=4_000, alias="OPSPILOT_TOOL_OUTPUT_MAX_CHARS"
    )
    # replay by default: spending money (live/record) is always an explicit
    # opt-in -- see opspilot/models/factory.py:resolve_mode.
    opspilot_model_mode: Literal["live", "record", "replay"] = Field(
        default="replay", alias="OPSPILOT_MODEL_MODE"
    )
    opspilot_store: Literal["memory", "mongo"] = Field(default="memory", alias="OPSPILOT_STORE")
    mongodb_uri: str = Field(default="", alias="MONGODB_URI")

    # --- web demo guards (opspilot/web/guards.py) ---
    # Always on: run starts per client IP per minute (replay costs CPU, not money).
    opspilot_rate_limit_per_min: int = Field(default=10, alias="OPSPILOT_RATE_LIMIT_PER_MIN")
    # Live mode only. The web app refuses to serve live runs unless this is
    # true -- a second, deliberate switch on top of OPSPILOT_MODEL_MODE.
    opspilot_demo_live: bool = Field(default=False, alias="OPSPILOT_DEMO_LIVE")
    # Live mode only: starting a run requires this token (owner-only). Empty =
    # live runs can't be started from the web at all.
    opspilot_live_token: str = Field(default="", alias="OPSPILOT_LIVE_TOKEN")
    opspilot_daily_run_cap: int = Field(default=20, alias="OPSPILOT_DAILY_RUN_CAP")
    # Per-run token budget for web live runs -- lower than the CLI default.
    opspilot_live_token_budget: int = Field(default=30_000, alias="OPSPILOT_LIVE_TOKEN_BUDGET")
    # Shared secret for POST /webhooks/alert (HMAC). Empty = endpoint disabled.
    opspilot_webhook_secret: str = Field(default="", alias="OPSPILOT_WEBHOOK_SECRET")


@lru_cache
def get_settings() -> Settings:
    return Settings()
