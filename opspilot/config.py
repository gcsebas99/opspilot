from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
