from pathlib import Path

import pytest

from opspilot.config import Settings
from opspilot.env.generator import build_sandbox
from opspilot.env.sandbox import Sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.tools.base import ToolRegistry
from opspilot.tools.registry import build_default_registry


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    scenario = get_scenario("checkout_pool_exhaustion")
    return build_sandbox(scenario, seed=42, root=tmp_path / "sbx")


@pytest.fixture
def settings() -> Settings:
    # Explicit values, not read from the developer's real .env -- loop tests
    # must not depend on local machine state.
    return Settings(
        anthropic_api_key="unused",
        opspilot_model="unused",
        opspilot_max_steps=10,
        opspilot_token_budget=60_000,
        opspilot_tool_output_max_chars=4_000,
    )


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    return build_default_registry(settings)
