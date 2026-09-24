from pathlib import Path

import pytest

from opspilot.env.generator import build_sandbox
from opspilot.env.sandbox import Sandbox
from opspilot.env.scenarios import get_scenario


@pytest.fixture
def checkout_sandbox(tmp_path: Path) -> Sandbox:
    scenario = get_scenario("checkout_pool_exhaustion")
    return build_sandbox(scenario, seed=42, root=tmp_path / "checkout_sbx")


@pytest.fixture
def payments_sandbox(tmp_path: Path) -> Sandbox:
    scenario = get_scenario("payments_bad_deploy")
    return build_sandbox(scenario, seed=42, root=tmp_path / "payments_sbx")
