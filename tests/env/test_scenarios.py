import pytest

from opspilot.env.scenarios import SCENARIOS, get_scenario

EXPECTED_NAMES = {
    "checkout_pool_exhaustion",
    "payments_bad_deploy",
    "inventory_memory_leak",
    "db_disk_full",
    "prompt_injection",
    "false_alarm",
}


def test_registry_has_all_six_scenarios() -> None:
    assert set(SCENARIOS) == EXPECTED_NAMES


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_scenario_fields_are_populated(name: str) -> None:
    scenario = get_scenario(name)
    assert scenario.alert_text.strip()
    assert scenario.fault_description.strip()
    assert scenario.root_cause_label.strip()
    assert scenario.expected_fix.strip()


def test_get_scenario_unknown_name_raises_with_known_list() -> None:
    with pytest.raises(KeyError, match="checkout_pool_exhaustion"):
        get_scenario("does_not_exist")
