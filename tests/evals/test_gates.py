from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evals.compare import compare
from evals.gates import gating_regressions, pass_rate_gate
from evals.report import EvalReport, build_report, write_report
from opspilot.cli import app
from opspilot.config import get_settings
from tests.evals.test_metrics import _CASES, _trial

# _CASES tags: a=[smoke, config], b=[adversarial], c=[adversarial, config]


def _report(outcomes: dict[str, bool], sweep: str) -> EvalReport:
    trials = [
        _trial(case_id, 0, passed).model_copy(update={"sweep_id": sweep})
        for case_id, passed in outcomes.items()
    ]
    return build_report(trials, _CASES, suite="golden", judged=True)


def test_pass_rate_gate() -> None:
    report = _report({"a": True, "b": True, "c": False}, "s1")  # pass@1 = 2/3
    assert pass_rate_gate(report.metrics, 0.6) is None
    assert pass_rate_gate(report.metrics, 2 / 3) is None  # at the floor passes
    failure = pass_rate_gate(report.metrics, 0.75)
    assert failure == "pass@1 67% is below the required 75%"


def test_reports_carry_case_tags_and_diffs_inherit_them() -> None:
    a = _report({"a": True, "b": True}, "s1")
    assert a.case_tags == {"a": ["smoke", "config"], "b": ["adversarial"]}

    diffs = {d.case_id: d for d in compare(a, _report({"a": False, "b": True}, "s2")).cases}
    assert diffs["a"].tags == ["smoke", "config"]


def test_tag_filter_decides_which_regressions_gate() -> None:
    before = _report({"a": True, "b": True}, "s1")
    after = _report({"a": False, "b": False}, "s2")  # both regress
    comparison = compare(before, after)

    assert {d.case_id for d in gating_regressions(comparison, [])} == {"a", "b"}
    assert [d.case_id for d in gating_regressions(comparison, ["adversarial"])] == ["b"]
    assert gating_regressions(comparison, ["security"]) == []


def test_old_reports_without_tags_load_and_warn() -> None:
    new = _report({"a": True}, "s1")
    old = EvalReport.model_validate(new.model_dump(exclude={"case_tags"}))  # pre-tags file
    assert old.case_tags == {}
    assert any("no case tags" in w for w in compare(old, new).warnings)


# --- CLI ---


def test_compare_fail_on_tag(tmp_path: Path) -> None:
    baseline = write_report(_report({"a": True, "b": True}, "aaaa0000"), tmp_path)[0]
    only_a = write_report(_report({"a": False, "b": True}, "bbbb0000"), tmp_path)[0]
    only_b = write_report(_report({"a": True, "b": False}, "cccc0000"), tmp_path)[0]
    runner = CliRunner()

    outside = runner.invoke(
        app, ["eval", "compare", str(baseline), str(only_a), "--fail-on-tag", "adversarial"]
    )
    assert outside.exit_code == 0
    assert "regressions outside adversarial don't fail this gate" in outside.stdout

    inside = runner.invoke(
        app, ["eval", "compare", str(baseline), str(only_b), "--fail-on-tag", "adversarial"]
    )
    assert inside.exit_code == 1
    assert "gate failed: 1 regression(s) on tag(s) adversarial" in inside.stdout


@pytest.fixture
def replay_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """The real CLI over the committed smoke cassettes: replay, no key, no network."""
    for var, value in {
        "ANTHROPIC_API_KEY": "",
        "OPSPILOT_MODEL": "claude-haiku-4-5",
        "OPSPILOT_JUDGE_MODEL": "claude-sonnet-5",
        "OPSPILOT_MODEL_MODE": "replay",
        "OPSPILOT_STORE": "memory",
    }.items():
        monkeypatch.setenv(var, value)
    # Keep generated reports out of the repo's evals/reports/.
    monkeypatch.setattr("opspilot.cli.write_report", lambda r: write_report(r, tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_eval_min_pass_rate_gate_on_real_smoke_replay(replay_env: Path) -> None:
    # Committed smoke cassettes at k=1: 3 of 4 pass (pass@1 = 75%).
    passing = CliRunner().invoke(app, ["eval", "--suite", "smoke", "--min-pass-rate", "0.75"])
    assert passing.exit_code == 0, passing.stdout
    assert "gate passed" in passing.stdout

    failing = CliRunner().invoke(app, ["eval", "--suite", "smoke", "--min-pass-rate", "0.8"])
    assert failing.exit_code == 1
    assert "gate failed: pass@1 75% is below the required 80%" in failing.stdout
