from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evals.compare import classify, compare, resolve_report
from evals.report import EvalReport, build_report, write_report
from opspilot.cli import app
from opspilot.store.models import EvalTrialDoc
from tests.evals.test_metrics import _CASES, _trial


def _report(
    outcomes: dict[str, list[bool]], *, sweep: str = "aaaaaaaa-1", **trial_kw: Any
) -> EvalReport:
    trials: list[EvalTrialDoc] = [
        _trial(case_id, i, passed, **trial_kw).model_copy(update={"sweep_id": sweep})
        for case_id, results in outcomes.items()
        for i, passed in enumerate(results)
    ]
    return build_report(trials, _CASES, suite="golden", judged=True)


@pytest.mark.parametrize(
    ("before", "after", "kind"),
    [
        ("pass", "fail", "regression"),
        ("pass", "flaky", "regression"),  # lost reliability counts
        ("fail", "pass", "fix"),
        ("flaky", "pass", "fix"),
        ("fail", "flaky", "changed"),
        ("flaky", "fail", "changed"),
        ("pass", "pass", "unchanged"),
        (None, "pass", "added"),
        ("fail", None, "removed"),
    ],
)
def test_classify(before: Any, after: Any, kind: str) -> None:
    assert classify(before, after) == kind


def test_compare_finds_regressions_fixes_and_deltas() -> None:
    a = _report({"a": [True, True], "b": [False, False], "c": [True, False]})
    b = _report({"a": [True, False], "b": [True, True], "c": [False, False]}, cost=0.04)

    result = compare(a, b)

    assert [c.case_id for c in result.of_kind("regression")] == ["a"]
    assert [c.case_id for c in result.of_kind("fix")] == ["b"]
    assert [c.case_id for c in result.of_kind("changed")] == ["c"]
    regression = result.of_kind("regression")[0]
    assert (regression.before_rate, regression.after_rate) == (1.0, 0.5)
    deltas = {d.name: d for d in result.deltas}
    assert deltas["agent cost mean"].change == pytest.approx(0.02)
    assert deltas["pass@1"].change == pytest.approx(0.0)  # flat aggregate, real churn underneath


def test_added_and_removed_cases() -> None:
    result = compare(_report({"a": [True]}), _report({"b": [True]}))
    assert [c.case_id for c in result.of_kind("removed")] == ["a"]
    assert [c.case_id for c in result.of_kind("added")] == ["b"]


def test_warnings_for_apples_to_oranges() -> None:
    a = _report({"a": [True]})
    b = _report({"a": [True]}, mode="replay")
    b = b.model_copy(
        update={"meta": b.meta.model_copy(update={"judge_prompt_versions": ["new"], "k": 3})}
    )
    a = a.model_copy(update={"meta": a.meta.model_copy(update={"judge_prompt_versions": ["old"]})})

    result = compare(a, b)
    text = " ".join(result.warnings)

    assert "judge rubric changed (old -> new)" in text
    assert "different modes" in text
    assert "different k" in text
    latency = next(d for d in result.deltas if d.name == "latency mean (s)")
    assert latency.after is None and latency.change is None  # replay: n/a, not "-10s"


def test_expected_ablation_differences_are_notes_not_warnings() -> None:
    a = _report({"a": [True]})
    b = _report({"a": [True]})
    b = b.model_copy(update={"meta": b.meta.model_copy(update={"model": "claude-sonnet-5"})})
    result = compare(a, b)
    assert result.warnings == []
    assert result.notes == ["model: m -> claude-sonnet-5"]


def test_judge_on_vs_off_is_a_warning() -> None:
    a = _report({"a": [True]})
    b = _report({"a": [True]})
    b = b.model_copy(update={"meta": b.meta.model_copy(update={"judged": False})})
    assert "without the judge" in " ".join(compare(a, b).warnings)


# --- resolving report references ---


def _write(tmp_path: Path, sweep: str, when: str) -> Path:
    report = _report({"a": [True]}, sweep=sweep)
    # model_copy doesn't re-validate, so hand it a real datetime.
    meta = report.meta.model_copy(update={"created_at": datetime.fromisoformat(when)})
    report = report.model_copy(update={"meta": meta})
    return write_report(report, tmp_path)[0]


def test_resolve_by_path_prefix_latest_previous(tmp_path: Path) -> None:
    old = _write(tmp_path, "11112222-x", "2026-01-01T00:00:00Z")
    regraded = _write(tmp_path, "11112222-x", "2026-01-02T00:00:00Z")
    new = _write(tmp_path, "33334444-y", "2026-01-03T00:00:00Z")

    assert resolve_report(str(old), tmp_path) == old
    assert resolve_report("1111", tmp_path) == regraded  # newest report of that sweep
    assert resolve_report("11112222-x-full-id", tmp_path) == regraded
    assert resolve_report("latest", tmp_path) == new
    assert resolve_report("previous", tmp_path) == regraded


def test_resolve_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too short"):
        resolve_report("ab", tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_report("ffff", tmp_path)
    with pytest.raises(FileNotFoundError, match="not enough reports"):
        resolve_report("previous", tmp_path)


# --- CLI ---


def test_cli_exits_1_on_regression_and_0_otherwise(tmp_path: Path) -> None:
    good = _write(tmp_path, "aaaa0000-a", "2026-01-01T00:00:00Z")
    report = _report({"a": [False]}, sweep="bbbb0000-b")
    bad = write_report(report, tmp_path)[0]

    regressed = CliRunner().invoke(app, ["eval", "compare", str(good), str(bad)])
    assert regressed.exit_code == 1
    assert "REGRESSIONS (1)" in regressed.stdout
    assert "a: pass -> fail" in regressed.stdout

    fixed = CliRunner().invoke(app, ["eval", "compare", str(bad), str(good)])
    assert fixed.exit_code == 0
    assert "fixes (1)" in fixed.stdout


def test_delta_formatting_by_unit() -> None:
    from evals.compare import Delta

    rate = Delta(name="pass@1", unit="rate", before=0.75, after=0.5)
    assert (rate.fmt(rate.before), rate.fmt(rate.change, signed=True)) == ("75%", "-25pp")
    cost = Delta(name="c", unit="usd", before=0.02, after=0.05)
    assert (cost.fmt(cost.after), cost.fmt(cost.change, signed=True)) == ("$0.0500", "+$0.0300")
    tokens = Delta(name="t", unit="count", before=33166.75, after=None)
    assert (tokens.fmt(tokens.before), tokens.fmt(tokens.after)) == ("33,167", "n/a")
