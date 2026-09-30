from pathlib import Path

from evals.report import build_report, failed_checks, load_report, render_markdown, write_report
from tests.evals.test_metrics import _CASES, _sweep, _trial


def test_report_round_trips_through_json(tmp_path: Path) -> None:
    report = build_report(_sweep(), _CASES, suite="golden", judged=True)

    json_path, md_path = write_report(report, tmp_path)

    assert load_report(json_path) == report
    assert json_path.stem == md_path.stem
    assert json_path.name.endswith(f"-golden-{report.meta.sweep_id[:8]}.json")


def test_meta_and_per_trial_summaries() -> None:
    report = build_report(_sweep(), _CASES, suite="golden", judged=True)

    assert report.meta.k == 3 and report.meta.judged
    assert report.meta.prompt_versions == ["v"]
    assert [(t.case_id, t.trial) for t in report.trials][:3] == [("a", 0), ("a", 1), ("a", 2)]
    failing = next(t for t in report.trials if not t.passed)
    assert failing.scores["trajectory"] == 0.5
    assert failing.judge_avg == 4.0 and failing.judge_cost_usd == 0.01


def test_markdown_has_summary_tables_and_verdicts() -> None:
    md = render_markdown(build_report(_sweep(), _CASES, suite="golden", judged=True))

    assert "| 44% | 67% | 33% | 3 | 9 | 0 |" in md  # pass@1 | pass@k | pass^k
    assert "| b | flaky | 1/3 |" in md
    assert "| adversarial | 2 |" in md
    assert "Mean judge score: **4.00**" in md
    assert "k=1" not in md


def test_markdown_labels_replay_numbers_and_k1() -> None:
    trials = [_trial("a", 0, True, mode="replay")]
    md = render_markdown(build_report(trials, _CASES, suite="smoke", judged=False))

    assert "agent cost (recorded)¹" in md
    assert "spent $0" in md
    assert "| latency (s)² | n/a | n/a |" in md
    assert "k=1: pass@1, pass@k and pass^k are the same number" in md
    assert "judge:** off" in md


def test_failure_reasons_are_listed_and_pipes_escaped() -> None:
    grades = {
        "trajectory": {
            "passed": False,
            "score": 0.5,
            "details": {
                "checks": [{"name": "must_call: grep_logs", "passed": False, "detail": "a|b"}]
            },
        },
        "outcome": {"passed": False, "score": 0.0, "details": {"match": "wrong service"}},
    }
    assert failed_checks(grades) == [
        "trajectory: must_call: grep_logs (a|b)",
        "outcome: wrong service",
    ]
    trial = _trial("c", 0, False).model_copy(update={"grades": grades})
    md = render_markdown(build_report([trial], _CASES, suite="golden", judged=False))
    assert "must_call: grep_logs (a\\|b); outcome: wrong service" in md
