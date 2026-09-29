import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from evals.graders.judge import CRITERIA, JUDGE_PROMPT_VERSION, PASS_THRESHOLD, grade_judge
from evals.loader import load_cases
from evals.models import EvalCase
from opspilot.config import Settings
from opspilot.env.generator import build_sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.models.base import ModelClient
from opspilot.store.models import EvalTrialDoc
from opspilot.tools.registry import build_default_registry

CALIBRATION_PATH = Path(__file__).parent / "judge_calibration.yaml"
CALIBRATION_CASSETTE_ROOT = Path(__file__).parent / "cassettes" / "judge-calibration"

# Agreement bars for calling the judge trustworthy. Within-1 (not exact)
# because humans disagree with *each other* by a point on a 1-5 scale;
# pass/fail must match everywhere because that's the decision evals act on.
MIN_WITHIN_ONE = 0.8
MIN_PASS_AGREEMENT = 1.0


class CalibrationItem(BaseModel):
    id: str
    case: str
    tier: Literal["good", "mediocre", "bad"]
    why: str
    outcome: Literal["completed", "escalated"]
    tool_calls: list[tuple[str, dict[str, Any]]]
    report: dict[str, Any]
    # None = not scored yet by a human.
    expected: dict[str, int | None] = Field(default_factory=dict)

    def is_scored(self) -> bool:
        return all(self.expected.get(c) is not None for c in CRITERIA)


def load_calibration(path: Path = CALIBRATION_PATH) -> list[CalibrationItem]:
    return [CalibrationItem.model_validate(raw) for raw in yaml.safe_load(path.read_text())]


def build_trial(item: CalibrationItem, case: EvalCase, settings: Settings) -> EvalTrialDoc:
    """Run the item's tool calls against a fresh sandbox for its case, so the
    judge sees genuine tool output and final state -- a hand-typed trajectory
    could drift from what the tools actually return."""
    registry = build_default_registry(settings)
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = build_sandbox(get_scenario(case.scenario), case.seed, Path(tmp) / "sbx")
        tool_calls = []
        for name, args in item.tool_calls:
            result = registry.execute(name, args, sandbox)
            if not result.ok:
                raise ValueError(f"{item.id}: {name}({args}) failed: {result.content}")
            tool_calls.append({"name": name, "input": args, "ok": True, "content": result.content})
        facts = sandbox.facts()
    return EvalTrialDoc(
        trial_id=f"calibration:{item.id}",
        sweep_id="calibration",
        suite="calibration",
        case_id=case.id,
        trial=0,
        run_id=f"calibration:{item.id}",
        git_sha="",
        prompt_version="",
        model="hand-written",
        strategy="raw",
        started_at=datetime.now(UTC),
        outcome=item.outcome,
        tool_calls=tool_calls,
        report=item.report,
        sandbox_facts=facts,
    )


class ItemResult(BaseModel):
    id: str
    tier: str
    expected: dict[str, int | None]
    judged: dict[str, int]
    judge_error: str | None = None

    @property
    def expected_avg(self) -> float | None:
        values = [v for v in self.expected.values() if v is not None]
        return sum(values) / len(values) if len(values) == len(CRITERIA) else None

    @property
    def judged_avg(self) -> float:
        return sum(self.judged.values()) / len(self.judged) if self.judged else 0.0


class CalibrationReport(BaseModel):
    judge_model: str
    judge_prompt_version: str
    items: list[ItemResult]
    within_one: float
    mean_abs_error: float
    pass_agreement: float
    per_criterion_mae: dict[str, float]

    @property
    def trusted(self) -> bool:
        return self.within_one >= MIN_WITHIN_ONE and self.pass_agreement >= MIN_PASS_AGREEMENT


def agreement(items: list[ItemResult]) -> tuple[float, float, float, dict[str, float]]:
    """(within-1 rate, mean abs error, pass/fail agreement, per-criterion MAE)
    over items that are both human-scored and successfully judged."""
    scored = [i for i in items if i.expected_avg is not None and i.judged]
    if not scored:
        return 0.0, 0.0, 0.0, {}
    diffs: dict[str, list[int]] = {c: [] for c in CRITERIA}
    for item in scored:
        for c in CRITERIA:
            expected = item.expected[c]
            assert expected is not None
            diffs[c].append(abs(item.judged[c] - expected))
    all_diffs = [d for ds in diffs.values() for d in ds]
    pass_matches = sum(
        (item.judged_avg >= PASS_THRESHOLD) == ((item.expected_avg or 0) >= PASS_THRESHOLD)
        for item in scored
    )
    return (
        sum(d <= 1 for d in all_diffs) / len(all_diffs),
        sum(all_diffs) / len(all_diffs),
        pass_matches / len(scored),
        {c: sum(ds) / len(ds) for c, ds in diffs.items()},
    )


# [HARNESS:EVAL] Judge calibration -- the meta-eval that makes a judge usable.
# WHY: an LLM judge is itself a model with biases (leniency, verbosity
# preference, missing hallucinations). Before its scores gate anything, it
# must agree with human scores on reports whose quality we *know* --
# including a deliberately hallucinated one. Humans score first, blind to
# the judge, so the judge can't anchor them.
# INTERVIEW: "How do you know your LLM judge is trustworthy?" -> a
# human-labeled calibration set; report within-1 agreement, MAE per
# criterion, and pass/fail agreement; re-run it whenever the judge prompt
# or judge model changes.
async def run_judge_check(
    items: list[CalibrationItem],
    *,
    settings: Settings,
    judge_client_for: Callable[[CalibrationItem], ModelClient],
) -> CalibrationReport:
    cases = {case.id: case for case in load_cases("golden")}
    results: list[ItemResult] = []
    for item in items:
        case = cases[item.case]
        trial = build_trial(item, case, settings)
        grade = await grade_judge(
            case, trial, judge_client_for(item), settings.opspilot_judge_model
        )
        criteria = grade.details.get("criteria") or {}
        results.append(
            ItemResult(
                id=item.id,
                tier=item.tier,
                expected=item.expected,
                judged={c: criteria[c]["score"] for c in criteria},
                judge_error=grade.details.get("error"),
            )
        )
    within_one, mae, pass_agreement, per_criterion = agreement(results)
    return CalibrationReport(
        judge_model=settings.opspilot_judge_model,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        items=results,
        within_one=within_one,
        mean_abs_error=mae,
        pass_agreement=pass_agreement,
        per_criterion_mae=per_criterion,
    )
