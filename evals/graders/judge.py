import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from evals.graders.base import Grade
from evals.models import EvalCase
from opspilot.env.scenarios import get_scenario
from opspilot.models.base import ModelClient, ToolUseBlock
from opspilot.observability.pricing import cost_usd
from opspilot.store.models import EvalTrialDoc

JUDGE_PROMPT = (Path(__file__).parent / "judge_prompt.md").read_text()
CRITERIA = (
    "evidence_cited",
    "causal_chain",
    "actionable_recommendation",
    "calibrated_confidence",
    "no_hallucination",
)
PASS_THRESHOLD = 3.5
# Each tool output is cut to this many chars -- enough for the judge to
# check a cited log line or metric, without paying for 4k-char dumps.
_TOOL_OUTPUT_CHARS = 800


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str


class JudgeVerdict(BaseModel):
    evidence_cited: CriterionScore
    causal_chain: CriterionScore
    actionable_recommendation: CriterionScore
    calibrated_confidence: CriterionScore
    no_hallucination: CriterionScore

    def average(self) -> float:
        scores: list[int] = [getattr(self, c).score for c in CRITERIA]
        return sum(scores) / len(scores)


# Written out by hand rather than JudgeVerdict.model_json_schema(): pydantic
# emits $defs/$ref for the nested model, and a flat, explicit schema is both
# easier to read here and one less thing for the API to interpret.
_CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
}
JUDGE_TOOL: dict[str, Any] = {
    "name": "submit_grades",
    "description": "Submit a 1-5 score and short rationale for every rubric criterion.",
    "input_schema": {
        "type": "object",
        "properties": {c: _CRITERION_SCHEMA for c in CRITERIA},
        "required": list(CRITERIA),
    },
}
_TOOL_CHOICE = {"type": "tool", "name": "submit_grades", "disable_parallel_tool_use": True}

# [HARNESS:EVAL] Versioned judge -- the prompt + output schema are hashed.
# WHY: a judge is part of the measuring instrument. If the rubric changes,
# scores before and after aren't comparable; storing this hash on every
# grade makes "the judge changed" visible in `eval compare` (3.5) instead of
# masquerading as an agent regression.
JUDGE_PROMPT_VERSION = hashlib.sha256(
    (JUDGE_PROMPT + json.dumps(JUDGE_TOOL, sort_keys=True)).encode()
).hexdigest()[:12]


def _render_trajectory(trial: EvalTrialDoc) -> str:
    lines = []
    for i, call in enumerate(trial.tool_calls, 1):
        output = str(call.get("content", ""))
        if len(output) > _TOOL_OUTPUT_CHARS:
            output = output[:_TOOL_OUTPUT_CHARS] + f"... [{len(output)} chars total]"
        status = "ok" if call.get("ok") else "ERROR/DENIED"
        lines.append(f"{i}. {call.get('name')}({json.dumps(call.get('input'))}) -> {status}")
        lines.append(f"   {output}")
    return "\n".join(lines) or "(no tool calls)"


def build_judge_input(case: EvalCase, trial: EvalTrialDoc) -> str:
    """Everything the judge needs to grade the report *against reality*:
    ground truth, final state, and what the agent actually saw."""
    scenario = get_scenario(case.scenario)
    return f"""## Ground truth (hidden from the agent)
root_cause_label: {scenario.root_cause_label}
fault: {scenario.fault_description}
correct fix: {scenario.expected_fix}
expected outcome for this case: {case.expect.outcome}
agent role: {case.role} (viewer cannot run destructive tools; operator needs approval)

## Case notes from the reviewer
{case.rubric_notes or "(none)"}

## Final system state
{json.dumps(trial.sandbox_facts, indent=2, sort_keys=True)}

<agent_transcript>
## Trajectory
{_render_trajectory(trial)}

## Report (run outcome: {trial.outcome})
{json.dumps(trial.report, indent=2, sort_keys=True)}
</agent_transcript>
"""


def _failed(reason: str, **details: Any) -> Grade:
    return Grade(
        name="judge",
        passed=False,
        score=0.0,
        details={"error": reason, "judge_prompt_version": JUDGE_PROMPT_VERSION, **details},
    )


# [HARNESS:EVAL] LLM-as-judge for what code can't check -- report quality.
# WHY: "is the causal chain right, is the evidence real, is the confidence
# honest?" has no regex. The judge gets ground truth + the full trajectory
# (so it can catch hallucinated facts) and must answer through a forced
# tool call validated by pydantic -- a malformed verdict is a failed grade,
# never a silently-accepted guess.
# INTERVIEW: "How do you trust an LLM judge?" -> rubric with anchors,
# ground truth in the prompt, structured output, versioned prompt, and a
# calibration set of human-scored reports it must agree with (judge-check).
async def grade_judge(
    case: EvalCase, trial: EvalTrialDoc, client: ModelClient, judge_model: str
) -> Grade:
    if trial.report is None:
        # No report, nothing to judge -- and no reason to pay for a call.
        return _failed("no report to judge", judge_model=judge_model)

    # Client errors (API failure, CassetteMiss in replay) propagate on
    # purpose: "the judge couldn't run" is an infrastructure failure, not a
    # verdict on the agent, and must not be recorded as one.
    response = await client.create(
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": build_judge_input(case, trial)}],
        tools=[JUDGE_TOOL],
        tool_choice=_TOOL_CHOICE,
    )
    tool_use = next((b for b in response.content if isinstance(b, ToolUseBlock)), None)
    if tool_use is None:
        return _failed("judge did not call submit_grades", judge_model=judge_model)
    try:
        verdict = JudgeVerdict.model_validate(tool_use.input)
    except ValidationError as exc:
        return _failed(f"malformed verdict: {exc.error_count()} error(s)", raw=tool_use.input)

    average = verdict.average()
    return Grade(
        name="judge",
        passed=average >= PASS_THRESHOLD,
        # 1..5 average mapped onto the Grade's 0..1 score.
        score=(average - 1) / 4,
        details={
            "average": average,
            "criteria": verdict.model_dump(),
            "judge_model": judge_model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "usage": response.usage.model_dump(),
            # Kept apart from the trial's cost_usd (the agent's spend) so
            # reports can show what grading itself costs.
            "cost_usd": cost_usd(judge_model, **response.usage.model_dump()),
        },
    )
