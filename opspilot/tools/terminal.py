from pydantic import BaseModel, Field

from opspilot.env.sandbox import Sandbox
from opspilot.tools.base import Tool, ToolResult


class SubmitReportInput(BaseModel):
    root_cause: str = Field(
        description="Concise root cause label, e.g. 'config_change:checkout:db_pool_size'."
    )
    evidence: list[str] = Field(
        description="Specific observations backing the root cause (log lines, metrics, diffs)."
    )
    actions_taken: list[str] = Field(
        default_factory=list, description="Tool calls actually made to fix the issue, if any."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the root cause, 0-1.")
    recommendation: str = Field(
        description="What should happen next (follow-up, monitoring, none)."
    )


def _submit_report(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    assert isinstance(args, SubmitReportInput)
    return ToolResult(ok=True, content="report submitted", data=args.model_dump())


SUBMIT_REPORT = Tool(
    name="submit_report",
    description=(
        "Submit your final incident report and end the investigation. Call this once "
        "you've identified the root cause (or confirmed there isn't one) and, if "
        "applicable, applied a fix. This is a terminal action -- no more tool calls "
        "happen after this."
    ),
    input_model=SubmitReportInput,
    risk="terminal",
    fn=_submit_report,
)


class EscalateInput(BaseModel):
    reason: str = Field(
        description="Why this needs a human (e.g. no safe tool fixes it, or cause is ambiguous)."
    )


def _escalate(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    assert isinstance(args, EscalateInput)
    return ToolResult(ok=True, content=f"escalated: {args.reason}", data={"reason": args.reason})


ESCALATE = Tool(
    name="escalate",
    description=(
        "Hand this incident off to a human instead of taking further action. Use when "
        "no available tool can safely fix the root cause (e.g. disk full), or when "
        "you're not confident enough to act. This is a terminal action -- no more tool "
        "calls happen after this."
    ),
    input_model=EscalateInput,
    risk="terminal",
    fn=_escalate,
)

TERMINAL_TOOLS = [SUBMIT_REPORT, ESCALATE]
