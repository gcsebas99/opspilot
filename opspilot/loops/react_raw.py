import json
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

from opspilot.config import Settings
from opspilot.context.assembler import build_initial_messages, build_system_blocks
from opspilot.env.sandbox import Sandbox
from opspilot.models.base import ContentBlock, ModelClient, TextBlock, ToolUseBlock
from opspilot.policy.permissions import Allow, Role, decide
from opspilot.tools.base import ToolRegistry, ToolResult

Outcome = Literal[
    "completed",
    "escalated",
    "max_steps",
    "budget_exceeded",
    "stuck",
    "no_report",
    "awaiting_approval",
]
EventType = Literal["step_start", "model_call", "tool_call", "exit"]


class TokenTotals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ToolCallRecord(BaseModel):
    name: str
    input: dict[str, Any]
    ok: bool
    content: str


class LoopEvent(BaseModel):
    type: EventType
    data: dict[str, Any]


class RunResult(BaseModel):
    outcome: Outcome
    report: dict[str, Any] | None
    steps: int
    tokens: TokenTotals
    tool_calls: list[ToolCallRecord]
    sandbox_snapshot: dict[str, str]
    # Set only when outcome == "awaiting_approval" (graph strategy only --
    # see opspilot/loops/graph.py). The raw loop never produces this.
    pending_approval: dict[str, Any] | None = None


def _serialize_content(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            serialized.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            serialized.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
    return serialized


async def run_react_loop(
    *,
    model: ModelClient,
    registry: ToolRegistry,
    sandbox: Sandbox,
    alert: str,
    settings: Settings,
    role: Role = "viewer",
    max_steps: int | None = None,
    on_event: Callable[[LoopEvent], None] | None = None,
) -> RunResult:
    """think -> act -> observe -> repeat, against a live or scripted model.

    This function IS the agent, top to bottom. Every exit condition below is
    explicit, tagged, and has its own test -- no implicit "it just stops
    eventually" behavior.
    """
    effective_max_steps = settings.opspilot_max_steps if max_steps is None else max_steps
    token_budget = settings.opspilot_token_budget

    system = build_system_blocks()
    tools_schema = registry.to_anthropic_schema()
    messages: list[dict[str, Any]] = build_initial_messages(alert)

    totals = TokenTotals()
    tool_calls: list[ToolCallRecord] = []
    steps = 0
    no_tool_call_strikes = 0
    last_signature: tuple[str, str] | None = None
    repeat_count = 0
    nudged_for_stuck = False

    def emit(event_type: EventType, **data: Any) -> None:
        if on_event is not None:
            on_event(LoopEvent(type=event_type, data=data))

    def finish(outcome: Outcome, report: dict[str, Any] | None = None) -> RunResult:
        emit("exit", outcome=outcome, steps=steps)
        return RunResult(
            outcome=outcome,
            report=report,
            steps=steps,
            tokens=totals,
            tool_calls=tool_calls,
            sandbox_snapshot=sandbox.snapshot(),
        )

    while True:
        # [HARNESS:LOOP] Exit condition #2 -- hard step cap.
        # WHY: a confused model can loop forever and burn tokens. This is
        # the cheapest guardrail -- checked before spending another model
        # call, not after.
        # INTERVIEW: "How do you stop runaway agents?" -> step cap, token
        # budget, stuck detection, and a required terminal tool call --
        # four independent backstops, not just one.
        if steps >= effective_max_steps:
            return finish("max_steps")

        steps += 1
        emit("step_start", step=steps)

        # think
        response = await model.create(system=system, messages=messages, tools=tools_schema)
        emit(
            "model_call",
            step=steps,
            stop_reason=response.stop_reason,
            usage=response.usage.model_dump(),
            latency_ms=response.latency_ms,
        )

        totals.input_tokens += response.usage.input_tokens
        totals.output_tokens += response.usage.output_tokens
        totals.cache_creation_input_tokens += response.usage.cache_creation_input_tokens
        totals.cache_read_input_tokens += response.usage.cache_read_input_tokens

        # [HARNESS:LOOP] Exit condition #3 -- cumulative token budget.
        # WHY: the step cap alone doesn't bound cost -- a single step can
        # burn a huge number of tokens (e.g. a giant tool result). Track
        # spend across the whole run, independent of step count, and
        # include cache tokens: they're cheaper per-token but still
        # represent real context growth.
        # INTERVIEW: "What if the model is expensive but doesn't loop
        # forever?" -> a separate cumulative token budget, because steps
        # and tokens are not the same resource to bound.
        cumulative = (
            totals.input_tokens
            + totals.output_tokens
            + totals.cache_creation_input_tokens
            + totals.cache_read_input_tokens
        )
        if cumulative > token_budget:
            return finish("budget_exceeded")

        messages.append({"role": "assistant", "content": _serialize_content(response.content)})

        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]

        # [HARNESS:LOOP] Exit condition #5 -- plain text, no tool call.
        # WHY: this agent's contract is "always end with submit_report or
        # escalate." A model that just chats instead of acting has drifted
        # off-task -- nudge once (models often self-correct when told
        # explicitly what's expected), but don't nudge forever.
        # INTERVIEW: "What if the model just talks instead of calling a
        # tool?" -> one explicit nudge back to the contract, then a
        # distinct outcome (no_report) so evals can tell this apart from a
        # genuine completed/escalated run.
        if not tool_use_blocks:
            no_tool_call_strikes += 1
            if no_tool_call_strikes >= 2:
                return finish("no_report")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You did not call a tool. Investigate using the available tools, then "
                        "finish with submit_report (or escalate if you can't safely fix this)."
                    ),
                }
            )
            continue
        no_tool_call_strikes = 0

        # act + observe
        # [HARNESS:LOOP] Parallel tool calls.
        # WHY: the model may return several tool_use blocks in one turn.
        # All of them run, and all their results go back in ONE user
        # message, each paired to its tool_use_id -- splitting tool_result
        # blocks across multiple messages is not what the API expects and
        # discourages the model from making parallel calls again.
        # INTERVIEW: "How do you handle multiple tool calls in one turn?"
        # -> execute every one, collect every tool_result into a single
        # user message preserving tool_use_id pairing.
        result_blocks: list[dict[str, Any]] = []
        terminal: tuple[str, ToolResult] | None = None
        stuck = False

        for block in tool_use_blocks:
            signature = (block.name, json.dumps(block.input, sort_keys=True))
            repeat_count = repeat_count + 1 if signature == last_signature else 1
            if signature != last_signature:
                nudged_for_stuck = False
            last_signature = signature

            tool_start = time.monotonic()
            tool = registry.get(block.name) if block.name in registry else None
            # [HARNESS:PERM] Enforcement point: a decision from decide() is
            # a block, not a suggestion -- the model never sees the policy
            # table, only the tool_result it produces. See
            # opspilot/policy/permissions.py for why this lives here and
            # not in the prompt. RequireApproval is treated like Deny here,
            # permanently -- this hand-written loop has no checkpointer to
            # pause on, so it can't offer real HITL. The graph strategy
            # (opspilot/loops/graph.py) implements the real interrupt/resume
            # flow for RequireApproval (2.5); that's one of the concrete
            # reasons this project ported to LangGraph in the first place.
            if tool is None:
                result = registry.execute(block.name, block.input, sandbox)
            else:
                decision = decide(role, tool)
                if isinstance(decision, Allow):
                    result = registry.execute(block.name, block.input, sandbox)
                else:
                    result = ToolResult(ok=False, content=decision.reason)
            tool_duration_ms = (time.monotonic() - tool_start) * 1000

            # Emitted after execution (not before, as in earlier Day 1 code)
            # so observability consumers (2.2) get real duration/result data
            # in the same event, rather than needing a second "tool_call_end".
            emit(
                "tool_call",
                step=steps,
                name=block.name,
                input=block.input,
                ok=result.ok,
                duration_ms=tool_duration_ms,
                truncated=result.truncated,
                output_size=len(result.content),
            )

            tool_calls.append(
                ToolCallRecord(
                    name=block.name, input=block.input, ok=result.ok, content=result.content
                )
            )
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.content,
                    "is_error": not result.ok,
                }
            )

            # A terminal tool only ends the run if it actually succeeded --
            # e.g. submit_report called with an out-of-range confidence
            # fails pydantic validation (ok=False) and the loop continues,
            # letting the model self-correct rather than exiting on a
            # failed terminal call.
            if tool is not None and tool.risk == "terminal" and result.ok:
                terminal = (block.name, result)

            # [HARNESS:LOOP] Exit condition #4 -- stuck detection.
            # WHY: a model can loop calling the identical tool with
            # identical arguments, learning nothing new each time (e.g.
            # re-running the same grep because it misread the output). One
            # nudge gives it a chance to self-correct; a 4th identical call
            # after that nudge means it's genuinely stuck, not re-verifying.
            # INTERVIEW: "How do you detect an agent stuck in a loop?" ->
            # track (tool_name, canonical_args) of the last call; 3 in a
            # row triggers one nudge, a 4th ends the run as `stuck` instead
            # of burning the rest of the step/token budget on repetition.
            if repeat_count == 3 and not nudged_for_stuck:
                result_blocks.append(
                    {
                        "type": "text",
                        "text": (
                            f"You've called {block.name} with the same arguments 3 times in a "
                            "row. Try a different tool or approach, or submit_report/escalate "
                            "if you're stuck."
                        ),
                    }
                )
                nudged_for_stuck = True
            elif repeat_count >= 4:
                stuck = True

        messages.append({"role": "user", "content": result_blocks})

        if stuck:
            return finish("stuck")

        # [HARNESS:LOOP] Exit condition #1 -- terminal tool called.
        # WHY: submit_report/escalate are the agent's only intended way to
        # finish -- everything else above is a guardrail against NOT
        # reaching this point.
        # INTERVIEW: "What's the happy path exit?" -> the model calls
        # submit_report (outcome=completed) or escalate (outcome=escalated)
        # and the loop returns immediately with that tool's structured data
        # as the report.
        if terminal is not None:
            name, result = terminal
            outcome: Outcome = "completed" if name == "submit_report" else "escalated"
            return finish(outcome, report=result.data)
