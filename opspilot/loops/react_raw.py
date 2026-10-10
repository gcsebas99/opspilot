import json
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

from opspilot.config import Settings
from opspilot.context.assembler import build_initial_messages, build_system_blocks
from opspilot.env.sandbox import Sandbox
from opspilot.models.base import ContentBlock, ModelClient, TextBlock, ToolUseBlock
from opspilot.policy.guardrails import (
    INJECTION_WARNING,
    detect_prompt_injection,
    frame_tool_output,
    is_in_scope,
)
from opspilot.policy.permissions import Allow, Deny, RequireApproval, Role, decide
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
    # [HARNESS:MEMORY] Short-term memory, by hand: the messages list *is* the context.
    # WHY: every turn appends the model's reply and tool results, and the whole list
    # is resent each call -- the model remembers nothing between calls on its own.
    messages: list[dict[str, Any]] = build_initial_messages(alert)

    totals = TokenTotals()
    tool_calls: list[ToolCallRecord] = []
    steps = 0
    no_tool_call_strikes = 0
    last_signature: tuple[str, str] | None = None
    repeat_count = 0
    nudged_for_stuck = False
    # [HARNESS:GUARD] Services a *successful read* touched this run (scope evidence).
    # WHY: only successful reads count, never a destructive attempt -- otherwise one
    # blocked call could "unlock" an out-of-scope service for the next try.
    known_services: set[str] = set()

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
        # WHY: a confused model can loop forever and burn tokens; this is the cheapest
        # backstop, checked before each model call rather than after.
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
        # WHY: steps don't bound cost (one step can carry a huge tool result). Cache
        # tokens count too: cheaper per token, but still real context growth.
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
        # WHY: the contract is "end with submit_report or escalate". Nudge once (models
        # often self-correct), then stop as no_report so evals can tell it apart.
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
        # [HARNESS:LOOP] Parallel tool calls: run them all, answer in ONE message.
        # WHY: the API pairs each tool_result to its tool_use_id within one user turn;
        # splitting them across messages breaks that and discourages parallel calls.
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
            # [HARNESS:PERM] Enforcement point: the policy decision is applied, not suggested.
            # WHY: the model never sees the policy table, only the tool_result. This loop has
            # no checkpointer to pause on, so RequireApproval acts like Deny here; the graph
            # strategy implements the real pause/resume.
            policy_decision: str | None = None
            denial_reason: str | None = None
            if tool is None:
                result = registry.execute(block.name, block.input, sandbox)
            else:
                service = block.input.get("service")
                in_scope = (
                    is_in_scope(service, alert_text=alert, known_services=known_services)
                    if tool.risk == "destructive" and service is not None
                    else True
                )
                decision = decide(role, tool, in_scope=in_scope)
                if isinstance(decision, Allow):
                    policy_decision = "allow"
                    result = registry.execute(block.name, block.input, sandbox)
                elif isinstance(decision, Deny):
                    policy_decision = "deny"
                    denial_reason = decision.reason
                    result = ToolResult(ok=False, content=decision.reason)
                else:
                    assert isinstance(decision, RequireApproval)
                    policy_decision = "require_approval"
                    denial_reason = decision.reason
                    result = ToolResult(ok=False, content=decision.reason)
                if tool.risk == "read" and result.ok and service:
                    known_services.add(service)
            tool_duration_ms = (time.monotonic() - tool_start) * 1000

            # [HARNESS:GUARD] Every tool result is framed as untrusted data.
            # WHY: framing applies always; the injection warning is prepended only when a
            # known pattern matches (detection is a signal, not the defense).
            injection_patterns = detect_prompt_injection(result.content)
            framed_content = frame_tool_output(block.name, result.content)
            if injection_patterns:
                framed_content = f"{INJECTION_WARNING}\n{framed_content}"

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
                injection_patterns=injection_patterns,
                risk=tool.risk if tool is not None else None,
                policy_decision=policy_decision,
                denial_reason=denial_reason,
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
                    "content": framed_content,
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
            # WHY: the same tool with the same args learns nothing new. Three in a row gets
            # one nudge; a fourth ends the run as `stuck` instead of burning the budget.
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

        # [HARNESS:LOOP] Exit condition #1 -- terminal tool called (the happy path).
        # WHY: submit_report / escalate are the only intended way to finish; every
        # other exit above is a backstop against never getting here.
        if terminal is not None:
            name, result = terminal
            outcome: Outcome = "completed" if name == "submit_report" else "escalated"
            return finish(outcome, report=result.data)
