import json
import operator
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict, cast

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pymongo import MongoClient

from opspilot.config import Settings
from opspilot.context.assembler import build_system_blocks
from opspilot.env.sandbox import Sandbox
from opspilot.loops.react_raw import RunResult, TokenTotals, ToolCallRecord
from opspilot.observability.audit import record_audit
from opspilot.observability.tracer import Tracer, redact_if_large
from opspilot.policy.guardrails import (
    INJECTION_WARNING,
    detect_prompt_injection,
    frame_tool_output,
    is_in_scope,
)
from opspilot.policy.permissions import Allow, Deny, Role, decide
from opspilot.store.base import Store
from opspilot.store.models import ApprovalDoc
from opspilot.tools.base import ToolRegistry, ToolResult

_MARK_MAX_STEPS = "mark_max_steps"
_MARK_BUDGET_EXCEEDED = "mark_budget_exceeded"
_MARK_NO_REPORT = "mark_no_report"


class ApprovalAlreadyDecided(RuntimeError):
    """Another request decided this approval first (see resume_react_graph)."""

    def __init__(self, approval_id: str, *, status: str, approver: str | None) -> None:
        super().__init__(f"approval {approval_id} was already {status} by {approver or 'someone'}")
        self.approval_id = approval_id
        self.status = status
        self.approver = approver


# [HARNESS:MEMORY] Short-term memory: the run's state, checkpointed after every node.
# WHY: the conversation, counters and pending decisions persist through the
# checkpointer, so a paused run resumes intact -- even in another process.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step: int
    tokens_used: dict[str, int]
    outcome: str | None
    report: dict[str, Any] | None
    # None = allowed; a string = blocked (Deny, or a rejected RequireApproval),
    # and is the tool_result content the model sees.
    policy_decisions: dict[str, str | None] | None
    # Populated only for an approved RequireApproval with decision="edit" --
    # tools node uses these args instead of the model's original ones.
    policy_edited_args: dict[str, dict[str, Any]] | None
    role: Role
    run_id: str
    alert: str
    # [HARNESS:GUARD] Services a *successful read* touched this run (scope evidence).
    # WHY: same rule as react_raw.py, one step coarser: the policy node runs before
    # the tools node, so a read can't unlock a destructive call in the same batch.
    known_services: list[str]
    last_tool_signature: tuple[str, str] | None
    repeat_count: int
    nudged_for_stuck: bool
    no_tool_call_strikes: int
    tool_call_records: Annotated[list[dict[str, Any]], operator.add]


def _cumulative_tokens(tokens: dict[str, int]) -> int:
    return sum(
        tokens.get(key, 0)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )


def build_checkpointer(
    settings: Settings,
) -> tuple[BaseCheckpointSaver[Any], MongoClient[Any] | None]:
    """InMemorySaver by default; MongoDBSaver when OPSPILOT_STORE=mongo.

    [HARNESS:OBS] Checkpointer: in-memory, or MongoDBSaver when OPSPILOT_STORE=mongo.
    WHY: only Mongo lets another process resume a paused run. LangGraph 1.x has no
    async Mongo checkpointer, so this takes a sync client (run in a thread executor),
    separate from our async MongoStore. The caller closes the returned client.
    """
    if settings.opspilot_store == "mongo":
        client: MongoClient[Any] = MongoClient(settings.mongodb_uri)
        return MongoDBSaver(client, db_name="opspilot"), client
    return InMemorySaver(), None


def _make_agent_node(model: LanguageModelLike, tracer: Tracer) -> Any:
    async def agent_node(state: AgentState) -> dict[str, Any]:
        async with tracer.span("model_call", "model.create", step=state["step"] + 1) as handle:
            response = await model.ainvoke(state["messages"])
            if not isinstance(response, AIMessage):
                raise TypeError(f"expected AIMessage from model, got {type(response).__name__}")
            usage: dict[str, Any] = dict(response.usage_metadata or {})
            details: dict[str, Any] = dict(usage.get("input_token_details") or {})
            # LangChain's input_tokens is the TOTAL prompt (uncached + cache
            # reads + cache writes) -- unlike Anthropic's raw usage, which is
            # what TokenTotals/cost_usd expect. Subtract the cached parts so
            # cached tokens aren't counted (and priced) twice. ChatAnthropic
            # zeroes `cache_creation` when it reports the per-TTL split.
            output_tokens = usage.get("output_tokens", 0)
            cache_read = details.get("cache_read") or 0
            cache_creation = details.get("cache_creation") or (
                (details.get("ephemeral_5m_input_tokens") or 0)
                + (details.get("ephemeral_1h_input_tokens") or 0)
            )
            input_tokens = usage.get("input_tokens", 0) - cache_read - cache_creation
            handle.set_attr("input_tokens", input_tokens)
            handle.set_attr("output_tokens", output_tokens)
            handle.set_attr("cache_creation_input_tokens", cache_creation)
            handle.set_attr("cache_read_input_tokens", cache_read)
            handle.set_attr("stop_reason", response.response_metadata.get("stop_reason"))

        tokens = dict(state["tokens_used"])
        tokens["input_tokens"] = tokens.get("input_tokens", 0) + input_tokens
        tokens["output_tokens"] = tokens.get("output_tokens", 0) + output_tokens
        tokens["cache_creation_input_tokens"] = (
            tokens.get("cache_creation_input_tokens", 0) + cache_creation
        )
        tokens["cache_read_input_tokens"] = tokens.get("cache_read_input_tokens", 0) + cache_read

        return {"messages": [response], "step": state["step"] + 1, "tokens_used": tokens}

    return agent_node


def _make_route_after_agent(max_steps: int, token_budget: int) -> Any:
    def route_after_agent(state: AgentState) -> str:
        # [HARNESS:LOOP] Step cap and token budget are checked first, on every turn.
        # WHY: a plain-text-only conversation never reaches route_after_tools (the other
        # check), so without this it could run past both limits.
        if state["step"] >= max_steps:
            return _MARK_MAX_STEPS
        if _cumulative_tokens(state["tokens_used"]) > token_budget:
            return _MARK_BUDGET_EXCEEDED
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if tool_calls:
            return "policy"
        if state["no_tool_call_strikes"] >= 1:
            return _MARK_NO_REPORT
        return "nudge_and_retry"

    return route_after_agent


async def _nudge_and_retry_node(state: AgentState) -> dict[str, Any]:
    # [HARNESS:LOOP] Exit condition #5 -- plain text, no tool call.
    # WHY: same contract as the raw loop: one nudge back to "use a tool or finish",
    # then mark_no_report on the next plain-text turn.
    nudge = HumanMessage(
        content=(
            "You did not call a tool. Investigate using the available tools, then "
            "finish with submit_report (or escalate if you can't safely fix this)."
        )
    )
    return {"messages": [nudge], "no_tool_call_strikes": state["no_tool_call_strikes"] + 1}


def _make_policy_node(
    registry: ToolRegistry, tracer: Tracer, store: Store, *, model_name: str, prompt_version: str
) -> Any:
    # [HARNESS:HITL] RequireApproval pauses the run here, via interrupt().
    # WHY: on resume LangGraph re-runs this node from the top, replaying earlier
    # interrupts. So the ApprovalDoc insert is check-then-insert, and the span and
    # audit writes happen once at the end -- never around an interrupt() call.
    async def policy_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            raise TypeError(f"expected AIMessage with tool_calls, got {type(last).__name__}")
        role = state["role"]
        run_id = state["run_id"]
        alert = state["alert"]
        known_services = state["known_services"]
        node_start = datetime.now(UTC)
        decisions: dict[str, str | None] = {}
        edited_args: dict[str, dict[str, Any]] = {}
        # [HARNESS:AUDIT] Denials are collected here and flushed once, at the end.
        # WHY: this node is replayed on resume (see the interrupt note above); writing
        # immediately could audit the same denial twice.
        denial_audit: list[tuple[str, str]] = []

        for tool_call in last.tool_calls:
            call_id = tool_call["id"]
            if call_id is None:
                raise ValueError(f"tool_call for {tool_call['name']!r} is missing an id")
            tool = registry.get(tool_call["name"]) if tool_call["name"] in registry else None
            if tool is None:
                decisions[call_id] = None  # unknown tool -> let registry.execute say so
                continue

            service = tool_call["args"].get("service")
            in_scope = (
                is_in_scope(service, alert_text=alert, known_services=known_services)
                if tool.risk == "destructive" and service is not None
                else True
            )
            decision = decide(role, tool, in_scope=in_scope)
            if isinstance(decision, Allow):
                decisions[call_id] = None
                continue
            if isinstance(decision, Deny):
                decisions[call_id] = decision.reason
                denial_audit.append((tool.name, decision.reason))
                continue

            # RequireApproval: pause for a human.
            approval_id = f"{run_id}:{call_id}"
            existing = await store.get_approval(approval_id)
            if existing is None:
                await store.insert_approval(
                    ApprovalDoc(
                        approval_id=approval_id,
                        run_id=run_id,
                        tool=tool.name,
                        args=tool_call["args"],
                        reason=decision.reason,
                        status="pending",
                        requested_at=datetime.now(UTC),
                    )
                )
            resume = interrupt(
                {
                    "approval_id": approval_id,
                    "tool": tool.name,
                    "args": tool_call["args"],
                    "reason": decision.reason,
                }
            )
            if resume["decision"] == "reject":
                approver = resume.get("approver") or "a human"
                reject_reason = resume.get("reason") or "no reason given"
                decisions[call_id] = f"Action rejected by {approver}: {reject_reason}"
            else:
                decisions[call_id] = None
                if resume["decision"] == "edit":
                    edited_args[call_id] = resume["args"]

        # Only reached by the pass that completes without hitting a new
        # interrupt -- see the WHY note above for why this can't be a
        # context manager wrapping the loop.
        duration_ms = (datetime.now(UTC) - node_start).total_seconds() * 1000
        await tracer.record_span(
            "policy_check",
            "policy",
            start=node_start,
            duration_ms=duration_ms,
            role=role,
            decisions=decisions,
        )
        for tool_name, reason in denial_audit:
            await record_audit(
                store,
                actor=role,
                action="permission_denied",
                target=tool_name,
                decision=reason,
                run_id=run_id,
                prompt_version=prompt_version,
                model=model_name,
                ts=node_start,
            )
        return {
            "policy_decisions": decisions,
            "policy_edited_args": edited_args,
            "no_tool_call_strikes": 0,
        }

    return policy_node


def _make_tools_node(
    registry: ToolRegistry,
    sandbox: Sandbox,
    tracer: Tracer,
    store: Store,
    *,
    model_name: str,
    prompt_version: str,
) -> Any:
    async def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            raise TypeError(f"expected AIMessage with tool_calls, got {type(last).__name__}")
        decisions = state.get("policy_decisions") or {}
        edited_args = state.get("policy_edited_args") or {}
        role = state["role"]
        run_id = state["run_id"]
        tool_messages: list[BaseMessage] = []
        tool_call_records: list[dict[str, Any]] = []
        outcome: str | None = None
        report: dict[str, Any] | None = None
        last_signature = state["last_tool_signature"]
        repeat_count = state["repeat_count"]
        nudged_for_stuck = state["nudged_for_stuck"]
        known_services = list(state["known_services"])

        for tool_call in last.tool_calls:
            name = tool_call["name"]
            call_id = tool_call["id"]
            if call_id is None:
                raise ValueError(f"tool_call for {name!r} is missing an id")
            # A human editing the args during approval (decision="edit")
            # takes precedence over what the model originally sent.
            args = edited_args.get(call_id, tool_call["args"])
            signature = (name, json.dumps(args, sort_keys=True))
            repeat_count = repeat_count + 1 if signature == last_signature else 1
            if signature != last_signature:
                nudged_for_stuck = False
            last_signature = signature

            tool = registry.get(name) if name in registry else None
            denial_reason = decisions.get(call_id)
            async with tracer.span(
                "tool_call", name, tool=name, args=redact_if_large(args)
            ) as handle:
                if denial_reason is not None:
                    result = ToolResult(ok=False, content=denial_reason)
                else:
                    result = registry.execute(name, args, sandbox)
                handle.set_attr("ok", result.ok)
                handle.set_attr("truncated", result.truncated)
                handle.set_attr("output_size", len(result.content))
                if not result.ok:
                    handle.status = "error"

                # [HARNESS:GUARD] Injection detection is a signal, recorded as a guardrail span.
                # WHY: written inside the open tool_call span so it nests under that call.
                injection_patterns = detect_prompt_injection(result.content)
                if injection_patterns:
                    handle.set_attr("injection_patterns", injection_patterns)
                    await tracer.record_span(
                        "guardrail",
                        name,
                        start=datetime.now(UTC),
                        duration_ms=0.0,
                        tool=name,
                        patterns=injection_patterns,
                    )

            service = args.get("service")
            if (
                tool is not None
                and tool.risk == "read"
                and result.ok
                and service
                and service not in known_services
            ):
                known_services.append(service)

            # [HARNESS:AUDIT] Audit every destructive action that actually ran.
            # WHY: "not denied" covers a plain Allow and an approved request alike; this node
            # is never replayed, so it can write the entry directly.
            if tool is not None and tool.risk == "destructive" and denial_reason is None:
                await record_audit(
                    store,
                    actor=role,
                    action="destructive_tool_executed",
                    target=name,
                    decision="executed" if result.ok else "execution failed",
                    run_id=run_id,
                    prompt_version=prompt_version,
                    model=model_name,
                    ts=datetime.now(UTC),
                )

            framed_content = frame_tool_output(name, result.content)
            if injection_patterns:
                framed_content = f"{INJECTION_WARNING}\n{framed_content}"

            tool_call_records.append(
                {"name": name, "input": args, "ok": result.ok, "content": result.content}
            )
            tool_messages.append(
                ToolMessage(
                    content=framed_content,
                    tool_call_id=call_id,
                    status="error" if not result.ok else "success",
                )
            )

            # Mirrors Day 1: a terminal tool only ends the run if it
            # actually succeeded (invalid submit_report args -> ok=False ->
            # not terminal -> model gets a chance to self-correct).
            if tool is not None and tool.risk == "terminal" and result.ok:
                outcome = "completed" if name == "submit_report" else "escalated"
                report = result.data

            # [HARNESS:LOOP] Exit condition #4 -- stuck detection.
            # WHY: same shape as the raw loop: three identical calls get one nudge, a fourth
            # ends the run as `stuck`.
            if repeat_count == 3 and not nudged_for_stuck:
                tool_messages.append(
                    HumanMessage(
                        content=(
                            f"You've called {name} with the same arguments 3 times in a "
                            "row. Try a different tool or approach, or submit_report/"
                            "escalate if you're stuck."
                        )
                    )
                )
                nudged_for_stuck = True
            elif repeat_count >= 4:
                outcome = "stuck"

        return {
            "messages": tool_messages,
            "tool_call_records": tool_call_records,
            "last_tool_signature": last_signature,
            "repeat_count": repeat_count,
            "nudged_for_stuck": nudged_for_stuck,
            "outcome": outcome,
            "report": report,
            "known_services": known_services,
        }

    return tools_node


def _make_route_after_tools(max_steps: int, token_budget: int) -> Any:
    def route_after_tools(state: AgentState) -> str:
        # [HARNESS:LOOP] Exit condition #1 -- the run already has an outcome.
        # WHY: the tools node sets it (terminal tool, or stuck); routing only reads it, so
        # there's one source of truth for *why* the run ended.
        if state["outcome"] is not None:
            return END
        # [HARNESS:LOOP] Exit condition #2 -- hard step cap.
        # WHY: the cheapest backstop against a model that never finishes.
        if state["step"] >= max_steps:
            return _MARK_MAX_STEPS
        # [HARNESS:LOOP] Exit condition #3 -- cumulative token budget (cache included).
        # WHY: steps don't bound cost; one step can carry a huge tool result.
        if _cumulative_tokens(state["tokens_used"]) > token_budget:
            return _MARK_BUDGET_EXCEEDED
        return "agent"

    return route_after_tools


def _mark(outcome: str) -> Any:
    async def node(state: AgentState) -> dict[str, Any]:
        return {"outcome": outcome}

    return node


def build_graph(
    *,
    model: LanguageModelLike,
    registry: ToolRegistry,
    sandbox: Sandbox,
    settings: Settings,
    tracer: Tracer,
    store: Store,
    prompt_version: str,
    max_steps: int | None = None,
    checkpointer: BaseCheckpointSaver[Any],
) -> Any:
    effective_max_steps = settings.opspilot_max_steps if max_steps is None else max_steps
    token_budget = settings.opspilot_token_budget
    model_name = settings.opspilot_model

    graph = StateGraph(AgentState)
    graph.add_node("agent", _make_agent_node(model, tracer))
    graph.add_node("nudge_and_retry", _nudge_and_retry_node)
    graph.add_node(
        "policy",
        _make_policy_node(
            registry, tracer, store, model_name=model_name, prompt_version=prompt_version
        ),
    )
    graph.add_node(
        "tools",
        _make_tools_node(
            registry,
            sandbox,
            tracer,
            store,
            model_name=model_name,
            prompt_version=prompt_version,
        ),
    )
    graph.add_node(_MARK_MAX_STEPS, _mark("max_steps"))
    graph.add_node(_MARK_BUDGET_EXCEEDED, _mark("budget_exceeded"))
    graph.add_node(_MARK_NO_REPORT, _mark("no_report"))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _make_route_after_agent(effective_max_steps, token_budget),
        {
            "policy": "policy",
            "nudge_and_retry": "nudge_and_retry",
            _MARK_NO_REPORT: _MARK_NO_REPORT,
            _MARK_MAX_STEPS: _MARK_MAX_STEPS,
            _MARK_BUDGET_EXCEEDED: _MARK_BUDGET_EXCEEDED,
        },
    )
    graph.add_edge("nudge_and_retry", "agent")
    graph.add_edge("policy", "tools")
    graph.add_conditional_edges(
        "tools",
        _make_route_after_tools(effective_max_steps, token_budget),
        {
            "agent": "agent",
            _MARK_MAX_STEPS: _MARK_MAX_STEPS,
            _MARK_BUDGET_EXCEEDED: _MARK_BUDGET_EXCEEDED,
            END: END,
        },
    )
    graph.add_edge(_MARK_MAX_STEPS, END)
    graph.add_edge(_MARK_BUDGET_EXCEEDED, END)
    graph.add_edge(_MARK_NO_REPORT, END)

    return graph.compile(checkpointer=checkpointer)


def _finalize_or_pause(raw_result: dict[str, Any], sandbox: Sandbox) -> RunResult:
    tokens = TokenTotals(**(raw_result.get("tokens_used") or {}))
    tool_calls = [ToolCallRecord(**tc) for tc in raw_result.get("tool_call_records") or []]
    steps = raw_result.get("step", 0)

    if "__interrupt__" in raw_result:
        # ainvoke() paused mid-node (a policy check hit RequireApproval) --
        # this is not a final outcome, just a snapshot of whatever earlier
        # nodes in this turn already completed.
        payload = dict(raw_result["__interrupt__"][0].value)
        return RunResult(
            outcome="awaiting_approval",
            report=None,
            steps=steps,
            tokens=tokens,
            tool_calls=tool_calls,
            sandbox_snapshot=sandbox.snapshot(),
            pending_approval=payload,
        )

    return RunResult(
        outcome=raw_result["outcome"],
        report=raw_result["report"],
        steps=steps,
        tokens=tokens,
        tool_calls=tool_calls,
        sandbox_snapshot=sandbox.snapshot(),
        pending_approval=None,
    )


async def run_react_graph(
    *,
    model: LanguageModelLike,
    registry: ToolRegistry,
    sandbox: Sandbox,
    alert: str,
    settings: Settings,
    tracer: Tracer,
    store: Store,
    checkpointer: BaseCheckpointSaver[Any],
    run_id: str,
    prompt_version: str,
    role: Role = "viewer",
    max_steps: int | None = None,
) -> RunResult:
    """Build and start the graph, returning the same RunResult shape as the
    raw loop (opspilot/loops/react_raw.py) -- so the CLI, store writes, and
    eval graders (Day 3) treat both strategies identically. If a tool call
    needs human approval, returns with outcome="awaiting_approval" instead
    of blocking forever; call resume_react_graph() once a decision exists.
    `prompt_version` is passed in rather than recomputed here -- the CLI
    computes it once (for RunDoc) and this keeps that the single source of
    truth for what gets attributed on every audit entry this run writes.
    """
    compiled = build_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        settings=settings,
        tracer=tracer,
        store=store,
        prompt_version=prompt_version,
        max_steps=max_steps,
        checkpointer=checkpointer,
    )

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=cast("list[str | dict[Any, Any]]", build_system_blocks())),
            HumanMessage(content=f"New alert:\n{alert}"),
        ],
        "step": 0,
        "tokens_used": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "outcome": None,
        "report": None,
        "policy_decisions": None,
        "policy_edited_args": None,
        "role": role,
        "run_id": run_id,
        "alert": alert,
        "known_services": [],
        "last_tool_signature": None,
        "repeat_count": 0,
        "nudged_for_stuck": False,
        "no_tool_call_strikes": 0,
        "tool_call_records": [],
    }
    config = {"configurable": {"thread_id": run_id}}
    raw_result = await compiled.ainvoke(initial_state, config=config)
    return _finalize_or_pause(raw_result, sandbox)


async def resume_react_graph(
    *,
    model: LanguageModelLike,
    registry: ToolRegistry,
    sandbox: Sandbox,
    settings: Settings,
    tracer: Tracer,
    store: Store,
    checkpointer: BaseCheckpointSaver[Any],
    run_id: str,
    decision: dict[str, Any],
    prompt_version: str,
    max_steps: int | None = None,
) -> RunResult:
    """Resume a run paused at outcome="awaiting_approval".

    `decision` is {"decision": "approve"|"reject"|"edit", "approver": str,
    "reason": str (for reject), "args": dict (for edit)}.

    [HARNESS:HITL] Resume runs OUTSIDE the graph, so it is never replayed.
    WHY: that makes it the one safe place to record the decision exactly once
    (ApprovalDoc, audit entry) and to time the human's wait (approval_wait span),
    even when the approval arrives much later, from another process.
    """
    compiled = build_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        settings=settings,
        tracer=tracer,
        store=store,
        prompt_version=prompt_version,
        max_steps=max_steps,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": run_id}}

    state = await compiled.aget_state(config)
    if not state.tasks or not state.tasks[0].interrupts:
        raise ValueError(f"run {run_id!r} has no pending approval to resume")
    interrupt_payload = state.tasks[0].interrupts[0].value
    approval_id = interrupt_payload["approval_id"]

    approval = await store.get_approval(approval_id)
    if approval is None:
        raise ValueError(f"no ApprovalDoc found for {approval_id!r}")

    decided_at = datetime.now(UTC)
    status = "rejected" if decision["decision"] == "reject" else "approved"
    # [HARNESS:HITL] Exactly-once decision: claim the approval atomically.
    # WHY: decisions can race (double-click, two tabs, two operators); without the
    # claim both would resume and the destructive tool could run twice. Only the
    # request that flips pending -> decided may resume.
    claimed = await store.claim_approval(
        approval_id,
        {"status": status, "decided_at": decided_at, "approver": decision.get("approver")},
    )
    if not claimed:
        current = await store.get_approval(approval_id)
        raise ApprovalAlreadyDecided(
            approval_id,
            status=current.status if current else "unknown",
            approver=current.approver if current else None,
        )
    wait_ms = (decided_at - approval.requested_at).total_seconds() * 1000
    # [HARNESS:OBS] A resumed run gets its own run span, so its spans nest again.
    # WHY: the original run span closed when the run paused (maybe in another
    # process); without this, everything after an approval lands at the top level.
    async with tracer.span(
        "run", "react_graph.resume", decision=decision["decision"], approval_id=approval_id
    ):
        await tracer.record_span(
            "approval_wait",
            approval.tool,
            start=approval.requested_at,
            duration_ms=wait_ms,
            approval_id=approval_id,
            decision=decision["decision"],
        )
        await record_audit(
            store,
            actor=str(decision.get("approver") or "unknown"),
            action="approval_decision",
            target=approval.tool,
            decision=decision["decision"],
            run_id=run_id,
            prompt_version=prompt_version,
            model=settings.opspilot_model,
            ts=decided_at,
        )
        raw_result = await compiled.ainvoke(Command(resume=decision), config=config)
    return _finalize_or_pause(raw_result, sandbox)
