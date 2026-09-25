import json
import operator
from typing import Annotated, Any, TypedDict, cast

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pymongo import MongoClient

from opspilot.config import Settings
from opspilot.context.assembler import build_system_blocks
from opspilot.env.sandbox import Sandbox
from opspilot.loops.react_raw import RunResult, TokenTotals, ToolCallRecord
from opspilot.observability.tracer import Tracer, redact_if_large
from opspilot.policy.permissions import Allow, Role, decide
from opspilot.tools.base import ToolRegistry, ToolResult

_MARK_MAX_STEPS = "mark_max_steps"
_MARK_BUDGET_EXCEEDED = "mark_budget_exceeded"
_MARK_NO_REPORT = "mark_no_report"


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step: int
    tokens_used: dict[str, int]
    outcome: str | None
    report: dict[str, Any] | None
    # Unused until 2.5 wires real interrupt()/Command HITL -- kept in the
    # schema now so that sub-task doesn't need a state migration.
    pending_action: dict[str, Any] | None
    # None = allowed; a string = blocked, and is the tool_result content
    # the model sees (works for both Deny and the interim RequireApproval
    # behavior -- see opspilot/policy/permissions.py).
    policy_decisions: dict[str, str | None] | None
    role: Role
    run_id: str
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

    [HARNESS:OBS] No async MongoDB checkpointer exists in the currently
    installed langgraph-checkpoint-mongodb (AsyncMongoDBSaver was removed
    going into LangGraph 1.0 -- confirmed by trying the import, not assumed
    from docs). MongoDBSaver needs a *sync* pymongo.MongoClient, separate
    from this project's own async MongoStore (2.1) -- two different
    concerns (LangGraph's internal thread/state persistence vs. our own
    runs/spans/audit/approvals). Verified empirically that `await
    graph.ainvoke(...)` works fine with this sync checkpointer -- LangGraph
    bridges sync checkpointer methods through a thread executor.
    The returned client (if any) is the caller's to close.
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
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_creation = details.get("cache_creation", 0)
            cache_read = details.get("cache_read", 0)
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
        # [HARNESS:LOOP] Checked first, unconditionally -- same guarantee as
        # Day 1's "check before every model call": max_steps and the token
        # budget always win, regardless of whether this turn had a tool
        # call, a plain-text reply, or anything else. Without this, a
        # plain-text-only conversation would never visit route_after_tools
        # (the only other place these were checked) and could run past
        # both limits.
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
    # [HARNESS:LOOP] Exit condition #5 -- plain text, no tool call. Mirrors
    # Day 1's raw loop exactly: one nudge back to the contract, then (via
    # route_after_agent, on the *next* no-tool-call turn) mark_no_report.
    nudge = HumanMessage(
        content=(
            "You did not call a tool. Investigate using the available tools, then "
            "finish with submit_report (or escalate if you can't safely fix this)."
        )
    )
    return {"messages": [nudge], "no_tool_call_strikes": state["no_tool_call_strikes"] + 1}


def _make_policy_node(registry: ToolRegistry, tracer: Tracer) -> Any:
    # [HARNESS:PERM] Real permission policy, replacing 2.3's Day-1-parity
    # stub. Role comes from state (set once, for the whole run) rather than
    # a closure param, consistent with "everything flows through state" --
    # see opspilot/policy/permissions.py for why this table, not the
    # prompt, is the actual enforcement point.
    async def policy_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            raise TypeError(f"expected AIMessage with tool_calls, got {type(last).__name__}")
        role = state["role"]
        decisions: dict[str, str | None] = {}
        async with tracer.span("policy_check", "policy", step=state["step"], role=role) as handle:
            for tool_call in last.tool_calls:
                call_id = tool_call["id"]
                if call_id is None:
                    raise ValueError(f"tool_call for {tool_call['name']!r} is missing an id")
                tool = registry.get(tool_call["name"]) if tool_call["name"] in registry else None
                if tool is None:
                    decisions[call_id] = None  # unknown tool -> let registry.execute say so
                else:
                    decision = decide(role, tool)
                    decisions[call_id] = None if isinstance(decision, Allow) else decision.reason
            handle.set_attr("decisions", decisions)
        return {"policy_decisions": decisions, "no_tool_call_strikes": 0}

    return policy_node


def _make_tools_node(registry: ToolRegistry, sandbox: Sandbox, tracer: Tracer) -> Any:
    async def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            raise TypeError(f"expected AIMessage with tool_calls, got {type(last).__name__}")
        decisions = state.get("policy_decisions") or {}
        tool_messages: list[BaseMessage] = []
        tool_call_records: list[dict[str, Any]] = []
        outcome: str | None = None
        report: dict[str, Any] | None = None
        last_signature = state["last_tool_signature"]
        repeat_count = state["repeat_count"]
        nudged_for_stuck = state["nudged_for_stuck"]

        for tool_call in last.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]
            if call_id is None:
                raise ValueError(f"tool_call for {name!r} is missing an id")
            signature = (name, json.dumps(args, sort_keys=True))
            repeat_count = repeat_count + 1 if signature == last_signature else 1
            if signature != last_signature:
                nudged_for_stuck = False
            last_signature = signature

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

            tool_call_records.append(
                {"name": name, "input": args, "ok": result.ok, "content": result.content}
            )
            tool_messages.append(
                ToolMessage(
                    content=result.content,
                    tool_call_id=call_id,
                    status="error" if not result.ok else "success",
                )
            )

            # Mirrors Day 1: a terminal tool only ends the run if it
            # actually succeeded (invalid submit_report args -> ok=False ->
            # not terminal -> model gets a chance to self-correct).
            tool = registry.get(name) if name in registry else None
            if tool is not None and tool.risk == "terminal" and result.ok:
                outcome = "completed" if name == "submit_report" else "escalated"
                report = result.data

            # [HARNESS:LOOP] Exit condition #4 -- stuck detection, same
            # 3-strikes-then-nudge-then-stuck shape as the raw loop.
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
        }

    return tools_node


def _make_route_after_tools(max_steps: int, token_budget: int) -> Any:
    def route_after_tools(state: AgentState) -> str:
        # [HARNESS:LOOP] Exit condition #1 -- terminal tool called (or #4,
        # stuck) already set outcome directly in the tools node; this is
        # just the routing decision, never the source of truth for *why*.
        if state["outcome"] is not None:
            return END
        # [HARNESS:LOOP] Exit condition #2 -- hard step cap.
        if state["step"] >= max_steps:
            return _MARK_MAX_STEPS
        # [HARNESS:LOOP] Exit condition #3 -- cumulative token budget,
        # including cache tokens (see react_raw.py for why).
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
    max_steps: int | None = None,
    checkpointer: BaseCheckpointSaver[Any],
) -> Any:
    effective_max_steps = settings.opspilot_max_steps if max_steps is None else max_steps
    token_budget = settings.opspilot_token_budget

    graph = StateGraph(AgentState)
    graph.add_node("agent", _make_agent_node(model, tracer))
    graph.add_node("nudge_and_retry", _nudge_and_retry_node)
    graph.add_node("policy", _make_policy_node(registry, tracer))
    graph.add_node("tools", _make_tools_node(registry, sandbox, tracer))
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


async def run_react_graph(
    *,
    model: LanguageModelLike,
    registry: ToolRegistry,
    sandbox: Sandbox,
    alert: str,
    settings: Settings,
    tracer: Tracer,
    checkpointer: BaseCheckpointSaver[Any],
    run_id: str,
    role: Role = "viewer",
    max_steps: int | None = None,
) -> RunResult:
    """Build and run the graph once, returning the same RunResult shape as
    the raw loop (opspilot/loops/react_raw.py) -- so the CLI, store writes,
    and eval graders (Day 3) treat both strategies identically.
    """
    compiled = build_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        settings=settings,
        tracer=tracer,
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
        "pending_action": None,
        "policy_decisions": None,
        "role": role,
        "run_id": run_id,
        "last_tool_signature": None,
        "repeat_count": 0,
        "nudged_for_stuck": False,
        "no_tool_call_strikes": 0,
        "tool_call_records": [],
    }
    config = {"configurable": {"thread_id": run_id}}
    final_state = await compiled.ainvoke(initial_state, config=config)

    return RunResult(
        outcome=final_state["outcome"],
        report=final_state["report"],
        steps=final_state["step"],
        tokens=TokenTotals(**final_state["tokens_used"]),
        tool_calls=[ToolCallRecord(**tc) for tc in final_state["tool_call_records"]],
        sandbox_snapshot=sandbox.snapshot(),
    )
