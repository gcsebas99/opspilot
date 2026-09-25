"""Module-level compiled graph for `langgraph dev` (Studio) visual debugging.

Not used by the CLI -- `opspilot run` builds a fresh graph per invocation,
tied to a specific model/registry/sandbox/tracer/checkpointer for one real
run. This module exists only so `langgraph dev` has something importable to
load, using a fixed default scenario/seed and a throwaway sandbox dir.
Kept separate from graph.py itself so importing graph.py (as every test in
this project does) never has this module's side effects (a real API key,
a sandbox directory on disk) attached.
"""

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

from opspilot.config import get_settings
from opspilot.env.generator import build_sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.loops.graph import build_graph
from opspilot.observability.tracer import Tracer
from opspilot.store.memory import MemoryStore
from opspilot.tools.registry import build_default_registry

_settings = get_settings()
_registry = build_default_registry(_settings)
_scenario = get_scenario("checkout_pool_exhaustion")
_sandbox = build_sandbox(_scenario, seed=42, root=Path("tmp/studio-sbx"))
_chat_model = ChatAnthropic(
    model=_settings.opspilot_model, max_tokens=8192, api_key=_settings.anthropic_api_key
)
_bound_model = _chat_model.bind_tools(_registry.to_anthropic_schema())
_store = MemoryStore()
_tracer = Tracer(_store, run_id="studio")

GRAPH = build_graph(
    model=_bound_model,
    registry=_registry,
    sandbox=_sandbox,
    settings=_settings,
    tracer=_tracer,
    store=_store,
    checkpointer=InMemorySaver(),
)
