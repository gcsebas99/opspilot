import hashlib
import json
from pathlib import Path
from typing import Any

from opspilot.context.runbooks import RUNBOOK_INDEX

CONTEXT_DIR = Path(__file__).parent


def _read_agents_md(context_dir: Path) -> str:
    return (context_dir / "AGENTS.md").read_text()


def _render_runbook_index(runbook_index: dict[str, str]) -> str:
    lines = [f"- {name}: {description}" for name, description in sorted(runbook_index.items())]
    return "Available runbooks (load full text with load_runbook):\n" + "\n".join(lines)


def build_system_prompt(
    context_dir: Path = CONTEXT_DIR, runbook_index: dict[str, str] | None = None
) -> str:
    """Plain-text system prompt: AGENTS.md + the runbook index.

    [HARNESS:CONTEXT] Progressive disclosure of skills.
    WHY: a runbook's full text (diagnostic steps, fix guidance) only earns
    its place in context once the model has a specific hypothesis. Putting
    all 4 runbooks in the system prompt up front would burn tokens on 3
    that never get used in a given incident. Instead only the index (name +
    one-line description) is always-present; full text loads on demand via
    the load_runbook tool.
    INTERVIEW: "What is progressive disclosure and why does it save
    context?" -> the model sees a menu, not the full manual; it pays the
    token cost of a runbook's full text only for the one(s) it actually
    decides to load.
    """
    index = RUNBOOK_INDEX if runbook_index is None else runbook_index
    agents_md = _read_agents_md(context_dir)
    return f"{agents_md}\n\n{_render_runbook_index(index)}"


def build_system_blocks(
    context_dir: Path = CONTEXT_DIR, runbook_index: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """System prompt wrapped for the API with a prompt-caching breakpoint.

    [HARNESS:CONTEXT] Prompt caching breakpoint.
    WHY: a Messages API request renders as tools -> system -> messages, so
    one cache_control marker on the last (and only) system block caches
    BOTH the tool schemas and the system prompt together -- no separate
    marker on the tools list is needed. Caching is a prefix match: any byte
    change anywhere before this marker (a reordered/changed tool set, a
    non-deterministic system prompt) invalidates the whole cached prefix,
    which is why build_system_prompt is a pure function of on-disk files --
    no timestamps, no per-request IDs.
    INTERVIEW: "What does prompt caching cache here, and what breaks it?"
    -> everything rendered before the marker (tools + system prompt);
    a `datetime.now()` or changing tool set anywhere in that prefix breaks
    it silently -- verify hits via response.usage.cache_read_input_tokens.
    """
    text = build_system_prompt(context_dir, runbook_index)
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def build_initial_messages(alert: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": f"New alert:\n{alert}"}]


def compute_prompt_version(
    agents_md: str, runbook_index_text: str, tool_schemas: list[dict[str, Any]]
) -> str:
    """Pure hash function -- no file I/O, so it's trivial to test for sensitivity."""
    payload = agents_md + runbook_index_text + json.dumps(tool_schemas, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def prompt_version(
    context_dir: Path,
    tool_schemas: list[dict[str, Any]],
    runbook_index: dict[str, str] | None = None,
) -> str:
    """Reads the real on-disk context, then hashes it via compute_prompt_version."""
    index = RUNBOOK_INDEX if runbook_index is None else runbook_index
    return compute_prompt_version(
        _read_agents_md(context_dir), _render_runbook_index(index), tool_schemas
    )
