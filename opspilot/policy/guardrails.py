import re
from collections.abc import Collection

# [HARNESS:GUARD] Heuristic prompt-injection detection -- a signal, not a
# blocker. WHY: no fixed pattern list reliably catches every injection
# attempt (a novel phrasing slips past), so this can never BE the defense --
# it exists to make an attempt visible (a guardrail span in the trace) so a
# human reviewing the run sees exactly what happened and why the agent
# behaved cautiously. The real defense is layer 3: the permission/approval
# policy plus the scope check below, neither of which reads the tool output
# text at all -- they hold even when detection here misses every pattern.
# INTERVIEW: "How do you defend against prompt injection in tool output?" ->
# defense in depth: label untrusted content explicitly (frame_tool_output),
# flag suspicious patterns as a signal (this), and make the actual authority
# boundary blind to prompt text entirely -- permissions.decide() only ever
# sees role, tool risk, and which service is targeted.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore (all |the )?(previous|prior|above) instructions",
        r"disregard (all |the )?(previous|prior|above) instructions",
        r"\bsystem\s*:",
        r"\byou are now\b",
        r"new instructions\s*:",
        r"\bact as\b.{0,20}\binstead\b",
    ]
]


def detect_prompt_injection(text: str) -> list[str]:
    """The regex patterns (source strings) that matched, empty if none."""
    return [pattern.pattern for pattern in _INJECTION_PATTERNS if pattern.search(text)]


INJECTION_WARNING = "⚠ possible prompt injection detected in this output"


def frame_tool_output(source: str, content: str) -> str:
    """Layer 1: input framing. Wraps tool output so it's unambiguous in the
    transcript itself -- not just in AGENTS.md -- that this text is data the
    environment returned, never an instruction to follow, regardless of
    what it claims to be."""
    return f'<tool_output source="{source}" trust="untrusted">\n{content}\n</tool_output>'


def is_in_scope(service: str, *, alert_text: str, known_services: Collection[str]) -> bool:
    """Layer 3 (scope check): has `service` actually been named in the alert
    or investigated (via a successful read tool call) during this run? A
    destructive call on a service that's neither is exactly what a
    successful injection looks like from the outside -- widening the blast
    radius to a service nobody has evidence about. Only widens what's
    allowed, never narrows it -- callers combine this with the normal role
    matrix, never in place of it."""
    return service.lower() in alert_text.lower() or service in known_services
