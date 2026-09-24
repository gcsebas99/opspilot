from opspilot.config import Settings, get_settings
from opspilot.tools.base import ToolRegistry
from opspilot.tools.destructive import DESTRUCTIVE_TOOLS
from opspilot.tools.read import READ_TOOLS
from opspilot.tools.terminal import TERMINAL_TOOLS


def build_default_registry(settings: Settings | None = None) -> ToolRegistry:
    """Wire up every implemented tool into one registry.

    `load_runbook` isn't here yet -- it ships in 1.4 alongside the runbook
    content it reads.
    """
    settings = settings or get_settings()
    registry = ToolRegistry(max_output_chars=settings.opspilot_tool_output_max_chars)
    for tool in [*READ_TOOLS, *DESTRUCTIVE_TOOLS, *TERMINAL_TOOLS]:
        registry.register(tool)
    return registry
