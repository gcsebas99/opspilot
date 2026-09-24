from pydantic import BaseModel

_PER_MILLION = 1_000_000


class ModelPricing(BaseModel):
    """USD per million tokens. `cache_write_1h` exists for completeness --
    this project only ever requests the 5-minute (ephemeral) cache, via
    context/assembler.py's cache_control breakpoint, so cost_usd() below
    never reads it.
    """

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float


# Verified against https://platform.claude.com/docs/en/about-claude/pricing
# on 2026-09-24 -- re-check that page before trusting stale numbers here.
PRICING: dict[str, ModelPricing] = {
    "claude-fable-5-1": ModelPricing(
        input=10.0, cache_write_5m=12.50, cache_write_1h=20.0, cache_read=0.25, output=50.0
    ),
    "claude-fable-5": ModelPricing(
        input=10.0, cache_write_5m=12.50, cache_write_1h=20.0, cache_read=1.0, output=50.0
    ),
    "claude-opus-5-5": ModelPricing(
        input=4.0, cache_write_5m=5.0, cache_write_1h=8.0, cache_read=0.20, output=20.0
    ),
    "claude-opus-5": ModelPricing(
        input=5.0, cache_write_5m=6.25, cache_write_1h=10.0, cache_read=0.50, output=25.0
    ),
    "claude-opus-4-8": ModelPricing(
        input=5.0, cache_write_5m=6.25, cache_write_1h=10.0, cache_read=0.50, output=25.0
    ),
    "claude-opus-4-7": ModelPricing(
        input=5.0, cache_write_5m=6.25, cache_write_1h=10.0, cache_read=0.50, output=25.0
    ),
    "claude-opus-4-6": ModelPricing(
        input=5.0, cache_write_5m=6.25, cache_write_1h=10.0, cache_read=0.50, output=25.0
    ),
    "claude-sonnet-5": ModelPricing(
        input=2.0, cache_write_5m=2.50, cache_write_1h=4.0, cache_read=0.20, output=10.0
    ),
    "claude-sonnet-4-6": ModelPricing(
        input=3.0, cache_write_5m=3.75, cache_write_1h=6.0, cache_read=0.30, output=15.0
    ),
    "claude-haiku-4-5": ModelPricing(
        input=1.0, cache_write_5m=1.25, cache_write_1h=2.0, cache_read=0.10, output=5.0
    ),
}


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """Cost of one model call in USD, given its usage breakdown."""
    try:
        pricing = PRICING[model]
    except KeyError as exc:
        known = ", ".join(sorted(PRICING))
        raise KeyError(f"no pricing for model {model!r}; known models: {known}") from exc

    return (
        input_tokens * pricing.input
        + output_tokens * pricing.output
        + cache_creation_input_tokens * pricing.cache_write_5m
        + cache_read_input_tokens * pricing.cache_read
    ) / _PER_MILLION
