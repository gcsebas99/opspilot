import pytest

from opspilot.observability.pricing import cost_usd


def test_cost_usd_matches_docs_worked_example_opus_5() -> None:
    # platform.claude.com/docs/en/about-claude/pricing's worked example:
    # 50,000 input + 15,000 output on Opus 5 -> $0.25 + $0.375
    result = cost_usd("claude-opus-5", input_tokens=50_000, output_tokens=15_000)
    assert result == pytest.approx(0.25 + 0.375)


def test_cost_usd_matches_docs_cache_example_opus_5() -> None:
    # Same doc, cached variant: 10,000 uncached input + 40,000 cache read + 15,000 output
    result = cost_usd(
        "claude-opus-5",
        input_tokens=10_000,
        output_tokens=15_000,
        cache_read_input_tokens=40_000,
    )
    assert result == pytest.approx(0.05 + 0.02 + 0.375)


def test_cost_usd_includes_cache_write() -> None:
    result = cost_usd(
        "claude-sonnet-5", input_tokens=0, output_tokens=0, cache_creation_input_tokens=1_000_000
    )
    assert result == pytest.approx(2.50)


def test_cost_usd_unknown_model_raises() -> None:
    with pytest.raises(KeyError, match="claude-sonnet-5"):
        cost_usd("not-a-real-model", input_tokens=1, output_tokens=1)
