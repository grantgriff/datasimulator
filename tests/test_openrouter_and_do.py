"""Tests for OpenRouter and DigitalOcean Serverless Inference routing."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datasimulator.core.models.llm_client import (
    OpenAICompatibleClient,
    UnifiedLLMClient,
    ClaudeClient,
    OpenAIClient,
    GeminiClient,
    OllamaClient,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Strip any provider env vars so tests are deterministic."""
    for k in ("OPENROUTER_API_KEY", "DO_INFERENCE_KEY", "OPENAI_API_KEY",
              "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_openrouter_prefix_routes_to_compatible_client():
    client = UnifiedLLMClient("openrouter/anthropic/claude-3.5-sonnet",
                              openrouter_api_key="sk-or-test")
    assert isinstance(client.client, OpenAICompatibleClient)
    # The "openrouter/" prefix is stripped before being passed to the API
    assert client.client.model == "anthropic/claude-3.5-sonnet"
    assert client.client._provider_label == "OpenRouter"


def test_do_prefix_routes_to_compatible_client():
    client = UnifiedLLMClient("do/llama3.3-70b-instruct", do_api_key="do-test")
    assert isinstance(client.client, OpenAICompatibleClient)
    assert client.client.model == "llama3.3-70b-instruct"
    assert client.client._provider_label == "DigitalOcean Serverless Inference"


def test_openrouter_prefix_wins_over_gpt_routing():
    """openrouter/openai/gpt-4o must NOT route to direct OpenAI."""
    client = UnifiedLLMClient("openrouter/openai/gpt-4o",
                              openrouter_api_key="sk-or-test",
                              openai_api_key="sk-openai")
    assert isinstance(client.client, OpenAICompatibleClient)
    # If it had hit the direct OpenAI path, it'd be a plain OpenAIClient
    assert not isinstance(client.client.__class__, type(OpenAIClient)) or \
           isinstance(client.client, OpenAICompatibleClient)


def test_direct_openai_still_works():
    client = UnifiedLLMClient("gpt-5.4-mini", openai_api_key="sk-test")
    # Direct OpenAI returns OpenAIClient, NOT the compatible subclass
    assert type(client.client) is OpenAIClient


def test_direct_anthropic_still_works():
    client = UnifiedLLMClient("claude-3-5-sonnet-20241022",
                              anthropic_api_key="sk-ant-test")
    assert isinstance(client.client, ClaudeClient)


def test_direct_gemini_still_works():
    client = UnifiedLLMClient("gemini-2.5-flash", google_api_key="g-test")
    assert isinstance(client.client, GeminiClient)


def test_openrouter_requires_key_when_no_env(monkeypatch):
    """Missing OPENROUTER_API_KEY should fail with a helpful message."""
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        UnifiedLLMClient("openrouter/anthropic/claude-3.5-sonnet")


def test_do_requires_key_when_no_env(monkeypatch):
    with pytest.raises(ValueError, match="DO_INFERENCE_KEY"):
        UnifiedLLMClient("do/llama3.3-70b-instruct")


def test_openrouter_picks_up_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    client = UnifiedLLMClient("openrouter/openai/gpt-4o")
    assert client.client.api_key == "from-env"


def test_do_picks_up_env_var(monkeypatch):
    monkeypatch.setenv("DO_INFERENCE_KEY", "do-env")
    client = UnifiedLLMClient("do/llama3.3-70b-instruct")
    assert client.client.api_key == "do-env"


def test_openrouter_uses_correct_base_url():
    client = UnifiedLLMClient("openrouter/anthropic/claude-3.5-sonnet",
                              openrouter_api_key="sk-or-test")
    # AsyncOpenAI stores base_url with a trailing slash sometimes; just check prefix
    assert str(client.client.client.base_url).startswith("https://openrouter.ai/api/v1")


def test_do_uses_correct_base_url():
    client = UnifiedLLMClient("do/llama3.3-70b-instruct", do_api_key="do-test")
    assert str(client.client.client.base_url).startswith("https://inference.do-ai.run/v1")


def test_cost_uses_provider_reported_value_when_present():
    """OpenRouter returns usage.cost — we should prefer that over pricing tables."""
    client = OpenAICompatibleClient(
        "anthropic/claude-3.5-sonnet",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
    )
    client._reported_cost = 0.00342  # OpenRouter would have set this in generate()
    assert client.estimate_cost(input_tokens=1000, output_tokens=500) == 0.00342


def test_cost_falls_back_to_pricing_table_for_known_do_model():
    client = OpenAICompatibleClient(
        "llama3.3-70b-instruct",
        base_url="https://inference.do-ai.run/v1",
        api_key="do-test",
    )
    # 1M input + 1M output × DO rates ($0.59 + $0.79)
    cost = client.estimate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.59 + 0.79)


def test_cost_returns_zero_for_unknown_model_with_warning(caplog):
    OpenAICompatibleClient._missing_price_warned.clear()
    client = OpenAICompatibleClient(
        "some-exotic-model-we-dont-know",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        provider_label="OpenRouter",
    )
    with caplog.at_level("WARNING"):
        cost = client.estimate_cost(input_tokens=1000, output_tokens=500)
    assert cost == 0.0
    assert any("No pricing known" in r.message for r in caplog.records)


def test_warning_only_fires_once_per_model(caplog):
    OpenAICompatibleClient._missing_price_warned.clear()
    client = OpenAICompatibleClient(
        "exotic-model-warn-once",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
    )
    with caplog.at_level("WARNING"):
        client.estimate_cost(100, 100)
        client.estimate_cost(100, 100)
        client.estimate_cost(100, 100)
    warns = [r for r in caplog.records if "No pricing known" in r.message]
    assert len(warns) == 1
