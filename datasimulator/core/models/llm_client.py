"""
Unified LLM client supporting multiple providers.

Supports:
- Anthropic Claude (claude-3-5-sonnet, etc.)
- OpenAI (gpt-4, gpt-4o, gpt-4o-mini, etc.)
- Google Gemini (gemini-2.5-pro, etc.)
- OpenRouter (openrouter/<provider>/<model>) — OpenAI-compatible, one bill
  for any model. Cost is read directly from response.usage.cost.
- Cloudflare Workers AI (cf/<model>) — OpenAI-compatible, edge-hosted
  open-source models. Requires CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID.
- DigitalOcean Serverless Inference (do/<model>) — OpenAI-compatible, hosts
  open-source models (Llama, Mistral, DeepSeek, Qwen).
- Ollama (local models: qwen2.5, llama3, etc.)
"""

import os
import json
import logging
import asyncio
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for token usage."""
        pass

    def count_tokens(self, text: str) -> int:
        """Rough token count estimate (4 chars ≈ 1 token)."""
        return len(text) // 4


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude API client."""

    # Pricing per 1M tokens (as of Jan 2025)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env variable."
            )

        self.client = AsyncAnthropic(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate text using Claude with streaming for large requests."""
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Use streaming for requests with high max_tokens (>10k)
                if max_tokens > 10000:
                    full_response = ""
                    input_tokens = 0
                    output_tokens = 0

                    async with self.client.messages.stream(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    ) as stream:
                        async for text in stream.text_stream:
                            full_response += text

                        # Get final message for token usage
                        final_message = await stream.get_final_message()
                        input_tokens = final_message.usage.input_tokens
                        output_tokens = final_message.usage.output_tokens

                    # Track token usage
                    self.last_input_tokens = input_tokens
                    self.last_output_tokens = output_tokens

                    return full_response
                else:
                    # Use non-streaming for smaller requests
                    response = await self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )

                    # Track token usage
                    self.last_input_tokens = response.usage.input_tokens
                    self.last_output_tokens = response.usage.output_tokens

                    return response.content[0].text

            except Exception as e:
                error_msg = str(e).lower()
                # Retry on connection/network errors
                if attempt < max_retries - 1 and any(x in error_msg for x in [
                    "peer closed connection",
                    "incomplete chunked read",
                    "connection",
                    "timeout",
                    "reset"
                ]):
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Claude API error: {e}")
                    raise

        # Should never reach here, but just in case
        raise Exception("Max retries exceeded")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        pricing = self.PRICING.get(
            self.model,
            {"input": 3.00, "output": 15.00}  # Default to Sonnet pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost


def _openai_compatible_output_ceiling(model: str) -> int:
    """Safe max output tokens for an OpenAI-compatible model.

    Newer OpenAI models reject requests with `max_tokens` above their
    actual completion ceiling (HTTP 400, no completion). Callers are
    encouraged by CLAUDE.md to ask for very high values for headroom, so
    we clamp here rather than pushing the limit knowledge upstream. Per
    family ceilings reflect OpenAI's published limits as of May 2026.
    """
    m = model.lower()
    # OpenRouter "openai/<model>" — strip provider prefix for matching.
    if "/" in m:
        m = m.split("/")[-1]
    # GPT-5.x and 4.1 families: 32K completion tokens
    if m.startswith("gpt-5") or m.startswith("gpt-4.1"):
        return 32768
    # Reasoning models
    if m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return 32768
    # GPT-4o family: 16K
    if m.startswith("gpt-4o") or m.startswith("gpt-4-turbo") or m.startswith("gpt-4"):
        return 16384
    # Cloudflare-routed open models — most cap at 4-8K
    if "llama-3.1-8b" in m or "llama-3.2" in m:
        return 8192
    if "llama-3.1-70b" in m or "llama-3.3" in m or "mistral" in m or "qwen" in m or "gemma" in m:
        return 8192
    # Default conservative fallback
    return 16384


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    # Pricing per 1M tokens (refreshed May 2026)
    PRICING = {
        # GPT-5.5 family
        "gpt-5.5": {"input": 5.00, "output": 30.00},
        "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
        # GPT-5.4 family
        "gpt-5.4": {"input": 2.50, "output": 15.00},
        "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
        "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
        # GPT-4.1 family
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        # Reasoning models
        "o4-mini": {"input": 0.55, "output": 2.20},
        # Legacy entries kept for callers still pinning these strings
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env variable."
            )

        self.client = AsyncOpenAI(api_key=self.api_key)

    @staticmethod
    def _uses_new_params(model: str) -> bool:
        """
        OpenAI renamed `max_tokens` -> `max_completion_tokens` for the
        GPT-5.x, GPT-4.1, and o-series families. Older models (gpt-4o,
        gpt-4-turbo, gpt-3.5-turbo) still use `max_tokens`.
        """
        m = model.lower()
        return (
            m.startswith("gpt-5")
            or m.startswith("gpt-4.1")
            or m.startswith("o1")
            or m.startswith("o3")
            or m.startswith("o4")
        )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate text using OpenAI."""
        try:
            max_tokens = min(max_tokens, _openai_compatible_output_ceiling(self.model))
            call_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            }
            if self._uses_new_params(self.model):
                call_kwargs["max_completion_tokens"] = max_tokens
                # GPT-5.x / o-series reject non-default temperature on many
                # endpoints; pass it through only when the caller deviates
                # from the default. Most of our callsites use the default,
                # so this is rarely sent.
                if temperature != 1.0:
                    call_kwargs["temperature"] = temperature
            else:
                call_kwargs["max_tokens"] = max_tokens
                call_kwargs["temperature"] = temperature

            response = await self.client.chat.completions.create(**call_kwargs)

            # Track token usage
            self.last_input_tokens = response.usage.prompt_tokens
            self.last_output_tokens = response.usage.completion_tokens

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        pricing = self.PRICING.get(
            self.model,
            {"input": 2.50, "output": 10.00}  # Default to gpt-4o pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost


class OpenAICompatibleClient(OpenAIClient):
    """
    Generic client for OpenAI-compatible endpoints (OpenRouter, DigitalOcean
    Serverless Inference, vLLM, LM Studio, etc.).

    Inherits OpenAIClient's request shaping so model-family detection (the
    `max_tokens` vs `max_completion_tokens` split) still works. The only
    difference is `base_url` and where the API key comes from.

    Cost accounting:
    - If the provider returns a `cost` field in `response.usage` (OpenRouter
      does this), we use that as the authoritative cost.
    - Otherwise, we look up the model in `EXTRA_PRICING`, falling back to
      "we don't know" (0.0) with a one-time warning. Better to under-report
      than to invent a number.
    """

    # Published per-1M-token rates for models where we don't get usage.cost
    # back from the provider. Keep this short — OpenRouter handles its own
    # billing surface, so this is mostly for DO + Cloudflare.
    EXTRA_PRICING: Dict[str, Dict[str, float]] = {
        # DigitalOcean Serverless Inference (May 2026 rates)
        "llama3.3-70b-instruct": {"input": 0.59, "output": 0.79},
        "llama-3.3-70b-instruct": {"input": 0.59, "output": 0.79},
        "mistral-nemo-instruct-2407": {"input": 0.15, "output": 0.15},
        "deepseek-r1-distill-llama-70b": {"input": 0.75, "output": 1.00},
        "qwen2.5-72b-instruct": {"input": 0.59, "output": 0.79},
        # Cloudflare Workers AI (May 2026 rates). CF bills in "neurons" but
        # publishes per-token equivalents for the OpenAI-compatible models.
        # The "@cf/..." prefix is part of the model ID as CF returns it.
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast": {"input": 0.29, "output": 2.25},
        "@cf/meta/llama-3.1-70b-instruct": {"input": 0.29, "output": 2.25},
        "@cf/meta/llama-3.1-8b-instruct": {"input": 0.28, "output": 0.83},
        "@cf/openai/gpt-oss-120b": {"input": 0.35, "output": 0.75},
        "@cf/openai/gpt-oss-20b": {"input": 0.20, "output": 0.30},
        "@cf/google/gemma-3-12b-it": {"input": 0.35, "output": 0.56},
        "@cf/mistralai/mistral-small-3.1-24b-instruct": {"input": 0.35, "output": 0.56},
        "@cf/qwen/qwq-32b": {"input": 0.66, "output": 1.00},
    }

    _missing_price_warned: set = set()

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        env_var: str = "OPENAI_API_KEY",
        provider_label: str = "openai-compatible",
    ):
        # Bypass OpenAIClient.__init__ — we need a custom base_url AND a
        # custom env var name. Build the AsyncOpenAI client ourselves.
        BaseLLMClient.__init__(self, model)
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv(env_var)
        if not self.api_key:
            raise ValueError(
                f"API key required for {provider_label}. "
                f"Set {env_var} env variable or pass the key explicitly."
            )

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
        self._provider_label = provider_label
        self._reported_cost: Optional[float] = None

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        # Reuse OpenAIClient.generate but capture the provider-reported cost
        # before returning. We do this by reading usage off the response
        # directly; OpenAIClient stores last_input/output tokens but not cost.
        try:
            max_tokens = min(max_tokens, _openai_compatible_output_ceiling(self.model))
            call_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            }
            if self._uses_new_params(self.model):
                call_kwargs["max_completion_tokens"] = max_tokens
                if temperature != 1.0:
                    call_kwargs["temperature"] = temperature
            else:
                call_kwargs["max_tokens"] = max_tokens
                call_kwargs["temperature"] = temperature

            response = await self.client.chat.completions.create(**call_kwargs)

            usage = response.usage
            self.last_input_tokens = usage.prompt_tokens
            self.last_output_tokens = usage.completion_tokens
            # OpenRouter reports cost directly. Some providers tuck it under
            # different attribute names; check a few.
            self._reported_cost = (
                getattr(usage, "cost", None)
                or getattr(usage, "total_cost", None)
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"{self._provider_label} API error: {e}")
            raise

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Prefer the cost the provider told us directly.
        if self._reported_cost is not None:
            return float(self._reported_cost)

        pricing = self.EXTRA_PRICING.get(self.model)
        if pricing is None:
            if self.model not in OpenAICompatibleClient._missing_price_warned:
                logger.warning(
                    f"No pricing known for {self._provider_label} model "
                    f"'{self.model}'. Cost reported as $0; check your provider "
                    f"dashboard for actual usage."
                )
                OpenAICompatibleClient._missing_price_warned.add(self.model)
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""

    # Pricing per 1M tokens (paid tier; <=200k context for 2.5 pro)
    # Refreshed May 2026 — gemini-2.0-* shuts down June 1, 2026.
    PRICING = {
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        # Legacy entries kept for callers still pinning these strings:
        "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. "
                "Set GOOGLE_API_KEY environment variable or pass google_api_key parameter."
            )

        self._client = genai.Client(api_key=self.api_key)
        self._types = genai_types
        self._model_name = model

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking_budget: int = 0,
        **kwargs
    ) -> str:
        """
        Generate text using Gemini.

        Args:
            prompt: User prompt.
            temperature: Sampling temperature.
            max_tokens: Max output tokens. With thinking enabled, this budget
                is shared between thinking and the final response, so set it
                generously or disable thinking.
            thinking_budget: Thinking tokens for Gemini 2.5+ "thinking" models.
                Defaults to 0 (disabled) — our prompts ask for JSON or short
                scores, so thinking wastes tokens and burns rate-limit budget.
                Pass -1 for dynamic thinking, or a positive int for a fixed
                thinking budget.
        """
        try:
            cfg_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            # Only attach a thinking_config for models that support it (2.5+).
            # For older models the field is silently ignored by the API, but
            # we still avoid the import-time cost if ThinkingConfig is absent.
            ThinkingConfig = getattr(self._types, "ThinkingConfig", None)
            if ThinkingConfig is not None:
                cfg_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=thinking_budget)

            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._types.GenerateContentConfig(**cfg_kwargs),
            )

            # Track token usage
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                self.last_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                self.last_output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            else:
                self.last_input_tokens = len(prompt) // 4
                self.last_output_tokens = len(response.text or "") // 4

            return response.text or ""

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Extract base model name (e.g., "gemini-2.5-flash" from "gemini-2.5-flash-exp")
        base_model = self.model.split("-exp")[0]
        pricing = self.PRICING.get(
            base_model,
            {"input": 0.30, "output": 2.50}  # Fallback to gemini-2.5-flash pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost


class OllamaClient(BaseLLMClient):
    """Ollama local model client."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        super().__init__(model)
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package not installed. "
                "Install with: pip install ollama"
            )

        self.ollama = ollama
        self.host = host
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate text using Ollama."""
        try:
            response = self.ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )

            # Estimate tokens (Ollama doesn't always return usage)
            self.last_input_tokens = self.count_tokens(prompt)
            self.last_output_tokens = self.count_tokens(response["response"])

            return response["response"]

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local models are free."""
        return 0.0


class UnifiedLLMClient:
    """
    Unified client that routes requests to appropriate provider.

    Automatically detects provider based on model name:
    - "claude-*" → Anthropic
    - "gpt-*" → OpenAI
    - Everything else → Ollama (local)
    """

    # OpenAI-compatible providers. Add new ones here.
    OPENAI_COMPATIBLE_PROVIDERS = {
        "openrouter/": {
            "base_url": "https://openrouter.ai/api/v1",
            "env_var": "OPENROUTER_API_KEY",
            "key_kwarg": "openrouter_api_key",
            "label": "OpenRouter",
        },
        "do/": {
            "base_url": "https://inference.do-ai.run/v1",
            "env_var": "DO_INFERENCE_KEY",
            "key_kwarg": "do_api_key",
            "label": "DigitalOcean Serverless Inference",
        },
    }

    def __init__(
        self,
        model: str,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        do_api_key: Optional[str] = None,
        cloudflare_api_key: Optional[str] = None,
        cloudflare_account_id: Optional[str] = None,
        ollama_host: str = "http://localhost:11434"
    ):
        self.model = model
        self.client = self._create_client(
            model,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            openrouter_api_key=openrouter_api_key,
            do_api_key=do_api_key,
            cloudflare_api_key=cloudflare_api_key,
            cloudflare_account_id=cloudflare_account_id,
            ollama_host=ollama_host,
        )

    def _create_client(
        self,
        model: str,
        *,
        anthropic_api_key: Optional[str],
        openai_api_key: Optional[str],
        google_api_key: Optional[str],
        openrouter_api_key: Optional[str],
        do_api_key: Optional[str],
        cloudflare_api_key: Optional[str],
        cloudflare_account_id: Optional[str],
        ollama_host: str
    ) -> BaseLLMClient:
        """Create appropriate client based on model name."""
        # Cloudflare Workers AI — its base URL contains the account ID, so
        # it can't live in the static OPENAI_COMPATIBLE_PROVIDERS registry.
        # Handle it explicitly before the generic dispatch.
        if model.startswith("cf/"):
            account_id = cloudflare_account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
            if not account_id:
                raise ValueError(
                    "Cloudflare Workers AI requires an account ID. Set "
                    "CLOUDFLARE_ACCOUNT_ID env var or pass cloudflare_account_id."
                )
            routed_model = model[len("cf/"):]
            base_url = (
                f"https://api.cloudflare.com/client/v4/accounts/"
                f"{account_id}/ai/v1"
            )
            logger.info(f"Using Cloudflare Workers AI: {routed_model}")
            return OpenAICompatibleClient(
                model=routed_model,
                base_url=base_url,
                api_key=cloudflare_api_key,
                env_var="CLOUDFLARE_API_TOKEN",
                provider_label="Cloudflare Workers AI",
            )

        # OpenAI-compatible providers — check prefixes first so they win
        # over the generic "claude-*"/"gpt-*" detection below (an OpenRouter
        # model name like `openrouter/openai/gpt-4o` shouldn't route to the
        # direct OpenAI client).
        provider_keys = {
            "openrouter_api_key": openrouter_api_key,
            "do_api_key": do_api_key,
        }
        for prefix, cfg in self.OPENAI_COMPATIBLE_PROVIDERS.items():
            if model.startswith(prefix):
                routed_model = model[len(prefix):]
                logger.info(f"Using {cfg['label']}: {routed_model}")
                return OpenAICompatibleClient(
                    model=routed_model,
                    base_url=cfg["base_url"],
                    api_key=provider_keys[cfg["key_kwarg"]],
                    env_var=cfg["env_var"],
                    provider_label=cfg["label"],
                )

        if model.startswith("claude"):
            logger.info(f"Using Anthropic Claude: {model}")
            return ClaudeClient(model, anthropic_api_key)

        elif model.startswith("gpt"):
            logger.info(f"Using OpenAI: {model}")
            return OpenAIClient(model, openai_api_key)

        elif model.startswith("gemini"):
            logger.info(f"Using Google Gemini: {model}")
            return GeminiClient(model, google_api_key)

        else:
            logger.info(f"Using Ollama (local): {model}")
            return OllamaClient(model, ollama_host)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate text."""
        return await self.client.generate(prompt, temperature, max_tokens, **kwargs)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate generation cost."""
        return self.client.estimate_cost(input_tokens, output_tokens)

    def get_last_usage(self) -> Dict[str, int]:
        """Get token usage from last generation."""
        return {
            "input_tokens": getattr(self.client, "last_input_tokens", 0),
            "output_tokens": getattr(self.client, "last_output_tokens", 0),
        }

    def get_last_cost(self) -> float:
        """Get cost from last generation."""
        usage = self.get_last_usage()
        return self.estimate_cost(usage["input_tokens"], usage["output_tokens"])


class ModelRouter:
    """
    Routes different tasks to different models.

    Allows using different models for:
    - Generation (main data creation)
    - Verification (quality checks)
    - Diversity (similarity checks)
    """

    def __init__(
        self,
        generator_model: str = "openrouter/google/gemini-flash-latest",
        verifier_model: str = "openrouter/google/gemini-flash-latest",
        diversity_model: str = "openrouter/google/gemini-flash-latest",
        **api_keys
    ):
        # api_keys may include: anthropic_api_key, openai_api_key,
        # google_api_key, openrouter_api_key, do_api_key. We forward
        # the same dict to all three clients — routing happens per-model.
        self.generator = UnifiedLLMClient(generator_model, **api_keys)
        self.verifier = UnifiedLLMClient(verifier_model, **api_keys)
        self.diversity = UnifiedLLMClient(diversity_model, **api_keys)

        self.total_cost = 0.0

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate data using generator model."""
        response = await self.generator.generate(prompt, **kwargs)
        self.total_cost += self.generator.get_last_cost()
        return response

    async def verify(self, prompt: str, **kwargs) -> str:
        """Verify quality using verifier model."""
        response = await self.verifier.generate(prompt, **kwargs)
        self.total_cost += self.verifier.get_last_cost()
        return response

    async def check_diversity(self, prompt: str, **kwargs) -> str:
        """Check diversity using diversity model."""
        response = await self.diversity.generate(prompt, **kwargs)
        self.total_cost += self.diversity.get_last_cost()
        return response

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        return self.total_cost

    def reset_cost(self):
        """Reset cost counter."""
        self.total_cost = 0.0
