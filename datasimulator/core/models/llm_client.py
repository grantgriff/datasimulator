"""
Unified LLM client supporting multiple providers.

Supports:
- Anthropic Claude (claude-3-5-sonnet, etc.)
- OpenAI (gpt-4, gpt-4o, gpt-4o-mini, etc.)
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


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    # Pricing per 1M tokens (as of Jan 2025)
    PRICING = {
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

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate text using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

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


class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""

    # Pricing per 1M tokens (as of Jan 2025)
    PRICING = {
        "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. "
                "Set GOOGLE_API_KEY environment variable or pass google_api_key parameter."
            )

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate text using Gemini."""
        try:
            # Configure generation settings
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Generate response
            response = await self.client.generate_content_async(
                prompt,
                generation_config=generation_config
            )

            # Track token usage
            if hasattr(response, 'usage_metadata'):
                self.last_input_tokens = response.usage_metadata.prompt_token_count
                self.last_output_tokens = response.usage_metadata.candidates_token_count
            else:
                # Estimate if usage not available
                self.last_input_tokens = len(prompt) // 4
                self.last_output_tokens = len(response.text) // 4

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Extract base model name (e.g., "gemini-2.0-flash" from "gemini-2.0-flash-exp")
        base_model = self.model.split("-exp")[0]
        pricing = self.PRICING.get(
            base_model,
            {"input": 0.075, "output": 0.30}  # Default to Flash pricing
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

    def __init__(
        self,
        model: str,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        ollama_host: str = "http://localhost:11434"
    ):
        self.model = model
        self.client = self._create_client(
            model,
            anthropic_api_key,
            openai_api_key,
            google_api_key,
            ollama_host
        )

    def _create_client(
        self,
        model: str,
        anthropic_api_key: Optional[str],
        openai_api_key: Optional[str],
        google_api_key: Optional[str],
        ollama_host: str
    ) -> BaseLLMClient:
        """Create appropriate client based on model name."""
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
        generator_model: str = "gemini-2.0-flash",
        verifier_model: str = "gpt-4o-mini-2024-07-18",
        diversity_model: str = "gpt-4o-mini-2024-07-18",
        **api_keys
    ):
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
