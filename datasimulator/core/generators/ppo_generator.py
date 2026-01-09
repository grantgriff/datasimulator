"""
PPO (Proximal Policy Optimization) data generator.

Generates prompts only - rewards come from a reward model at training time.
"""

import json
import logging
from typing import List, Dict, Any, Type

from .base_generator import BaseGenerator
from ..data_models import PPOPrompt, TrainingDataFormat

logger = logging.getLogger(__name__)


class PPOGenerator(BaseGenerator):
    """
    Generate PPO (Proximal Policy Optimization) training data.

    Generates prompts only. The reward signal comes from:
    - A separately trained reward model, OR
    - A rule-based reward function at training time
    """

    def __init__(self, prompt_style: str = "open_ended", **kwargs):
        """
        Initialize PPO generator.

        Args:
            prompt_style: Style of prompts to generate
                - "open_ended": Open questions requiring creative responses
                - "specific": Specific questions with clear answers
                - "task": Task-oriented instructions
            **kwargs: Passed to BaseGenerator
        """
        super().__init__(**kwargs)
        self.prompt_style = prompt_style

        # Infer domain from source content if provided
        self.domain_context = self.config.domain_context or self._infer_domain()

    @property
    def data_format(self) -> Type[TrainingDataFormat]:
        """Return Pydantic model for PPO format."""
        return PPOPrompt

    @property
    def data_type_name(self) -> str:
        """Return data type name."""
        return "ppo"

    def _infer_domain(self) -> str:
        """Infer domain from source content."""
        if not self.source_content:
            return "general knowledge"

        # Simple keyword-based domain detection
        content_lower = self.source_content.lower()

        if any(kw in content_lower for kw in ["account", "debit", "credit", "ledger"]):
            return "accounting"
        elif any(kw in content_lower for kw in ["finance", "investment", "stock"]):
            return "finance"
        elif any(kw in content_lower for kw in ["code", "function", "variable"]):
            return "programming"
        else:
            return "general knowledge"

    async def _generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Generate a batch of PPO prompts.

        Args:
            batch_size: Number of prompts to generate

        Returns:
            List of raw prompt dictionaries
        """
        prompt = self._build_generation_prompt(batch_size)

        # Generate batch
        try:
            response = await self.model_router.generate(
                prompt,
                temperature=0.8,
                max_tokens=2048 * (batch_size // 20 + 1)  # Prompts are shorter
            )

            # Parse JSON response
            samples = self._parse_batch_response(response)

            if len(samples) < batch_size:
                logger.warning(
                    f"Generated {len(samples)} samples, expected {batch_size}"
                )

            return samples

        except Exception as e:
            logger.error(f"Error generating batch: {e}")
            return []

    def _build_generation_prompt(self, batch_size: int) -> str:
        """Build prompt for PPO prompt generation."""
        style_instructions = self._get_style_instructions()

        prompt = f"""
Generate {batch_size} diverse prompts for Proximal Policy Optimization (PPO) training.

**Domain:** {self.domain_context}

**Prompt Style:** {self.prompt_style}
{style_instructions}

**Format:** Each example should have only a "prompt" field containing the question/instruction.
The model will generate responses at training time, which will be scored by a reward model.

**Quality Requirements:**
- Prompts should be clear and specific
- Vary difficulty and complexity
- Cover different aspects of the domain
- Suitable for reinforcement learning (responses can be ranked/scored)
- {self.prompt_style.replace('_', ' ').title()} style

**Source Context (use as inspiration):**
{self.source_content[:2000] if self.source_content else "Generate diverse prompts in the specified domain."}

**Examples of good PPO prompts:**
- "Explain how to calculate depreciation for a new asset"
- "Write a function to sort a list in Python"
- "Describe the process of recording a journal entry"

Return a JSON array of {batch_size} prompts with this structure:
{{
  "prompt": "Question or instruction here"
}}

Return ONLY the JSON array, no other text.
"""
        return prompt

    def _get_style_instructions(self) -> str:
        """Get instructions for the prompt style."""
        if self.prompt_style == "open_ended":
            return """
**Open-Ended Style:**
- Questions that allow for creative, varied responses
- No single "correct" answer
- Encourage explanation and reasoning
- Examples: "Explain...", "Describe...", "What are the benefits of..."
"""
        elif self.prompt_style == "specific":
            return """
**Specific Style:**
- Questions with clear, definable answers
- Focused on particular concepts or procedures
- Can be verified for correctness
- Examples: "How do you...", "What is...", "Calculate..."
"""
        elif self.prompt_style == "task":
            return """
**Task Style:**
- Action-oriented instructions
- Clear task to be completed
- Results can be evaluated
- Examples: "Write...", "Create...", "Implement...", "Solve..."
"""
        else:
            return ""

    def _parse_batch_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response from model."""
        try:
            # Try direct JSON parse
            samples = json.loads(response)

            if isinstance(samples, list):
                return samples
            elif isinstance(samples, dict) and "samples" in samples:
                return samples["samples"]
            else:
                logger.error(f"Unexpected response format: {type(samples)}")
                return []

        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            if "```json" in response:
                try:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    samples = json.loads(json_str)
                    return samples if isinstance(samples, list) else []
                except Exception as e:
                    logger.error(f"Error parsing markdown JSON: {e}")

            elif "```" in response:
                try:
                    json_str = response.split("```")[1].split("```")[0].strip()
                    samples = json.loads(json_str)
                    return samples if isinstance(samples, list) else []
                except Exception as e:
                    logger.error(f"Error parsing code block JSON: {e}")

            logger.error(f"Could not parse JSON from response")
            return []

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate PPO sample format."""
        try:
            # Validate using Pydantic model
            PPOPrompt(**sample)
            return True

        except Exception as e:
            logger.debug(f"Validation failed: {e}")
            return False

    def set_prompt_style(self, style: str):
        """Update the prompt style."""
        self.prompt_style = style
