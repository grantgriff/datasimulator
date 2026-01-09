"""
GRPO (Group Relative Policy Optimization) data generator.

Generates prompts for multi-completion generation with relative ranking.
Similar to PPO but designed for comparing multiple completions per prompt.
"""

import json
import logging
from typing import List, Dict, Any, Type

from .base_generator import BaseGenerator
from ..data_models import GRPOPrompt, TrainingDataFormat

logger = logging.getLogger(__name__)


class GRPOGenerator(BaseGenerator):
    """
    Generate GRPO (Group Relative Policy Optimization) training data.

    Generates prompts that will have multiple completions generated at training time.
    GRPO uses relative ranking within the group rather than absolute reward scores.

    Particularly effective for:
    - Verifiable tasks (math, coding, factual questions)
    - Tasks where multiple valid approaches exist
    - Scenarios requiring comparison between responses
    """

    def __init__(
        self,
        num_completions: int = 4,
        task_type: str = "verifiable",
        **kwargs
    ):
        """
        Initialize GRPO generator.

        Args:
            num_completions: Number of completions to generate per prompt at training time
            task_type: Type of tasks to generate
                - "verifiable": Tasks with objectively correct answers
                - "open_ended": Tasks where quality is subjective
                - "creative": Creative tasks with multiple valid approaches
            **kwargs: Passed to BaseGenerator
        """
        super().__init__(**kwargs)
        self.num_completions = num_completions
        self.task_type = task_type

        # Infer domain from source content if provided
        self.domain_context = self.config.domain_context or self._infer_domain()

    @property
    def data_format(self) -> Type[TrainingDataFormat]:
        """Return Pydantic model for GRPO format."""
        return GRPOPrompt

    @property
    def data_type_name(self) -> str:
        """Return data type name."""
        return "grpo"

    def _infer_domain(self) -> str:
        """Infer domain from source content."""
        if not self.source_content:
            return "general knowledge"

        content_lower = self.source_content.lower()

        if any(kw in content_lower for kw in ["account", "debit", "credit", "ledger"]):
            return "accounting"
        elif any(kw in content_lower for kw in ["finance", "investment", "stock"]):
            return "finance"
        elif any(kw in content_lower for kw in ["code", "function", "algorithm"]):
            return "programming"
        elif any(kw in content_lower for kw in ["math", "calculate", "equation"]):
            return "mathematics"
        else:
            return "general knowledge"

    async def _generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Generate a batch of GRPO prompts.

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
                max_tokens=2048 * (batch_size // 20 + 1)
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
        """Build prompt for GRPO prompt generation."""
        task_instructions = self._get_task_instructions()

        prompt = f"""
Generate {batch_size} diverse prompts for Group Relative Policy Optimization (GRPO) training.

**Domain:** {self.domain_context}

**Task Type:** {self.task_type}
{task_instructions}

**GRPO Training Process:**
At training time, {self.num_completions} different completions will be generated for each prompt.
These completions will be ranked relative to each other (not scored absolutely).
This is particularly effective for tasks where responses can be compared and ranked.

**Format:** Each example should have:
1. A "prompt" field with the question/task
2. A "num_completions" field (default: {self.num_completions})

**Quality Requirements:**
- Prompts should have multiple possible approaches/answers
- Should be suitable for comparison and ranking
- Clear enough that different quality levels can be distinguished
- Appropriate for generating {self.num_completions} diverse completions
- Vary in difficulty and complexity

**Source Context (use as inspiration):**
{self.source_content[:2000] if self.source_content else "Generate diverse prompts in the specified domain."}

**Good GRPO Prompts:**
- "Calculate the net present value of an investment with..."
- "Write a function to find the longest common subsequence"
- "Explain the accounting treatment for inventory valuation"
- "Solve for x in the equation: 3x² + 5x - 2 = 0"

Return a JSON array of {batch_size} prompts with this structure:
{{
  "prompt": "Question or task here",
  "num_completions": {self.num_completions}
}}

Return ONLY the JSON array, no other text.
"""
        return prompt

    def _get_task_instructions(self) -> str:
        """Get instructions for the task type."""
        if self.task_type == "verifiable":
            return """
**Verifiable Tasks:**
- Questions with objectively correct answers
- Mathematical calculations, factual questions
- Tasks where correctness can be verified
- Different approaches can be compared for accuracy and efficiency
- Examples: "Calculate...", "What year...", "How many...", "Solve..."
"""
        elif self.task_type == "open_ended":
            return """
**Open-Ended Tasks:**
- Questions with multiple valid answers
- Quality is subjective but can be compared
- Completions can be ranked by depth, clarity, completeness
- Examples: "Explain...", "Describe...", "Compare..."
"""
        elif self.task_type == "creative":
            return """
**Creative Tasks:**
- Tasks requiring creative problem-solving
- Multiple valid approaches exist
- Can be ranked by creativity, effectiveness, elegance
- Examples: "Design...", "Create...", "Propose..."
"""
        else:
            return ""

    def _parse_batch_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response from model."""
        try:
            # Try direct JSON parse
            samples = json.loads(response)

            if isinstance(samples, list):
                # Ensure num_completions is set
                for sample in samples:
                    if "num_completions" not in sample:
                        sample["num_completions"] = self.num_completions
                return samples
            elif isinstance(samples, dict) and "samples" in samples:
                samples_list = samples["samples"]
                for sample in samples_list:
                    if "num_completions" not in sample:
                        sample["num_completions"] = self.num_completions
                return samples_list
            else:
                logger.error(f"Unexpected response format: {type(samples)}")
                return []

        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            if "```json" in response:
                try:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    samples = json.loads(json_str)
                    if isinstance(samples, list):
                        for sample in samples:
                            if "num_completions" not in sample:
                                sample["num_completions"] = self.num_completions
                        return samples
                except Exception as e:
                    logger.error(f"Error parsing markdown JSON: {e}")

            elif "```" in response:
                try:
                    json_str = response.split("```")[1].split("```")[0].strip()
                    samples = json.loads(json_str)
                    if isinstance(samples, list):
                        for sample in samples:
                            if "num_completions" not in sample:
                                sample["num_completions"] = self.num_completions
                        return samples
                except Exception as e:
                    logger.error(f"Error parsing code block JSON: {e}")

            logger.error("Could not parse JSON from response")
            return []

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate GRPO sample format."""
        try:
            # Validate using Pydantic model
            GRPOPrompt(**sample)
            return True

        except Exception as e:
            logger.debug(f"Validation failed: {e}")
            return False

    def set_task_type(self, task_type: str):
        """Update the task type."""
        self.task_type = task_type

    def set_num_completions(self, num_completions: int):
        """Update the number of completions."""
        self.num_completions = num_completions
