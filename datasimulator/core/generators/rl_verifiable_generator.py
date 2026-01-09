"""
RL with Verifiable Rewards generator.

Generates prompts with ground truth answers for verifiable reinforcement learning.
Ideal for tasks with objectively correct answers (math, coding, factual questions).
"""

import json
import logging
import re
from typing import List, Dict, Any, Type, Literal

from .base_generator import BaseGenerator
from ..data_models import RLVerifiable, TrainingDataFormat

logger = logging.getLogger(__name__)


class RLVerifiableGenerator(BaseGenerator):
    """
    Generate RL training data with verifiable rewards.

    Generates prompts with ground truth answers that can be automatically verified.
    Perfect for:
    - Mathematical calculations
    - Coding problems with test cases
    - Factual questions with definitive answers
    - Accounting calculations
    """

    def __init__(
        self,
        verification_type: Literal[
            "numeric_match",
            "exact_match",
            "semantic_match",
            "contains",
            "regex"
        ] = "exact_match",
        include_explanation: bool = True,
        **kwargs
    ):
        """
        Initialize RL verifiable generator.

        Args:
            verification_type: How to verify the ground truth
                - "numeric_match": Compare numbers (handles decimals, formatting)
                - "exact_match": Exact string match
                - "semantic_match": Meaning-based match
                - "contains": Check if answer contains ground truth
                - "regex": Regex pattern matching
            include_explanation: Whether to include explanation with answer
            **kwargs: Passed to BaseGenerator
        """
        super().__init__(**kwargs)
        self.verification_type = verification_type
        self.include_explanation = include_explanation

        # Infer domain from source content if provided
        self.domain_context = self.config.domain_context or self._infer_domain()

    @property
    def data_format(self) -> Type[TrainingDataFormat]:
        """Return Pydantic model for RL verifiable format."""
        return RLVerifiable

    @property
    def data_type_name(self) -> str:
        """Return data type name."""
        return "rl_verifiable"

    def _infer_domain(self) -> str:
        """Infer domain from source content."""
        if not self.source_content:
            return "general knowledge"

        content_lower = self.source_content.lower()

        if any(kw in content_lower for kw in ["account", "debit", "credit", "calculate"]):
            return "accounting"
        elif any(kw in content_lower for kw in ["math", "equation", "calculate", "solve"]):
            return "mathematics"
        elif any(kw in content_lower for kw in ["code", "function", "algorithm"]):
            return "programming"
        elif any(kw in content_lower for kw in ["finance", "investment", "interest rate"]):
            return "finance"
        else:
            return "general knowledge"

    async def _generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Generate a batch of RL verifiable samples.

        Args:
            batch_size: Number of samples to generate

        Returns:
            List of raw sample dictionaries
        """
        prompt = self._build_generation_prompt(batch_size)

        # Generate batch
        try:
            response = await self.model_router.generate(
                prompt,
                temperature=0.7,  # Lower temp for more precise answers
                max_tokens=4096 * (batch_size // 10 + 1)
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
        """Build prompt for RL verifiable generation."""
        verification_instructions = self._get_verification_instructions()

        prompt = f"""
Generate {batch_size} verifiable training examples for Reinforcement Learning.

**Domain:** {self.domain_context}

**Verification Type:** {self.verification_type}
{verification_instructions}

**Format:** Each example must have:
1. A "prompt" field with the question/problem
2. A "ground_truth" field with the correct answer
3. A "verification_type" field (set to "{self.verification_type}")
4. Optional "metadata" field with additional context

**Critical Requirements:**
- Prompts must have OBJECTIVE, VERIFIABLE answers
- Ground truth must be the CORRECT answer
- Answers must be verifiable using the specified verification type
- Include only questions with definitive answers
- Vary difficulty and topics within the domain

**Source Context (use as inspiration):**
{self.source_content[:2000] if self.source_content else "Generate verifiable examples in the specified domain."}

**Example Formats:**

For numeric_match:
{{
  "prompt": "A company has $500,000 in accounts receivable and estimates 3% will be uncollectible. What is the bad debt expense?",
  "ground_truth": "15000",
  "verification_type": "numeric_match",
  "metadata": {{"calculation": "500000 * 0.03 = 15000"}}
}}

For exact_match:
{{
  "prompt": "What year was the FASB established?",
  "ground_truth": "1973",
  "verification_type": "exact_match"
}}

For contains:
{{
  "prompt": "What are the three main financial statements?",
  "ground_truth": "income statement, balance sheet, cash flow statement",
  "verification_type": "contains"
}}

Return a JSON array of {batch_size} examples following the format above.
Return ONLY the JSON array, no other text.
"""
        return prompt

    def _get_verification_instructions(self) -> str:
        """Get instructions for verification type."""
        if self.verification_type == "numeric_match":
            return """
**Numeric Match Verification:**
- Ground truth should be a number (can be string formatted)
- Handles different number formats (1000, 1,000, 1000.00)
- Perfect for: calculations, quantities, years, percentages
- Example ground truth: "15000" or "15000.00" or "15,000"
"""
        elif self.verification_type == "exact_match":
            return """
**Exact Match Verification:**
- Ground truth must exactly match the answer
- Case-sensitive string comparison
- Perfect for: dates, names, specific terms, short answers
- Example ground truth: "1973" or "Generally Accepted Accounting Principles"
"""
        elif self.verification_type == "semantic_match":
            return """
**Semantic Match Verification:**
- Ground truth should capture the meaning
- Allows for paraphrasing
- Perfect for: definitions, explanations (keep brief)
- Example ground truth: "expenses recorded in same period as related revenues"
"""
        elif self.verification_type == "contains":
            return """
**Contains Verification:**
- Ground truth should be key terms that must appear
- Checks if answer contains the ground truth
- Perfect for: lists, multiple correct elements
- Example ground truth: "income statement, balance sheet, cash flow"
"""
        elif self.verification_type == "regex":
            return """
**Regex Verification:**
- Ground truth should be a regex pattern
- Advanced verification for formatted answers
- Perfect for: structured responses, specific formats
- Example ground truth: "^\\d{4}-\\d{2}-\\d{2}$" (for dates)
"""
        else:
            return ""

    def _parse_batch_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response from model."""
        try:
            # Try direct JSON parse
            samples = json.loads(response)

            if isinstance(samples, list):
                # Ensure verification_type is set
                for sample in samples:
                    if "verification_type" not in sample:
                        sample["verification_type"] = self.verification_type
                return samples
            elif isinstance(samples, dict) and "samples" in samples:
                samples_list = samples["samples"]
                for sample in samples_list:
                    if "verification_type" not in sample:
                        sample["verification_type"] = self.verification_type
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
                            if "verification_type" not in sample:
                                sample["verification_type"] = self.verification_type
                        return samples
                except Exception as e:
                    logger.error(f"Error parsing markdown JSON: {e}")

            elif "```" in response:
                try:
                    json_str = response.split("```")[1].split("```")[0].strip()
                    samples = json.loads(json_str)
                    if isinstance(samples, list):
                        for sample in samples:
                            if "verification_type" not in sample:
                                sample["verification_type"] = self.verification_type
                        return samples
                except Exception as e:
                    logger.error(f"Error parsing code block JSON: {e}")

            logger.error("Could not parse JSON from response")
            return []

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate RL verifiable sample format."""
        try:
            # Validate using Pydantic model
            RLVerifiable(**sample)

            # Additional validation: ground truth should not be empty
            if not sample.get("ground_truth", "").strip():
                logger.debug("Ground truth is empty")
                return False

            return True

        except Exception as e:
            logger.debug(f"Validation failed: {e}")
            return False

    def set_verification_type(
        self,
        verification_type: Literal[
            "numeric_match",
            "exact_match",
            "semantic_match",
            "contains",
            "regex"
        ]
    ):
        """Update the verification type."""
        self.verification_type = verification_type
