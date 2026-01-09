"""
DPO (Direct Preference Optimization) data generator.

Generates preference pairs with chosen (better) and rejected (worse) responses
to the same prompt.
"""

import json
import logging
from typing import List, Dict, Any, Type, Literal

from .base_generator import BaseGenerator
from ..data_models import (
    DPOPreference,
    DPOMessages,
    TrainingDataFormat
)

logger = logging.getLogger(__name__)


class DPOGenerator(BaseGenerator):
    """
    Generate DPO (Direct Preference Optimization) training data.

    Generates preference pairs where each sample has:
    - A prompt
    - A chosen (better) response
    - A rejected (worse) response
    """

    def __init__(
        self,
        format_type: Literal["preference", "messages"] = "preference",
        preference_strategy: Literal["quality", "style", "length"] = "quality",
        **kwargs
    ):
        """
        Initialize DPO generator.

        Args:
            format_type: Output format ("preference" for strings, "messages" for message arrays)
            preference_strategy: How to differentiate chosen/rejected
                - "quality": Better vs worse quality response
                - "style": Formal vs informal
                - "length": Detailed vs brief
            **kwargs: Passed to BaseGenerator
        """
        super().__init__(**kwargs)
        self.format_type = format_type
        self.preference_strategy = preference_strategy

        # Infer domain from source content if provided
        self.domain_context = self.config.domain_context or self._infer_domain()

    @property
    def data_format(self) -> Type[TrainingDataFormat]:
        """Return appropriate Pydantic model based on format type."""
        if self.format_type == "messages":
            return DPOMessages
        else:
            return DPOPreference

    @property
    def data_type_name(self) -> str:
        """Return data type name."""
        return "dpo"

    def _infer_domain(self) -> str:
        """Infer domain from source content."""
        if not self.source_content:
            return "general knowledge"

        # Simple keyword-based domain detection
        content_lower = self.source_content.lower()

        if any(kw in content_lower for kw in ["account", "debit", "credit", "ledger", "gaap"]):
            return "accounting"
        elif any(kw in content_lower for kw in ["finance", "investment", "stock", "bond"]):
            return "finance"
        elif any(kw in content_lower for kw in ["code", "function", "variable", "class"]):
            return "programming"
        elif any(kw in content_lower for kw in ["medical", "patient", "diagnosis", "treatment"]):
            return "medical"
        else:
            return "general knowledge"

    async def _generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Generate a batch of DPO preference pairs.

        Args:
            batch_size: Number of preference pairs to generate

        Returns:
            List of raw preference pair dictionaries
        """
        # Build generation prompt based on format type
        if self.format_type == "messages":
            prompt = self._build_messages_prompt(batch_size)
        else:
            prompt = self._build_preference_prompt(batch_size)

        # Generate batch
        try:
            response = await self.model_router.generate(
                prompt,
                temperature=0.9,  # Higher temp for diverse responses
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

    def _build_preference_prompt(self, batch_size: int) -> str:
        """Build prompt for preference format generation."""
        strategy_instructions = self._get_strategy_instructions()

        prompt = f"""
Generate {batch_size} diverse preference pairs for Direct Preference Optimization (DPO).

**Domain:** {self.domain_context}

**Preference Strategy:** {self.preference_strategy}
{strategy_instructions}

**Format:** Each example should have:
1. A "prompt" field with a clear question/instruction
2. A "chosen" field with the BETTER/PREFERRED response
3. A "rejected" field with the WORSE/LESS PREFERRED response

**Quality Requirements:**
- Prompts should be specific and relevant to the domain
- Chosen responses should be high quality, accurate, and complete
- Rejected responses should be noticeably worse but still plausible:
  * If quality strategy: Less accurate, incomplete, or unclear
  * If style strategy: Too informal, casual, or unprofessional
  * If length strategy: Too brief or lacking detail
- Make the difference clear but subtle (not obviously wrong)
- Vary question difficulty and topics

**Source Context (use as inspiration):**
{self.source_content[:2000] if self.source_content else "Generate diverse examples in the specified domain."}

Return a JSON array of {batch_size} preference pairs. Each must have this structure:
{{
  "prompt": "Question or instruction here",
  "chosen": "High-quality, preferred response",
  "rejected": "Lower-quality, less preferred response"
}}

Return ONLY the JSON array, no other text.
"""
        return prompt

    def _build_messages_prompt(self, batch_size: int) -> str:
        """Build prompt for messages format generation."""
        strategy_instructions = self._get_strategy_instructions()

        prompt = f"""
Generate {batch_size} diverse preference pairs for Direct Preference Optimization (DPO).

**Domain:** {self.domain_context}

**Preference Strategy:** {self.preference_strategy}
{strategy_instructions}

**Format:** Each example should have:
1. A "prompt" array with context messages (system, user)
2. A "chosen" array with the BETTER assistant response
3. A "rejected" array with the WORSE assistant response

**Source Context:**
{self.source_content[:2000] if self.source_content else "Generate diverse examples in the specified domain."}

Return a JSON array of {batch_size} examples with this structure:
{{
  "prompt": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}}
  ],
  "chosen": [
    {{"role": "assistant", "content": "High-quality response"}}
  ],
  "rejected": [
    {{"role": "assistant", "content": "Lower-quality response"}}
  ]
}}

Return ONLY the JSON array, no other text.
"""
        return prompt

    def _get_strategy_instructions(self) -> str:
        """Get instructions for the preference strategy."""
        if self.preference_strategy == "quality":
            return """
**Quality Strategy:**
- Chosen: Accurate, complete, well-structured, professional
- Rejected: Less accurate, incomplete, poorly structured, or vague
"""
        elif self.preference_strategy == "style":
            return """
**Style Strategy:**
- Chosen: Professional, formal, appropriate tone
- Rejected: Too casual, informal, or inappropriate tone (but not completely wrong)
"""
        elif self.preference_strategy == "length":
            return """
**Length Strategy:**
- Chosen: Comprehensive, detailed, thorough explanation
- Rejected: Too brief, lacks detail, or oversimplified (but not incorrect)
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
            # Try to extract JSON from markdown code blocks
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

            logger.error(f"Could not parse JSON from response: {response[:200]}")
            return []

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate DPO sample format."""
        try:
            # Validate using Pydantic model
            if self.format_type == "messages":
                DPOMessages(**sample)
            else:
                DPOPreference(**sample)

            # Additional check: chosen and rejected should be different
            if self.format_type == "preference":
                if sample.get("chosen") == sample.get("rejected"):
                    logger.debug("Chosen and rejected are identical")
                    return False

            return True

        except Exception as e:
            logger.debug(f"Validation failed: {e}")
            return False

    def set_preference_strategy(self, strategy: Literal["quality", "style", "length"]):
        """Update the preference strategy."""
        self.preference_strategy = strategy

    def set_domain_context(self, domain_context: str):
        """Update the domain context."""
        self.domain_context = domain_context
