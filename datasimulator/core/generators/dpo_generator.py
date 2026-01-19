"""
DPO (Direct Preference Optimization) data generator.

Generates preference pairs with chosen (better) and rejected (worse) responses
to the same prompt.
"""

import json
import logging
from typing import List, Dict, Any, Type, Literal, Optional

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

    async def _generate_batch(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of DPO preference pairs.

        Args:
            batch_size: Number of preference pairs to generate
            batch_spec: Optional batch specification from generation plan

        Returns:
            List of raw preference pair dictionaries
        """
        # Build generation prompt based on format type
        if self.format_type == "messages":
            prompt = self._build_messages_prompt(batch_size, batch_spec)
        else:
            prompt = self._build_preference_prompt(batch_size, batch_spec)

        # Generate batch
        try:
            response = await self.model_router.generate(
                prompt,
                temperature=0.9,  # Higher temp for diverse responses
                max_tokens=64000  # Hard cap at Sonnet max output tokens
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

    def _build_preference_prompt(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for preference format generation with optional topic guidance."""
        strategy_instructions = self._get_strategy_instructions()

        if batch_spec:
            topic = batch_spec.get("topic", "General Content")
            subtopic = batch_spec.get("subtopic", "")
            guidance = batch_spec.get("guidance", "")
            focus_areas = batch_spec.get("focus_areas", [])
            relevant_files = batch_spec.get("relevant_files", [])

            source_context = self._extract_relevant_source(relevant_files) if relevant_files else self.source_content
            focus_list = "\n".join([f"- {area}" for area in focus_areas]) if focus_areas else ""

            prompt = f"""
Generate {batch_size} high-quality DPO preference pairs.

=== TOPIC CONTEXT ===
MAJOR TOPIC: {topic}
SUBTOPIC: {subtopic}
FOCUS AREAS: {focus_list}
GUIDANCE: {guidance}

=== SOURCE MATERIAL ===
RELEVANT DOCUMENTS: {", ".join(relevant_files) if relevant_files else "All documents"}

**IMPORTANT:** Use source content as PRIMARY inspiration - extract topics, examples, and terminology directly from the material below.

SOURCE CONTENT:
{source_context[:4000] if source_context else "No source content."}

=== REQUIREMENTS ===

1. **FORMAT TYPE: DPO (Direct Preference Optimization) - Preference Format**
   Each sample has "prompt", "chosen", "rejected" fields

2. EXAMPLE FORMAT:
{{
  "prompt": "What is the journal entry for recording bad debt expense?",
  "chosen": "The journal entry is:\\nDR Bad Debt Expense $X\\n   CR Allowance for Doubtful Accounts $X\\n\\nThis follows the matching principle and is required under GAAP.",
  "rejected": "You just write off the bad debt when someone doesn't pay."
}}

3. **FEW-SHOT EXAMPLES OF APPROPRIATE DETAIL LEVEL**:

EXAMPLE 1:
{{
  "prompt": "I bought a $35,000 piece of equipment for my manufacturing business. Should I use Section 179 or depreciate it over time?",
  "chosen": "Section 179 allows you to deduct the full $35,000 immediately in 2024, giving you the largest first-year tax benefit.\\n\\nAlternatives:\\n1. Section 179: Deduct $35,000 now (requires sufficient income)\\n2. Bonus Depreciation: Deduct 60% ($21,000) in 2024, depreciate remaining $14,000 over 5-7 years\\n3. Regular MACRS: Spread $35,000 over 5-7 years\\n\\nRecommendation: Use Section 179 if you have the income for maximum immediate tax savings.\\n\\nSee IRS Publication 946, Chapter 2 for Section 179 rules.",
  "rejected": "Just depreciate it over 5 years like normal equipment. Section 179 is complicated and not worth the hassle for most businesses."
}}

EXAMPLE 2:
{{
  "prompt": "What is the difference between tangible and intangible property for depreciation purposes?",
  "chosen": "Tangible property is physical property you can touch. Intangible property is non-physical property with value.\\n\\nTangible Property (depreciable):\\n- Equipment, machinery, vehicles\\n- Buildings and structures\\n- Depreciated under MACRS over 5-7 years\\n\\nIntangible Property (amortizable):\\n- Patents, copyrights, trademarks\\n- Goodwill, software, licenses\\n- Amortized over 15 years under Section 197\\n\\nExample: A $10,000 computer is tangible (5-year MACRS). A $10,000 software license is intangible (36-month amortization).\\n\\nSee IRS Publication 946 for tangible property.",
  "rejected": "Tangible assets are things like equipment and buildings that you depreciate. Intangible assets are things like patents that don't depreciate."
}}

Your responses should match this level of detail - the 'chosen' response should be CONCISE but comprehensive with key examples and calculations (150-250 words), while the 'rejected' response should be incomplete, oversimplified, or missing key details.

4. **CONTENT SOURCE**: Pull questions/examples DIRECTLY from source material above

5. **PREFERENCE STRATEGY**: {self.preference_strategy}
{strategy_instructions}

6. Generate EXACTLY {batch_size} samples about "{topic} → {subtopic}"

7. Difficulty: {batch_size//3} basic, {batch_size//3} intermediate, {batch_size//3} advanced

Return JSON array: [{{"prompt": "...", "chosen": "...", "rejected": "..."}}, ...]
ONLY JSON, no other text.
"""
        else:
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
- Rejected responses should be noticeably worse but still plausible
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

    def _build_messages_prompt(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for messages format generation with optional topic guidance."""
        strategy_instructions = self._get_strategy_instructions()

        if batch_spec:
            topic = batch_spec.get("topic", "General Content")
            subtopic = batch_spec.get("subtopic", "")
            guidance = batch_spec.get("guidance", "")
            focus_areas = batch_spec.get("focus_areas", [])
            relevant_files = batch_spec.get("relevant_files", [])

            source_context = self._extract_relevant_source(relevant_files) if relevant_files else self.source_content
            focus_list = "\n".join([f"- {area}" for area in focus_areas]) if focus_areas else ""

            prompt = f"""
Generate {batch_size} high-quality DPO preference pairs (messages format).

=== TOPIC CONTEXT ===
MAJOR TOPIC: {topic}
SUBTOPIC: {subtopic}
FOCUS AREAS: {focus_list}
GUIDANCE: {guidance}

=== SOURCE MATERIAL ===
RELEVANT DOCUMENTS: {", ".join(relevant_files) if relevant_files else "All documents"}

**IMPORTANT:** Use source content as PRIMARY inspiration.

SOURCE CONTENT:
{source_context[:4000] if source_context else "No source content."}

=== REQUIREMENTS ===

1. **FORMAT TYPE: DPO - Messages Format**
   Each sample has "prompt", "chosen", "rejected" arrays

2. EXAMPLE FORMAT:
{{
  "prompt": [
    {{"role": "system", "content": "You are an expert accountant."}},
    {{"role": "user", "content": "What is the journal entry for bad debt expense?"}}
  ],
  "chosen": [
    {{"role": "assistant", "content": "DR Bad Debt Expense $X\\n   CR Allowance for Doubtful Accounts $X\\nThis follows the matching principle under GAAP."}}
  ],
  "rejected": [
    {{"role": "assistant", "content": "Just write it off when they don't pay."}}
  ]
}}

3. **CONTENT SOURCE**: Pull questions/examples DIRECTLY from source material

4. **PREFERENCE STRATEGY**: {self.preference_strategy}
{strategy_instructions}

5. Generate EXACTLY {batch_size} samples about "{topic} → {subtopic}"

Return JSON array with the format shown above.
ONLY JSON, no other text.
"""
        else:
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

    def _extract_relevant_source(self, relevant_files: List[str]) -> str:
        """
        Extract ONLY relevant source content based on file names.

        Args:
            relevant_files: List of relevant source file names from batch_spec

        Returns:
            Combined content from ONLY the specified files
        """
        if not relevant_files or not self.source_content_by_file:
            return self.source_content

        relevant_content = []
        for file_path in relevant_files:
            matched = False
            for stored_key, content in self.source_content_by_file.items():
                if stored_key.endswith(file_path) or file_path.endswith(stored_key):
                    relevant_content.append(f"\n\n=== {file_path} ===\n\n{content}")
                    matched = True
                    break

            if not matched:
                logger.warning(f"Could not find content for file: {file_path}")

        if not relevant_content:
            logger.warning(f"No matching files found for {relevant_files}, using all content")
            return self.source_content

        combined = "\n\n".join(relevant_content)
        logger.debug(f"Extracted {len(combined)} chars from {len(relevant_content)} relevant files")
        return combined

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
