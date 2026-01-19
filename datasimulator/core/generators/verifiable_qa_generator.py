"""
Verifiable QA generator.

Generates question-answer pairs with ground truth for automatic verification.
Ideal for tasks with objectively correct answers (math, coding, factual questions).
"""

import json
import logging
import re
from typing import List, Dict, Any, Type, Literal, Optional

from .base_generator import BaseGenerator
from ..data_models import RLVerifiable, TrainingDataFormat

logger = logging.getLogger(__name__)


class VerifiableQAGenerator(BaseGenerator):
    """
    Generate verifiable question-answer training data.

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
        Initialize Verifiable QA generator.

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
        """Return Pydantic model for verifiable QA format."""
        return RLVerifiable

    @property
    def data_type_name(self) -> str:
        """Return data type name."""
        return "verifiable_qa"

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

    async def _generate_batch(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of verifiable Q&A samples.

        Args:
            batch_size: Number of samples to generate
            batch_spec: Optional batch specification from generation plan

        Returns:
            List of raw sample dictionaries
        """
        prompt = self._build_generation_prompt(batch_size, batch_spec)

        # Generate batch
        try:
            response = await self.model_router.generate(
                prompt,
                temperature=0.7,  # Lower temp for more precise answers
                max_tokens=batch_size * 5000  # ~5000 tokens per sample - generous headroom to prevent truncation
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

    def _build_generation_prompt(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for verifiable Q&A generation with optional topic guidance."""
        verification_instructions = self._get_verification_instructions()

        if batch_spec:
            topic = batch_spec.get("topic", "General Content")
            subtopic = batch_spec.get("subtopic", "")
            guidance = batch_spec.get("guidance", "")
            focus_areas = batch_spec.get("focus_areas", [])
            relevant_files = batch_spec.get("relevant_files", [])

            source_context = self._extract_relevant_source(relevant_files) if relevant_files else self.source_content
            focus_list = "\n".join([f"- {area}" for area in focus_areas]) if focus_areas else ""

            prompt = f"""
Generate {batch_size} verifiable Q&A examples.

=== TOPIC CONTEXT ===
MAJOR TOPIC: {topic}
SUBTOPIC: {subtopic}
FOCUS AREAS: {focus_list}
GUIDANCE: {guidance}

=== SOURCE MATERIAL ===
RELEVANT DOCUMENTS: {", ".join(relevant_files) if relevant_files else "All documents"}

**IMPORTANT:** Use source content as PRIMARY inspiration - extract factual questions with verifiable answers directly from the material below.

SOURCE CONTENT:
{source_context[:4000] if source_context else "No source content."}

=== REQUIREMENTS ===

1. **FORMAT TYPE: Verifiable Q&A**
   Each sample has "prompt", "ground_truth", "verification_type" fields

2. EXAMPLE FORMAT (for {self.verification_type}):
{{
  "prompt": "A company has $500,000 in AR and estimates 3% uncollectible. What is bad debt expense?",
  "ground_truth": "15000",
  "verification_type": "{self.verification_type}",
  "metadata": {{"calculation": "500000 * 0.03 = 15000"}}
}}

3. **FEW-SHOT EXAMPLES OF APPROPRIATE DETAIL LEVEL**:

EXAMPLE 1:
{{
  "prompt": "A manufacturing business purchases equipment for $35,000 and uses Section 179. If they have $35,000 in taxable income, how much can they deduct in year 1?",
  "ground_truth": "35000",
  "verification_type": "{self.verification_type}",
  "metadata": {{"explanation": "Section 179 allows immediate deduction of full equipment cost"}}
}}

EXAMPLE 2:
{{
  "prompt": "A company buys a computer system for $10,000 under 5-year MACRS with half-year convention. What is the Year 1 depreciation percentage?",
  "ground_truth": "20",
  "verification_type": "{self.verification_type}",
  "metadata": {{"calculation": "200% / 5 years = 40%, half-year = 20%"}}
}}

Your questions should include specific dollar amounts, percentages, or calculations with objectively verifiable numeric answers.

4. **CONTENT SOURCE**: Pull questions/answers DIRECTLY from source material
   - Extract facts, calculations, definitions from the sources
   - Ensure ground_truth answers are objectively verifiable
   - Base numeric questions on calculations shown in sources

5. **VERIFICATION TYPE**: {self.verification_type}
{verification_instructions}

6. Generate EXACTLY {batch_size} samples about "{topic} → {subtopic}"

7. Difficulty: {batch_size//3} basic, {batch_size//3} intermediate, {batch_size//3} advanced

Return JSON array: [{{"prompt": "...", "ground_truth": "...", "verification_type": "{self.verification_type}"}}, ...]
ONLY JSON, no other text.
"""
        else:
            prompt = f"""
Generate {batch_size} verifiable training examples.

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

Return a JSON array of {batch_size} examples.
Return ONLY the JSON array, no other text.
"""
        return prompt

    def _extract_relevant_source(self, relevant_files: List[str]) -> str:
        """Extract ONLY relevant source content based on file names."""
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
