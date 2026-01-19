"""
SFT (Supervised Fine-Tuning) data generator.

Supports both message-based and completion-based formats:
- Messages: Conversation-style with system, user, assistant roles
- Completion: Simple prompt/completion pairs
"""

import json
import logging
from typing import List, Dict, Any, Type, Literal, Optional

from .base_generator import BaseGenerator
from ..data_models import (
    SFTMessages,
    SFTCompletion,
    Message,
    TrainingDataFormat
)

logger = logging.getLogger(__name__)


class SFTGenerator(BaseGenerator):
    """
    Generate SFT (Supervised Fine-Tuning) training data.

    Generates high-quality instruction-response pairs suitable for
    supervised fine-tuning of language models.
    """

    def __init__(
        self,
        format_type: Literal["messages", "completion"] = "messages",
        system_prompt: str = None,
        **kwargs
    ):
        """
        Initialize SFT generator.

        Args:
            format_type: Output format ("messages" or "completion")
            system_prompt: Optional custom system prompt for messages format
            **kwargs: Passed to BaseGenerator
        """
        super().__init__(**kwargs)
        self.format_type = format_type
        self.system_prompt = system_prompt

        # Infer domain from source content if provided
        self.domain_context = self.config.domain_context or self._infer_domain()

    @property
    def data_format(self) -> Type[TrainingDataFormat]:
        """Return appropriate Pydantic model based on format type."""
        if self.format_type == "messages":
            return SFTMessages
        else:
            return SFTCompletion

    @property
    def data_type_name(self) -> str:
        """Return data type name."""
        return "sft"

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
        Generate a batch of SFT samples.

        Args:
            batch_size: Number of samples to generate
            batch_spec: Optional batch specification from generation plan

        Returns:
            List of raw sample dictionaries
        """
        # Build generation prompt based on format type
        if self.format_type == "messages":
            prompt = self._build_messages_prompt(batch_size, batch_spec)
        else:
            prompt = self._build_completion_prompt(batch_size, batch_spec)

        # Generate batch
        try:
            response = await self.model_router.generate(
                prompt,
                temperature=0.8,  # Higher temp for diversity
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

    def _build_messages_prompt(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for messages format generation with optional topic guidance."""
        system_prompt = self.system_prompt or self._get_default_system_prompt()

        # If we have batch_spec, use topic-specific guidance
        if batch_spec:
            topic = batch_spec.get("topic", "General Content")
            subtopic = batch_spec.get("subtopic", "")
            guidance = batch_spec.get("guidance", "")
            focus_areas = batch_spec.get("focus_areas", [])
            relevant_files = batch_spec.get("relevant_files", [])

            # Extract relevant source content (will implement next)
            source_context = self._extract_relevant_source(relevant_files) if relevant_files else self.source_content

            focus_list = "\n".join([f"- {area}" for area in focus_areas]) if focus_areas else ""

            prompt = f"""
You are an expert at generating high-quality training data for fine-tuning language models.

=== GENERATION TASK ===

DATA TYPE: SFT (Supervised Fine-Tuning)
FORMAT: Messages (system/user/assistant)
BATCH SIZE: {batch_size} samples

=== TOPIC CONTEXT ===

MAJOR TOPIC: {topic}
SUBTOPIC: {subtopic}

FOCUS AREAS:
{focus_list}

GENERATION GUIDANCE:
{guidance}

=== SOURCE MATERIAL ===

RELEVANT DOCUMENTS: {", ".join(relevant_files) if relevant_files else "All source documents"}

**IMPORTANT:** Use the source content below as your PRIMARY inspiration. Extract:
- Topics and themes directly from the source material
- Specific examples, scenarios, and calculations from the documents
- Terminology, definitions, and concepts as written in the sources
- Real-world applications and use cases mentioned in the sources

SOURCE CONTENT:
{source_context[:4000] if source_context else "No source content provided - generate from general knowledge."}

=== GENERATION REQUIREMENTS ===

1. **FORMAT TYPE: SFT (Supervised Fine-Tuning) - Messages Format**
   Each sample must have "messages" array with system/user/assistant roles

2. Generate EXACTLY {batch_size} samples in this exact structure:

EXAMPLE FORMAT:
{{
  "messages": [
    {{
      "role": "system",
      "content": "{system_prompt}"
    }},
    {{
      "role": "user",
      "content": "What is the journal entry to record bad debt expense using the allowance method?"
    }},
    {{
      "role": "assistant",
      "content": "The journal entry to record bad debt expense using the allowance method is:\\n\\nDR Bad Debt Expense    $X\\n   CR Allowance for Doubtful Accounts    $X\\n\\nThis entry:\\n1. Records the estimated uncollectible accounts\\n2. Matches the expense to the period when revenue was recognized (matching principle)\\n3. Reduces accounts receivable to net realizable value\\n\\nThe allowance method is preferred under GAAP because it follows the matching principle, unlike the direct write-off method which violates matching."
    }}
  ]
}}

3. **CRITICAL - CONTENT SOURCE**:
   - Pull questions, examples, and scenarios DIRECTLY from the source material above
   - Use exact terminology, definitions, and concepts from the sources
   - Base numerical examples on calculations shown in the source documents
   - Reference specific methods, procedures, and standards mentioned in sources

4. **TOPIC FOCUS**: All samples MUST be about "{topic} → {subtopic}"
   - Use the FOCUS AREAS as specific concepts to cover:
{focus_list}

5. **DIFFICULTY DISTRIBUTION**:
   - Samples 1-{batch_size//3}: Basic (definitions, simple concepts)
   - Samples {batch_size//3+1}-{2*batch_size//3}: Intermediate (calculations, procedures)
   - Samples {2*batch_size//3+1}-{batch_size}: Advanced (complex scenarios, analysis)

6. **QUESTION VARIETY**:
   - Conceptual, Calculation, Procedural, Analysis, Application questions
   - Mix formats: definitions, comparisons, step-by-step, problem-solving

7. **RESPONSE QUALITY**:
   - Detailed and educational (not just brief answers)
   - Include examples, formulas, journal entries when relevant
   - Explain the "why" not just the "what"

8. **RESPONSE LENGTH AND DETAIL REQUIREMENTS**:
   - Each assistant response should be CONCISE but COMPREHENSIVE (150-250 words typical)
   - Focus on key points with specific examples
   - Include essential calculations and formulas
   - Reference relevant standards when applicable
   - Be thorough but avoid unnecessary elaboration

   FEW-SHOT EXAMPLES OF APPROPRIATE DETAIL LEVEL:

   EXAMPLE 1:
   User: "I bought a $35,000 piece of equipment for my manufacturing business. Should I use Section 179 or depreciate it over time?"

   Assistant: "Section 179 allows you to deduct the full $35,000 immediately in 2024, giving you the largest first-year tax benefit.

Alternatives:
1. Section 179: Deduct $35,000 now (requires sufficient income)
2. Bonus Depreciation: Deduct 60% ($21,000) in 2024, depreciate remaining $14,000 over 5-7 years
3. Regular MACRS: Spread $35,000 over 5-7 years

Recommendation: Use Section 179 if you have the income for maximum immediate tax savings.

See IRS Publication 946, Chapter 2 for Section 179 rules."

   EXAMPLE 2:
   User: "What is the difference between tangible and intangible property for depreciation purposes?"

   Assistant: "Tangible property is physical property you can touch. Intangible property is non-physical property with value.

Tangible Property (depreciable):
- Equipment, machinery, vehicles
- Buildings and structures
- Depreciated under MACRS over 5-7 years

Intangible Property (amortizable):
- Patents, copyrights, trademarks
- Goodwill, software, licenses
- Amortized over 15 years under Section 197

Example: A $10,000 computer system is tangible (5-year MACRS). A $10,000 software license is intangible (36-month amortization).

See IRS Publication 946 for tangible property."

   Your responses should match this level of detail with real calculations, examples, and references.

OUTPUT: JSON array with EXACTLY {batch_size} objects in the format shown above.
Provide ONLY the JSON array, no other text.
"""
        else:
            # Fallback to generic prompt if no batch_spec
            prompt = f"""
Generate {batch_size} diverse, high-quality training examples for supervised fine-tuning.

**Domain:** {self.domain_context}

**Format:** Each example should have a "messages" array with:
1. A "system" message (optional, but recommended)
2. A "user" message with a question/instruction
3. An "assistant" message with a detailed, accurate response

**Quality Requirements:**
- Questions should be specific and relevant to the domain
- Responses should be comprehensive, accurate, and well-structured
- Vary question difficulty (easy, medium, hard)
- Cover different aspects of the topic
- Make examples diverse and realistic

**Source Context (use as inspiration):**
{self.source_content[:2000] if self.source_content else "Generate diverse examples in the specified domain."}

**Example System Prompt:** "{system_prompt}"

Return a JSON array of {batch_size} examples. Each example must have this exact structure:
{{
  "messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
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
            # Fallback: return all content if no per-file mapping
            return self.source_content

        # Extract content from ONLY the specified files
        relevant_content = []
        for file_path in relevant_files:
            # Try to match by filename (handle both full paths and just filenames)
            matched = False
            for stored_key, content in self.source_content_by_file.items():
                # Match if stored_key ends with the file_path (handles both cases)
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

    def _build_completion_prompt(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for completion format generation with optional topic guidance."""
        # Similar structure to messages prompt, but for completion format
        if batch_spec:
            topic = batch_spec.get("topic", "General Content")
            subtopic = batch_spec.get("subtopic", "")
            guidance = batch_spec.get("guidance", "")
            focus_areas = batch_spec.get("focus_areas", [])
            relevant_files = batch_spec.get("relevant_files", [])

            source_context = self._extract_relevant_source(relevant_files) if relevant_files else self.source_content
            focus_list = "\n".join([f"- {area}" for area in focus_areas]) if focus_areas else ""

            prompt = f"""
Generate {batch_size} high-quality SFT training examples (completion format).

=== TOPIC CONTEXT ===
MAJOR TOPIC: {topic}
SUBTOPIC: {subtopic}
FOCUS AREAS: {focus_list}
GUIDANCE: {guidance}

=== SOURCE MATERIAL ===
RELEVANT DOCUMENTS: {", ".join(relevant_files) if relevant_files else "All documents"}

**IMPORTANT:** Use source content as PRIMARY inspiration - extract topics, examples, calculations, and terminology directly from the material below.

SOURCE CONTENT:
{source_context[:4000] if source_context else "No source content."}

=== REQUIREMENTS ===

1. **FORMAT TYPE: SFT (Supervised Fine-Tuning) - Completion Format**
   Each sample has "prompt" and "completion" fields

2. EXAMPLE FORMAT:
{{
  "prompt": "Question: What is the journal entry for bad debt expense using the allowance method?",
  "completion": "DR Bad Debt Expense $X\\n   CR Allowance for Doubtful Accounts $X\\n\\nThis records estimated uncollectible accounts and follows the matching principle."
}}

3. **CONTENT SOURCE**: Pull questions/examples DIRECTLY from source material above

4. Generate EXACTLY {batch_size} samples about "{topic} → {subtopic}"

5. Difficulty: {batch_size//3} basic, {batch_size//3} intermediate, {batch_size//3} advanced

Return JSON array: [{{"prompt": "...", "completion": "..."}}, ...]
ONLY JSON, no other text.
"""
        else:
            # Fallback to generic prompt if no batch_spec
            prompt = f"""
Generate {batch_size} diverse, high-quality training examples for supervised fine-tuning.

**Domain:** {self.domain_context}

**Format:** Each example should have:
1. A "prompt" field with a clear question/instruction
2. A "completion" field with a detailed, accurate response

**Quality Requirements:**
- Prompts should be specific and relevant to the domain
- Completions should be comprehensive, accurate, and well-structured
- Vary question difficulty (easy, medium, hard)
- Cover different aspects of the topic
- Make examples diverse and realistic

**Source Context (use as inspiration):**
{self.source_content[:2000] if self.source_content else "Generate diverse examples in the specified domain."}

Return a JSON array of {batch_size} examples. Each example must have this exact structure:
{{
  "prompt": "Question: ... Answer:",
  "completion": "..."
}}

Return ONLY the JSON array, no other text.
"""
        return prompt

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on domain."""
        domain_prompts = {
            "accounting": "You are an expert accountant with deep knowledge of GAAP, IFRS, and financial reporting standards.",
            "finance": "You are a financial expert with expertise in investments, markets, and financial analysis.",
            "programming": "You are an expert programmer with deep knowledge of software development best practices.",
            "medical": "You are a medical expert with comprehensive knowledge of healthcare and medicine.",
            "general knowledge": "You are a knowledgeable assistant that provides accurate and helpful information."
        }

        return domain_prompts.get(self.domain_context, domain_prompts["general knowledge"])

    def _parse_batch_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSON response from model.

        Handles various response formats and extracts JSON array.
        """
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
        """
        Validate sample format.

        Args:
            sample: Raw sample dictionary

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate using Pydantic model
            if self.format_type == "messages":
                SFTMessages(**sample)
            else:
                SFTCompletion(**sample)

            return True

        except Exception as e:
            logger.debug(f"Validation failed: {e}")
            return False

    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for messages format."""
        self.system_prompt = system_prompt

    def set_domain_context(self, domain_context: str):
        """Update the domain context."""
        self.domain_context = domain_context
