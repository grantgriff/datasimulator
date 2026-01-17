"""
SFT (Supervised Fine-Tuning) data generator.

Supports both message-based and completion-based formats:
- Messages: Conversation-style with system, user, assistant roles
- Completion: Simple prompt/completion pairs
"""

import json
import logging
from typing import List, Dict, Any, Type, Literal

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
                max_tokens=4096 * (batch_size // 10 + 1)  # Scale with batch size
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

SOURCE CONTENT (use as factual basis for your examples):
{source_context[:4000] if source_context else "No source content provided - generate from general knowledge."}

=== GENERATION REQUIREMENTS ===

1. Generate EXACTLY {batch_size} SFT training samples
2. Each sample must include:
   - system: Role description for the AI assistant
   - user: A question or instruction
   - assistant: A detailed, accurate response

3. **CRITICAL**: All samples MUST focus on the topic/subtopic above
   - Topic: {topic}
   - Subtopic: {subtopic}
   - Do NOT generate samples about other topics

4. Use the FOCUS AREAS as your guide:
{focus_list}

5. Vary the difficulty across the {batch_size} samples:
   - Samples 1-{batch_size//3}: Basic/foundational (definitions, concepts, simple examples)
   - Samples {batch_size//3+1}-{2*batch_size//3}: Intermediate (calculations, procedures, method comparisons)
   - Samples {2*batch_size//3+1}-{batch_size}: Advanced (complex scenarios, multi-step problems, analysis)

6. Vary the question types:
   - Conceptual: "What is...?", "Explain..."
   - Calculation: "Calculate...", "Compute..."
   - Procedural: "How do you...?", "What are the steps to...?"
   - Analysis: "Compare...", "Explain why..."
   - Application: "Given this scenario..."

7. Ensure all information is accurate and based on the source material provided

8. Make responses detailed and educational (not just brief answers)

9. Use appropriate system prompts like: "{system_prompt}"

OUTPUT FORMAT (JSON only):
Return a JSON array with EXACTLY {batch_size} objects in this format:

[
  {{
    "messages": [
      {{
        "role": "system",
        "content": "You are an expert in {topic.lower()}..."
      }},
      {{
        "role": "user",
        "content": "Question or instruction here"
      }},
      {{
        "role": "assistant",
        "content": "Detailed, accurate response here..."
      }}
    ]
  }},
  ... {batch_size - 1} more samples ...
]

CRITICAL: Provide ONLY the JSON array. No introduction, no explanation, just the JSON.
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
        Extract relevant portions of source content based on file names.

        For now, returns all source content. In future, could:
        - Store per-file content mapping
        - Extract only sections from specified files
        - Use RAG/embeddings to find relevant sections

        Args:
            relevant_files: List of relevant source file names

        Returns:
            Extracted source content
        """
        # TODO: Implement per-file extraction
        # For now, return all source content
        # In a future enhancement, we could:
        # 1. Store content per file during loading
        # 2. Only return content from files in relevant_files list
        return self.source_content

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

TOPIC: {topic}
SUBTOPIC: {subtopic}

FOCUS AREAS:
{focus_list}

GUIDANCE: {guidance}

SOURCE CONTENT:
{source_context[:4000] if source_context else "No source content."}

REQUIREMENTS:
1. Generate EXACTLY {batch_size} samples about {topic} → {subtopic}
2. Each sample has "prompt" and "completion" fields
3. Use focus areas as guide
4. Vary difficulty: {batch_size//3} easy, {batch_size//3} medium, {batch_size//3} hard
5. Ensure accuracy based on source material

Return JSON array of {batch_size} examples:
[
  {{"prompt": "Question: ...", "completion": "..."}},
  ...
]

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
