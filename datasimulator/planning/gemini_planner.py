"""
Gemini-powered planning layer for intelligent data generation.

Analyzes source documents and creates a structured generation plan
with topic extraction and sample allocation.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GeminiPlanner:
    """
    Uses Gemini's large context window to analyze sources and plan generation.

    Features:
    - Reviews and summarizes source documents
    - Extracts 5-50 key topics based on content
    - Allocates samples per topic (guaranteed to sum to total)
    - Provides domain-specific guidance for generation
    - Handles large documents with chunking failsafe
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro-latest"):
        """
        Initialize Gemini planner.

        Args:
            api_key: Google API key (or uses GOOGLE_API_KEY env var)
            model: Gemini model to use (pro for 1M+ token context)
        """
        import os

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
                "Google API key required. Set GOOGLE_API_KEY env variable or pass api_key"
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

        # Gemini 1.5 Pro has ~2M token context, but leave buffer
        self.max_chars = 1_500_000  # ~375K tokens

        logger.info(f"Gemini planner initialized with model: {model}")

    def _chunk_content(self, content: str, max_chars: int) -> List[str]:
        """
        Chunk content if too large for single API call.

        Args:
            content: Full content
            max_chars: Maximum characters per chunk

        Returns:
            List of content chunks
        """
        if len(content) <= max_chars:
            return [content]

        # Split by double newlines (paragraph breaks) to keep context
        chunks = []
        current_chunk = []
        current_size = 0

        paragraphs = content.split("\n\n")

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > max_chars:
                # Save current chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                # Start new chunk
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add final chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        logger.info(f"Content split into {len(chunks)} chunks (max {max_chars:,} chars each)")
        return chunks

    async def create_generation_plan(
        self,
        source_content: str,
        total_samples: int,
        data_type: str = "sft",
        source_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze sources and create intelligent generation plan.

        Args:
            source_content: Combined content from all source documents
            total_samples: Total number of samples to generate
            data_type: Type of training data (sft, dpo, etc.)
            source_files: List of source file names (optional)

        Returns:
            Dictionary with:
            - file_review: Summary of source files
            - topics: List of 5-50 extracted topics
            - total_allocated: Total samples (guaranteed to equal total_samples)
            - domain: Overall domain/field
        """
        logger.info(f"Creating generation plan for {total_samples} {data_type.upper()} samples")
        logger.info(f"Analyzing {len(source_content):,} characters of source content")

        # Handle large documents with chunking
        if len(source_content) > self.max_chars:
            logger.warning(f"Content exceeds {self.max_chars:,} chars, using chunked analysis")
            return await self._create_plan_chunked(source_content, total_samples, data_type, source_files)

        # Single-pass planning for smaller documents
        return await self._create_plan_single(source_content, total_samples, data_type, source_files)

    async def _create_plan_single(
        self,
        source_content: str,
        total_samples: int,
        data_type: str,
        source_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create plan with single Gemini API call."""

        file_list = "\n".join([f"  - {f}" for f in source_files]) if source_files else "Multiple source files"

        planning_prompt = f"""
You are an expert at analyzing technical documents and creating training data generation plans.

TASK: Analyze the source documents below and create a structured plan for generating {total_samples} {data_type.upper()} training samples.

SOURCE FILES:
{file_list}

SOURCE CONTENT:
{source_content[:self.max_chars]}

INSTRUCTIONS:
1. First, provide a brief review of the source files (1-2 paragraphs summarizing key content areas)
2. Identify 5-50 key topics/themes from the documents (choose number based on content diversity)
3. For each topic, provide:
   - Topic name
   - Brief description (1 sentence)
   - Number of samples to generate
   - Specific guidance for generating questions/prompts for this topic
4. CRITICAL: Ensure sample counts add up EXACTLY to {total_samples}
5. Allocate samples based on topic importance and coverage in the source material

OUTPUT FORMAT (JSON only, no explanation):
{{
  "file_review": "Brief 1-2 paragraph summary of source content, key themes, and areas of focus",
  "topics": [
    {{
      "name": "Topic Name",
      "description": "One sentence description",
      "sample_count": 100,
      "guidance": "Specific instructions for generating samples about this topic"
    }}
  ],
  "total_allocated": {total_samples},
  "domain": "Overall domain/field (e.g., accounting, finance, healthcare)"
}}

IMPORTANT: The sum of all sample_count values must equal {total_samples} exactly.

Provide ONLY the JSON output, nothing else.
"""

        try:
            # Generate plan using Gemini
            response = self.model.generate_content(planning_prompt)
            plan_text = response.text.strip()

            # Extract JSON from response
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_text:
                plan_text = plan_text.split("```")[1].split("```")[0].strip()

            plan = json.loads(plan_text)

            # Validate and fix sample allocation
            plan = self._validate_and_fix_plan(plan, total_samples)

            logger.info(f"✓ Plan created with {len(plan['topics'])} topics")
            logger.info(f"✓ File review: {plan['file_review'][:100]}...")
            for topic in plan["topics"]:
                logger.info(f"  - {topic['name']}: {topic['sample_count']} samples")

            return plan

        except Exception as e:
            logger.error(f"Error creating plan with Gemini: {e}")
            logger.warning("Falling back to simple plan")
            return self._fallback_plan(total_samples, source_files)

    async def _create_plan_chunked(
        self,
        source_content: str,
        total_samples: int,
        data_type: str,
        source_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create plan for large documents using chunked analysis.

        Strategy:
        1. Summarize each chunk
        2. Combine summaries
        3. Create plan from combined summaries
        """
        logger.info("Using chunked analysis strategy for large documents")

        chunks = self._chunk_content(source_content, self.max_chars)
        summaries = []

        # Summarize each chunk
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Summarizing chunk {i}/{len(chunks)}")

            summary_prompt = f"""
Analyze this section of source documents and provide a structured summary.

SECTION {i} of {len(chunks)}:
{chunk}

Provide a summary covering:
1. Main topics/themes in this section
2. Key concepts and areas of focus
3. Types of information present (calculations, procedures, concepts, etc.)

Keep summary to 3-5 paragraphs maximum.
"""

            try:
                response = self.model.generate_content(summary_prompt)
                summaries.append(response.text.strip())
                logger.info(f"✓ Chunk {i} summarized")
            except Exception as e:
                logger.error(f"Error summarizing chunk {i}: {e}")
                summaries.append(f"[Chunk {i}: Could not summarize]")

        # Combine summaries and create plan
        combined_summary = "\n\n".join(summaries)
        logger.info(f"Combined summaries: {len(combined_summary):,} characters")

        # Now create plan from summaries
        return await self._create_plan_single(combined_summary, total_samples, data_type, source_files)

    def _validate_and_fix_plan(self, plan: Dict[str, Any], target_samples: int) -> Dict[str, Any]:
        """
        Validate plan and fix sample allocation to match target exactly.

        Args:
            plan: Generated plan
            target_samples: Required total samples

        Returns:
            Fixed plan with exact sample allocation
        """
        # Ensure required fields
        if "topics" not in plan or not plan["topics"]:
            return self._fallback_plan(target_samples)

        if "file_review" not in plan:
            plan["file_review"] = "Source content analyzed for training data generation."

        # Calculate current total
        current_total = sum(t.get("sample_count", 0) for t in plan["topics"])

        if current_total == target_samples:
            # Perfect! No adjustment needed
            plan["total_allocated"] = target_samples
            return plan

        # Need to adjust allocation
        logger.warning(f"Sample allocation mismatch: {current_total} != {target_samples}, adjusting...")

        if current_total == 0:
            # No samples allocated, distribute evenly
            per_topic = target_samples // len(plan["topics"])
            remainder = target_samples % len(plan["topics"])

            for i, topic in enumerate(plan["topics"]):
                topic["sample_count"] = per_topic + (1 if i < remainder else 0)
        else:
            # Scale proportionally and fix rounding errors
            scale = target_samples / current_total

            adjusted_counts = []
            for topic in plan["topics"]:
                adjusted = int(topic["sample_count"] * scale)
                adjusted_counts.append(adjusted)
                topic["sample_count"] = adjusted

            # Fix rounding error by adjusting largest topics
            current_sum = sum(adjusted_counts)
            difference = target_samples - current_sum

            if difference != 0:
                # Sort topics by sample count (descending)
                sorted_indices = sorted(
                    range(len(plan["topics"])),
                    key=lambda i: plan["topics"][i]["sample_count"],
                    reverse=True
                )

                # Distribute difference across top topics
                for i in range(abs(difference)):
                    idx = sorted_indices[i % len(sorted_indices)]
                    plan["topics"][idx]["sample_count"] += 1 if difference > 0 else -1

        # Final validation
        final_total = sum(t["sample_count"] for t in plan["topics"])
        assert final_total == target_samples, f"Failed to fix allocation: {final_total} != {target_samples}"

        plan["total_allocated"] = target_samples
        logger.info(f"✓ Sample allocation fixed: {final_total} samples")

        return plan

    def _fallback_plan(self, total_samples: int, source_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create simple fallback plan when Gemini fails."""
        file_info = f"Analyzed {len(source_files)} source files" if source_files else "Source content analyzed"

        return {
            "file_review": f"{file_info} for training data generation. Unable to create detailed plan, using general approach.",
            "topics": [{
                "name": "General Content",
                "description": "General questions covering all source material",
                "sample_count": total_samples,
                "guidance": "Generate diverse questions covering all aspects of the source documents"
            }],
            "total_allocated": total_samples,
            "domain": "General"
        }

    def save_plan(self, plan: Dict[str, Any], output_path: str):
        """Save generation plan to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        logger.info(f"Plan saved to: {output_path}")

    def load_plan(self, plan_path: str) -> Dict[str, Any]:
        """Load generation plan from file."""
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan = json.load(f)

        logger.info(f"Plan loaded from: {plan_path}")
        return plan
