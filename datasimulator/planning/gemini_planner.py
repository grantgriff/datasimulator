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

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        chunk_overlap: float = 0.1
    ):
        """
        Initialize Gemini planner.

        Args:
            api_key: Google API key (or uses GOOGLE_API_KEY env var)
            model: Gemini model to use (pro for 1M+ token context)
            chunk_overlap: Fraction of chunk to overlap with next (0.0-0.5, default 0.1 = 10%)
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
        self.chunk_overlap = max(0.0, min(0.5, chunk_overlap))  # Clamp to 0-50%

        logger.info(f"Gemini planner initialized with model: {model}")
        logger.info(f"Chunk overlap: {self.chunk_overlap*100:.0f}%")

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

            # Handle giant single paragraphs that exceed max_chars
            if para_size > max_chars:
                logger.warning(f"Paragraph exceeds {max_chars:,} chars ({para_size:,}), splitting into sentences")

                # Save current chunk before handling giant paragraph
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split giant paragraph and pack sentences into chunks
                sentence_chunks = self._split_giant_paragraph(para, max_chars)
                chunks.extend(sentence_chunks)

            elif current_size + para_size > max_chars:
                # Save current chunk
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(chunk_text)

                    # Add overlap: keep last N% of current chunk for next chunk
                    if self.chunk_overlap > 0:
                        overlap_size = int(len(chunk_text) * self.chunk_overlap)
                        overlap_text = chunk_text[-overlap_size:] if overlap_size > 0 else ""

                        # Find paragraph boundary in overlap region
                        # Try to start at a paragraph break for cleaner overlap
                        last_para_break = overlap_text.find("\n\n")
                        if last_para_break > 0:
                            overlap_text = overlap_text[last_para_break + 2:]

                        # Start new chunk with overlap + current paragraph
                        current_chunk = [overlap_text, para] if overlap_text else [para]
                        current_size = len(overlap_text) + para_size
                    else:
                        # No overlap - start fresh
                        current_chunk = [para]
                        current_size = para_size
                else:
                    # First chunk, no overlap possible
                    current_chunk = [para]
                    current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add final chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        logger.info(f"Content split into {len(chunks)} chunks (max {max_chars:,} chars each, {self.chunk_overlap*100:.0f}% overlap)")
        return chunks

    def _split_giant_paragraph(self, paragraph: str, max_chars: int) -> List[str]:
        """
        Split a giant paragraph that exceeds max_chars.

        Strategy:
        1. Try splitting by sentences (on ". ", "! ", "? ")
        2. Pack sentences into chunks of max_chars
        3. If individual sentence still too big, hard-chunk it

        Args:
            paragraph: The oversized paragraph
            max_chars: Maximum characters per chunk

        Returns:
            List of chunks from this paragraph
        """
        import re

        # Split on sentence boundaries (. ! ?) but keep the punctuation
        # This regex splits on sentence endings while preserving them
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If single sentence exceeds limit, hard-chunk it
            if sentence_size > max_chars:
                logger.warning(f"Single sentence exceeds {max_chars:,} chars, hard-chunking")

                # Save current accumulated sentences first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Hard-chunk the giant sentence
                for i in range(0, sentence_size, max_chars):
                    chunk = sentence[i:i + max_chars]
                    chunks.append(chunk)

            elif current_size + sentence_size + 1 > max_chars:  # +1 for space
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Start new chunk with this sentence
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info(f"Giant paragraph split into {len(chunks)} sentence-based chunks")
        return chunks

    async def create_generation_plan(
        self,
        source_content: str,
        total_samples: int,
        data_type: str = "sft",
        source_files: Optional[List[str]] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze sources and create batch-level generation plan.

        Args:
            source_content: Combined content from all source documents
            total_samples: Total number of samples to generate
            data_type: Type of training data (sft, dpo, etc.)
            source_files: List of source file names (optional)
            batch_size: Number of samples per batch (default: 10)

        Returns:
            Dictionary with:
            - file_review: Summary of source files
            - batches: List of batch specifications (one per batch of samples)
            - total_samples: Total samples (equals total_samples parameter)
            - batch_size: Samples per batch
            - num_batches: Total number of batches
            - domain: Overall domain/field
        """
        num_batches = (total_samples + batch_size - 1) // batch_size
        logger.info(f"Creating batch-level plan for {total_samples} {data_type.upper()} samples")
        logger.info(f"Plan will have {num_batches} batches of {batch_size} samples each")
        logger.info(f"Analyzing {len(source_content):,} characters of source content")

        # Handle large documents with chunking
        if len(source_content) > self.max_chars:
            logger.warning(f"Content exceeds {self.max_chars:,} chars, using chunked analysis")
            return await self._create_plan_chunked(source_content, total_samples, data_type, source_files, batch_size)

        # Single-pass planning for smaller documents
        return await self._create_plan_single(source_content, total_samples, data_type, source_files, batch_size)

    async def _create_plan_single(
        self,
        source_content: str,
        total_samples: int,
        data_type: str,
        source_files: Optional[List[str]] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """Create batch-level plan with single Gemini API call."""

        file_list = "\n".join([f"  - {f}" for f in source_files]) if source_files else "Multiple source files"
        num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division

        planning_prompt = f"""
You are an expert at analyzing technical documents and creating detailed training data generation plans.

TASK: Analyze the source documents and create a BATCH-LEVEL plan for generating {total_samples} {data_type.upper()} training samples in batches of {batch_size}.

SOURCE FILES:
{file_list}

SOURCE CONTENT:
{source_content[:self.max_chars]}

REQUIREMENTS:
1. Provide a brief review of the source files (1-2 paragraphs)
2. Create EXACTLY {num_batches} batches (each batch generates {batch_size} samples)
3. For each batch, specify:
   - Topic (major theme from the documents)
   - Subtopic (specific aspect of the topic for this batch)
   - Detailed generation guidance (what to focus on in these {batch_size} samples)
   - Relevant source files (which documents contain this content)
   - Key focus areas (3-5 specific concepts to cover)

STRATEGY:
- Group related batches under major topics (e.g., batches 1-25 on "Accounts Receivable", batches 26-65 on "Journal Entries")
- Progress from fundamental to advanced concepts within each topic
- Ensure diverse subtopics to avoid repetition
- Reference specific source files that contain relevant content

OUTPUT FORMAT (JSON only, no explanation):
{{
  "file_review": "1-2 paragraph summary of source content and key themes",
  "domain": "Overall domain/field (e.g., accounting, finance, healthcare)",
  "total_samples": {total_samples},
  "batch_size": {batch_size},
  "num_batches": {num_batches},
  "batches": [
    {{
      "batch_number": 1,
      "topic": "Major Topic Name",
      "subtopic": "Specific Subtopic for this batch",
      "guidance": "Detailed instructions: Generate {batch_size} samples focusing on [specific concepts]. Include questions about [X, Y, Z]. Vary difficulty from basic to advanced.",
      "relevant_files": ["file1.pdf", "file2.pdf"],
      "focus_areas": ["concept1", "concept2", "concept3"]
    }},
    {{
      "batch_number": 2,
      "topic": "Major Topic Name",
      "subtopic": "Different Subtopic",
      "guidance": "Generate {batch_size} samples about...",
      "relevant_files": ["file3.pdf"],
      "focus_areas": ["concept4", "concept5"]
    }}
    // ... continue for all {num_batches} batches
  ]
}}

CRITICAL: You must provide EXACTLY {num_batches} batch entries. The last batch should generate the remaining samples if {total_samples} is not evenly divisible by {batch_size}.

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

            # Validate batch plan
            plan = self._validate_batch_plan(plan, total_samples, batch_size)

            logger.info(f"✓ Plan created with {len(plan['batches'])} batches")
            logger.info(f"✓ File review: {plan['file_review'][:100]}...")

            # Log first few batches as preview
            for i, batch in enumerate(plan["batches"][:5], 1):
                logger.info(f"  Batch {batch['batch_number']}: {batch['topic']} → {batch['subtopic']}")
            if len(plan["batches"]) > 5:
                logger.info(f"  ... and {len(plan['batches']) - 5} more batches")

            return plan

        except Exception as e:
            logger.error(f"Error creating plan with Gemini: {e}")
            logger.warning("Falling back to simple batch plan")
            return self._fallback_batch_plan(total_samples, batch_size, source_files)

    async def _create_plan_chunked(
        self,
        source_content: str,
        total_samples: int,
        data_type: str,
        source_files: Optional[List[str]] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Create batch-level plan for large documents using chunked analysis.

        Strategy:
        1. Summarize each chunk in parallel
        2. Combine summaries
        3. Create batch plan from combined summaries
        """
        logger.info("Using chunked analysis strategy for large documents")

        chunks = self._chunk_content(source_content, self.max_chars)

        # Summarize all chunks in parallel
        logger.info(f"Summarizing {len(chunks)} chunks in parallel...")
        import asyncio
        summary_tasks = [
            self._summarize_chunk(chunk, i, len(chunks))
            for i, chunk in enumerate(chunks, 1)
        ]
        summaries = await asyncio.gather(*summary_tasks)

        # Combine summaries and create plan
        combined_summary = "\n\n".join(summaries)
        logger.info(f"✓ All chunks summarized. Combined: {len(combined_summary):,} characters")

        # Now create batch plan from summaries
        return await self._create_plan_single(combined_summary, total_samples, data_type, source_files, batch_size)

    async def _summarize_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> str:
        """
        Summarize a single chunk (async for parallel processing).

        Args:
            chunk: Content chunk to summarize
            chunk_num: Chunk number (1-indexed)
            total_chunks: Total number of chunks

        Returns:
            Summary text or error placeholder
        """
        logger.info(f"Summarizing chunk {chunk_num}/{total_chunks}...")

        summary_prompt = f"""
Analyze this section of source documents and provide a structured summary.

SECTION {chunk_num} of {total_chunks}:
{chunk}

Provide a summary covering:
1. Main topics/themes in this section
2. Key concepts and areas of focus
3. Types of information present (calculations, procedures, concepts, etc.)

Keep summary to 3-5 paragraphs maximum.
"""

        try:
            # Note: Gemini SDK doesn't have native async, but we can use sync in executor
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(summary_prompt)
            )
            logger.info(f"✓ Chunk {chunk_num} summarized")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error summarizing chunk {chunk_num}: {e}")
            return f"[Chunk {chunk_num}: Could not summarize due to error: {str(e)[:100]}]"

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

    def _validate_batch_plan(
        self,
        plan: Dict[str, Any],
        target_samples: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """
        Validate batch-level plan.

        Args:
            plan: Generated batch plan
            target_samples: Required total samples
            batch_size: Samples per batch

        Returns:
            Validated plan
        """
        # Ensure required fields
        if "batches" not in plan or not plan["batches"]:
            logger.warning("No batches in plan, creating fallback")
            return self._fallback_batch_plan(target_samples, batch_size)

        if "file_review" not in plan:
            plan["file_review"] = "Source content analyzed for training data generation."

        # Validate number of batches
        expected_batches = (target_samples + batch_size - 1) // batch_size
        actual_batches = len(plan["batches"])

        if actual_batches != expected_batches:
            logger.warning(
                f"Batch count mismatch: expected {expected_batches}, got {actual_batches}. "
                f"Adjusting..."
            )

            if actual_batches < expected_batches:
                # Add missing batches (duplicate last batch pattern)
                last_batch = plan["batches"][-1] if plan["batches"] else {
                    "batch_number": 1,
                    "topic": "General Content",
                    "subtopic": "Additional Coverage",
                    "guidance": f"Generate {batch_size} diverse samples",
                    "relevant_files": [],
                    "focus_areas": []
                }

                for i in range(actual_batches, expected_batches):
                    new_batch = last_batch.copy()
                    new_batch["batch_number"] = i + 1
                    plan["batches"].append(new_batch)

            elif actual_batches > expected_batches:
                # Trim excess batches
                plan["batches"] = plan["batches"][:expected_batches]

        # Ensure batch numbers are sequential
        for i, batch in enumerate(plan["batches"], 1):
            batch["batch_number"] = i
            # Ensure all required fields exist
            batch.setdefault("topic", "General Content")
            batch.setdefault("subtopic", f"Batch {i}")
            batch.setdefault("guidance", f"Generate {batch_size} samples")
            batch.setdefault("relevant_files", [])
            batch.setdefault("focus_areas", [])

        # Add metadata
        plan["total_samples"] = target_samples
        plan["batch_size"] = batch_size
        plan["num_batches"] = len(plan["batches"])

        logger.info(f"✓ Batch plan validated: {len(plan['batches'])} batches for {target_samples} samples")

        return plan

    def _fallback_batch_plan(
        self,
        total_samples: int,
        batch_size: int,
        source_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create simple fallback batch plan when Gemini fails."""
        num_batches = (total_samples + batch_size - 1) // batch_size
        file_info = f"Analyzed {len(source_files)} source files" if source_files else "Source content analyzed"

        batches = []
        for i in range(1, num_batches + 1):
            batches.append({
                "batch_number": i,
                "topic": "General Content",
                "subtopic": f"Batch {i} Coverage",
                "guidance": f"Generate {batch_size} diverse samples covering various aspects of the source material",
                "relevant_files": source_files or [],
                "focus_areas": ["general coverage", "varied difficulty", "diverse question types"]
            })

        return {
            "file_review": f"{file_info} for training data generation. Using general approach across all batches.",
            "domain": "General",
            "total_samples": total_samples,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "batches": batches
        }

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
