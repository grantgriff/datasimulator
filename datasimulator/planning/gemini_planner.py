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
    - Extracts key topics from combined source documents
    - Determines sample allocation per topic
    - Provides domain-specific guidance for generation
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

        logger.info(f"Gemini planner initialized with model: {model}")

    async def create_generation_plan(
        self,
        source_content: str,
        total_samples: int,
        data_type: str = "sft"
    ) -> Dict[str, Any]:
        """
        Analyze sources and create intelligent generation plan.

        Args:
            source_content: Combined content from all source documents
            total_samples: Total number of samples to generate
            data_type: Type of training data (sft, dpo, etc.)

        Returns:
            Dictionary with:
            - topics: List of extracted topics
            - samples_per_topic: Allocation of samples to each topic
            - domain_guidance: Specific instructions per topic
        """
        logger.info(f"Creating generation plan for {total_samples} {data_type.upper()} samples")
        logger.info(f"Analyzing {len(source_content)} characters of source content")

        planning_prompt = f"""
You are an expert at analyzing technical documents and creating training data generation plans.

TASK: Analyze the source documents below and create a structured plan for generating {total_samples} {data_type.upper()} training samples.

SOURCE DOCUMENTS:
{source_content[:500000]}  # Use first 500K chars for planning

INSTRUCTIONS:
1. Identify 5-10 key topics/themes from the documents
2. For each topic, provide:
   - Topic name
   - Brief description
   - Number of samples to generate (allocate {total_samples} total)
   - Specific guidance for generating questions/prompts for this topic
3. Ensure topic allocation reflects the importance and coverage in the source

OUTPUT FORMAT (JSON only, no explanation):
{{
  "topics": [
    {{
      "name": "Topic Name",
      "description": "Brief description",
      "sample_count": 200,
      "guidance": "Specific instructions for generating samples about this topic"
    }}
  ],
  "total_allocated": {total_samples},
  "domain": "Overall domain/field"
}}

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

            # Validate and normalize
            total_allocated = sum(t.get("sample_count", 0) for t in plan.get("topics", []))
            if total_allocated != total_samples:
                logger.warning(
                    f"Plan allocated {total_allocated} samples, adjusting to {total_samples}"
                )
                # Proportionally adjust
                scale = total_samples / total_allocated if total_allocated > 0 else 1
                for topic in plan["topics"]:
                    topic["sample_count"] = int(topic["sample_count"] * scale)

            logger.info(f"✓ Plan created with {len(plan['topics'])} topics")
            for topic in plan["topics"]:
                logger.info(
                    f"  - {topic['name']}: {topic['sample_count']} samples"
                )

            return plan

        except Exception as e:
            logger.error(f"Error creating plan with Gemini: {e}")
            logger.warning("Falling back to simple plan")

            # Fallback: single topic with all samples
            return {
                "topics": [{
                    "name": "General",
                    "description": "General content from source documents",
                    "sample_count": total_samples,
                    "guidance": "Generate diverse questions covering all source material"
                }],
                "total_allocated": total_samples,
                "domain": "General"
            }

    def save_plan(self, plan: Dict[str, Any], output_path: str):
        """Save generation plan to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(plan, f, indent=2)

        logger.info(f"Plan saved to: {output_path}")

    def load_plan(self, plan_path: str) -> Dict[str, Any]:
        """Load generation plan from file."""
        with open(plan_path, 'r') as f:
            plan = json.load(f)

        logger.info(f"Plan loaded from: {plan_path}")
        return plan
