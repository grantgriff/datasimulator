"""
Advanced quality scoring system.

Provides multi-dimensional quality assessment beyond basic 1-10 scoring:
- Relevance to source material
- Accuracy and correctness
- Clarity and completeness
- Instruction-following quality
- Appropriate difficulty level
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Advanced quality scoring with detailed breakdown.

    Scores samples on multiple dimensions and provides
    detailed feedback for improvement.
    """

    def __init__(self, verifier_client, threshold: float = 6.0):
        """
        Initialize quality scorer.

        Args:
            verifier_client: LLM client for verification
            threshold: Minimum acceptable quality score (1-10)
        """
        self.verifier_client = verifier_client
        self.threshold = threshold

    async def score_sample(
        self,
        sample: Dict[str, Any],
        source_context: Optional[str] = None,
        data_type: str = "sft"
    ) -> Dict[str, Any]:
        """
        Score a sample with detailed breakdown.

        Args:
            sample: Sample to score
            source_context: Optional source material for relevance checking
            data_type: Type of training data (sft, dpo, etc.)

        Returns:
            Dictionary with overall score and dimension scores
        """
        scoring_prompt = self._build_scoring_prompt(sample, source_context, data_type)

        try:
            response = await self.verifier_client.generate(
                scoring_prompt,
                temperature=0.2,  # Low temp for consistent scoring
                max_tokens=500
            )

            # Parse response
            scores = self._parse_scoring_response(response)

            # Calculate overall score
            overall_score = self._calculate_overall_score(scores)

            result = {
                "overall_score": overall_score,
                "dimension_scores": scores,
                "passes_threshold": overall_score >= self.threshold,
                "timestamp": datetime.now().isoformat()
            }

            logger.debug(f"Quality score: {overall_score:.1f}/10 (threshold: {self.threshold})")

            return result

        except Exception as e:
            logger.error(f"Error scoring sample: {e}")
            # Return below-threshold score to trigger regeneration
            return {
                "overall_score": 5.0,
                "dimension_scores": {},
                "passes_threshold": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _build_scoring_prompt(
        self,
        sample: Dict[str, Any],
        source_context: Optional[str],
        data_type: str
    ) -> str:
        """Build detailed scoring prompt."""
        prompt = f"""
Score this training data sample on multiple dimensions.

**Data Type:** {data_type.upper()}

**Sample:**
{json.dumps(sample, indent=2)}

**Source Context (if provided):**
{source_context[:1000] if source_context else "No source context provided"}

**Scoring Dimensions (each 1-10):**

1. **Relevance** (1-10): How relevant is this to the source material/domain?
   - 1-3: Off-topic or irrelevant
   - 4-6: Somewhat relevant but could be better
   - 7-8: Relevant and on-topic
   - 9-10: Highly relevant and well-aligned

2. **Accuracy** (1-10): How accurate is the information?
   - 1-3: Contains errors or misinformation
   - 4-6: Mostly accurate with minor issues
   - 7-8: Accurate and correct
   - 9-10: Exceptionally accurate with proper depth

3. **Clarity** (1-10): How clear and well-structured is the content?
   - 1-3: Confusing or poorly structured
   - 4-6: Understandable but could be clearer
   - 7-8: Clear and well-organized
   - 9-10: Exceptionally clear and polished

4. **Completeness** (1-10): How complete and thorough is the response?
   - 1-3: Incomplete or superficial
   - 4-6: Covers basics but lacks depth
   - 7-8: Complete and satisfactory
   - 9-10: Comprehensive and thorough

5. **Instruction_Quality** (1-10): How well does the instruction/question guide the response?
   - 1-3: Vague or poorly formed
   - 4-6: Acceptable but could be improved
   - 7-8: Good and clear instruction
   - 9-10: Excellent, specific instruction

**Output Format (JSON only, no other text):**
{{
  "relevance": <score>,
  "accuracy": <score>,
  "clarity": <score>,
  "completeness": <score>,
  "instruction_quality": <score>,
  "feedback": "<brief explanation of scores>"
}}
"""
        return prompt

    def _parse_scoring_response(self, response: str) -> Dict[str, float]:
        """Parse scoring response from LLM."""
        try:
            # Try direct JSON parse
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                data = json.loads(json_str)
            else:
                # Fallback: extract numbers from text
                logger.warning("Could not parse JSON, using fallback")
                return self._fallback_parse(response)

        # Extract scores
        scores = {}
        for key in ["relevance", "accuracy", "clarity", "completeness", "instruction_quality"]:
            score = data.get(key, 5.0)
            # Clamp to 1-10 range
            scores[key] = max(1.0, min(10.0, float(score)))

        # Store feedback if available
        if "feedback" in data:
            scores["feedback"] = data["feedback"]

        return scores

    def _fallback_parse(self, response: str) -> Dict[str, float]:
        """Fallback parsing if JSON parsing fails."""
        # Return middle-of-road scores
        return {
            "relevance": 6.0,
            "accuracy": 6.0,
            "clarity": 6.0,
            "completeness": 6.0,
            "instruction_quality": 6.0,
            "feedback": "Parsing failed, using default scores"
        }

    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        Calculate overall score from dimension scores.

        Weighted average with emphasis on accuracy and relevance.
        """
        weights = {
            "relevance": 0.25,
            "accuracy": 0.30,
            "clarity": 0.15,
            "completeness": 0.20,
            "instruction_quality": 0.10
        }

        total_score = 0.0
        total_weight = 0.0

        for dimension, weight in weights.items():
            if dimension in dimension_scores:
                total_score += dimension_scores[dimension] * weight
                total_weight += weight

        if total_weight > 0:
            return total_score / total_weight
        else:
            return 5.0  # Default middle score

    def get_improvement_suggestions(
        self,
        dimension_scores: Dict[str, float]
    ) -> List[str]:
        """
        Get specific improvement suggestions based on low-scoring dimensions.

        Args:
            dimension_scores: Dictionary of dimension scores

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        if dimension_scores.get("relevance", 10) < 7:
            suggestions.append(
                "Improve relevance: Ensure content aligns with source material and domain"
            )

        if dimension_scores.get("accuracy", 10) < 7:
            suggestions.append(
                "Improve accuracy: Verify facts and ensure technical correctness"
            )

        if dimension_scores.get("clarity", 10) < 7:
            suggestions.append(
                "Improve clarity: Restructure for better readability and organization"
            )

        if dimension_scores.get("completeness", 10) < 7:
            suggestions.append(
                "Improve completeness: Add more detail and cover the topic thoroughly"
            )

        if dimension_scores.get("instruction_quality", 10) < 7:
            suggestions.append(
                "Improve instruction: Make the question/prompt more specific and clear"
            )

        return suggestions


class QualityFilter:
    """
    Filter samples based on quality criteria.

    Provides various filtering strategies.
    """

    @staticmethod
    def filter_by_threshold(
        samples: List[Dict[str, Any]],
        threshold: float = 6.0
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter samples by quality threshold.

        Args:
            samples: List of samples with quality metrics
            threshold: Minimum quality score

        Returns:
            Tuple of (passing_samples, failing_samples)
        """
        passing = []
        failing = []

        for sample in samples:
            score = sample.get("metrics", {}).get("quality_score", 0.0)
            if score >= threshold:
                passing.append(sample)
            else:
                failing.append(sample)

        logger.info(
            f"Quality filter: {len(passing)}/{len(samples)} passed "
            f"(threshold: {threshold})"
        )

        return passing, failing

    @staticmethod
    def filter_by_dimension(
        samples: List[Dict[str, Any]],
        dimension: str,
        min_score: float = 7.0
    ) -> List[Dict[str, Any]]:
        """
        Filter samples by specific quality dimension.

        Args:
            samples: List of samples
            dimension: Dimension to filter on (relevance, accuracy, etc.)
            min_score: Minimum score for that dimension

        Returns:
            Filtered samples
        """
        filtered = []

        for sample in samples:
            dimension_scores = sample.get("quality_details", {}).get("dimension_scores", {})
            score = dimension_scores.get(dimension, 0.0)

            if score >= min_score:
                filtered.append(sample)

        logger.info(
            f"Dimension filter ({dimension}>={min_score}): "
            f"{len(filtered)}/{len(samples)} passed"
        )

        return filtered

    @staticmethod
    def get_top_k(
        samples: List[Dict[str, Any]],
        k: int,
        by: str = "overall_score"
    ) -> List[Dict[str, Any]]:
        """
        Get top K samples by quality metric.

        Args:
            samples: List of samples
            k: Number of samples to return
            by: Metric to sort by

        Returns:
            Top K samples
        """
        sorted_samples = sorted(
            samples,
            key=lambda x: x.get("metrics", {}).get("quality_score", 0.0),
            reverse=True
        )

        return sorted_samples[:k]
