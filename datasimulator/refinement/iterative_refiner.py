"""
Iterative refinement system.

Automatically regenerates low-quality samples with improvements.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import asyncio

logger = logging.getLogger(__name__)


class IterativeRefiner:
    """
    Iteratively refine generated samples.

    Identifies low-quality samples and regenerates them with
    improvements based on feedback.
    """

    def __init__(
        self,
        quality_threshold: float = 6.0,
        max_retries: int = 3,
        improvement_strategy: str = "feedback"
    ):
        """
        Initialize iterative refiner.

        Args:
            quality_threshold: Minimum acceptable quality score
            max_retries: Maximum regeneration attempts per sample
            improvement_strategy: Strategy for improvement ("feedback", "rewrite", "enhance")
        """
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        self.improvement_strategy = improvement_strategy

        # Track refinement stats
        self.stats = {
            "samples_refined": 0,
            "total_retries": 0,
            "success_rate": 0.0
        }

    async def refine_samples(
        self,
        samples: List[Dict[str, Any]],
        generator_func: Callable,
        quality_scorer: Any
    ) -> List[Dict[str, Any]]:
        """
        Refine samples that don't meet quality threshold.

        Args:
            samples: List of samples with quality scores
            generator_func: Async function to generate new samples
            quality_scorer: Quality scoring system

        Returns:
            Refined samples (all above threshold or max retries reached)
        """
        refined_samples = []
        samples_to_refine = []

        # Separate good and bad samples
        for sample in samples:
            quality_score = sample.get("metrics", {}).get("quality_score", 0.0)

            if quality_score >= self.quality_threshold:
                refined_samples.append(sample)
            else:
                samples_to_refine.append(sample)
                logger.debug(
                    f"Sample needs refinement: quality={quality_score:.1f} "
                    f"< {self.quality_threshold}"
                )

        if not samples_to_refine:
            logger.info("All samples meet quality threshold")
            return refined_samples

        logger.info(
            f"Refining {len(samples_to_refine)}/{len(samples)} samples "
            f"(below threshold: {self.quality_threshold})"
        )

        # Refine each low-quality sample
        for sample in samples_to_refine:
            refined = await self._refine_single_sample(
                sample,
                generator_func,
                quality_scorer
            )

            if refined:
                refined_samples.append(refined)
                self.stats["samples_refined"] += 1
            else:
                # Keep original if refinement failed
                refined_samples.append(sample)
                logger.warning("Refinement failed, keeping original sample")

        # Update success rate
        self.stats["success_rate"] = (
            self.stats["samples_refined"] / len(samples_to_refine)
            if samples_to_refine else 1.0
        )

        logger.info(
            f"Refinement complete: {self.stats['samples_refined']} improved "
            f"({self.stats['success_rate']*100:.1f}% success rate)"
        )

        return refined_samples

    async def _refine_single_sample(
        self,
        sample: Dict[str, Any],
        generator_func: Callable,
        quality_scorer: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Refine a single sample with retries.

        Args:
            sample: Sample to refine
            generator_func: Generator function
            quality_scorer: Quality scorer

        Returns:
            Refined sample or None if failed
        """
        original_score = sample.get("metrics", {}).get("quality_score", 0.0)

        # Extract improvement feedback
        feedback = self._get_improvement_feedback(sample)

        for retry in range(self.max_retries):
            self.stats["total_retries"] += 1

            logger.debug(
                f"Refinement attempt {retry + 1}/{self.max_retries} "
                f"(original score: {original_score:.1f})"
            )

            # Generate improved version
            improved = await generator_func(
                count=1,
                improvement_context=feedback,
                original_sample=sample
            )

            if not improved:
                continue

            # Score improved sample
            improved_sample = improved[0]
            score_result = await quality_scorer.score_sample(improved_sample)

            new_score = score_result.get("overall_score", 0.0)

            logger.debug(f"Improved score: {new_score:.1f} (was {original_score:.1f})")

            # Check if improvement is sufficient
            if new_score >= self.quality_threshold:
                logger.info(
                    f"✓ Refinement successful: {original_score:.1f} → {new_score:.1f}"
                )
                # Update metrics
                improved_sample["metrics"]["quality_score"] = new_score
                improved_sample["metrics"]["refinement_attempts"] = retry + 1
                improved_sample["metrics"]["original_score"] = original_score

                return improved_sample

            # Update feedback for next iteration
            feedback = self._update_feedback(feedback, score_result)

        logger.warning(
            f"✗ Refinement failed after {self.max_retries} attempts: "
            f"{original_score:.1f} → {new_score:.1f}"
        )

        return None

    def _get_improvement_feedback(self, sample: Dict[str, Any]) -> str:
        """
        Extract improvement feedback from sample scores.

        Args:
            sample: Sample with quality scores

        Returns:
            Feedback string for improvement
        """
        feedback_parts = []

        # Get dimension scores
        dimension_scores = sample.get("metrics", {}).get("dimension_scores", {})

        if dimension_scores:
            # Identify weak dimensions
            for dimension, score in dimension_scores.items():
                if isinstance(score, (int, float)) and score < 7.0:
                    feedback_parts.append(
                        f"Improve {dimension} (current: {score:.1f}/10)"
                    )

            # Add specific feedback if available
            if "feedback" in dimension_scores:
                feedback_parts.append(dimension_scores["feedback"])

        if not feedback_parts:
            feedback_parts.append("Improve overall quality and relevance")

        return " | ".join(feedback_parts)

    def _update_feedback(
        self,
        current_feedback: str,
        score_result: Dict[str, Any]
    ) -> str:
        """Update feedback based on new scores."""
        # Extract new low-scoring dimensions
        new_feedback = []

        dimension_scores = score_result.get("dimension_scores", {})
        for dimension, score in dimension_scores.items():
            if isinstance(score, (int, float)) and score < 7.0:
                new_feedback.append(f"Still needs work on {dimension}")

        if new_feedback:
            return " | ".join(new_feedback)
        else:
            return current_feedback

    def get_stats(self) -> Dict[str, Any]:
        """Get refinement statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset refinement statistics."""
        self.stats = {
            "samples_refined": 0,
            "total_retries": 0,
            "success_rate": 0.0
        }


class AdaptiveRefiner(IterativeRefiner):
    """
    Adaptive refiner that learns from successful refinements.

    Adjusts strategy based on what works.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Track successful strategies
        self.strategy_success: Dict[str, int] = {
            "feedback": 0,
            "rewrite": 0,
            "enhance": 0
        }

    async def _refine_single_sample(
        self,
        sample: Dict[str, Any],
        generator_func: Callable,
        quality_scorer: Any
    ) -> Optional[Dict[str, Any]]:
        """Refine with adaptive strategy selection."""

        # Try different strategies in order of past success
        strategies = sorted(
            self.strategy_success.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for strategy_name, _ in strategies:
            self.improvement_strategy = strategy_name

            result = await super()._refine_single_sample(
                sample,
                generator_func,
                quality_scorer
            )

            if result:
                # Track successful strategy
                self.strategy_success[strategy_name] += 1
                logger.debug(f"Successful strategy: {strategy_name}")
                return result

        return None
