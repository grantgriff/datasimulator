"""
Abstract base class for all data generators.

Provides common functionality for batch generation, quality checking,
and iterative refinement.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from ..data_models import (
    TrainingDataFormat,
    QualityMetrics,
    DatasetSample,
    GenerationConfig
)
from ..models.llm_client import ModelRouter
from ...utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base class for all training data generators.

    Subclasses must implement:
    - _generate_batch(): Generate a batch of samples
    - _validate_sample(): Validate a single sample
    - data_format: Pydantic model for the data format
    """

    def __init__(
        self,
        model_router: ModelRouter,
        cost_tracker: CostTracker,
        config: GenerationConfig,
        source_content: Optional[str] = None
    ):
        """
        Initialize generator.

        Args:
            model_router: Router for different model tasks
            cost_tracker: Cost tracking system
            config: Generation configuration
            source_content: Source material for generation context
        """
        self.model_router = model_router
        self.cost_tracker = cost_tracker
        self.config = config
        self.source_content = source_content or ""

        self.generated_samples = []
        self.failed_samples = []
        self.total_regenerations = 0

    @property
    @abstractmethod
    def data_format(self) -> Type[TrainingDataFormat]:
        """Return the Pydantic model for this data format."""
        pass

    @property
    @abstractmethod
    def data_type_name(self) -> str:
        """Return the name of this data type (e.g., 'sft', 'dpo')."""
        pass

    @abstractmethod
    async def _generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Generate a batch of samples.

        Args:
            batch_size: Number of samples to generate

        Returns:
            List of raw sample dictionaries
        """
        pass

    @abstractmethod
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample matches the expected format.

        Args:
            sample: Raw sample dictionary

        Returns:
            True if valid, False otherwise
        """
        pass

    async def _score_quality(self, sample: Dict[str, Any]) -> float:
        """
        Score sample quality from 1-10.

        Uses verifier model to evaluate:
        - Relevance to source material
        - Accuracy and correctness
        - Clarity and completeness
        - Instruction-following quality

        Args:
            sample: Sample to score

        Returns:
            Quality score from 1.0 to 10.0
        """
        scoring_prompt = f"""
Score this training data sample from 1-10 based on:
1. Relevance to source material (if provided)
2. Accuracy and correctness
3. Clarity and completeness
4. Instruction-following quality
5. Appropriate difficulty level

Source Context:
{self.source_content[:1000] if self.source_content else "No source context provided"}

Data Type: {self.data_type_name.upper()}

Sample:
{json.dumps(sample, indent=2)}

Provide ONLY a single number from 1-10 (can use decimals like 7.5).
No explanation needed, just the number.
"""

        try:
            response = await self.model_router.verify(
                scoring_prompt,
                temperature=0.3,
                max_tokens=10
            )

            # Extract numeric score
            score_str = response.strip()
            score = float(score_str)

            # Clamp to 1-10 range
            score = max(1.0, min(10.0, score))

            logger.debug(f"Quality score: {score}/10")
            return score

        except Exception as e:
            logger.error(f"Error scoring quality: {e}")
            # Default to below threshold to trigger regeneration
            return 5.0

    async def generate(
        self,
        num_samples: int,
        show_progress: bool = True
    ) -> List[DatasetSample]:
        """
        Generate dataset samples with quality checking and refinement.

        Args:
            num_samples: Number of samples to generate
            show_progress: Whether to show progress information

        Returns:
            List of validated, high-quality samples
        """
        samples_generated = 0
        samples_needed = num_samples

        if show_progress:
            print(f"\n🚀 Starting generation of {num_samples} {self.data_type_name.upper()} samples")
            print(f"📊 Quality threshold: {self.config.quality_threshold}/10")
            print(f"💰 Cost limit: ${self.config.max_cost:.2f}\n")

        while samples_generated < num_samples:
            if not self.cost_tracker.can_continue():
                logger.warning("Cost limit reached, stopping generation")
                break

            # Calculate batch size
            remaining = num_samples - samples_generated
            batch_size = min(self.config.batch_size, remaining)

            if show_progress:
                print(f"📦 Generating batch of {batch_size} samples... ", end="", flush=True)

            try:
                # Generate batch
                start_time = datetime.now()
                raw_samples = await self._generate_batch(batch_size)
                generation_time = (datetime.now() - start_time).total_seconds()

                # Track cost for this batch
                last_cost = self.model_router.get_total_cost() - self.cost_tracker.total_cost
                if not self.cost_tracker.add_cost(last_cost, operation="generation"):
                    logger.warning("User stopped generation")
                    break

                if show_progress:
                    print(f"✓ Generated (${last_cost:.3f})")

                # Validate and score each sample
                batch_samples = []
                for i, raw_sample in enumerate(raw_samples):
                    if show_progress:
                        print(f"  └─ Sample {i+1}/{len(raw_samples)}: ", end="", flush=True)

                    # Validate format
                    if not self._validate_sample(raw_sample):
                        logger.warning(f"Sample {i} failed validation")
                        self.failed_samples.append({
                            "sample": raw_sample,
                            "reason": "validation_failed"
                        })
                        if show_progress:
                            print("❌ Invalid format")
                        continue

                    # Score quality
                    quality_score = await self._score_quality(raw_sample)

                    # Track verification cost
                    verify_cost = self.model_router.get_total_cost() - self.cost_tracker.total_cost
                    if not self.cost_tracker.add_cost(verify_cost, operation="verification"):
                        break

                    # Check if meets quality threshold
                    if quality_score < self.config.quality_threshold:
                        logger.info(
                            f"Sample {i} below quality threshold: "
                            f"{quality_score:.1f} < {self.config.quality_threshold}"
                        )
                        self.failed_samples.append({
                            "sample": raw_sample,
                            "reason": "low_quality",
                            "score": quality_score
                        })
                        if show_progress:
                            print(f"❌ Low quality ({quality_score:.1f}/10)")
                        continue

                    # Create dataset sample with metrics
                    try:
                        validated_data = self.data_format(**raw_sample)
                        metrics = QualityMetrics(
                            quality_score=quality_score,
                            token_count=len(json.dumps(raw_sample)) // 4,
                            generation_cost=last_cost / len(raw_samples),
                            model_used=self.model_router.generator.model,
                            generation_time=generation_time / len(raw_samples),
                            regeneration_count=0
                        )

                        dataset_sample = DatasetSample(
                            data=validated_data,
                            metrics=metrics
                        )

                        batch_samples.append(dataset_sample)
                        if show_progress:
                            print(f"✓ Quality: {quality_score:.1f}/10")

                    except Exception as e:
                        logger.error(f"Error creating dataset sample: {e}")
                        if show_progress:
                            print(f"❌ Error: {e}")

                # Add successful samples
                self.generated_samples.extend(batch_samples)
                samples_generated += len(batch_samples)

                if show_progress:
                    success_rate = len(batch_samples) / len(raw_samples) * 100
                    print(
                        f"\n✓ Batch complete: {len(batch_samples)}/{len(raw_samples)} samples passed "
                        f"({success_rate:.0f}% success rate)"
                    )
                    print(
                        f"📈 Progress: {samples_generated}/{num_samples} "
                        f"({samples_generated/num_samples*100:.0f}%) | "
                        f"Cost: ${self.cost_tracker.total_cost:.2f}\n"
                    )

                # Handle failed samples (regenerate if needed)
                failed_in_batch = batch_size - len(batch_samples)
                if failed_in_batch > 0 and self.cost_tracker.can_continue():
                    logger.info(f"Regenerating {failed_in_batch} failed samples")
                    # Don't increment samples_generated yet - will retry
                    continue

            except Exception as e:
                logger.error(f"Error generating batch: {e}")
                if show_progress:
                    print(f"❌ Error: {e}")

                # Break if we hit too many errors
                if len(self.failed_samples) > num_samples * 2:
                    logger.error("Too many failures, stopping generation")
                    break

        if show_progress:
            print(f"\n{'='*60}")
            print(f"✅ Generation complete!")
            print(f"   Samples generated: {samples_generated}/{num_samples}")
            print(f"   Failed samples: {len(self.failed_samples)}")
            print(f"   Total cost: ${self.cost_tracker.total_cost:.2f}")
            print(f"{'='*60}\n")

        return self.generated_samples[:num_samples]

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        if not self.generated_samples:
            return {
                "total_samples": 0,
                "average_quality": 0.0,
                "total_cost": self.cost_tracker.total_cost,
                "failed_samples": len(self.failed_samples)
            }

        quality_scores = [s.metrics.quality_score for s in self.generated_samples]

        return {
            "total_samples": len(self.generated_samples),
            "average_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "total_cost": self.cost_tracker.total_cost,
            "failed_samples": len(self.failed_samples),
            "total_regenerations": self.total_regenerations,
            "success_rate": len(self.generated_samples) / (len(self.generated_samples) + len(self.failed_samples)) * 100
        }
