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
        source_content: Optional[str] = None,
        source_content_by_file: Optional[Dict[str, str]] = None,
        max_retries: int = 10,
        quality_check_batch_size: int = 50
    ):
        """
        Initialize generator.

        Args:
            model_router: Router for different model tasks
            cost_tracker: Cost tracking system
            config: Generation configuration
            source_content: Combined source material for generation context
            source_content_by_file: Per-file source content mapping (filename -> content)
            max_retries: Maximum retry attempts for failed samples
            quality_check_batch_size: Number of samples to check per API call
        """
        self.model_router = model_router
        self.cost_tracker = cost_tracker
        self.config = config
        self.source_content = source_content or ""
        self.source_content_by_file = source_content_by_file or {}
        self.max_retries = max_retries
        self.quality_check_batch_size = quality_check_batch_size

        self.generated_samples = []
        self.failed_samples = []
        self.total_regenerations = 0
        self.retry_count = 0

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
    async def _generate_batch(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of samples.

        Args:
            batch_size: Number of samples to generate
            batch_spec: Optional batch specification from generation plan containing:
                - topic: Major topic name
                - subtopic: Specific subtopic for this batch
                - guidance: Detailed generation instructions
                - relevant_files: List of relevant source files
                - focus_areas: Key concepts to cover

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

    async def _score_quality_batch(self, samples: List[Dict[str, Any]]) -> List[float]:
        """
        Score multiple samples in a single API call (batched quality checking).

        Args:
            samples: List of samples to score

        Returns:
            List of quality scores (1.0-10.0) for each sample
        """
        if not samples:
            return []

        # Create batched scoring prompt
        samples_text = "\n\n".join([
            f"=== SAMPLE {i+1} ===\n{json.dumps(sample, indent=2)}"
            for i, sample in enumerate(samples)
        ])

        scoring_prompt = f"""
Score each of the following {len(samples)} training data samples from 1-10 based on:
1. Relevance to source material
2. Accuracy and correctness
3. Clarity and completeness
4. Instruction-following quality
5. Appropriate difficulty level

Source Context:
{self.source_content[:1000] if self.source_content else "No source context provided"}

Data Type: {self.data_type_name.upper()}

SAMPLES TO SCORE:
{samples_text}

Provide ONLY the scores as a JSON array of numbers, nothing else.
Example: [7.5, 8.2, 6.3, 9.0, ...]

Output (JSON array only):
"""

        try:
            response = await self.model_router.verify(
                scoring_prompt,
                temperature=0.3,
                max_tokens=500
            )

            # Extract JSON array
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0].strip()

            scores = json.loads(response_clean)

            # Validate and clamp scores
            if not isinstance(scores, list) or len(scores) != len(samples):
                logger.warning(f"Batch scoring returned {len(scores)} scores for {len(samples)} samples")
                # Fall back to default scores
                return [5.0] * len(samples)

            # Clamp each score to 1-10 range
            scores = [max(1.0, min(10.0, float(s))) for s in scores]

            logger.debug(f"Batch quality scores: {scores}")
            return scores

        except Exception as e:
            logger.error(f"Error in batch scoring: {e}")
            logger.warning("Falling back to individual scoring")
            # Fall back to individual scoring
            return [await self._score_quality(sample) for sample in samples]

    async def generate(
        self,
        num_samples: int,
        show_progress: bool = True,
        checkpoint_dir: Optional[Any] = None,
        checkpoint_interval: int = 100,
        generation_plan: Optional[Dict[str, Any]] = None
    ) -> List[DatasetSample]:
        """
        Generate dataset samples with quality checking and refinement.

        Args:
            num_samples: Number of samples to generate
            show_progress: Whether to show progress information
            checkpoint_dir: Directory to save checkpoints (Path object or None)
            checkpoint_interval: Save checkpoint every N samples
            generation_plan: Optional batch-level plan from GeminiPlanner with topic-specific guidance

        Returns:
            List of validated, high-quality samples
        """
        samples_generated = 0
        samples_needed = num_samples

        if show_progress:
            print(f"\n🚀 Starting generation of {num_samples} {self.data_type_name.upper()} samples")
            print(f"📊 Quality threshold: {self.config.quality_threshold}/10")
            print(f"💰 Cost limit: ${self.config.max_cost:.2f}")
            if checkpoint_dir:
                print(f"💾 Checkpointing enabled: every {checkpoint_interval} samples")
            if generation_plan and "batches" in generation_plan:
                print(f"📋 Using Gemini plan: {len(generation_plan['batches'])} batches")
                print(f"📚 Domain: {generation_plan.get('domain', 'General')}")
            print()

        # If we have a plan, use batch-by-batch generation with topic guidance
        if generation_plan and "batches" in generation_plan:
            return await self._generate_with_plan(
                generation_plan,
                num_samples,
                show_progress,
                checkpoint_dir,
                checkpoint_interval
            )

        # Otherwise, use standard chunked generation (no topic guidance)
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

                # Validate all samples first
                valid_samples = []
                valid_indices = []
                for i, raw_sample in enumerate(raw_samples):
                    if self._validate_sample(raw_sample):
                        valid_samples.append(raw_sample)
                        valid_indices.append(i)
                    else:
                        logger.warning(f"Sample {i} failed validation")
                        self.failed_samples.append({
                            "sample": raw_sample,
                            "reason": "validation_failed"
                        })

                if show_progress:
                    print(f"  └─ Validated: {len(valid_samples)}/{len(raw_samples)} samples passed")

                # Batch quality scoring (process in chunks)
                all_quality_scores = []
                for chunk_start in range(0, len(valid_samples), self.quality_check_batch_size):
                    chunk_end = min(chunk_start + self.quality_check_batch_size, len(valid_samples))
                    chunk = valid_samples[chunk_start:chunk_end]

                    if show_progress:
                        print(f"  └─ Quality check: batch {chunk_start//self.quality_check_batch_size + 1} ({len(chunk)} samples)...", end="", flush=True)

                    # Score batch
                    chunk_scores = await self._score_quality_batch(chunk)
                    all_quality_scores.extend(chunk_scores)

                    # Track verification cost
                    verify_cost = self.model_router.get_total_cost() - self.cost_tracker.total_cost
                    if not self.cost_tracker.add_cost(verify_cost, operation="verification"):
                        break

                    if show_progress:
                        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
                        print(f" ✓ Avg: {avg_score:.1f}/10")

                # Create dataset samples from quality-passing samples
                batch_samples = []
                for i, (raw_sample, quality_score) in enumerate(zip(valid_samples, all_quality_scores)):
                    # Check if meets quality threshold
                    if quality_score < self.config.quality_threshold:
                        logger.info(
                            f"Sample below quality threshold: {quality_score:.1f} < {self.config.quality_threshold}"
                        )
                        self.failed_samples.append({
                            "sample": raw_sample,
                            "reason": "low_quality",
                            "score": quality_score
                        })
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

                    except Exception as e:
                        logger.error(f"Error creating dataset sample: {e}")

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

                # Save checkpoint if needed
                if checkpoint_dir and samples_generated % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_dir, samples_generated, show_progress)

                # Handle failed samples (regenerate if needed)
                failed_in_batch = batch_size - len(batch_samples)
                if failed_in_batch > 0 and self.cost_tracker.can_continue():
                    self.retry_count += 1
                    if self.retry_count >= self.max_retries:
                        logger.warning(f"Max retry limit reached ({self.max_retries}), stopping regeneration")
                        break
                    logger.info(f"Regenerating {failed_in_batch} failed samples (retry {self.retry_count}/{self.max_retries})")
                    # Don't increment samples_generated yet - will retry
                    continue
                else:
                    # Reset retry count on successful batch
                    self.retry_count = 0

            except Exception as e:
                logger.error(f"Error generating batch: {e}")
                if show_progress:
                    print(f"❌ Error: {e}")

                # Break if we hit too many errors
                if len(self.failed_samples) > num_samples * 2:
                    logger.error("Too many failures, stopping generation")
                    break

        # Save final checkpoint if checkpointing is enabled
        if checkpoint_dir and samples_generated > 0:
            self._save_checkpoint(checkpoint_dir, samples_generated, show_progress, final=True)

        if show_progress:
            print(f"\n{'='*60}")
            print(f"✅ Generation complete!")
            print(f"   Samples generated: {samples_generated}/{num_samples}")
            print(f"   Failed samples: {len(self.failed_samples)}")
            print(f"   Total cost: ${self.cost_tracker.total_cost:.2f}")
            print(f"{'='*60}\n")

        return self.generated_samples[:num_samples]

    async def _generate_with_plan(
        self,
        generation_plan: Dict[str, Any],
        num_samples: int,
        show_progress: bool,
        checkpoint_dir: Optional[Any],
        checkpoint_interval: int
    ) -> List[DatasetSample]:
        """
        Generate samples using batch-level plan with topic guidance.

        This method loops through generation_plan["batches"] and generates
        each batch with specific topic/subtopic guidance.

        Args:
            generation_plan: Batch-level plan from GeminiPlanner
            num_samples: Total samples to generate
            show_progress: Show progress output
            checkpoint_dir: Checkpoint directory
            checkpoint_interval: Checkpoint frequency

        Returns:
            List of generated samples
        """
        batches = generation_plan["batches"]
        samples_generated = 0

        for batch_spec in batches:
            if samples_generated >= num_samples:
                break

            if not self.cost_tracker.can_continue():
                logger.warning("Cost limit reached, stopping generation")
                break

            batch_num = batch_spec.get("batch_number", 0)
            topic = batch_spec.get("topic", "General")
            subtopic = batch_spec.get("subtopic", "")
            batch_size = generation_plan.get("batch_size", 20)

            # Adjust batch size for last batch
            remaining = num_samples - samples_generated
            actual_batch_size = min(batch_size, remaining)

            if show_progress:
                print(f"\n📦 Batch {batch_num}/{len(batches)}: {topic} → {subtopic}")
                print(f"   Generating {actual_batch_size} samples... ", end="", flush=True)

            try:
                # Generate batch with topic-specific guidance
                start_time = datetime.now()
                raw_samples = await self._generate_batch(actual_batch_size, batch_spec)
                generation_time = (datetime.now() - start_time).total_seconds()

                # Track cost
                last_cost = self.model_router.get_total_cost() - self.cost_tracker.total_cost
                if not self.cost_tracker.add_cost(last_cost, operation="generation"):
                    logger.warning("User stopped generation")
                    break

                if show_progress:
                    print(f"✓ Generated (${last_cost:.3f})")

                # Validate samples
                valid_samples = []
                for i, raw_sample in enumerate(raw_samples):
                    if self._validate_sample(raw_sample):
                        valid_samples.append(raw_sample)
                    else:
                        logger.warning(f"Sample {i} failed validation")
                        self.failed_samples.append({
                            "sample": raw_sample,
                            "reason": "validation_failed"
                        })

                if show_progress:
                    print(f"   └─ Validated: {len(valid_samples)}/{len(raw_samples)} samples")

                # Batch quality scoring
                all_quality_scores = []
                for chunk_start in range(0, len(valid_samples), self.quality_check_batch_size):
                    chunk_end = min(chunk_start + self.quality_check_batch_size, len(valid_samples))
                    chunk = valid_samples[chunk_start:chunk_end]

                    if show_progress:
                        print(f"   └─ Quality check ({len(chunk)} samples)... ", end="", flush=True)

                    chunk_scores = await self._score_quality_batch(chunk)
                    all_quality_scores.extend(chunk_scores)

                    verify_cost = self.model_router.get_total_cost() - self.cost_tracker.total_cost
                    if not self.cost_tracker.add_cost(verify_cost, operation="verification"):
                        break

                    if show_progress:
                        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
                        print(f"✓ Avg: {avg_score:.1f}/10")

                # Create dataset samples from quality-passing samples
                batch_samples = []
                for raw_sample, quality_score in zip(valid_samples, all_quality_scores):
                    if quality_score < self.config.quality_threshold:
                        logger.info(f"Sample below quality threshold: {quality_score:.1f}")
                        self.failed_samples.append({
                            "sample": raw_sample,
                            "reason": "low_quality",
                            "score": quality_score
                        })
                        continue

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
                        batch_samples.append(DatasetSample(data=validated_data, metrics=metrics))
                    except Exception as e:
                        logger.error(f"Error creating dataset sample: {e}")
                        continue

                # Smart regeneration: only regenerate failed samples
                regeneration_attempt = 0
                while len(batch_samples) < actual_batch_size and regeneration_attempt < self.max_retries:
                    samples_needed = actual_batch_size - len(batch_samples)
                    regeneration_attempt += 1

                    if show_progress:
                        print(f"   └─ Regenerating {samples_needed} failed samples (attempt {regeneration_attempt}/{self.max_retries})... ", end="", flush=True)

                    try:
                        # Generate only the missing samples with same batch_spec
                        regen_start_time = datetime.now()
                        regen_raw_samples = await self._generate_batch(samples_needed, batch_spec)
                        regen_generation_time = (datetime.now() - regen_start_time).total_seconds()

                        # Track regeneration cost
                        regen_cost = self.model_router.get_total_cost() - self.cost_tracker.total_cost
                        if not self.cost_tracker.add_cost(regen_cost, operation="regeneration"):
                            logger.warning("Cost limit reached during regeneration")
                            break

                        if show_progress:
                            print(f"✓ (${regen_cost:.3f})")

                        # Validate regenerated samples
                        regen_valid_samples = []
                        for i, raw_sample in enumerate(regen_raw_samples):
                            if self._validate_sample(raw_sample):
                                regen_valid_samples.append(raw_sample)
                            else:
                                logger.warning(f"Regenerated sample {i} failed validation")
                                self.failed_samples.append({
                                    "sample": raw_sample,
                                    "reason": "validation_failed",
                                    "regeneration_attempt": regeneration_attempt
                                })

                        if show_progress:
                            print(f"   └─ Validated: {len(regen_valid_samples)}/{len(regen_raw_samples)} regenerated samples")

                        # Batch quality scoring for regenerated samples
                        regen_quality_scores = []
                        for chunk_start in range(0, len(regen_valid_samples), self.quality_check_batch_size):
                            chunk_end = min(chunk_start + self.quality_check_batch_size, len(regen_valid_samples))
                            chunk = regen_valid_samples[chunk_start:chunk_end]

                            if show_progress:
                                print(f"   └─ Quality check ({len(chunk)} regenerated samples)... ", end="", flush=True)

                            chunk_scores = await self._score_quality_batch(chunk)
                            regen_quality_scores.extend(chunk_scores)

                            verify_cost = self.model_router.get_total_cost() - self.cost_tracker.total_cost
                            if not self.cost_tracker.add_cost(verify_cost, operation="verification"):
                                break

                            if show_progress:
                                avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
                                print(f"✓ Avg: {avg_score:.1f}/10")

                        # Add passing regenerated samples to batch
                        regen_passed = 0
                        for raw_sample, quality_score in zip(regen_valid_samples, regen_quality_scores):
                            if quality_score < self.config.quality_threshold:
                                logger.info(f"Regenerated sample below quality threshold: {quality_score:.1f}")
                                self.failed_samples.append({
                                    "sample": raw_sample,
                                    "reason": "low_quality",
                                    "score": quality_score,
                                    "regeneration_attempt": regeneration_attempt
                                })
                                continue

                            try:
                                validated_data = self.data_format(**raw_sample)
                                metrics = QualityMetrics(
                                    quality_score=quality_score,
                                    token_count=len(json.dumps(raw_sample)) // 4,
                                    generation_cost=regen_cost / len(regen_raw_samples),
                                    model_used=self.model_router.generator.model,
                                    generation_time=regen_generation_time / len(regen_raw_samples),
                                    regeneration_count=regeneration_attempt
                                )
                                batch_samples.append(DatasetSample(data=validated_data, metrics=metrics))
                                regen_passed += 1
                            except Exception as e:
                                logger.error(f"Error creating regenerated dataset sample: {e}")
                                continue

                        if show_progress:
                            print(f"   └─ Accepted: {regen_passed} regenerated samples")

                        # If we got no passing samples this attempt, no point continuing
                        if regen_passed == 0:
                            logger.warning(f"No passing samples in regeneration attempt {regeneration_attempt}, stopping regeneration")
                            break

                    except Exception as e:
                        logger.error(f"Error during regeneration attempt {regeneration_attempt}: {e}")
                        break

                if len(batch_samples) < actual_batch_size:
                    shortfall = actual_batch_size - len(batch_samples)
                    logger.warning(f"Batch incomplete: {len(batch_samples)}/{actual_batch_size} samples (short by {shortfall})")
                    if show_progress:
                        print(f"   ⚠️  Batch incomplete: {len(batch_samples)}/{actual_batch_size} samples")

                self.generated_samples.extend(batch_samples)
                samples_generated += len(batch_samples)

                if show_progress:
                    print(f"   └─ Accepted: {len(batch_samples)} samples (total: {samples_generated}/{num_samples})")

                # Checkpointing
                if checkpoint_dir and samples_generated % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_dir, samples_generated, show_progress)

            except Exception as e:
                logger.error(f"Error generating batch {batch_num}: {e}")
                continue

        # Final checkpoint
        if checkpoint_dir and samples_generated > 0:
            self._save_checkpoint(checkpoint_dir, samples_generated, show_progress, final=True)

        if show_progress:
            print(f"\n{'='*60}")
            print(f"✅ Generation complete!")
            print(f"   Generated samples: {samples_generated}")
            print(f"   Failed samples: {len(self.failed_samples)}")
            print(f"   Total cost: ${self.cost_tracker.total_cost:.2f}")
            print(f"{'='*60}\n")

        return self.generated_samples[:num_samples]

    def _save_checkpoint(self, checkpoint_dir, samples_count: int, show_progress: bool = True, final: bool = False):
        """
        Save checkpoint of generated samples.

        Args:
            checkpoint_dir: Directory to save checkpoints (Path object)
            samples_count: Number of samples generated so far
            show_progress: Whether to show progress messages
            final: Whether this is the final checkpoint
        """
        try:
            checkpoint_name = f"checkpoint_final.jsonl" if final else f"checkpoint_{samples_count}.jsonl"
            checkpoint_path = checkpoint_dir / checkpoint_name

            # Save samples in JSONL format
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                for sample in self.generated_samples[:samples_count]:
                    data_dict = sample.data.model_dump()
                    f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')

            # Also save metadata
            metadata_path = checkpoint_dir / f"checkpoint_{samples_count}_meta.json"
            metadata = {
                "samples_count": samples_count,
                "total_cost": self.cost_tracker.total_cost,
                "failed_samples": len(self.failed_samples),
                "data_type": self.data_type_name,
                "timestamp": datetime.now().isoformat()
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            if show_progress:
                checkpoint_label = "FINAL" if final else samples_count
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
                print(f"   💾 Checkpoint saved: {checkpoint_label} samples -> {checkpoint_path}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            if show_progress:
                print(f"   ⚠️  Warning: Failed to save checkpoint: {e}")

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
