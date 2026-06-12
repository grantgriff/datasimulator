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
        quality_check_batch_size: int = 50,
        progress_callback: Optional[Any] = None
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
            progress_callback: Optional callable invoked with a dict for each
                lifecycle event (see BaseGenerator._emit). Can be sync or async.
                Exceptions are logged and swallowed.
        """
        self.model_router = model_router
        self.cost_tracker = cost_tracker
        self.config = config
        self.source_content = source_content or ""
        self.source_content_by_file = source_content_by_file or {}
        self.max_retries = max_retries
        self.quality_check_batch_size = quality_check_batch_size
        self.progress_callback = progress_callback

        self.generated_samples = []
        self.failed_samples = []
        self.total_regenerations = 0
        self.retry_count = 0

        # Lazy-init dedup state. Diversity check runs after quality
        # threshold passes; samples too similar (cosine >= diversity_threshold)
        # to any already-accepted sample get rejected as duplicates.
        self._diversity_checker = None
        self._accepted_comparable_texts: List[str] = []

    async def _emit(self, event: str, **payload) -> None:
        """
        Fire a progress event to the user-supplied callback, if any.

        The callback receives a single dict: {"event": <name>, ...payload}.
        Both sync and async callbacks are supported. Any exception raised
        by the callback is logged and swallowed so it cannot break generation.
        """
        if not self.progress_callback:
            return
        payload["event"] = event
        try:
            import asyncio as _asyncio
            result = self.progress_callback(payload)
            if _asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"progress_callback raised on event {event!r}: {e}")

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

    # ----- Generator prompt injection: diversity + factuality guard -----

    @staticmethod
    def _anti_duplication_block(n: int) -> str:
        """Return a generator-prompt block that explicitly tells the model
        to produce distinct, factually correct samples — and NOT to produce
        near-duplicates, restatements of the question, or factual errors.

        Used by every generator. Keeping the language in one place makes it
        easy to tune (and means the generator side mirrors the failure
        modes the verifier explicitly downgrades to <=3).
        """
        return f"""
=== CRITICAL: DIVERSITY & FACTUAL CORRECTNESS ===

The {n} samples in this batch MUST be UNIQUE AND DISTINCT from each other.
Each sample should explore a genuinely different angle, scenario, calculation,
edge case, or detail — not just rephrase the same idea with surface wording
changes.

ABSOLUTE FAILURES — do NOT produce any of the following:
1. Two or more samples that share essentially the same scenario, numbers, or
   framing with only minor wording differences. Each sample must be
   substantively different from every other sample in this batch.
2. Any sample whose response RESTATES THE QUESTION without adding new
   information — the answer must contribute concrete facts, calculations,
   or reasoning beyond what's already in the prompt.
3. Any sample containing a FACTUAL ERROR — wrong numbers, wrong rules,
   misattributed standards, fabricated citations, or claims that contradict
   the source material.

If you cannot produce {n} genuinely distinct, factually correct samples on
this topic, produce FEWER rather than padding with near-duplicates or
restatements. A short batch of strong samples is better than a full batch
of weak ones — the downstream verifier will reject the weak ones anyway.
"""

    # ----- Diversity / dedup -----

    def _get_diversity_checker(self):
        """Lazy-init the local DiversityChecker (sentence-transformers).

        The pipeline used to ignore diversity entirely — the threshold was
        plumbed through but never consulted by any filter. Now wired in
        between the quality-threshold check and sample acceptance.
        """
        if self._diversity_checker is None:
            from ...quality.diversity_checker import DiversityChecker
            self._diversity_checker = DiversityChecker(
                similarity_threshold=self.config.diversity_threshold,
                use_local=True,
            )
        return self._diversity_checker

    def _get_comparable_text(self, sample: Dict[str, Any]) -> str:
        """Extract the text used for sample-to-sample similarity comparison.

        Default implementation: concatenate every string value in the sample
        (recursing into lists/dicts) and return the joined text. Works for
        any data shape; subclasses can override if they want a tighter
        signal (e.g. SFT comparing user+assistant only).
        """
        parts: List[str] = []

        def walk(v):
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    walk(vv)
            elif isinstance(v, list):
                for vv in v:
                    walk(vv)

        walk(sample)
        return "\n".join(parts).strip()

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
You are a STRICT quality assessor for ML training data. Score this
sample from 1-10 using these anchors (default to skepticism):

  1-3  FAIL: errors, off-topic, plagiarized, or no training signal.
        Specifically score ≤3 if the response contains a FACTUAL ERROR,
        or if it RESTATES THE QUESTION without adding information.
  4-5  WEAK: valid but adds nothing the source doesn't already say —
        DEFAULT here unless you can name a specific strength
  6    PASSING: basic understanding, adds at least one useful detail
  7-8  GOOD: clearly above median, depth/specificity beyond paraphrase
  9    VERY GOOD: exemplary on multiple dimensions; rare
  10   PERFECT: top-tier training example; very rare (<5%)

Dimensions: relevance to source, factual accuracy, clarity, useful
depth beyond source, instruction-following, difficulty calibration.

Source Context:
{self.source_content[:1000] if self.source_content else "No source context provided"}

Data Type: {self.data_type_name.upper()}

Sample:
{json.dumps(sample, indent=2)}

Provide ONLY a single number from 1-10 (decimals like 5.5 are fine).
No explanation, just the number. Default to 5 unless a specific
strength justifies higher.
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
        n = len(samples)
        samples_text = "\n\n".join([
            f"=== SAMPLE {i+1} ===\n{json.dumps(sample, indent=2)}"
            for i, sample in enumerate(samples)
        ])

        # IMPORTANT: do NOT include "..." in any example. Models mimic the
        # example pattern verbatim — a literal "..." in the example
        # teaches them to truncate the output. This was the root cause of
        # "Batch scoring returned 5 scores for 10 samples" in earlier runs.
        # The example below uses a concrete N=3 case to anchor the format
        # without inviting truncation. The example scores deliberately
        # include a fail (3.5), a pass-but-weak (5.5), and a strong (8.0)
        # so the verifier sees the expected full range.
        scoring_prompt = f"""
You are a STRICT quality assessor for ML training data. Default to
skepticism — most training data in the wild is mediocre, and your
scores should reflect that. Generous scoring lets bad samples into
training sets.

Score each of the {n} samples below from 1-10 on a STRICT rubric:

  1-3  FAIL: factual errors, off-topic, unparsable, plagiarized verbatim
        from the source, or so short/generic it provides no training
        signal. Use freely for anything that shouldn't ship.
        SPECIFIC FAILURE MODES — score ≤3 if any apply:
          • Response contains a FACTUAL ERROR (wrong claim, wrong number,
            wrong rule, misattribution, fabricated citation).
          • Response RESTATES THE QUESTION without adding information
            (e.g., echoes the prompt's premise back as the answer).
          • Sample is a NEAR-DUPLICATE of another sample in this batch —
            same scenario, same numbers, same framing with only surface
            wording changes. Even a well-written near-duplicate is a fail
            because it provides no additional training signal.
  4-5  WEAK: technically valid but adds nothing the source doesn't
        already contain. Generic phrasing. Could be replaced by N
        similar samples with no loss. THIS IS THE DEFAULT for
        unremarkable samples — only go higher with specific evidence.
  6    PASSING: shows basic understanding, no clear flaws, adds at
        least ONE specific detail or framing useful for training.
  7-8  GOOD: clearly above median. Demonstrates depth, specificity, or
        pedagogical structure beyond a paraphrase of the source. Earn
        each point above 6 with a concrete reason.
  9    VERY GOOD: exemplary on multiple dimensions; hard to improve.
        Rare.
  10   PERFECT: would headline a published training set. Should be very
        rare — fewer than 1 in 20 samples in a typical batch.

Dimensions to weigh: relevance to source, factual accuracy, clarity,
useful depth beyond source, instruction-following, difficulty
calibration.

Source Context:
{self.source_content[:1000] if self.source_content else "No source context provided"}

Data Type: {self.data_type_name.upper()}

SAMPLES TO SCORE:
{samples_text}

CRITICAL CALIBRATION RULES — read before scoring:
- DEFAULT to 5 unless you can name a specific reason to go higher.
- Aim for a realistic spread: most samples land 4-7. A typical batch
  has 1-2 samples below 6 (filterable) and 1-2 samples at 8+.
- Do not score generously out of politeness. Mediocre samples get 5.
  Generic restatements of the source get 4.
- For each sample, silently identify ONE specific weakness OR strength
  before assigning. If you cannot name a strength, score ≤6.

CRITICAL OUTPUT RULES:
- Output a JSON array of EXACTLY {n} numbers — one score per sample,
  in order.
- No commentary, no markdown, no trailing comments or ellipses.
- Decimals are OK (e.g. 5.5, 7.2).
- If you are about to output fewer than {n} numbers, STOP and produce
  all {n}.

Example for a hypothetical 3-sample batch (yours has {n}) — note the
range across fail / weak-pass / strong:
[3.5, 5.5, 8.0]

Now output your JSON array of EXACTLY {n} scores:
"""

        try:
            response = await self.model_router.verify(
                scoring_prompt,
                temperature=0.3,
                # Per CLAUDE.md: never use a low cap. 128K is the project
                # default for any structured/batch response.
                max_tokens=128000
            )

            # Extract JSON array
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0].strip()

            scores = json.loads(response_clean)

            # Validate and clamp scores
            if not isinstance(scores, list) or len(scores) != n:
                # On length mismatch: keep the partial scores we got and
                # individually score the rest. Never silently assign 5.0
                # to everything — that masks real quality and (combined
                # with a quality_threshold) drops the whole batch.
                got = len(scores) if isinstance(scores, list) else 0
                logger.warning(
                    f"Batch scoring returned {got} scores for {n} samples — "
                    f"individually scoring the remaining {n - got}"
                )
                partial = []
                if isinstance(scores, list):
                    partial = [max(1.0, min(10.0, float(s))) for s in scores[:n]]
                missing_samples = samples[len(partial):]
                tail = [await self._score_quality(s) for s in missing_samples]
                return partial + tail

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

        await self._emit(
            "generation_started",
            num_samples=num_samples,
            data_type=self.data_type_name,
            quality_threshold=self.config.quality_threshold,
            max_cost=self.config.max_cost,
            num_planned_batches=(
                len(generation_plan["batches"])
                if generation_plan and "batches" in generation_plan
                else None
            ),
            domain=(generation_plan.get("domain") if generation_plan else None),
        )

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
                await self._emit(
                    "cost_limit_reached",
                    total_cost=self.cost_tracker.total_cost,
                    max_cost=self.config.max_cost,
                    samples_generated=samples_generated,
                    samples_target=num_samples,
                )
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

                    # Diversity / dedup check: cosine sim against already-accepted
                    # samples must be below diversity_threshold (default 0.85).
                    sample_text = self._get_comparable_text(raw_sample)
                    is_diverse, max_sim = self._get_diversity_checker().is_diverse(
                        sample_text, self._accepted_comparable_texts,
                    )
                    if not is_diverse:
                        logger.info(
                            f"Sample rejected as duplicate: similarity={max_sim:.3f} "
                            f">= {self.config.diversity_threshold}"
                        )
                        self.failed_samples.append({
                            "sample": raw_sample,
                            "reason": "low_diversity",
                            "similarity": max_sim,
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

                # Add successful samples + register their text for future
                # dedup comparison
                self.generated_samples.extend(batch_samples)
                for ds in batch_samples:
                    raw = ds.data.model_dump() if hasattr(ds.data, "model_dump") else dict(ds.data)
                    self._accepted_comparable_texts.append(self._get_comparable_text(raw))
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

                batch_avg_quality = (
                    sum(all_quality_scores) / len(all_quality_scores)
                    if all_quality_scores else 0.0
                )
                await self._emit(
                    "batch_completed",
                    samples_in_batch=len(raw_samples),
                    samples_passed=len(batch_samples),
                    batch_cost=last_cost,
                    average_quality=batch_avg_quality,
                    samples_generated=samples_generated,
                    samples_target=num_samples,
                    total_cost=self.cost_tracker.total_cost,
                )

                # Save checkpoint if needed
                if checkpoint_dir and samples_generated % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_dir, samples_generated, show_progress)
                    await self._emit(
                        "checkpoint_saved",
                        samples_generated=samples_generated,
                        checkpoint_dir=str(checkpoint_dir),
                    )

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

        await self._emit(
            "generation_completed",
            samples_generated=samples_generated,
            samples_target=num_samples,
            failed=len(self.failed_samples),
            total_cost=self.cost_tracker.total_cost,
        )

        return self.generated_samples[:num_samples]

    async def _process_single_batch(
        self,
        batch_spec: Dict[str, Any],
        generation_plan: Dict[str, Any],
        remaining_samples: int,
        show_progress: bool
    ) -> List[DatasetSample]:
        """
        Process a single batch - generate, validate, score, and regenerate if needed.

        Args:
            batch_spec: Batch specification with topic/subtopic
            generation_plan: Full generation plan
            remaining_samples: Number of samples still needed
            show_progress: Show progress output

        Returns:
            List of generated DatasetSamples for this batch
        """
        batch_num = batch_spec.get("batch_number", 0)
        topic = batch_spec.get("topic", "General")
        subtopic = batch_spec.get("subtopic", "")
        batch_size = generation_plan.get("batch_size", 20)

        # Adjust batch size for remaining samples
        actual_batch_size = min(batch_size, remaining_samples)

        if show_progress:
            print(f"\n📦 Batch {batch_num}: {topic} → {subtopic}")
            print(f"   Generating {actual_batch_size} samples... ", end="", flush=True)

        try:
            # Generate batch with topic-specific guidance
            start_time = datetime.now()
            raw_samples = await self._generate_batch(actual_batch_size, batch_spec)
            generation_time = (datetime.now() - start_time).total_seconds()

            # Track cost
            last_cost = self.model_router.get_total_cost() - self.cost_tracker.total_cost
            if not self.cost_tracker.add_cost(last_cost, operation="generation"):
                logger.warning("Cost limit reached during batch generation")
                return []

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

                # Diversity / dedup check against already-accepted samples
                sample_text = self._get_comparable_text(raw_sample)
                is_diverse, max_sim = self._get_diversity_checker().is_diverse(
                    sample_text, self._accepted_comparable_texts,
                )
                if not is_diverse:
                    logger.info(
                        f"Sample rejected as duplicate: similarity={max_sim:.3f} "
                        f">= {self.config.diversity_threshold}"
                    )
                    self.failed_samples.append({
                        "sample": raw_sample,
                        "reason": "low_diversity",
                        "similarity": max_sim,
                    })
                    continue

                try:
                    validated_data = self.data_format(**raw_sample)
                    metrics = QualityMetrics(
                        quality_score=quality_score,
                        token_count=len(json.dumps(raw_sample)) // 4,
                        generation_cost=last_cost / len(raw_samples) if len(raw_samples) > 0 else 0,
                        model_used=self.model_router.generator.model,
                        generation_time=generation_time / len(raw_samples) if len(raw_samples) > 0 else 0,
                        regeneration_count=0,
                        topic=topic,
                        subtopic=subtopic,
                    )
                    batch_samples.append(DatasetSample(data=validated_data, metrics=metrics))
                    # Register accepted sample for future dedup comparison
                    self._accepted_comparable_texts.append(sample_text)
                except Exception as e:
                    logger.error(f"Error creating dataset sample: {e}")
                    continue

            # Smart regeneration
            regeneration_attempt = 0
            consecutive_failures = 0

            while len(batch_samples) < actual_batch_size and regeneration_attempt < self.max_retries:
                samples_needed = actual_batch_size - len(batch_samples)
                regeneration_attempt += 1

                if show_progress:
                    print(f"   └─ Regenerating {samples_needed} failed samples (attempt {regeneration_attempt}/{self.max_retries})... ", end="", flush=True)

                try:
                    regen_start_time = datetime.now()
                    regen_raw_samples = await self._generate_batch(samples_needed, batch_spec)
                    regen_generation_time = (datetime.now() - regen_start_time).total_seconds()

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
                            self.failed_samples.append({
                                "sample": raw_sample,
                                "reason": "validation_failed",
                                "regeneration_attempt": regeneration_attempt
                            })

                    if show_progress:
                        print(f"   └─ Validated: {len(regen_valid_samples)}/{len(regen_raw_samples)} regenerated samples")

                    # Quality scoring
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

                    # Add passing regenerated samples
                    regen_passed = 0
                    for raw_sample, quality_score in zip(regen_valid_samples, regen_quality_scores):
                        if quality_score < self.config.quality_threshold:
                            self.failed_samples.append({
                                "sample": raw_sample,
                                "reason": "low_quality",
                                "score": quality_score,
                                "regeneration_attempt": regeneration_attempt
                            })
                            continue

                        # Diversity check on regenerated samples too
                        sample_text = self._get_comparable_text(raw_sample)
                        is_diverse, max_sim = self._get_diversity_checker().is_diverse(
                            sample_text, self._accepted_comparable_texts,
                        )
                        if not is_diverse:
                            self.failed_samples.append({
                                "sample": raw_sample,
                                "reason": "low_diversity",
                                "similarity": max_sim,
                                "regeneration_attempt": regeneration_attempt,
                            })
                            continue

                        try:
                            validated_data = self.data_format(**raw_sample)
                            metrics = QualityMetrics(
                                quality_score=quality_score,
                                token_count=len(json.dumps(raw_sample)) // 4,
                                generation_cost=regen_cost / len(regen_raw_samples) if len(regen_raw_samples) > 0 else 0,
                                model_used=self.model_router.generator.model,
                                generation_time=regen_generation_time / len(regen_raw_samples) if len(regen_raw_samples) > 0 else 0,
                                regeneration_count=regeneration_attempt,
                                topic=topic,
                                subtopic=subtopic,
                            )
                            batch_samples.append(DatasetSample(data=validated_data, metrics=metrics))
                            self._accepted_comparable_texts.append(sample_text)
                            regen_passed += 1
                        except Exception as e:
                            logger.error(f"Error creating regenerated dataset sample: {e}")
                            continue

                    if show_progress:
                        print(f"   └─ Accepted: {regen_passed} regenerated samples")

                    # Track consecutive failures
                    if regen_passed == 0:
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            logger.warning(f"Stopping regeneration after {consecutive_failures} consecutive failures")
                            break
                    else:
                        consecutive_failures = 0

                except Exception as e:
                    logger.error(f"Error during regeneration attempt {regeneration_attempt}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                    continue

            if len(batch_samples) < actual_batch_size:
                shortfall = actual_batch_size - len(batch_samples)
                logger.warning(f"Batch incomplete: {len(batch_samples)}/{actual_batch_size} samples (short by {shortfall})")
                if show_progress:
                    print(f"   ⚠️  Batch incomplete: {len(batch_samples)}/{actual_batch_size} samples")

            if show_progress:
                print(f"   └─ Accepted: {len(batch_samples)} samples")

            return batch_samples

        except Exception as e:
            logger.error(f"Error generating batch {batch_num}: {e}")
            return []

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

        # Get parallel batch count from config
        parallel_batches = getattr(self.config, 'parallel_batches', 1)

        # Process batches in parallel groups
        import asyncio
        batch_groups = [batches[i:i + parallel_batches] for i in range(0, len(batches), parallel_batches)]

        for batch_group in batch_groups:
            if samples_generated >= num_samples:
                break

            if not self.cost_tracker.can_continue():
                logger.warning("Cost limit reached, stopping generation")
                await self._emit(
                    "cost_limit_reached",
                    total_cost=self.cost_tracker.total_cost,
                    max_cost=self.config.max_cost,
                    samples_generated=samples_generated,
                    samples_target=num_samples,
                )
                break

            # Process batches in this group in parallel
            if len(batch_group) > 1 and show_progress:
                print(f"\n🔄 Processing {len(batch_group)} batches in parallel...")

            # Create tasks for parallel execution
            tasks = []
            for batch_spec in batch_group:
                if samples_generated >= num_samples:
                    break
                tasks.append(self._process_single_batch(
                    batch_spec,
                    generation_plan,
                    num_samples - samples_generated,
                    show_progress
                ))

            # Run batches in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    continue

                if result:  # result is list of DatasetSample
                    self.generated_samples.extend(result)
                    samples_generated += len(result)

                    batch_avg_quality = (
                        sum(s.metrics.quality_score for s in result) / len(result)
                        if result else 0.0
                    )
                    await self._emit(
                        "batch_completed",
                        samples_in_batch=len(result),
                        samples_passed=len(result),
                        batch_cost=sum(s.metrics.generation_cost for s in result),
                        average_quality=batch_avg_quality,
                        samples_generated=samples_generated,
                        samples_target=num_samples,
                        total_cost=self.cost_tracker.total_cost,
                    )

                    # Checkpointing
                    if checkpoint_dir and samples_generated % checkpoint_interval == 0:
                        self._save_checkpoint(checkpoint_dir, samples_generated, show_progress)
                        await self._emit(
                            "checkpoint_saved",
                            samples_generated=samples_generated,
                            checkpoint_dir=str(checkpoint_dir),
                        )

            if show_progress and len(batch_group) > 1:
                print(f"   ✓ Completed parallel group (total: {samples_generated}/{num_samples})")

        # ---- Top-up loop ----
        # The planned batches above may undershoot num_samples when samples
        # get filtered post-generation (low quality, low diversity, Pydantic
        # validation, etc.) and the within-batch regen budget runs out. Keep
        # generating extra batches — cycling through the planner's specs in
        # round-robin so topic coverage stays balanced — until we hit
        # num_samples or the cost cap. Bounded by max_topup_rounds as a
        # safety net against pathological all-reject loops.
        max_topup_rounds = max(num_samples, 50)
        topup_round = 0
        while samples_generated < num_samples and topup_round < max_topup_rounds:
            if not self.cost_tracker.can_continue():
                logger.warning("Cost limit reached during top-up; stopping")
                await self._emit(
                    "cost_limit_reached",
                    total_cost=self.cost_tracker.total_cost,
                    max_cost=self.config.max_cost,
                    samples_generated=samples_generated,
                    samples_target=num_samples,
                )
                break

            remaining = num_samples - samples_generated
            batch_spec = batches[topup_round % len(batches)]
            topup_round += 1

            if show_progress:
                print(
                    f"\n🔁 Top-up round {topup_round}: need {remaining} more sample(s) "
                    f"(cycling through plan, on batch '{batch_spec.get('topic','')}')"
                )

            result = await self._process_single_batch(
                batch_spec,
                generation_plan,
                remaining,
                show_progress,
            )
            if result:
                self.generated_samples.extend(result)
                samples_generated += len(result)
                await self._emit(
                    "batch_completed",
                    samples_in_batch=len(result),
                    samples_passed=len(result),
                    batch_cost=sum(s.metrics.generation_cost for s in result),
                    average_quality=(
                        sum(s.metrics.quality_score for s in result) / len(result)
                        if result else 0.0
                    ),
                    samples_generated=samples_generated,
                    samples_target=num_samples,
                    total_cost=self.cost_tracker.total_cost,
                    topup_round=topup_round,
                )

        if samples_generated < num_samples:
            logger.warning(
                f"Stopped short at {samples_generated}/{num_samples} after "
                f"{topup_round} top-up rounds (cost cap or safety bound)"
            )

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

        await self._emit(
            "generation_completed",
            samples_generated=samples_generated,
            samples_target=num_samples,
            failed=len(self.failed_samples),
            total_cost=self.cost_tracker.total_cost,
        )

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
