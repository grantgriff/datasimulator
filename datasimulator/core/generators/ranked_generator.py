"""
Ranked-response generator for GRPO-style training data.

For each prompt, generates N responses spanning a configurable quality spread,
scores each via the verifier model, and assigns 1..N ranks by descending score.
"""

import json
import logging
from typing import List, Dict, Any, Type, Optional, Literal

from .base_generator import BaseGenerator
from ..data_models import RankedSample, TrainingDataFormat

logger = logging.getLogger(__name__)


QualitySpread = Literal["wide", "narrow"]


class RankedGenerator(BaseGenerator):
    """
    Generate multi-response ranked records for GRPO training.

    Each output sample is one prompt + N ranked responses, where each
    response carries the verifier's quality_score and ranks are assigned
    by sorting those scores descending.
    """

    # Quality-spread thresholds (gap = rank1.quality_score - rankN.quality_score)
    WIDE_MIN_GAP = 5.0
    NARROW_MAX_GAP = 2.5
    SCORE_TIE_EPSILON = 0.05

    def __init__(
        self,
        num_responses: int = 4,
        quality_spread: QualitySpread = "wide",
        **kwargs
    ):
        """
        Args:
            num_responses: How many responses to generate per prompt (>=2).
            quality_spread: "wide" enforces a >5.0 score gap between best/worst;
                "narrow" enforces a <2.5 gap. Samples that don't meet the spread
                target are filtered (the regeneration loop in the base class
                will then top up the batch).
            **kwargs: Passed to BaseGenerator.
        """
        super().__init__(**kwargs)
        if num_responses < 2:
            raise ValueError(f"num_responses must be >= 2, got {num_responses}")
        if quality_spread not in ("wide", "narrow"):
            raise ValueError(f"quality_spread must be 'wide' or 'narrow', got {quality_spread!r}")

        self.num_responses = num_responses
        self.quality_spread = quality_spread

    @property
    def data_format(self) -> Type[TrainingDataFormat]:
        return RankedSample

    @property
    def data_type_name(self) -> str:
        return "ranked"

    async def _generate_batch(
        self,
        batch_size: int,
        batch_spec: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of ranked-response records.

        1. Prompt the generator for `batch_size` records, each with `num_responses`
           candidate responses at deliberately varied quality levels.
        2. Score each response via the verifier.
        3. Sort responses by score (descending), assign ranks 1..N.
        4. Drop records whose quality spread doesn't match the configured target.
        """
        prompt = self._build_prompt(batch_size, batch_spec)

        try:
            response = await self.model_router.generate(
                prompt,
                temperature=0.9,
                max_tokens=64000,
            )
        except Exception as e:
            logger.error(f"Error generating ranked batch: {e}")
            return []

        raw_records = self._parse_batch_response(response)
        if not raw_records:
            return []

        finalized: List[Dict[str, Any]] = []
        for record in raw_records:
            try:
                ranked = await self._score_and_rank_record(record, batch_spec)
            except Exception as e:
                logger.debug(f"Failed to score/rank record: {e}")
                continue

            if ranked is None:
                continue

            if not self._meets_quality_spread(ranked):
                logger.debug(
                    f"Record dropped: quality spread doesn't match '{self.quality_spread}' target"
                )
                continue

            finalized.append(ranked)

        if len(finalized) < batch_size:
            logger.warning(
                f"Ranked batch: kept {len(finalized)}/{batch_size} records "
                f"(spread='{self.quality_spread}')"
            )

        return finalized

    def _build_prompt(self, batch_size: int, batch_spec: Optional[Dict[str, Any]]) -> str:
        """Build the generator prompt asking for N varied-quality responses per prompt."""
        n = self.num_responses
        spread_instruction = self._spread_instruction()

        if batch_spec:
            topic = batch_spec.get("topic", "General Content")
            subtopic = batch_spec.get("subtopic", "")
            guidance = batch_spec.get("guidance", "")
            focus_areas = batch_spec.get("focus_areas", [])
            relevant_files = batch_spec.get("relevant_files", [])

            source_context = (
                self._extract_relevant_source(relevant_files)
                if relevant_files
                else self.source_content
            )
            focus_list = "\n".join(f"  - {area}" for area in focus_areas) if focus_areas else "  (none specified)"

            topic_block = f"""=== TOPIC CONTEXT ===
MAJOR TOPIC: {topic}
SUBTOPIC: {subtopic}
FOCUS AREAS:
{focus_list}
GUIDANCE: {guidance}
"""
            topic_field_hint = f'  "topic": "{topic}",\n  "subtopic": "{subtopic}",\n'
        else:
            topic_block = ""
            topic_field_hint = '  "topic": "<derived from prompt>",\n  "subtopic": "<derived from prompt>",\n'
            source_context = self.source_content

        return f"""
You are generating training data for GRPO (group-relative policy optimization).

Each record contains ONE prompt and EXACTLY {n} candidate responses spanning a deliberate range of quality.

{topic_block}

=== SOURCE MATERIAL ===
{source_context[:4000] if source_context else "No source material provided — generate from domain knowledge."}

=== QUALITY SPREAD: {self.quality_spread.upper()} ===
{spread_instruction}

=== TASK ===
Generate EXACTLY {batch_size} records. Each record must follow this JSON shape:

{{
{topic_field_hint}  "prompt": "<question or instruction>",
  "responses": [
    "<response intended as rank 1 (best)>",
    "<response intended as rank 2>",
{self._intermediate_rank_hints(n)}    "<response intended as rank {n} (worst)>"
  ]
}}

The ORDER of responses in the array indicates your INTENDED quality (index 0 = best, index {n-1} = worst).
The verifier will score them independently and may reorder; do not include scores or rank labels in the text.

Do NOT include ranks or quality scores in the response text itself — write each response as a natural, standalone answer.

OUTPUT: JSON array of EXACTLY {batch_size} records. ONLY the JSON array, no other text.
"""

    def _spread_instruction(self) -> str:
        if self.quality_spread == "wide":
            return (
                "Rank 1 should be EXPERT-quality: comprehensive, precise, with correct reasoning, examples, and references.\n"
                "Rank 2 should be MOSTLY correct but missing some nuance or a minor detail.\n"
                "Middle ranks should be PARTIALLY correct with a clear, identifiable error.\n"
                f"Rank {self.num_responses} should be PLAUSIBLE-SOUNDING but substantively WRONG.\n"
                "The best and worst should be CLEARLY distinguishable to a domain expert."
            )
        return (
            "All responses should be of similar overall quality with only SUBTLE differences.\n"
            "Differences should come from minor omissions, slightly less precise wording, or a small factual imprecision.\n"
            "No response should be obviously wrong; the ranking will reflect fine-grained discrimination."
        )

    def _intermediate_rank_hints(self, n: int) -> str:
        if n <= 2:
            return ""
        lines = []
        for r in range(3, n):
            lines.append(f'    "<response intended as rank {r}>",\n')
        return "".join(lines)

    async def _score_and_rank_record(
        self,
        record: Dict[str, Any],
        batch_spec: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Score each candidate response, sort, and emit a RankedSample-shaped dict."""
        prompt_text = record.get("prompt")
        responses = record.get("responses") or []

        if not isinstance(prompt_text, str) or not prompt_text.strip():
            return None
        if not isinstance(responses, list) or len(responses) != self.num_responses:
            logger.debug(
                f"Record has {len(responses) if isinstance(responses, list) else 'invalid'} "
                f"responses, expected {self.num_responses}"
            )
            return None
        if any(not isinstance(r, str) or not r.strip() for r in responses):
            return None

        scores = await self._score_responses(prompt_text, responses)
        if len(scores) != len(responses):
            return None

        # Sort by score descending. Resolve ties by index (stable) to keep original order.
        order = sorted(range(len(responses)), key=lambda i: (-scores[i], i))
        sorted_scores = [scores[i] for i in order]

        # If two adjacent scores are within epsilon, treat as a tie and signal regeneration
        # by dropping this record (per spec).
        for i in range(len(sorted_scores) - 1):
            if abs(sorted_scores[i] - sorted_scores[i + 1]) < self.SCORE_TIE_EPSILON:
                logger.debug(
                    f"Score tie within epsilon ({sorted_scores[i]} vs {sorted_scores[i+1]}); dropping record"
                )
                return None

        ranked_responses = [
            {
                "rank": new_rank,
                "text": responses[old_idx],
                "quality_score": scores[old_idx],
            }
            for new_rank, old_idx in enumerate(order, start=1)
        ]

        topic = record.get("topic") or (batch_spec.get("topic") if batch_spec else None)
        subtopic = record.get("subtopic") or (batch_spec.get("subtopic") if batch_spec else None)

        return {
            "prompt": prompt_text,
            "ranked_responses": ranked_responses,
            "topic": topic,
            "subtopic": subtopic,
        }

    async def _score_responses(self, prompt_text: str, responses: List[str]) -> List[float]:
        """Score N candidate responses to a single prompt via the verifier model."""
        responses_block = "\n\n".join(
            f"=== RESPONSE {i+1} ===\n{r}" for i, r in enumerate(responses)
        )

        scoring_prompt = f"""
Score each candidate response to the same prompt from 1-10 on overall quality
(accuracy, completeness, clarity, relevance). Each response is INDEPENDENT — score
them on absolute quality, not relative to each other.

Source Context:
{self.source_content[:1000] if self.source_content else "(none provided)"}

PROMPT:
{prompt_text}

CANDIDATE RESPONSES:
{responses_block}

Output a JSON array of {len(responses)} numbers in 1-10 range, one per response, in order.
Example: [8.5, 6.2, 4.1, 2.5]

Output (JSON array only):
"""

        try:
            raw = await self.model_router.verify(
                scoring_prompt,
                temperature=0.3,
                max_tokens=200,
            )
        except Exception as e:
            logger.error(f"Error scoring ranked responses: {e}")
            return []

        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            scores = json.loads(cleaned)
        except Exception as e:
            logger.error(f"Could not parse verifier scores: {e}; raw={raw[:200]}")
            return []

        if not isinstance(scores, list) or len(scores) != len(responses):
            logger.warning(
                f"Verifier returned {len(scores) if isinstance(scores, list) else 'invalid'} scores "
                f"for {len(responses)} responses"
            )
            return []

        return [max(1.0, min(10.0, float(s))) for s in scores]

    def _meets_quality_spread(self, record: Dict[str, Any]) -> bool:
        """Check that rank1 - rankN gap matches the configured quality_spread."""
        responses = record.get("ranked_responses") or []
        if len(responses) < 2:
            return False
        gap = responses[0]["quality_score"] - responses[-1]["quality_score"]
        if self.quality_spread == "wide":
            return gap > self.WIDE_MIN_GAP
        return gap < self.NARROW_MAX_GAP

    def _parse_batch_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract a JSON array of raw records from the model response."""
        cleaned = response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse ranked batch JSON: {e}; raw={response[:200]}")
            return []

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "records" in data:
            return data["records"]
        logger.error(f"Unexpected ranked batch shape: {type(data)}")
        return []

    def _extract_relevant_source(self, relevant_files: List[str]) -> str:
        """Pull only the source content for the named files (falls back to all)."""
        if not relevant_files or not self.source_content_by_file:
            return self.source_content

        chunks = []
        for file_path in relevant_files:
            for stored_key, content in self.source_content_by_file.items():
                if stored_key.endswith(file_path) or file_path.endswith(stored_key):
                    chunks.append(f"\n\n=== {file_path} ===\n\n{content}")
                    break

        return "\n\n".join(chunks) if chunks else self.source_content

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a ranked sample against the Pydantic schema."""
        try:
            RankedSample(**sample)
            return True
        except Exception as e:
            logger.debug(f"RankedSample validation failed: {e}")
            return False

    async def _score_quality_batch(self, samples: List[Dict[str, Any]]) -> List[float]:
        """
        For ranked records, each response was already scored individually by
        the verifier inside `_generate_batch`. The sample-level quality is
        just the rank-1 score (the best response we'd actually ship as a
        gold answer / chosen response).

        Overrides the base class's batch scoring, which serializes the whole
        record and sends it to the verifier — but the nested ranked_responses
        array confuses the verifier into scoring each inner response, so it
        returns the wrong number of scores and every sample falls back to
        the default 5.0 (which trips the quality threshold).
        """
        scores: List[float] = []
        for s in samples:
            try:
                top = s["ranked_responses"][0]["quality_score"]
                scores.append(max(1.0, min(10.0, float(top))))
            except (KeyError, IndexError, TypeError, ValueError):
                # Malformed record — give it a failing score so it gets rejected.
                scores.append(1.0)
        return scores
