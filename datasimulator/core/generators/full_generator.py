"""
Unified-format generator: SFT + DPO + GRPO views on a single record.

Builds on RankedGenerator. After producing the ranked group, derives:
  - `gold_answer` = rank-1 text (SFT view)
  - `chosen`      = rank-1 text (DPO view)
  - `rejected`    = rank-N text (DPO view)
  - `ranked_responses` is carried through unchanged (GRPO view)
"""

import logging
from typing import Type, Dict, Any

from .ranked_generator import RankedGenerator
from ..data_models import FullSample, TrainingDataFormat

logger = logging.getLogger(__name__)


class FullGenerator(RankedGenerator):
    """Generate FullSample records (SFT + DPO + GRPO on one record)."""

    @property
    def data_format(self) -> Type[TrainingDataFormat]:
        return FullSample

    @property
    def data_type_name(self) -> str:
        return "full"

    async def _score_and_rank_record(
        self,
        record: Dict[str, Any],
        batch_spec
    ):
        """Run the ranked pipeline, then layer on SFT/DPO views."""
        ranked = await super()._score_and_rank_record(record, batch_spec)
        if ranked is None:
            return None

        responses = ranked["ranked_responses"]
        best = responses[0]["text"]
        worst = responses[-1]["text"]

        ranked["gold_answer"] = best
        ranked["chosen"] = best
        ranked["rejected"] = worst
        return ranked

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        try:
            FullSample(**sample)
            return True
        except Exception as e:
            logger.debug(f"FullSample validation failed: {e}")
            return False
