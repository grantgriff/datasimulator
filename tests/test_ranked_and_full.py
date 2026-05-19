"""Tests for ranked / full data types (21.2, 21.3)."""

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from datasimulator.core.data_models import (
    RankedResponse,
    RankedSample,
    FullSample,
)
from datasimulator.core.generators.ranked_generator import RankedGenerator
from datasimulator.core.generators.full_generator import FullGenerator


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


def test_ranked_sample_accepts_well_formed_record():
    sample = RankedSample(
        prompt="Q?",
        ranked_responses=[
            RankedResponse(rank=1, text="best", quality_score=9.0),
            RankedResponse(rank=2, text="mid", quality_score=6.0),
            RankedResponse(rank=3, text="worst", quality_score=3.0),
        ],
        topic="t",
        subtopic="s",
    )
    assert sample.ranked_responses[0].rank == 1


def test_ranked_sample_rejects_non_sorted_ranks():
    with pytest.raises(ValueError, match="rank 1 first"):
        RankedSample(
            prompt="Q?",
            ranked_responses=[
                RankedResponse(rank=2, text="a", quality_score=8.0),
                RankedResponse(rank=1, text="b", quality_score=9.0),
            ],
        )


def test_ranked_sample_rejects_missing_rank():
    with pytest.raises(ValueError, match="ranks 1..3"):
        RankedSample(
            prompt="Q?",
            ranked_responses=[
                RankedResponse(rank=1, text="a", quality_score=9.0),
                RankedResponse(rank=2, text="b", quality_score=7.0),
                RankedResponse(rank=4, text="c", quality_score=3.0),  # gap
            ],
        )


def test_ranked_sample_rejects_score_inversion():
    with pytest.raises(ValueError, match="non-increasing"):
        RankedSample(
            prompt="Q?",
            ranked_responses=[
                RankedResponse(rank=1, text="a", quality_score=5.0),
                RankedResponse(rank=2, text="b", quality_score=9.0),  # > rank 1
            ],
        )


def test_full_sample_carries_all_views():
    sample = FullSample(
        prompt="Q?",
        gold_answer="best",
        chosen="best",
        rejected="worst",
        ranked_responses=[
            RankedResponse(rank=1, text="best", quality_score=9.0),
            RankedResponse(rank=2, text="ok", quality_score=6.0),
            RankedResponse(rank=3, text="worst", quality_score=2.0),
        ],
    )
    assert sample.gold_answer == sample.chosen == sample.ranked_responses[0].text
    assert sample.rejected == sample.ranked_responses[-1].text


# ---------------------------------------------------------------------------
# RankedGenerator behavior (mocked LLMs)
# ---------------------------------------------------------------------------


def _make_ranked_gen(quality_spread="wide", num_responses=4) -> RankedGenerator:
    from datasimulator.core.data_models import GenerationConfig

    return RankedGenerator(
        num_responses=num_responses,
        quality_spread=quality_spread,
        model_router=MagicMock(),
        cost_tracker=MagicMock(),
        config=GenerationConfig(num_samples=10, batch_size=2, quality_threshold=5.0),
        source_content="some source",
    )


def test_score_and_rank_sorts_descending():
    gen = _make_ranked_gen()
    # Make verifier return scrambled scores
    gen._score_responses = AsyncMock(return_value=[5.0, 9.0, 2.0, 7.0])

    record = {
        "prompt": "Q?",
        "responses": ["a", "b", "c", "d"],
        "topic": "t",
        "subtopic": "s",
    }
    out = asyncio.run(gen._score_and_rank_record(record, batch_spec=None))
    assert out is not None
    ranked = out["ranked_responses"]
    assert [r["text"] for r in ranked] == ["b", "d", "a", "c"]
    assert [r["rank"] for r in ranked] == [1, 2, 3, 4]
    assert [r["quality_score"] for r in ranked] == [9.0, 7.0, 5.0, 2.0]


def test_score_and_rank_drops_on_tie():
    gen = _make_ranked_gen()
    gen._score_responses = AsyncMock(return_value=[8.0, 8.02, 4.0, 2.0])  # tie within epsilon
    out = asyncio.run(
        gen._score_and_rank_record(
            {"prompt": "Q?", "responses": ["a", "b", "c", "d"]},
            batch_spec=None,
        )
    )
    assert out is None


def test_quality_spread_wide_filter():
    gen = _make_ranked_gen(quality_spread="wide")
    wide = {
        "ranked_responses": [
            {"rank": 1, "text": "a", "quality_score": 9.0},
            {"rank": 2, "text": "b", "quality_score": 3.0},
        ]
    }
    narrow = {
        "ranked_responses": [
            {"rank": 1, "text": "a", "quality_score": 8.0},
            {"rank": 2, "text": "b", "quality_score": 6.0},
        ]
    }
    assert gen._meets_quality_spread(wide) is True
    assert gen._meets_quality_spread(narrow) is False


def test_quality_spread_narrow_filter():
    gen = _make_ranked_gen(quality_spread="narrow")
    wide = {
        "ranked_responses": [
            {"rank": 1, "text": "a", "quality_score": 9.0},
            {"rank": 2, "text": "b", "quality_score": 3.0},
        ]
    }
    narrow = {
        "ranked_responses": [
            {"rank": 1, "text": "a", "quality_score": 8.0},
            {"rank": 2, "text": "b", "quality_score": 7.0},
        ]
    }
    assert gen._meets_quality_spread(wide) is False
    assert gen._meets_quality_spread(narrow) is True


def test_ranked_generator_rejects_bad_num_responses():
    with pytest.raises(ValueError):
        _make_ranked_gen(num_responses=1)


def test_full_generator_derives_sft_and_dpo_views():
    from datasimulator.core.data_models import GenerationConfig

    gen = FullGenerator(
        num_responses=3,
        quality_spread="wide",
        model_router=MagicMock(),
        cost_tracker=MagicMock(),
        config=GenerationConfig(num_samples=1, batch_size=1, quality_threshold=5.0),
        source_content="",
    )
    gen._score_responses = AsyncMock(return_value=[9.0, 6.0, 2.0])

    record = {"prompt": "Q?", "responses": ["best", "ok", "bad"]}
    out = asyncio.run(gen._score_and_rank_record(record, batch_spec=None))

    assert out is not None
    assert out["gold_answer"] == "best"
    assert out["chosen"] == "best"
    assert out["rejected"] == "bad"
    # Pydantic should accept the produced shape
    FullSample(**out)
