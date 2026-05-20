"""Tests for the per-sample metadata sidecar file."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datasimulator.core.data_models import (
    DatasetSample,
    QualityMetrics,
    SFTMessages,
    Message,
)
from datasimulator.sdk import GeneratedDataset


def _make_sample(quality=8.5, topic="Revenue Recognition", subtopic="ASC 606"):
    return DatasetSample(
        data=SFTMessages(messages=[
            Message(role="user", content="Q?"),
            Message(role="assistant", content="A."),
        ]),
        metrics=QualityMetrics(
            quality_score=quality,
            token_count=100,
            generation_cost=0.001,
            model_used="gpt-5.4-mini",
            generation_time=0.5,
            regeneration_count=0,
            topic=topic,
            subtopic=subtopic,
        ),
    )


def test_jsonl_save_writes_sidecar(tmp_path):
    samples = [_make_sample(quality=9.0, topic="Inventory", subtopic="LIFO"),
               _make_sample(quality=7.5, topic="PP&E", subtopic="Depreciation")]
    ds = GeneratedDataset(
        samples=samples,
        data_type="sft",
        generation_config={},
        cost_tracker=MagicMock(total_cost=0.002, costs_by_operation={}),
    )

    out = tmp_path / "dataset.jsonl"
    ds.save(str(out), format="jsonl")

    # Training file
    assert out.exists()
    training_records = [json.loads(line) for line in out.open()]
    assert len(training_records) == 2
    # No metrics polluting the training file
    assert "metrics" not in training_records[0]
    assert "quality_score" not in training_records[0]

    # Sidecar
    sidecar = out.with_suffix(".metadata.jsonl")
    assert sidecar.exists(), "Expected sidecar metadata file alongside .jsonl"
    meta_records = [json.loads(line) for line in sidecar.open()]
    assert len(meta_records) == 2

    # Same order as training file
    assert meta_records[0]["idx"] == 0
    assert meta_records[1]["idx"] == 1

    # All the fields we care about for QA
    expected_keys = {"idx", "quality_score", "topic", "subtopic",
                     "token_count", "generation_cost", "model_used",
                     "regeneration_count", "generation_time", "timestamp"}
    assert expected_keys.issubset(meta_records[0].keys())

    # Topic/subtopic survived round-trip
    assert meta_records[0]["topic"] == "Inventory"
    assert meta_records[0]["subtopic"] == "LIFO"
    assert meta_records[1]["topic"] == "PP&E"

    # Quality scores survived
    assert meta_records[0]["quality_score"] == 9.0
    assert meta_records[1]["quality_score"] == 7.5


def test_sidecar_handles_missing_topic_gracefully(tmp_path):
    # Standard (non-plan) generation path leaves topic/subtopic as None
    samples = [_make_sample(topic=None, subtopic=None)]
    ds = GeneratedDataset(
        samples=samples,
        data_type="sft",
        generation_config={},
        cost_tracker=MagicMock(total_cost=0.001, costs_by_operation={}),
    )
    out = tmp_path / "dataset.jsonl"
    ds.save(str(out), format="jsonl")

    sidecar = out.with_suffix(".metadata.jsonl")
    meta = json.loads(sidecar.read_text().strip())
    assert meta["topic"] is None
    assert meta["subtopic"] is None
