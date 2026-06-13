"""Tests for the live sample-preview readout (event + compact print)."""
import asyncio
from unittest.mock import MagicMock

import pytest

from datasimulator.core.data_models import GenerationConfig
from datasimulator.core.generators.base_generator import BaseGenerator
from datasimulator.core.generators.ranked_generator import RankedGenerator


def _gen(**kw):
    return RankedGenerator(
        num_responses=4,
        quality_spread="wide",
        model_router=MagicMock(),
        cost_tracker=MagicMock(),
        config=GenerationConfig(num_samples=10, batch_size=2, quality_threshold=5.0),
        **kw,
    )


# ----------------------------------------------------- _preview_fields shapes
def test_preview_fields_messages():
    f = BaseGenerator._preview_fields(
        {"messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]}
    )
    assert f == {"prompt": "Q?", "response": "A."}


def test_preview_fields_ranked():
    f = BaseGenerator._preview_fields(
        {"prompt": "P", "ranked_responses": [{"text": "best"}, {"text": "worse"}]}
    )
    assert f == {"prompt": "P", "response": "best"}


def test_preview_fields_dpo_and_verifiable():
    assert BaseGenerator._preview_fields({"prompt": "P", "chosen": "C", "rejected": "R"}) == {
        "prompt": "P", "response": "C"}
    assert BaseGenerator._preview_fields({"prompt": "P", "ground_truth": "42"}) == {
        "prompt": "P", "response": "42"}


def test_preview_fields_handles_garbage():
    assert BaseGenerator._preview_fields(None) == {"prompt": "", "response": ""}
    assert BaseGenerator._preview_fields(123) == {"prompt": "", "response": ""}


# ----------------------------------------------------- _emit_sample_preview
def test_emit_sample_preview_event_carries_full_and_truncated(capsys):
    events = []
    gen = _gen(progress_callback=lambda e: events.append(e))

    long_answer = "x" * 500
    batch = [{"prompt": "How do I run it?", "ranked_responses": [{"text": long_answer}]}]
    asyncio.run(gen._emit_sample_preview(batch, show_progress=True))

    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "sample_preview"
    assert ev["data_type"] == "ranked"
    assert ev["prompt"] == "How do I run it?"
    assert ev["response"] == long_answer                      # full text preserved
    assert ev["response_preview"].startswith("x" * 300)       # truncated for display
    assert "more chars" in ev["response_preview"]

    out = capsys.readouterr().out
    assert "sample preview (ranked)" in out
    assert "How do I run it?" in out


def test_emit_sample_preview_silent_print_when_disabled(capsys):
    events = []
    gen = _gen(progress_callback=lambda e: events.append(e))
    asyncio.run(gen._emit_sample_preview(
        [{"prompt": "P", "ranked_responses": [{"text": "short"}]}], show_progress=False))
    # Event still emitted (machine consumers), but nothing printed to stdout.
    assert len(events) == 1 and events[0]["response_preview"] == "short"
    assert capsys.readouterr().out == ""


def test_emit_sample_preview_noop_on_empty_batch():
    events = []
    gen = _gen(progress_callback=lambda e: events.append(e))
    asyncio.run(gen._emit_sample_preview([], show_progress=True))
    assert events == []
