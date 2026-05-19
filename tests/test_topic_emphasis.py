"""Tests for the topic_emphasis SDK extension (21.1)."""

import os
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from datasimulator.planning.gemini_planner import GeminiPlanner
from datasimulator.sdk import DataSimulator


# ---------------------------------------------------------------------------
# Validation / plumbing (no network)
# ---------------------------------------------------------------------------


def _make_sdk(enable_planning: bool, monkeypatch) -> DataSimulator:
    """Build a DataSimulator without hitting any network or loading sources."""
    monkeypatch.setattr(
        "datasimulator.sdk.ModelRouter",
        lambda *a, **kw: MagicMock(),
    )
    if enable_planning:
        # Stub out GeminiPlanner so we don't need a real API key
        with patch("datasimulator.planning.GeminiPlanner") as mock_planner_cls:
            mock_planner_cls.return_value = MagicMock()
            return DataSimulator(
                source=None,
                data_type="sft",
                enable_planning=True,
            )
    return DataSimulator(source=None, data_type="sft", enable_planning=False)


def test_emphasis_rejected_when_weights_exceed_one(monkeypatch):
    sdk = _make_sdk(enable_planning=True, monkeypatch=monkeypatch)
    with pytest.raises(ValueError, match="sum to <= 1.0"):
        sdk._validate_topic_emphasis({"A": 0.7, "B": 0.5})


def test_emphasis_rejected_for_bad_weight(monkeypatch):
    sdk = _make_sdk(enable_planning=True, monkeypatch=monkeypatch)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        sdk._validate_topic_emphasis({"A": 0.0})
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        sdk._validate_topic_emphasis({"A": 1.5})


def test_emphasis_rejected_for_bad_key(monkeypatch):
    sdk = _make_sdk(enable_planning=True, monkeypatch=monkeypatch)
    with pytest.raises(ValueError, match="non-empty strings"):
        sdk._validate_topic_emphasis({"": 0.5})


def test_emphasis_passed_through_when_valid(monkeypatch):
    sdk = _make_sdk(enable_planning=True, monkeypatch=monkeypatch)
    out = sdk._validate_topic_emphasis({"A": 0.4, "B": 0.3})
    assert out == {"A": 0.4, "B": 0.3}


def test_emphasis_dropped_with_warning_when_planning_disabled(monkeypatch, caplog):
    sdk = _make_sdk(enable_planning=False, monkeypatch=monkeypatch)
    with caplog.at_level("WARNING"):
        out = sdk._validate_topic_emphasis({"A": 0.5})
    assert out is None
    assert any("enable_planning=False" in r.message for r in caplog.records)


def test_emphasis_none_returns_none(monkeypatch):
    sdk = _make_sdk(enable_planning=False, monkeypatch=monkeypatch)
    assert sdk._validate_topic_emphasis(None) is None


def test_planner_emphasis_section_renders_topics():
    """`_build_emphasis_section` should produce a prompt block that names each topic."""
    planner = GeminiPlanner.__new__(GeminiPlanner)  # bypass __init__ (no API key needed)
    section = planner._build_emphasis_section(
        topic_emphasis={"ASC 606 revenue recognition": 0.4, "deferred tax assets": 0.3},
        num_batches=10,
    )
    assert "ASC 606 revenue recognition" in section
    assert "deferred tax assets" in section
    assert "40%" in section
    assert "30%" in section


def test_planner_emphasis_section_empty_when_no_emphasis():
    planner = GeminiPlanner.__new__(GeminiPlanner)
    assert planner._build_emphasis_section(None, num_batches=10) == ""


# ---------------------------------------------------------------------------
# Behavioral test against Gemini (skipped without API key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="Requires GOOGLE_API_KEY to call Gemini",
)
def test_emphasis_biases_batch_allocation():
    """With topic_emphasis={'X': 0.5}, at least 40% of batches should be tagged with topic 'X'."""
    planner = GeminiPlanner()
    source_content = (
        "This document covers two areas of accounting:\n\n"
        "Topic X: revenue recognition under ASC 606, including the 5-step model, "
        "performance obligations, transaction price allocation, and disclosure requirements.\n\n"
        "Topic Y: lease accounting under ASC 842, including operating vs finance lease "
        "classification, right-of-use assets, lease liability measurement, and disclosure."
    )

    plan = asyncio.run(
        planner.create_generation_plan(
            source_content=source_content,
            total_samples=200,
            data_type="sft",
            source_files=["accounting.pdf"],
            batch_size=20,
            topic_emphasis={"Topic X": 0.5},
        )
    )

    batches = plan["batches"]
    matched = sum(1 for b in batches if "topic x" in b["topic"].lower())
    fraction = matched / len(batches)
    assert fraction >= 0.4, f"Topic X only got {fraction:.0%} of batches (expected >= 40%)"
