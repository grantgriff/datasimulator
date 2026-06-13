"""Regression tests for robust JSON extraction from model responses.

The 2026-06-13 'posty-run0' incident: an accounting/CLI-domain run produced
valid JSON records whose response *text* contained markdown code fences
(```bash ... ```). The old split("```") cleaning sliced the JSON mid-string,
so json.loads failed at char 0 and EVERY batch was dropped (0/5 kept), looping
the regeneration path forever while spending OpenRouter budget. These tests
lock in that content containing backticks/code fences no longer corrupts
otherwise-valid JSON.
"""
import json

from datasimulator.core.generators.base_generator import extract_json_block


def test_plain_json_array():
    assert extract_json_block('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]


def test_plain_json_object():
    assert extract_json_block('{"batches": [1, 2, 3]}') == {"batches": [1, 2, 3]}


def test_fence_wrapped_json():
    assert extract_json_block('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json_block('```\n[{"a": 1}]\n```') == [{"a": 1}]


def test_prose_around_json():
    raw = 'Sure! Here is the JSON:\n[{"a": 1}]\nHope this helps!'
    assert extract_json_block(raw) == [{"a": 1}]


def test_code_fence_INSIDE_string_value_does_not_corrupt():
    """THE incident: a response string contains a ```bash code block. The old
    splitter sliced it apart; the bracket-matcher must treat it as opaque."""
    record = {
        "topic": "General Content",
        "subtopic": "Batch 1 Coverage",
        "prompt": "What command do I run after putting files in `accounting_docs/`?",
        "responses": [
            "Run the generator like this:\n```bash\npython generate.py --dir accounting_docs/\n```\nThat's it.",
            "Use ```python\nimport datasim\n``` then call it.",
            "I think you just double-click the file.",
            "There is no script; accounting data appears automatically.",
        ],
    }
    raw = json.dumps([record])
    out = extract_json_block(raw)
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["responses"][0].endswith("That's it.")
    assert "```bash" in out[0]["responses"][0]


def test_inline_backticks_in_values():
    raw = '[{"prompt": "Use the `ls -la` command", "answer": "Run `cat file.txt`"}]'
    out = extract_json_block(raw)
    assert out == [{"prompt": "Use the `ls -la` command", "answer": "Run `cat file.txt`"}]


def test_brackets_inside_string_values():
    """Brackets inside strings must not throw off bracket-matching."""
    raw = '[{"text": "an array looks like [1, 2, 3] or {a: b}"}]'
    out = extract_json_block(raw)
    assert out == [{"text": "an array looks like [1, 2, 3] or {a: b}"}]


def test_object_with_inner_array_returns_whole_object():
    raw = 'noise {"file_review": "ok", "batches": [{"n": 1}]} trailing'
    assert extract_json_block(raw) == {"file_review": "ok", "batches": [{"n": 1}]}


def test_truncated_json_returns_none():
    """A response cut off mid-structure has no complete value → None (the
    caller's regeneration loop handles it), not a corrupted partial."""
    assert extract_json_block('[{"a": 1}, {"a": 2') is None


def test_empty_and_nonjson_inputs():
    assert extract_json_block("") is None
    assert extract_json_block(None) is None
    assert extract_json_block("just some prose, no json here") is None


def test_ranked_parser_survives_code_fences_in_content():
    """End-to-end through RankedGenerator._parse_batch_response — the exact
    method that was failing in the incident."""
    from unittest.mock import MagicMock
    from datasimulator.core.data_models import GenerationConfig
    from datasimulator.core.generators.ranked_generator import RankedGenerator

    gen = RankedGenerator(
        num_responses=4,
        quality_spread="wide",
        model_router=MagicMock(),
        cost_tracker=MagicMock(),
        config=GenerationConfig(num_samples=10, batch_size=2, quality_threshold=5.0),
    )
    record = {
        "topic": "General Content",
        "subtopic": "Batch 1 Coverage",
        "prompt": "How do I run the pipeline?",
        "responses": [
            "```bash\npython run.py\n```",
            "Run `python run.py`.",
            "Click the icon.",
            "It runs itself.",
        ],
    }
    raw = json.dumps([record])
    records = gen._parse_batch_response(raw)
    assert len(records) == 1
    assert records[0]["prompt"] == "How do I run the pipeline?"
