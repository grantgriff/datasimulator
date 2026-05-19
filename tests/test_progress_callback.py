"""Tests for the progress_callback hook on BaseGenerator."""

import asyncio

import pytest

from datasimulator.core.generators.base_generator import BaseGenerator


class _StubGenerator(BaseGenerator):
    """Minimal concrete subclass — we only exercise _emit."""

    @property
    def data_format(self):
        return dict

    @property
    def data_type_name(self):
        return "stub"

    async def _generate_batch(self, batch_size, batch_spec=None):
        return []

    def _validate_sample(self, sample):
        return True


def _make_gen(callback=None):
    return _StubGenerator(
        model_router=None,
        cost_tracker=None,
        config=None,
        progress_callback=callback,
    )


def test_emit_no_callback_is_a_noop():
    gen = _make_gen(callback=None)
    asyncio.run(gen._emit("anything", foo=1))  # must not raise


def test_emit_calls_sync_callback_with_event_dict():
    received = []
    gen = _make_gen(callback=lambda payload: received.append(payload))

    asyncio.run(gen._emit("batch_completed", samples_in_batch=5, total_cost=0.12))

    assert received == [{
        "event": "batch_completed",
        "samples_in_batch": 5,
        "total_cost": 0.12,
    }]


def test_emit_awaits_async_callback():
    received = []

    async def cb(payload):
        received.append(payload)

    gen = _make_gen(callback=cb)
    asyncio.run(gen._emit("generation_started", num_samples=10))

    assert received == [{"event": "generation_started", "num_samples": 10}]


def test_emit_swallows_callback_exceptions():
    def angry(payload):
        raise RuntimeError("callback bug")

    gen = _make_gen(callback=angry)
    asyncio.run(gen._emit("checkpoint_saved", samples_generated=20))  # must not raise


def test_emit_swallows_async_callback_exceptions():
    async def angry(payload):
        raise RuntimeError("async callback bug")

    gen = _make_gen(callback=angry)
    asyncio.run(gen._emit("cost_limit_reached", total_cost=99))  # must not raise


def test_sdk_threads_callback_to_generator():
    """DataSimulator must hand the callback off to its underlying generator."""
    from unittest.mock import patch, MagicMock
    from datasimulator import DataSimulator

    cb = lambda _: None

    with patch("datasimulator.sdk.ModelRouter", lambda *a, **kw: MagicMock()):
        sdk = DataSimulator(
            source="ASC 606 is a revenue standard.\nIt has 5 steps.",
            data_type="sft",
            progress_callback=cb,
        )

    assert sdk.progress_callback is cb
