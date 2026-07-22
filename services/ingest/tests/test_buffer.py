"""Tests for the ring buffer."""

from __future__ import annotations

import pytest

from ingest.buffer import RingBuffer


def test_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        RingBuffer(capacity=0)


def test_snapshot_empty_buffer():
    buf = RingBuffer(capacity=3)
    assert buf.snapshot() == []
    assert len(buf) == 0


def test_append_and_snapshot_preserves_order():
    buf = RingBuffer(capacity=3)
    buf.append({"sequence_number": 1})
    buf.append({"sequence_number": 2})
    assert [f["sequence_number"] for f in buf.snapshot()] == [1, 2]


def test_capacity_evicts_oldest():
    buf = RingBuffer(capacity=2)
    buf.append({"sequence_number": 1})
    buf.append({"sequence_number": 2})
    buf.append({"sequence_number": 3})
    assert [f["sequence_number"] for f in buf.snapshot()] == [2, 3]
    assert len(buf) == 2


def test_snapshot_is_a_copy():
    buf = RingBuffer(capacity=3)
    buf.append({"sequence_number": 1})
    snap = buf.snapshot()
    buf.append({"sequence_number": 2})
    assert [f["sequence_number"] for f in snap] == [1]
