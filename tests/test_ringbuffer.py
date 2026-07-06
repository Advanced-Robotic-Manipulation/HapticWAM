"""SharedRingBuffer: wrap-around, drain cursors, window queries, attach."""

import numpy as np
import pytest

from phantom.recording.ringbuffer import SharedRingBuffer


@pytest.fixture
def ring():
    r = SharedRingBuffer("test_ring_phantom", 8,
                         {"a": ((2, 3), "float32"), "b": ((), "float64")},
                         create=True)
    yield r
    r.close()


def test_push_latest(ring):
    for i in range(5):
        ring.push(float(i), a=np.full((2, 3), i, dtype=np.float32), b=np.float64(i * 10))
    ts, data = ring.latest(3)
    assert list(ts) == [2.0, 3.0, 4.0]
    assert data["a"][-1][0, 0] == 4.0
    assert data["b"][0] == 20.0


def test_wraparound(ring):
    for i in range(20):
        ring.push(float(i), a=np.full((2, 3), i, dtype=np.float32), b=np.float64(i))
    ts, data = ring.latest(8)
    assert list(ts) == [float(i) for i in range(12, 20)]
    ts, _ = ring.latest(100)   # capped at capacity
    assert len(ts) == 8


def test_drain_cursor(ring):
    for i in range(4):
        ring.push(float(i), a=np.zeros((2, 3), np.float32), b=np.float64(i))
    nxt, ts, _ = ring.drain(0)
    assert nxt == 4 and len(ts) == 4
    nxt2, ts2, _ = ring.drain(nxt)
    assert nxt2 == 4 and len(ts2) == 0
    for i in range(4, 6):
        ring.push(float(i), a=np.zeros((2, 3), np.float32), b=np.float64(i))
    nxt3, ts3, _ = ring.drain(nxt2)
    assert list(ts3) == [4.0, 5.0]


def test_drain_after_overrun(ring):
    for i in range(30):
        ring.push(float(i), a=np.zeros((2, 3), np.float32), b=np.float64(i))
    nxt, ts, _ = ring.drain(0)   # reader far behind: resumes from oldest available
    assert nxt == 30 and list(ts) == [float(i) for i in range(22, 30)]


def test_window(ring):
    for i in range(8):
        ring.push(float(i), a=np.zeros((2, 3), np.float32), b=np.float64(i))
    ts, data = ring.window(2.5, 5.5)
    assert list(ts) == [3.0, 4.0, 5.0]


def test_attach(ring):
    ring.push(1.0, a=np.ones((2, 3), np.float32), b=np.float64(7))
    reader = SharedRingBuffer.attach(ring.spec_dict())
    ts, data = reader.latest(1)
    assert ts[0] == 1.0 and data["b"][0] == 7.0
    reader.close()
