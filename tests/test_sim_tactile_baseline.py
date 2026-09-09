"""Checkpoint-specific pad baseline affects model input, never raw safety data."""

from types import SimpleNamespace

import numpy as np
import pytest
from phantom_test_utils import make_small_hw
from test_sim_policy_adapter import Policy, observe

from phantom.sim.policy_adapter import SimulationPolicyAdapter


def make_adapter(rows=8, *, mode="teacher"):
    policy = Policy()
    policy.wrench_baseline_rows = rows
    return SimulationPolicyAdapter(make_small_hw(), policy, mode=mode)


def tactile(ad, values, t):
    h, w = ad.hw.recording.field_ds.hw
    kh, kw = ad.hw.recording.keyframe_ds.hw
    ih, iw, _ = ad.hw.tactile.infer_img.hwc
    return {
        sensor.name: {
            "t": t,
            "fields_ds": np.full((h, w, 8), 0.01, np.float32),
            "keyframe": np.full((kh, kw, 8), 0.02, np.float32),
            "infer_img": np.full((ih, iw), 90, np.uint8),
            "wrench": np.asarray(values[i], np.float32),
            "area": 0.1 + i,
        }
        for i, sensor in enumerate(ad.hw.tactile.sensors)
    }


def feed(ad, values, t):
    observe(ad, t, tactile=tactile(ad, values, t), ft=np.arange(6) + 3)


def test_checkpoint_baseline_is_latest_eight_per_pad_and_frozen_under_load():
    ad = make_adapter()
    history = np.arange(12 * 2 * 6, dtype=np.float32).reshape(12, 2, 6)
    history[6, 0] += 1000  # Median rejects a single offset outlier.
    for i, row in enumerate(history):
        feed(ad, row, i * 0.01)
    base = np.median(history[-8:], axis=0)
    first = ad.snapshot()
    np.testing.assert_array_equal(first.contact_state[:, :6], history[-1] - base)
    for i, sensor in enumerate(ad.hw.tactile.sensors):
        np.testing.assert_array_equal(ad.wrench_base[sensor.name], base[i])
        np.testing.assert_array_equal(
            ad.rings[f"tactile_{sensor.name}"].latest(1)[1]["wrench"][0], history[-1, i]
        )
    loaded = history[-1] + np.array([0, 0, -7, 0, 0, 0], np.float32)
    feed(ad, loaded, 0.12)
    later = ad.snapshot()
    np.testing.assert_array_equal(later.contact_state[:, :6], loaded - base)
    np.testing.assert_array_equal(later.contact_state[:, 6:], first.contact_state[:, 6:])
    np.testing.assert_array_equal(later.fields, first.fields)
    np.testing.assert_array_equal(later.gel, first.gel)
    np.testing.assert_array_equal(later.wrist_window[-1], np.arange(6) + 3)


def test_early_first_snapshot_warns_and_freezes_available_rows(caplog):
    ad = make_adapter()
    values = np.arange(3 * 2 * 6, dtype=np.float32).reshape(3, 2, 6)
    for i, value in enumerate(values):
        feed(ad, value, i * 0.01)
    first = ad.snapshot()
    base = np.median(values, axis=0)
    np.testing.assert_array_equal(first.contact_state[:, :6], values[-1] - base)
    assert "baseline from 3 rows (< 8)" in caplog.text
    for i in range(3, 9):
        feed(ad, values[-1] + 100, i * 0.01)
    np.testing.assert_array_equal(ad.snapshot().contact_state[:, :6], values[-1] + 100 - base)


def test_episode_reset_captures_new_offsets():
    ad = make_adapter()
    feed(ad, np.full((2, 6), 3), 0)
    np.testing.assert_array_equal(ad.snapshot().contact_state[:, :6], np.zeros((2, 6)))
    ad.reset()
    assert ad.wrench_base == {}
    assert ad.policy.resets == 1
    feed(ad, np.full((2, 6), 17), 0)
    np.testing.assert_array_equal(ad.snapshot().contact_state[:, :6], np.zeros((2, 6)))
    assert all(np.all(value == 17) for value in ad.wrench_base.values())


def test_fta_baseline_zero_preserves_raw_contact_state():
    ad = make_adapter(0)
    raw = np.arange(12, dtype=np.float32).reshape(2, 6) + 0.25
    feed(ad, raw, 0)
    np.testing.assert_array_equal(ad.snapshot().contact_state[:, :6], raw)
    assert ad.wrench_base == {}


def test_delayed_first_snapshot_captures_only_causally_available_rows():
    ad = make_adapter()
    history = np.arange(8 * 2 * 6, dtype=np.float32).reshape(8, 2, 6)
    for i, row in enumerate(history):
        feed(ad, row, i * 0.01)
    delayed = ad.snapshot(observation_delay_s=0.03)
    base = np.median(history[:5], axis=0)
    np.testing.assert_array_equal(delayed.contact_state[:, :6], history[4] - base)
    for i, sensor in enumerate(ad.hw.tactile.sensors):
        np.testing.assert_array_equal(ad.wrench_base[sensor.name], base[i])


@pytest.mark.parametrize("count", [3, 12])
def test_baseline_capture_and_reset_match_native_snapshot_builder(count):
    pytest.importorskip("torch")
    from phantom.deploy.planner import SnapshotBuilder

    ad = make_adapter()
    native = SnapshotBuilder(
        ad.hw, SimpleNamespace(rings={}), "teacher", wrench_baseline_rows=8
    )
    history = np.arange(count * 2 * 6, dtype=np.float32).reshape(count, 2, 6)
    for i, row in enumerate(history):
        feed(ad, row, i * 0.01)
    for sensor in ad.hw.tactile.sensors:
        ring = ad.rings[f"tactile_{sensor.name}"]
        np.testing.assert_array_equal(
            ad._baseline_for(sensor.name, ring), native._baseline_for(sensor.name, ring)
        )
    feed(ad, history[-1] + 100, count * 0.01)
    for sensor in ad.hw.tactile.sensors:
        ring = ad.rings[f"tactile_{sensor.name}"]
        np.testing.assert_array_equal(
            ad._baseline_for(sensor.name, ring), native._baseline_for(sensor.name, ring)
        )
    ad.reset_baseline()
    native.reset_baseline()
    for sensor in ad.hw.tactile.sensors:
        ring = ad.rings[f"tactile_{sensor.name}"]
        np.testing.assert_array_equal(
            ad._baseline_for(sensor.name, ring), native._baseline_for(sensor.name, ring)
        )
