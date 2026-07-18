"""Echo teleop protocol / Δ-EE derivation / dual-stream recording (device-free)."""

import numpy as np
import pytest

from phantom.config.hardware import EchoTeleopConfig
from phantom.data.derived import pose_delta, rotvec_nearest
from phantom.data.schema import STREAM_ACTIONS, STREAM_ACTIONS_QTARGET
from phantom.teleop.echo import TICK_TO_RAD, EchoTeleop, parse_r2_frame
from phantom_test_utils import make_hw

BASE = (1.427e-03, -1.5621, 1.5880, 0.00954, 1.5737, -3.0957)


def _cfg(**kw) -> EchoTeleopConfig:
    d = dict(base_pose=BASE, gripper_open_tick=5, gripper_closed_tick=155)
    d.update(kw)
    return EchoTeleopConfig(**d)


def _frame(ticks=None, sense=0, start=0) -> bytes:
    t = np.zeros(16, dtype=np.int16) if ticks is None else np.asarray(ticks, dtype=np.int16)
    return t.tobytes() + bytes([sense, start])


# ---------------------------------------------------------------------------
# protocol parser
# ---------------------------------------------------------------------------

def test_parse_r2_frame_roundtrip():
    ticks_in = np.arange(16, dtype=np.int16) * 100 - 800
    ticks, sense, start = parse_r2_frame(_frame(ticks_in, sense=2, start=1))
    assert np.array_equal(ticks, ticks_in)
    assert sense == 2 and start is True


def test_parse_r2_frame_rejects_short_read():
    with pytest.raises(ValueError, match="34"):
        parse_r2_frame(_frame()[:-1])


def test_tick_to_rad_factor():
    # device joints span 300 degrees over 4096 ticks (lab stack constant)
    assert TICK_TO_RAD == pytest.approx(0.0012783173232373046875, rel=1e-12)
    assert 4096 * TICK_TO_RAD == pytest.approx(np.deg2rad(300.0))


# ---------------------------------------------------------------------------
# EchoTeleop unit behavior (no serial device: exercise the pure parts)
# ---------------------------------------------------------------------------

def test_gripper_normalization_and_clip():
    dev = EchoTeleop(_cfg())
    assert dev._gripper_01(5) == 0.0
    assert dev._gripper_01(155) == 1.0
    assert dev._gripper_01(80) == pytest.approx((80 - 5) / 150)
    assert dev._gripper_01(-50) == 0.0          # clipped below
    assert dev._gripper_01(400) == 1.0          # clipped above


def test_gripper_normalization_inverted_calibration():
    dev = EchoTeleop(_cfg(gripper_open_tick=155, gripper_closed_tick=5))
    assert dev._gripper_01(155) == 0.0
    assert dev._gripper_01(5) == 1.0


def test_gripper_tick_validator():
    with pytest.raises(Exception, match="differ"):
        _cfg(gripper_open_tick=10, gripper_closed_tick=10)


def test_start_flag_level_to_edge():
    dev = EchoTeleop(_cfg())
    dev._have_sample = True
    # level held high across many polls -> exactly one start_stop
    dev._start_level = True
    fired = [dev.poll().buttons.get("start_stop", False) for _ in range(5)]
    assert fired == [True, False, False, False, False]
    # falling edge fires once more (recording stops)
    dev._start_level = False
    fired = [dev.poll().buttons.get("start_stop", False) for _ in range(3)]
    assert fired == [True, False, False]


def test_poll_holds_until_first_sample():
    dev = EchoTeleop(_cfg())
    cmd = dev.poll()
    assert cmd.q_target is None          # no device sample yet: loop keeps Δ-EE path
    assert cmd.gripper == 0.0


def test_sensitivity_divisor_indexing():
    cfg = _cfg()
    for flag, div in enumerate(cfg.sensitivity_divisors):
        ticks = np.zeros(16, dtype=np.int16)
        ticks[8:14] = 4096                          # 300 deg on every right-arm joint
        parsed, sense, _ = parse_r2_frame(_frame(ticks, sense=flag))
        offset = parsed[8:14].astype(np.float64) * TICK_TO_RAD
        expected = np.asarray(BASE) + offset / div
        # mirrors the reader-thread math exactly
        assert np.allclose(expected - np.asarray(BASE),
                           np.deg2rad(300.0) / div, atol=1e-9)


# ---------------------------------------------------------------------------
# Δ-EE derivation
# ---------------------------------------------------------------------------

def test_rotvec_nearest_antipodal_flip():
    # same rotation, antipodal representation: r and r*(1 - 2pi/|r|)
    prev = np.array([0.0, 0.0, np.pi - 1e-3])
    cur = prev * (1.0 - 2.0 * np.pi / np.linalg.norm(prev))   # flipped repr, ~ -pi
    fixed = rotvec_nearest(prev, cur)
    assert np.linalg.norm(fixed - prev) < 1e-2               # small once un-flipped


def test_pose_delta_small_and_cumsum_reconstructs():
    rng = np.random.default_rng(0)
    poses = [np.array([0.0, -0.45, 0.25, 0.0, 3.1, 0.0])]
    for _ in range(50):
        step = rng.normal(0, 0.005, 6)
        poses.append(poses[-1] + step)
    deltas = [pose_delta(a, b) for a, b in zip(poses[:-1], poses[1:])]
    assert max(np.abs(d).max() for d in deltas) < 0.05        # all small
    # executor model: pose = t0 + cumsum(deltas)
    recon = poses[0] + np.cumsum(np.stack(deltas), axis=0)
    assert np.allclose(recon[-1], poses[-1], atol=1e-5)


# ---------------------------------------------------------------------------
# recorder: dual action streams
# ---------------------------------------------------------------------------

class _StubSession:
    def __init__(self, hw):
        self.hw = hw
        self.rings = {}


def test_recorder_writes_both_action_streams(tmp_path):
    from phantom.data.episode_store import EpisodeReader
    from phantom.data.schema import EpisodeMeta
    from phantom.recording.recorder import EpisodeRecorder
    from phantom.timesync.clock import IdentityClock

    hw = make_hw()
    rec = EpisodeRecorder(_StubSession(hw), IdentityClock(), tmp_path)
    rec.start(EpisodeMeta(task="t", text="t", policy="teleop"), "ep_test")
    for k in range(5):
        rec.record_action(0.1 * k, np.full(7, k, dtype=np.float32))
        rec.record_action(0.1 * k, np.full(6, -k, dtype=np.float32),
                          stream=STREAM_ACTIONS_QTARGET)
    path = rec.stop(success=True)
    r = EpisodeReader(path)
    assert r.has(STREAM_ACTIONS) and r.has(STREAM_ACTIONS_QTARGET)
    assert r.n(STREAM_ACTIONS) == 5
    assert r.n(STREAM_ACTIONS_QTARGET) == 5
    a = np.asarray(r._g(STREAM_ACTIONS)["data"][:])
    q = np.asarray(r._g(STREAM_ACTIONS_QTARGET)["data"][:])
    assert a.shape == (5, 7) and q.shape == (5, 6)
    assert np.array_equal(r.ts(STREAM_ACTIONS), r.ts(STREAM_ACTIONS_QTARGET))


# ---------------------------------------------------------------------------
# smoothing: one-euro filter + accel-limited tracker + streamer
# ---------------------------------------------------------------------------

def test_one_euro_kills_noise_at_rest():
    """At rest (the tremor-suppression regime) noise must be strongly cut."""
    from phantom.teleop.filters import OneEuroFilter
    rng = np.random.default_rng(1)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.3)
    dt = 0.01
    t = np.arange(0, 4, dt)
    hold = 0.7                                          # operator holding a pose
    noisy = hold + rng.normal(0, 0.02, t.size)          # encoder noise / tremor
    out = np.array([f.filter(np.array([x]), ti)[0] for x, ti in zip(noisy, t)])
    tail = slice(50, None)                              # skip warm-up
    noise_in = np.std(noisy[tail] - hold)
    noise_out = np.std(out[tail] - hold)
    assert noise_out < 0.4 * noise_in                   # tremor attenuated
    assert abs(np.mean(out[tail]) - hold) < 0.005       # unbiased


def test_one_euro_adaptive_lag():
    """Fast motion must pass with less relative lag than a plain low-pass at rest."""
    from phantom.teleop.filters import OneEuroFilter
    dt = 0.01
    t = np.arange(0, 2, dt)
    ramp = 2.0 * t                                      # fast 2 rad/s sweep
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    out = np.array([f.filter(np.array([x]), ti)[0] for x, ti in zip(ramp, t)])
    lag = ramp[-1] - out[-1]
    assert lag < 0.15                                   # beta keeps up with motion


def test_tracker_no_overshoot_and_bounded():
    from phantom.teleop.filters import AccelLimitedTracker
    tr = AccelLimitedTracker(v_max=1.0, a_max=4.0)
    tr.reset_to(np.zeros(2))
    dt = 1.0 / 125.0
    target = np.array([0.5, -0.3])
    qs = [tr.step(target, dt) for _ in range(500)]
    qs = np.stack(qs)
    v = np.diff(qs, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    assert np.allclose(qs[-1], target, atol=1e-9)               # exact landing
    assert np.abs(v).max() <= 1.0 + 1e-6                        # vel bound
    # accel bound holds throughout the approach; only the final landing tick
    # may carry a one-off <=2*a_max transient (documented in filters.py —
    # imperceptible at the tiny landing velocity)
    assert np.abs(a).max() <= 2 * 4.0 + 1e-3
    for j in range(qs.shape[1]):
        settle = int(np.argmax(qs[:, j] == target[j]))
        assert np.abs(a[: settle - 3, j]).max() <= 4.0 + 1e-3
    assert qs[:, 0].max() <= target[0] + 1e-9                   # no overshoot
    assert qs[:, 1].min() >= target[1] - 1e-9


def test_tracker_engage_is_smooth():
    """A large engage step must start from rest (bounded initial accel)."""
    from phantom.teleop.filters import AccelLimitedTracker
    tr = AccelLimitedTracker(v_max=1.0, a_max=4.0)
    tr.reset_to(np.zeros(1))
    dt = 1.0 / 125.0
    q1 = tr.step(np.array([1.0]), dt)                   # first tick after engage
    assert abs(q1[0]) <= 4.0 * dt * dt + 1e-9           # a_max*dt^2, not a jump


def test_streamer_glides_mock_arm(default_hw):
    from phantom.drivers.mock.ur import MockArm
    from phantom.teleop.streamer import JointServoStreamer
    import time as _time
    hw = default_hw
    arm = MockArm(hw)
    arm.connect(control=True)
    st = JointServoStreamer(hw, arm)
    q0 = arm.get_state().q
    st.start()
    st.set_target(q0 + 0.2)
    _time.sleep(0.5)
    st.stop()
    q1 = arm.get_state().q
    moved = q1 - q0
    assert np.all(moved > 0.05)                         # clearly moving toward target
    assert np.all(moved <= 0.2 + 1e-6)                  # never past it
    arm.disconnect()
