"""Regression tests for the P8 tactile grasp-success rule.

Synthetic zarr episodes (real EpisodeWriter, real streams) exercise the four
cases the rig actually produces — a clean grasp, an under-grasp that closes on
air and lifts anyway, an over-squeeze that grasps AND stalls, and a correct
close with no lift — plus the invariant that Robotiq OBJ never enters
`grasp_ok` (REVIEW_SYNTHESIS.md P8: OBJ==2 flags over-squeeze, not grasp).
"""

from __future__ import annotations

import numpy as np
import pytest

from phantom.data.episode_store import EpisodeWriter
from phantom.data.schema import (STREAM_ARM_TCP_POSE, STREAM_GRIPPER,
                                 EpisodeMeta, tactile_stream)
from pathlib import Path

from phantom.eval.grasp_label import (close_attempts, confusion,
                                     label_episode, reason_key)
from phantom_test_utils import make_small_hw

T0 = 1000.0
DUR = 8.0
CLOSE_T = 2.0
GRIP_HZ = 50.0
TCP_HZ = 50.0


def _fields(hw, contact: bool) -> np.ndarray:
    """One field_ds frame: a contact blob on the depth channel, or nothing.

    Channel layout is the canonical 8-ch stack (deformation2d 0:2, depth 2:3,
    shear 3:5, dist_force 5:8) — contact_source is `depth`, so only ch 2 matters.
    The blob covers 64/768 = 8.3% of cells, above tau_contact_area (2.5%).
    """
    h, w = hw.recording.field_ds.h, hw.recording.field_ds.w
    f = np.zeros((h, w, hw.tactile.field_ch), dtype=np.float16)
    if contact:
        f[2:10, 2:10, 2] = 1.0        # |depth| >> tau_contact_depth (0.05)
    return f


def write_episode(path, hw, *, task="waffles", success=None, notes="",
                  z_close_mm=90.0, z_lift_mm=120.0, contact_after_close=True,
                  obj_value=3.0, close=True, tags=None, policy=""):
    """A minimal but real episode: gripper, arm_tcp_pose, both fields_ds."""
    w = EpisodeWriter(path, hw, EpisodeMeta(task=task, tags=list(tags or []),
                                            policy=policy))

    g_ts = T0 + np.arange(0.0, DUR, 1.0 / GRIP_HZ)
    pos = np.where(g_ts - T0 < CLOSE_T, 0.10, 0.70 if close else 0.10)
    obj = np.where(g_ts - T0 < CLOSE_T, 0.0, float(obj_value))
    w.append(STREAM_GRIPPER, g_ts, np.stack([pos, obj], axis=1).astype(np.float32))

    t_ts = T0 + np.arange(0.0, DUR, 1.0 / TCP_HZ)
    rel = t_ts - T0
    # descend to z_close by CLOSE_T, hold 0.5 s, then lift linearly over 2 s
    z = np.full_like(rel, z_close_mm)
    ramp = np.clip((rel - (CLOSE_T + 0.5)) / 2.0, 0.0, 1.0)
    z = z + ramp * z_lift_mm
    pose = np.zeros((len(t_ts), 6), dtype=np.float64)
    pose[:, 2] = z / 1000.0
    w.append(STREAM_ARM_TCP_POSE, t_ts, pose)

    f_hz = hw.recording.field_ds_rate_hz
    f_ts = T0 + np.arange(0.0, DUR, 1.0 / f_hz)
    on = _fields(hw, True)
    off = _fields(hw, False)
    frames = np.stack([on if (contact_after_close and t - T0 >= CLOSE_T) else off
                       for t in f_ts])
    for s in hw.tactile.sensors:
        w.append(tactile_stream(s.name, "fields_ds"), f_ts, frames)
    w.finalize(success=success, notes=notes)
    return path


@pytest.fixture(scope="module")
def hw():
    return make_small_hw()


# ---------------------------------------------------------------------------
# the four rig cases
# ---------------------------------------------------------------------------

def test_clean_success(tmp_path, hw):
    ep = write_episode(tmp_path / "ep_ok", hw, success=True, notes="clean")
    lab = label_episode(ep, hw)
    assert lab.grasp_ok, lab.reasons
    assert lab.reasons == []
    assert lab.t_close == pytest.approx(CLOSE_T, abs=0.05)
    assert lab.z_close_mm == pytest.approx(90.0, abs=1.0)
    assert lab.hold_s == pytest.approx(DUR - CLOSE_T - 0.5, abs=0.1)
    assert lab.c_hold == 1.0
    assert lab.lift_mm == pytest.approx(120.0, abs=1.0)
    assert lab.stall is False
    assert lab.operator_success is True and lab.notes == "clean"
    assert lab.z_max_mm == 103.0          # waffles, demo p95 + 15 mm


def test_undergrasp_closes_on_air_and_lifts(tmp_path, hw):
    """The 45 `*_fail` demos: close 60 mm too high, no contact, lift anyway."""
    ep = write_episode(tmp_path / "ep_air", hw, success=False,
                       z_close_mm=160.0, z_lift_mm=200.0,
                       contact_after_close=False)
    lab = label_episode(ep, hw)
    assert not lab.grasp_ok
    keys = {reason_key(r) for r in lab.reasons}
    assert keys == {"z_close", "c_hold", "lift"}   # hold is long enough
    assert lab.c_hold == 0.0
    assert lab.lift_mm == 0.0                      # no in-contact sample at all
    assert lab.stall is False


def test_over_squeeze_stall_is_a_grasp_plus_a_flag(tmp_path, hw):
    """Over-squeeze demos: genuinely holding (grasp_ok) AND stalled (flag)."""
    ep = write_episode(tmp_path / "ep_squeeze", hw, success=True, obj_value=2.0)
    lab = label_episode(ep, hw)
    assert lab.grasp_ok, lab.reasons
    assert lab.stall is True
    assert lab.obj2_frac_hold == 1.0


def test_close_at_right_height_but_no_lift(tmp_path, hw):
    ep = write_episode(tmp_path / "ep_nolift", hw, success=False, z_lift_mm=0.0)
    lab = label_episode(ep, hw)
    assert not lab.grasp_ok
    assert {reason_key(r) for r in lab.reasons} == {"lift"}
    assert lab.c_hold == 1.0 and lab.lift_mm == 0.0


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obj_value", [0.0, 1.0, 2.0, 3.0])
def test_obj_alone_never_flips_grasp_ok(tmp_path, hw, obj_value):
    """P8: gOBJ is an over-squeeze FLAG. It must not move the grasp verdict in
    either direction — not for a real grasp, not for a close on air."""
    good = label_episode(write_episode(tmp_path / f"ep_g{obj_value}", hw,
                                       obj_value=obj_value), hw)
    bad = label_episode(write_episode(tmp_path / f"ep_b{obj_value}", hw,
                                      obj_value=obj_value, z_close_mm=160.0,
                                      contact_after_close=False), hw)
    assert good.grasp_ok is True, good.reasons
    assert bad.grasp_ok is False
    assert good.stall is (obj_value == 2.0)
    assert bad.stall is (obj_value == 2.0)


def test_never_closed(tmp_path, hw):
    ep = write_episode(tmp_path / "ep_open", hw, close=False)
    lab = label_episode(ep, hw)
    assert not lab.grasp_ok and lab.reasons == ["never_closed"]
    assert lab.t_close is None


def test_unknown_task_blocks_positive_but_is_overridable(tmp_path, hw):
    ep = write_episode(tmp_path / "ep_newtask", hw, task="sponge")
    lab = label_episode(ep, hw)
    assert not lab.grasp_ok
    assert {reason_key(r) for r in lab.reasons} == {"z_max_unknown_task"}
    lab2 = label_episode(ep, hw, z_max_table={"sponge": 120.0})
    assert lab2.grasp_ok, lab2.reasons


def test_fail_task_suffix_reuses_the_base_task_ceiling(tmp_path, hw):
    ep = write_episode(tmp_path / "ep_failsuffix", hw, task="waffles_fail")
    assert label_episode(ep, hw).z_max_mm == 103.0


def test_contact_threshold_is_the_training_one(tmp_path, hw):
    """A single hot pixel is NOT contact (issue #1: `> 0` saturated events)."""
    hi = make_small_hw(derived={"tau_contact_area": 0.5})   # blob is 8.3%
    ep = write_episode(tmp_path / "ep_tau", hw)
    assert label_episode(ep, hw).c_hold == 1.0
    assert label_episode(ep, hi).c_hold == 0.0


def test_release_ends_the_hold_window(tmp_path, hw):
    """Reopening the gripper 1 s after close leaves a sub-2 s hold."""
    ep = tmp_path / "ep_release"
    w = EpisodeWriter(ep, hw, EpisodeMeta(task="waffles"))
    g_ts = T0 + np.arange(0.0, DUR, 1.0 / GRIP_HZ)
    rel = g_ts - T0
    pos = np.where(rel < CLOSE_T, 0.10, np.where(rel < CLOSE_T + 1.0, 0.70, 0.10))
    w.append(STREAM_GRIPPER, g_ts,
             np.stack([pos, np.full_like(pos, 3.0)], axis=1).astype(np.float32))
    t_ts = T0 + np.arange(0.0, DUR, 1.0 / TCP_HZ)
    pose = np.zeros((len(t_ts), 6), dtype=np.float64)
    pose[:, 2] = 0.090
    w.append(STREAM_ARM_TCP_POSE, t_ts, pose)
    f_ts = T0 + np.arange(0.0, DUR, 1.0 / hw.recording.field_ds_rate_hz)
    frames = np.stack([_fields(hw, True)] * len(f_ts))
    for s in hw.tactile.sensors:
        w.append(tactile_stream(s.name, "fields_ds"), f_ts, frames)
    w.finalize(success=False)
    lab = label_episode(ep, hw)
    assert lab.hold_s == pytest.approx(0.5, abs=0.1)
    assert "hold" in {reason_key(r) for r in lab.reasons}


def test_confusion_counts(tmp_path, hw):
    labels = [
        label_episode(write_episode(tmp_path / "ep_c1", hw, success=True), hw),
        label_episode(write_episode(tmp_path / "ep_c2", hw, success=False,
                                    z_close_mm=160.0,
                                    contact_after_close=False), hw),
        label_episode(write_episode(tmp_path / "ep_c3", hw, success=None), hw),
    ]
    c = confusion(labels)
    assert c == {"ok_s": 1, "ok_f": 0, "ok_none": 1,
                 "no_s": 0, "no_f": 1, "no_none": 0,
                 # the third row: episodes the rule ABSTAINS on (the recording
                 # ended inside the hold window) — none of these three
                 "trunc_s": 0, "trunc_f": 0, "trunc_none": 0}


def test_truncated_stream_is_a_reason_not_a_crash(tmp_path, hw):
    """Recordings can die mid-write, leaving a .zarr group with no arrays."""
    import shutil
    ep = write_episode(tmp_path / "ep_trunc", hw)
    shutil.rmtree(ep / "gripper.zarr" / "data")
    lab = label_episode(ep, hw)
    assert not lab.grasp_ok
    assert lab.reasons and lab.reasons[0].startswith("unreadable_stream")


def test_missing_stream_is_reported(tmp_path, hw):
    import shutil
    ep = write_episode(tmp_path / "ep_nopose", hw)
    shutil.rmtree(ep / "arm_tcp_pose.zarr")
    lab = label_episode(ep, hw)
    assert lab.reasons == ["missing_stream:arm_tcp_pose"]


# ---------------------------------------------------------------------------
# validation 2026-08-30: the two cases the rule used to get wrong
# ---------------------------------------------------------------------------

def write_grip_episode(path, hw, *, task="waffles", success=None, dur=8.0,
                       close_t=2.0, reopen_t=None, reclose_t=None,
                       z_close_mm=90.0, z_lift_mm=120.0, lift_from=None,
                       contact_from=None, tags=None):
    """An episode with an explicit gripper story: close [, reopen, close].

    `reopen_t`/`reclose_t` reproduce a `--terminal-veto` retry (the veto forces
    the gripper open on a suspected phantom grasp and lets the policy close
    again); `dur` shorter than close_t + 2.5 s reproduces `run_episode` ending
    the recording before the hold can be measured.
    """
    w = EpisodeWriter(path, hw, EpisodeMeta(task=task, tags=list(tags or [])))
    contact_from = close_t if contact_from is None else contact_from
    lift_from = (contact_from + 0.5) if lift_from is None else lift_from

    g_ts = T0 + np.arange(0.0, dur, 1.0 / GRIP_HZ)
    rel = g_ts - T0
    pos = np.where(rel < close_t, 0.10, 0.70)
    if reopen_t is not None:
        pos = np.where(rel >= reopen_t, 0.10, pos)
        if reclose_t is not None:
            pos = np.where(rel >= reclose_t, 0.70, pos)
    obj = np.where(rel < close_t, 0.0, 3.0)
    w.append(STREAM_GRIPPER, g_ts, np.stack([pos, obj], axis=1).astype(np.float32))

    t_ts = T0 + np.arange(0.0, dur, 1.0 / TCP_HZ)
    rel_t = t_ts - T0
    z = z_close_mm + np.clip((rel_t - lift_from) / 2.0, 0.0, 1.0) * z_lift_mm
    pose = np.zeros((len(t_ts), 6), dtype=np.float64)
    pose[:, 2] = z / 1000.0
    w.append(STREAM_ARM_TCP_POSE, t_ts, pose)

    f_hz = hw.recording.field_ds_rate_hz
    f_ts = T0 + np.arange(0.0, dur, 1.0 / f_hz)
    on, off = _fields(hw, True), _fields(hw, False)
    frames = np.stack([on if (t - T0) >= contact_from else off for t in f_ts])
    for s in hw.tactile.sensors:
        w.append(tactile_stream(s.name, "fields_ds"), f_ts, frames)
    w.finalize(success=success, notes="")
    return path


def test_hold_truncated_is_not_a_negative(tmp_path, hw):
    """`run_episode` stops the recorder right after `planner.run()` returns, so
    a grasp on one of the last replans has no 2.5 s tail. `hold 0.4s < 2.0s`
    there is TRUNCATION — flagged, and reported in its own confusion cell
    instead of as a failed grasp (G2/G3 are 16-episode decisions)."""
    ep = write_grip_episode(tmp_path / "ep_trunc_hold", hw, dur=3.0,
                            close_t=2.0, z_lift_mm=200.0, lift_from=2.1)
    lab = label_episode(ep, hw)
    assert lab.hold_truncated is True
    assert not lab.grasp_ok
    assert lab.hold_s < 2.0
    keys = {reason_key(r) for r in lab.reasons}
    assert "hold_truncated" in keys and "hold" not in keys
    assert "recording ends" in " ".join(lab.reasons)
    # ... and it is NOT counted as a negative
    c = confusion([lab])
    assert c["trunc_none"] == 1 and c["no_none"] == 0 and c["ok_none"] == 0


def test_a_real_reopen_is_still_a_short_hold_failure(tmp_path, hw):
    """The flag must not swallow the genuine case: the gripper DID let go."""
    ep = write_grip_episode(tmp_path / "ep_letgo", hw, dur=8.0, close_t=2.0,
                            reopen_t=3.0, z_lift_mm=200.0)
    lab = label_episode(ep, hw)
    assert lab.hold_truncated is False and not lab.grasp_ok
    keys = {reason_key(r) for r in lab.reasons}
    assert "hold" in keys and "hold_truncated" not in keys
    assert confusion([lab])["no_none"] == 1


def test_a_close_after_a_reopen_is_scored_on_the_last_close(tmp_path, hw):
    """The --terminal-veto retry: close, forced open, close again and hold.

    Scoring the FIRST close made every veto retry-success a failure — the exact
    number GATE G2 is decided on."""
    ep = write_grip_episode(tmp_path / "ep_retry", hw, success=True, dur=9.0,
                            close_t=2.0, reopen_t=2.6, reclose_t=3.2,
                            contact_from=3.2, lift_from=3.7, z_lift_mm=120.0)
    lab = label_episode(ep, hw)
    assert lab.n_close_attempts == 2
    assert lab.t_first_close_s == pytest.approx(2.0, abs=0.05)
    assert lab.t_close == pytest.approx(3.2, abs=0.05)     # the LAST close
    assert lab.hold_s == pytest.approx(9.0 - 3.2 - 0.5, abs=0.1)
    assert lab.c_hold == 1.0 and lab.lift_mm == pytest.approx(120.0, abs=1.0)
    assert lab.grasp_ok, lab.reasons
    assert lab.hold_truncated is False
    # the attempt the OLD rule scored: closed at 2.0 s, forced open 0.6 s later
    from phantom.data.episode_store import EpisodeReader
    rd = EpisodeReader(ep)
    g = np.asarray(rd.data(STREAM_GRIPPER)[:], dtype=np.float64)
    ts = rd.ts(STREAM_GRIPPER)
    atts = close_attempts(g[:, 0], ts, float(ts[-1]))
    assert len(atts) == 2
    ci0, t_rel0, released0 = atts[0]
    assert released0 is True
    assert t_rel0 - float(ts[ci0]) < 1.0, "the veto reopen must end attempt 1"
    assert atts[1][0] > ci0 and atts[1][2] is False        # still closed at the end


def test_a_reopen_with_no_second_close_still_fails(tmp_path, hw):
    """One attempt, released early — the veto's own failure mode."""
    ep = write_grip_episode(tmp_path / "ep_veto_fail", hw, dur=8.0, close_t=2.0,
                            reopen_t=2.8)
    lab = label_episode(ep, hw)
    assert lab.n_close_attempts == 1 and not lab.grasp_ok
    assert lab.hold_truncated is False


def test_label_grasps_cli_reports_the_truncated_row(tmp_path, hw, capsys,
                                                    monkeypatch):
    """The real entry point: tools/label_grasps.py --confusion must show the
    abstention row, not fold truncated episodes into `not ok`."""
    import sys
    import yaml
    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "tools"))
    import label_grasps

    root = tmp_path / "eps"
    root.mkdir()
    write_grip_episode(root / "ep_good", hw, success=True)
    write_grip_episode(root / "ep_cut", hw, dur=3.0, close_t=2.0,
                       z_lift_mm=200.0, lift_from=2.1)
    hw_yaml = tmp_path / "hw.yaml"
    hw_yaml.write_text(yaml.safe_dump(hw.model_dump(mode="json")))
    monkeypatch.setattr(sys, "argv", ["label_grasps.py", str(root),
                                      "--hw", str(hw_yaml), "--confusion"])
    assert label_grasps.main() == 0
    out = capsys.readouterr().out
    assert "truncated" in out
    assert "hold_truncated" in out          # the per-condition histogram
    lines = [l for l in out.splitlines() if l.startswith("truncated")]
    assert lines and lines[0].split()[-1] == "1"
