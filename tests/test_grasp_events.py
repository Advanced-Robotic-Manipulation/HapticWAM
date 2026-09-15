"""Synthetic-episode tests for tools/rig/analysis/grasp_events.py.

Every fixture is a hand-built zarr episode in `tmp_path` with the same layout
the recorder writes (`<ep>/<stream>.zarr/{data, ts}` + meta.json/stop.json), so
the classifier is exercised end to end without touching the rig or its data.

The scenarios are the six outcome classes plus the three things that are easy
to get wrong: the pad zero (the left pad carries a standing offset on the real
rig), the multi-close walk (a terminal-veto retry), and the operator
cross-check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "rig" / "analysis"))

import grasp_events as ge  # noqa: E402

DUR = 20.0
T0 = 1000.0


def _grid(rate: float) -> np.ndarray:
    return T0 + np.arange(0.0, DUR, 1.0 / rate)


def _write(ep: Path, stream: str, data: np.ndarray, ts: np.ndarray) -> None:
    g = zarr.open(str(ep / f"{stream}.zarr"), mode="w")
    g.create_dataset("data", data=np.asarray(data))
    g.create_dataset("ts", data=np.asarray(ts, dtype=np.float64))


def make_episode(tmp_path: Path, name: str, *,
                 closes: list[tuple[float, float]] = (),
                 contact: tuple[float, float, float] | None = None,
                 z_profile=None,
                 left_offset: float = 2.0,
                 success=None, tags=None, notes: str = "",
                 placement: dict | None = None,
                 stop_reason: str = "operator_stop",
                 with_area: bool = True) -> Path:
    """Build one synthetic take.

    closes:  [(t_close, t_reopen_or_inf)] windows where the COMMAND is closed.
    contact: (t_on, t_off, newtons) — the load both pads report while held.
    z_profile: callable t -> z in millimetres; default is a flat 80 mm.
    left_offset: standing bias, newtons, on the left pad's fz for the whole
                 take (the real rig's left pad reads 1-2.5 N with the gripper
                 open); the classifier must zero it out.
    """
    ep = tmp_path / name
    ep.mkdir(parents=True)

    t_tcp = _grid(125.0)
    t_act = _grid(12.5)
    t_grip = _grid(100.0)
    t_pad = _grid(8.0)

    if z_profile is None:
        def z_profile(t):
            return np.full_like(t, 80.0)

    # --- commanded gripper aperture: 0.20 open, 0.62 closed ---------------
    def cmd_at(ts):
        out = np.full_like(ts, 0.20)
        for t_c, t_o in closes:
            out[(ts >= T0 + t_c) & (ts < T0 + t_o)] = 0.62
        return out

    acts = np.zeros((len(t_act), 7), dtype=np.float32)
    acts[:, 6] = cmd_at(t_act - T0 + T0)
    _write(ep, "actions", acts, t_act)

    grip = np.zeros((len(t_grip), 2), dtype=np.float32)
    grip[:, 0] = cmd_at(t_grip)          # measured tracks the command here
    _write(ep, "gripper", grip, t_grip)

    tcp = np.zeros((len(t_tcp), 6), dtype=np.float64)
    tcp[:, 0] = -0.37
    tcp[:, 1] = -0.29
    tcp[:, 2] = z_profile(t_tcp - T0) / 1000.0
    _write(ep, "arm_tcp_pose", tcp, t_tcp)
    _write(ep, "arm_ft", np.zeros((len(t_tcp), 6)), t_tcp)

    load = np.zeros(len(t_pad))
    if contact is not None:
        t_on, t_off, newtons = contact
        load[(t_pad - T0 >= t_on) & (t_pad - T0 <= t_off)] = newtons

    lw = np.zeros((len(t_pad), 6), dtype=np.float32)
    rw = np.zeros((len(t_pad), 6), dtype=np.float32)
    lw[:, 2] = -(load + left_offset)     # compression is negative fz
    rw[:, 2] = -load
    _write(ep, "tactile_left_wrench", lw, t_pad)
    _write(ep, "tactile_right_wrench", rw, t_pad)
    if with_area:
        _write(ep, "tactile_left_area", load * 1.2, t_pad)
        _write(ep, "tactile_right_area", load * 1.1, t_pad)

    (ep / "meta.json").write_text(json.dumps({
        "task": "waffles", "policy": "student", "success": success,
        "notes": notes, "status": "finalized",
        "tags": list(tags or ["label:testarm", "seed:101"]),
        "deploy_overrides": {"z_floor_m": 0.0315},
    }))
    stop = {"stopped_reason": stop_reason, "n_replans": 30, "stop_state": {}}
    if placement is not None:
        stop["stop_state"]["placement_descent"] = placement
    (ep / "stop.json").write_text(json.dumps(stop))
    return ep


def ramp(points):
    """Piecewise-linear z(t) in mm from [(t, z), ...]."""
    ts = np.array([p[0] for p in points], dtype=float)
    zs = np.array([p[1] for p in points], dtype=float)

    def f(t):
        return np.interp(np.asarray(t, dtype=float), ts, zs)
    return f


# ---------------------------------------------------------------------------
# the six classes
# ---------------------------------------------------------------------------

def test_never_closed(tmp_path):
    ep = make_episode(tmp_path, "ep_a", closes=[], success=False)
    r = ge.analyse_episode(ep)
    assert r.cls == "never_closed"
    assert r.n_closes == 0
    assert r.closes == []
    assert r.arm == "testarm" and r.cell == "101"


def test_closed_on_air_is_the_under_grasp(tmp_path):
    """A close command with nothing between the pads."""
    ep = make_episode(tmp_path, "ep_b", closes=[(6.0, 1e9)], contact=None,
                      success=False)
    r = ge.analyse_episode(ep)
    assert r.cls == "closed_on_air"
    assert r.n_closes == 1
    assert r.closes[0].both_contact is False
    assert r.closes[0].pad_left_n < ge.CONTACT_N
    # height above table uses the episode's own z_floor (31.5 mm)
    assert r.closes[0].height_above_table_mm == pytest.approx(48.5, abs=0.5)
    assert r.hold_s < ge.MIN_HOLD_S


def test_momentary_touch_is_not_a_grasp(tmp_path):
    """A 0.2 s two-pad blip stays `closed_on_air`, not a hold."""
    ep = make_episode(tmp_path, "ep_b2", closes=[(6.0, 1e9)],
                      contact=(6.2, 6.35, 12.0), success=False)
    r = ge.analyse_episode(ep)
    assert r.cls == "closed_on_air"


def test_grasped_no_lift(tmp_path):
    ep = make_episode(tmp_path, "ep_c", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 11.0), success=False)
    r = ge.analyse_episode(ep)
    assert r.cls == "grasped_no_lift"
    assert r.hold_s > 5.0
    assert r.lift_mm < ge.LIFT_MIN_MM
    assert r.pad_peak_n == pytest.approx(11.0, abs=0.3)
    assert r.drop is False


def test_grasped_dropped(tmp_path):
    """Lifted 220 mm, then contact lost while still high."""
    ep = make_episode(tmp_path, "ep_d", closes=[(6.0, 1e9)],
                      contact=(6.2, 12.0, 11.0),
                      z_profile=ramp([(0, 300), (6, 80), (11, 300), (20, 300)]),
                      success=False)
    r = ge.analyse_episode(ep)
    assert r.cls == "grasped_dropped"
    assert r.drop is True
    assert r.released is True
    assert r.lift_mm > 150
    assert r.z_release_mm > ge.PLACE_Z_MM


def test_placed(tmp_path):
    """Lifted, then contact lost low over the crate."""
    ep = make_episode(tmp_path, "ep_e", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 12.0),
                      z_profile=ramp([(0, 300), (6, 80), (10, 320),
                                      (14, 120), (20, 300)]),
                      success=True)
    r = ge.analyse_episode(ep)
    assert r.cls == "placed"
    assert r.crush is False
    assert r.drop is False
    assert r.z_release_mm <= ge.PLACE_Z_MM
    assert r.lift_mm > 200


def test_placed_crushed(tmp_path):
    """Same trajectory, pad load above the clean-placed range."""
    ep = make_episode(tmp_path, "ep_f", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 21.0),
                      z_profile=ramp([(0, 300), (6, 80), (10, 320),
                                      (14, 120), (20, 300)]),
                      success=True, tags=["label:testarm", "seed:101", "crushed"],
                      notes="operator: c CRUSHED")
    r = ge.analyse_episode(ep)
    assert r.cls == "placed_crushed"
    assert r.crush is True
    assert r.verdict == "crushed"


def test_held_to_the_end_is_not_a_drop(tmp_path):
    """Contact runs to the last sample: inconclusive, not a drop or a place."""
    ep = make_episode(tmp_path, "ep_g", closes=[(6.0, 1e9)],
                      contact=(6.2, 19.99, 11.0),
                      z_profile=ramp([(0, 300), (6, 80), (12, 320), (20, 320)]),
                      success=False)
    r = ge.analyse_episode(ep)
    assert r.cls == "grasped_held_at_end"
    assert r.released is False
    assert r.drop is False


# ---------------------------------------------------------------------------
# the three easy-to-get-wrong details
# ---------------------------------------------------------------------------

def test_left_pad_standing_offset_is_zeroed(tmp_path):
    """A 6 N standing bias on the left pad must not read as contact."""
    ep = make_episode(tmp_path, "ep_h", closes=[(6.0, 1e9)], contact=None,
                      left_offset=6.0, success=False)
    r = ge.analyse_episode(ep)
    assert r.cls == "closed_on_air"
    assert r.closes[0].pad_left_n < ge.CONTACT_N


def test_terminal_veto_retry_counts_two_closes(tmp_path):
    """close -> forced open -> close again is two attempts, scored on the last."""
    ep = make_episode(tmp_path, "ep_i", closes=[(4.0, 6.0), (8.0, 1e9)],
                      contact=(8.3, 15.0, 12.0),
                      z_profile=ramp([(0, 300), (8, 80), (12, 320),
                                      (15, 120), (20, 300)]),
                      success=True)
    r = ge.analyse_episode(ep)
    assert r.n_closes == 2
    assert r.closes[0].t_reopen_s is not None
    assert r.closes[0].both_contact is False      # the first close caught air
    assert r.closes[1].both_contact is True
    # the hold is scored against the SECOND close, so the lift is measured
    # from the low z the retry closed at, not the first attempt's
    assert r.t_grasp_close_s == pytest.approx(r.closes[1].t_s, abs=0.2)
    assert r.cls == "placed"


def test_controller_release_record_beats_the_z_gate(tmp_path):
    """A `placement_descent.releases` count makes it a placement."""
    ep = make_episode(tmp_path, "ep_j", closes=[(6.0, 1e9)],
                      contact=(6.2, 12.0, 12.0),
                      z_profile=ramp([(0, 300), (6, 80), (12, 300), (20, 300)]),
                      success=True,
                      placement={"releases": 1, "release_gate_z_m": 0.16,
                                 "released_at_z_m": 0.141,
                                 "released_at_y_m": 0.061})
    r = ge.analyse_episode(ep)
    assert r.ctrl_releases == 1
    assert r.cls == "placed"          # despite losing contact at ~300 mm


def test_cross_check_flags_operator_disagreement(tmp_path):
    """Operator says success, sensors say the object was dropped."""
    ep = make_episode(tmp_path, "ep_k", closes=[(6.0, 1e9)],
                      contact=(6.2, 12.0, 11.0),
                      z_profile=ramp([(0, 300), (6, 80), (11, 300), (20, 300)]),
                      success=True, notes="operator: s")
    r = ge.cross_check(ge.analyse_episode(ep))
    assert r.verdict == "s"
    assert r.cls == "grasped_dropped"
    assert r.agree is False
    assert "operator success" in r.disagreement


def test_untagged_crush_is_flagged(tmp_path):
    ep = make_episode(tmp_path, "ep_l", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 22.0),
                      z_profile=ramp([(0, 300), (6, 80), (10, 320),
                                      (14, 120), (20, 300)]),
                      success=False)
    r = ge.cross_check(ge.analyse_episode(ep))
    assert r.crush is True
    assert r.agree is False
    assert "no crush tag" in r.disagreement


def test_missing_area_streams_are_tolerated(tmp_path):
    ep = make_episode(tmp_path, "ep_m", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 11.0), with_area=False,
                      success=False)
    r = ge.analyse_episode(ep)
    assert r.cls == "grasped_no_lift"
    assert np.isnan(r.closes[0].area_left)


def test_unreadable_episode_does_not_raise(tmp_path):
    ep = tmp_path / "ep_broken"
    ep.mkdir()
    (ep / "meta.json").write_text(json.dumps({"task": "waffles", "tags": []}))
    r = ge.analyse_episode(ep)
    assert r.cls == "unreadable"
    assert r.agree is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_csv_and_markdown(tmp_path):
    root = tmp_path / "deploy"
    root.mkdir()
    make_episode(root, "ep_x_000", closes=[], success=False,
                 tags=["label:armA", "seed:101"])
    make_episode(root, "ep_x_001", closes=[(6.0, 1e9)],
                 contact=(6.2, 14.0, 12.0),
                 z_profile=ramp([(0, 300), (6, 80), (10, 320), (14, 120),
                                 (20, 300)]),
                 success=True, tags=["label:armB", "seed:102"])
    out_csv = tmp_path / "out.csv"
    out_md = tmp_path / "out.md"
    out_closes = tmp_path / "closes.csv"
    rc = ge.main([str(root), "--csv", str(out_csv), "--md", str(out_md),
                  "--closes-csv", str(out_closes), "--quiet"])
    assert rc == 0
    text = out_csv.read_text()
    assert "never_closed" in text and "placed" in text
    assert "armA" in text and "armB" in text
    md = out_md.read_text()
    assert "Per-arm class counts" in md
    assert "`armA`" in md and "`armB`" in md
    assert "Contact threshold" in md
    assert "episode,arm,cell,verdict,index" in out_closes.read_text()


def test_cli_rejects_an_empty_folder(tmp_path):
    assert ge.main([str(tmp_path), "--csv", str(tmp_path / "o.csv")]) == 2
