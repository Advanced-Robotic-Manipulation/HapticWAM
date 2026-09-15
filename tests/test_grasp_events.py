"""Synthetic-episode tests for tools/rig/analysis/grasp_events.py.

Every fixture is a hand-built zarr episode in `tmp_path` with the same layout
the recorder writes (`<ep>/<stream>.zarr/{data, ts}` + meta.json/stop.json), so
the two-level classifier is exercised end to end without touching the rig or
its data.

Covered: each of the seven classes, the two outcome levels and their exact
arithmetic, all three Level A evidence signals separately, and the four things
that are easy to get wrong — the pad zero (the left pad carries a standing
offset on the real rig), the multi-close walk (a terminal-veto retry), the
run-time crush-band derivation, and the operator cross-check.
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
                 closes=(),
                 contact: tuple[float, float, float] | None = None,
                 area: tuple[float, float, float] | None = None,
                 obj2: tuple[float, float] | None = None,
                 z_profile=None,
                 left_offset: float = 2.0,
                 success=None, tags=None, notes: str = "",
                 placement: dict | None = None,
                 stop_reason: str = "operator_stop",
                 with_area: bool = True) -> Path:
    """Build one synthetic take.

    closes:  [(t_close, t_reopen_or_inf)] windows where the COMMAND is closed.
    contact: (t_on, t_off, newtons) load both pads report while held.
    area:    (t_on, t_off, mm2) vendor contact area, independent of `contact`.
    obj2:    (t_on, t_off) window where the Robotiq gOBJ code reads 2.
    left_offset: standing bias, newtons, on the left pad for the whole take
                 (the real rig's left pad reads 1-2.5 N with the gripper open);
                 the classifier must zero it out.
    """
    ep = tmp_path / name
    ep.mkdir(parents=True)

    t_tcp, t_act = _grid(125.0), _grid(12.5)
    t_grip, t_pad = _grid(100.0), _grid(8.0)

    if z_profile is None:
        def z_profile(t):
            return np.full_like(t, 80.0)

    def cmd_at(ts):
        out = np.full_like(ts, 0.20)
        for t_c, t_o in closes:
            out[(ts >= T0 + t_c) & (ts < T0 + t_o)] = 0.62
        return out

    acts = np.zeros((len(t_act), 7), dtype=np.float32)
    acts[:, 6] = cmd_at(t_act)
    _write(ep, "actions", acts, t_act)

    grip = np.zeros((len(t_grip), 2), dtype=np.float32)
    grip[:, 0] = cmd_at(t_grip)          # measured tracks the command here
    if obj2 is not None:
        grip[(t_grip - T0 >= obj2[0]) & (t_grip - T0 <= obj2[1]), 1] = 2.0
    _write(ep, "gripper", grip, t_grip)

    tcp = np.zeros((len(t_tcp), 6), dtype=np.float64)
    tcp[:, 0], tcp[:, 1] = -0.37, -0.29
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
        a = np.zeros(len(t_pad))
        if area is not None:
            a[(t_pad - T0 >= area[0]) & (t_pad - T0 <= area[1])] = area[2]
        elif contact is not None:
            a = load * 1.2
        _write(ep, "tactile_left_area", a, t_pad)
        _write(ep, "tactile_right_area", a, t_pad)

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
    return lambda t: np.interp(np.asarray(t, dtype=float), ts, zs)


PICK_PLACE = ramp([(0, 300), (6, 80), (10, 320), (14, 120), (20, 300)])
PICK_DROP = ramp([(0, 300), (6, 80), (11, 300), (20, 300)])


# ---------------------------------------------------------------------------
# the seven classes and the two levels
# ---------------------------------------------------------------------------

def test_never_reached(tmp_path):
    ep = make_episode(tmp_path, "ep_a", closes=[], success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "never_reached"
    assert r.n_closes == 0 and r.closes == []
    assert r.grasp_success is False and r.haptic_success is False
    assert r.arm == "testarm" and r.cell == "101"


def test_closed_on_air(tmp_path):
    """A close command with nothing between the pads: not even Level A."""
    ep = make_episode(tmp_path, "ep_b", closes=[(6.0, 1e9)], contact=None,
                      success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "closed_on_air"
    assert r.grasp_success is False
    assert r.n_closes == 1
    assert r.contact_a_evidence == ""
    # height above table uses the episode's own z_floor (31.5 mm)
    assert r.closes[0].height_above_table_mm == pytest.approx(48.5, abs=0.5)


def test_contact_no_hold_is_the_under_grasp(tmp_path):
    """A brief two-pad touch: Level A yes, Level B no."""
    ep = make_episode(tmp_path, "ep_c", closes=[(6.0, 1e9)],
                      contact=(6.2, 6.35, 12.0), success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "contact_no_hold"
    assert r.grasp_success is True and r.haptic_success is False


def test_hold_without_a_lift_is_contact_no_hold(tmp_path):
    """Held on the table but never picked up: Level A only."""
    ep = make_episode(tmp_path, "ep_d", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 11.0), success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "contact_no_hold"
    assert r.hold_s > 5.0
    assert r.lift_mm < ge.LIFT_MIN_MM
    assert r.grasp_success is True and r.haptic_success is False


def test_held_dropped(tmp_path):
    """Lifted 220 mm, then contact lost while still high."""
    ep = make_episode(tmp_path, "ep_e", closes=[(6.0, 1e9)],
                      contact=(6.2, 12.0, 11.0), z_profile=PICK_DROP,
                      success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "held_dropped"
    assert r.drop is True and r.released is True
    assert r.lift_mm > 150
    assert r.z_release_mm > ge.PLACE_Z_MM
    assert r.grasp_success is True and r.haptic_success is False


def test_placed_clean(tmp_path):
    """Lifted, then contact lost low over the crate, inside the clean band."""
    ep = make_episode(tmp_path, "ep_f", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 12.0), z_profile=PICK_PLACE,
                      success=True)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "placed_clean"
    assert r.crush is False and r.drop is False
    assert r.z_release_mm <= ge.PLACE_Z_MM
    assert r.lift_mm > 200
    assert r.grasp_success is True and r.haptic_success is True


def test_placed_crushed(tmp_path):
    """Same trajectory, peak load above the crush band: Level A, not Level B."""
    ep = make_episode(tmp_path, "ep_g", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 21.0), z_profile=PICK_PLACE,
                      success=True, tags=["label:testarm", "seed:101", "crushed"],
                      notes="operator: c CRUSHED")
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "placed_crushed"
    assert r.crush is True
    assert r.verdict == "crushed"
    assert r.grasp_success is True and r.haptic_success is False


def test_held_at_cut_is_not_a_drop(tmp_path):
    """Contact runs to the last sample: Level A yes, Level B abstains."""
    ep = make_episode(tmp_path, "ep_h", closes=[(6.0, 1e9)],
                      contact=(6.2, 19.99, 11.0),
                      z_profile=ramp([(0, 300), (6, 80), (12, 320), (20, 320)]),
                      success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "held_at_cut"
    assert r.released is False and r.drop is False
    assert r.grasp_success is True and r.haptic_success is False


def test_level_arithmetic_matches_the_class_sets(tmp_path):
    """grasp_success and haptic_success are exactly the documented sums."""
    assert ge.GRASP_OK_CLASSES == {"contact_no_hold", "held_dropped",
                                   "held_at_cut", "placed_clean",
                                   "placed_crushed"}
    assert ge.HAPTIC_OK_CLASSES == {"placed_clean"}
    assert ge.GRASP_OK_CLASSES < set(ge.CLASSES)
    assert set(ge.CLASSES) - ge.GRASP_OK_CLASSES == {"never_reached",
                                                     "closed_on_air"}


# ---------------------------------------------------------------------------
# the three Level A evidence signals, each on its own
# ---------------------------------------------------------------------------

def test_level_a_fires_on_pad_load_alone(tmp_path):
    ep = make_episode(tmp_path, "ep_i", closes=[(6.0, 1e9)],
                      contact=(6.2, 7.0, 4.0), area=(0, 0, 0), success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.grasp_success is True
    assert "load" in r.contact_a_evidence
    assert "area" not in r.contact_a_evidence


def test_level_a_fires_on_contact_area_alone(tmp_path):
    """Load stays under the threshold; the vendor area alone carries it."""
    ep = make_episode(tmp_path, "ep_j", closes=[(6.0, 1e9)], contact=None,
                      area=(6.2, 8.0, 3.5), success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "contact_no_hold"
    assert r.grasp_success is True
    assert r.contact_a_evidence.startswith("area")


def test_level_a_fires_on_the_gobj_stall_bit_alone(tmp_path):
    """No pad load, no area: the fingers stalled on something."""
    ep = make_episode(tmp_path, "ep_k", closes=[(6.0, 1e9)], contact=None,
                      area=(0, 0, 0), obj2=(6.5, 12.0), success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "contact_no_hold"
    assert r.grasp_success is True
    assert "gOBJ2" in r.contact_a_evidence


def test_level_a_threshold_is_respected(tmp_path):
    """A load just under the Level A threshold is not evidence."""
    ep = make_episode(tmp_path, "ep_l", closes=[(6.0, 1e9)],
                      contact=(6.2, 8.0, 1.0), area=(0, 0, 0), success=False)
    r = ge.finalise(ge.analyse_episode(ep, contact_a_n=2.5), 16.0)
    assert r.cls == "closed_on_air"
    assert r.grasp_success is False


# ---------------------------------------------------------------------------
# the four easy-to-get-wrong details
# ---------------------------------------------------------------------------

def test_left_pad_standing_offset_is_zeroed(tmp_path):
    """A 6 N standing bias on the left pad must not read as contact."""
    ep = make_episode(tmp_path, "ep_m", closes=[(6.0, 1e9)], contact=None,
                      area=(0, 0, 0), left_offset=6.0, success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "closed_on_air"
    assert r.grasp_success is False
    assert r.closes[0].pad_left_n < ge.CONTACT_A_N


def test_terminal_veto_retry_counts_two_closes(tmp_path):
    """close -> forced open -> close again is two attempts, scored on the last."""
    ep = make_episode(tmp_path, "ep_n", closes=[(4.0, 6.0), (8.0, 1e9)],
                      contact=(8.3, 15.0, 12.0),
                      z_profile=ramp([(0, 300), (8, 80), (12, 320),
                                      (15, 120), (20, 300)]),
                      success=True)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.n_closes == 2
    assert r.closes[0].t_reopen_s is not None
    assert r.closes[0].both_contact is False      # the first close caught air
    assert r.closes[1].both_contact is True
    # the hold is scored against the SECOND close, so the lift is measured
    # from the low z the retry closed at, not the first attempt's
    assert r.t_grasp_close_s == pytest.approx(r.closes[1].t_s, abs=0.2)
    assert r.cls == "placed_clean"


def test_controller_release_record_beats_the_z_gate(tmp_path):
    """A `placement_descent.releases` count makes it a placement."""
    ep = make_episode(tmp_path, "ep_o", closes=[(6.0, 1e9)],
                      contact=(6.2, 12.0, 12.0), z_profile=PICK_DROP,
                      success=True,
                      placement={"releases": 1, "release_gate_z_m": 0.16,
                                 "released_at_z_m": 0.141,
                                 "released_at_y_m": 0.061})
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.ctrl_releases == 1
    assert r.cls == "placed_clean"    # despite losing contact at ~300 mm


def test_crush_band_is_derived_from_the_reference_arm(tmp_path):
    """mean + k*sd over the reference arm's clean placements, not a constant."""
    rows = []
    for i, n in enumerate([10.0, 11.0, 12.0, 13.0]):      # teacher clean band
        ep = make_episode(tmp_path, f"ep_t{i}", closes=[(6.0, 1e9)],
                          contact=(6.2, 14.0, n), z_profile=PICK_PLACE,
                          success=True, tags=["label:teach", f"seed:10{i}"])
        rows.append(ge.analyse_episode(ep))
    arm, peaks, mean, sd = ge.clean_band(rows)
    assert arm == "teach"
    assert len(peaks) == 4
    assert mean == pytest.approx(11.5, abs=0.3)
    crush_n = mean + 3 * sd
    assert 14.0 < crush_n < 16.0
    ge.apply_crush_threshold(rows, crush_n)
    assert all(r.cls == "placed_clean" for r in rows)
    assert all(r.haptic_success for r in rows)


def test_crush_band_excludes_operator_tagged_crushes(tmp_path):
    """A take the operator tagged `crushed` must not widen the clean band."""
    rows = []
    for i, n in enumerate([10.0, 11.0, 12.0]):
        ep = make_episode(tmp_path, f"ep_u{i}", closes=[(6.0, 1e9)],
                          contact=(6.2, 14.0, n), z_profile=PICK_PLACE,
                          success=True, tags=["label:teach", f"seed:10{i}"])
        rows.append(ge.analyse_episode(ep))
    ep = make_episode(tmp_path, "ep_u9", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 30.0), z_profile=PICK_PLACE,
                      success=True, tags=["label:teach", "seed:109", "crushed"])
    rows.append(ge.analyse_episode(ep))
    arm, peaks, mean, sd = ge.clean_band(rows)
    assert len(peaks) == 3                    # the 30 N take is excluded
    assert max(peaks) < 13.0


def test_cross_check_flags_operator_disagreement(tmp_path):
    """Operator placed it, sensors say it was dropped."""
    ep = make_episode(tmp_path, "ep_p", closes=[(6.0, 1e9)],
                      contact=(6.2, 12.0, 11.0), z_profile=PICK_DROP,
                      success=True, notes="operator: s")
    r = ge.cross_check(ge.finalise(ge.analyse_episode(ep), 16.0))
    assert r.operator_placed is True
    assert r.cls == "held_dropped"
    assert r.agree is False
    assert "operator placed it" in r.disagreement


def test_untagged_crush_is_flagged(tmp_path):
    ep = make_episode(tmp_path, "ep_q", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 22.0), z_profile=PICK_PLACE,
                      success=True)
    r = ge.cross_check(ge.finalise(ge.analyse_episode(ep), 16.0))
    assert r.crush is True
    assert r.cls == "placed_crushed"
    assert r.agree is False
    assert "no crush tag" in r.disagreement


def test_crush_tag_without_the_load_is_flagged(tmp_path):
    ep = make_episode(tmp_path, "ep_r", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 11.0), z_profile=PICK_PLACE,
                      success=True, tags=["label:testarm", "seed:101", "crushed"])
    r = ge.cross_check(ge.finalise(ge.analyse_episode(ep), 16.0))
    assert r.crush is False
    assert r.agree is False
    assert "tagged crushed" in r.disagreement


def test_missing_area_streams_are_tolerated(tmp_path):
    ep = make_episode(tmp_path, "ep_s", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 11.0), with_area=False, success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "contact_no_hold"
    assert r.closes[0].area_left == 0.0


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
    for i, n in enumerate([11.0, 12.0, 13.0]):
        make_episode(root, f"ep_x_10{i}", closes=[(6.0, 1e9)],
                     contact=(6.2, 14.0, n), z_profile=PICK_PLACE,
                     success=True, tags=["label:armB", f"seed:10{i}"])
    out_csv, out_md = tmp_path / "out.csv", tmp_path / "out.md"
    out_closes = tmp_path / "closes.csv"
    rc = ge.main([str(root), "--csv", str(out_csv), "--md", str(out_md),
                  "--closes-csv", str(out_closes), "--sweep", "--quiet"])
    assert rc == 0
    text = out_csv.read_text()
    header = text.splitlines()[0]
    # the columns the team lead asked to be added, plus the ones kept
    for col in ("class", "grasp_success", "haptic_success", "arm", "cell",
                "verdict", "n_closes", "pad_peak_n", "hold_s", "lift_mm",
                "drop", "stop_reason"):
        assert col in header.split(","), col
    assert "never_reached" in text and "placed_clean" in text
    md = out_md.read_text()
    # thresholds must be stated BEFORE the per-arm table
    assert md.index("Thresholds, fixed before the table") < md.index("Outcome by arm")
    assert "grasp_success  = contact_no_hold" in md
    assert "`armA`" in md and "`armB`" in md
    assert "grasp 3/3, haptic 3/3" in md          # armB headline
    assert "episode,arm,cell,operator,index" in out_closes.read_text()


def test_cli_rejects_an_empty_folder(tmp_path):
    assert ge.main([str(tmp_path), "--csv", str(tmp_path / "o.csv")]) == 2
