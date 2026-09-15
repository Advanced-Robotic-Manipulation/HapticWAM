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
    # wording follows the 20:05 MSK retag: the operator flags force, the
    # threshold decides the crush, so the mismatch is named from the flag
    assert "operator flagged nothing" in r.disagreement


def test_crush_tag_without_the_load_is_flagged(tmp_path):
    ep = make_episode(tmp_path, "ep_r", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 11.0), z_profile=PICK_PLACE,
                      success=True, tags=["label:testarm", "seed:101", "crushed"])
    r = ge.cross_check(ge.finalise(ge.analyse_episode(ep), 16.0))
    assert r.crush is False
    assert r.agree is False
    assert "operator flagged excessive force" in r.disagreement


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


# ---------------------------------------------------------------------------
# under-grasp / over-grasp and the pooled crush band
# ---------------------------------------------------------------------------

def test_under_grasp_covers_touch_without_delivery(tmp_path):
    """contact_no_hold and held_dropped are under-grasps; a placement is not."""
    under = make_episode(tmp_path, "ep_ug1", closes=[(6.0, 1e9)],
                         contact=(6.2, 14.0, 3.0), success=False)
    dropped = make_episode(tmp_path, "ep_ug2", closes=[(6.0, 1e9)],
                           contact=(6.2, 12.0, 11.0), z_profile=PICK_DROP,
                           success=False)
    placed = make_episode(tmp_path, "ep_ug3", closes=[(6.0, 1e9)],
                          contact=(6.2, 14.0, 11.0), z_profile=PICK_PLACE,
                          success=True)
    a, b, c = (ge.finalise(ge.analyse_episode(e), 20.0)
               for e in (under, dropped, placed))
    assert a.cls == "contact_no_hold" and a.under_grasp is True
    assert b.cls == "held_dropped" and b.under_grasp is True
    assert c.cls == "placed_clean" and c.under_grasp is False
    # every under-grasp is a Level A success and a Level B failure
    for r in (a, b):
        assert r.grasp_success is True and r.haptic_success is False


def test_never_reached_is_not_an_under_grasp(tmp_path):
    """No close at all is a reach failure, not a grasp that was too light."""
    ep = make_episode(tmp_path, "ep_ug4", closes=[], success=False)
    r = ge.finalise(ge.analyse_episode(ep), 20.0)
    assert r.cls == "never_reached"
    assert r.under_grasp is False and r.over_grasp is False


def test_held_at_cut_is_not_an_under_grasp(tmp_path):
    """The recording ended mid-carry, so non-delivery was never observed."""
    ep = make_episode(tmp_path, "ep_ug5", closes=[(6.0, 1e9)],
                      contact=(6.2, 19.9, 11.0),
                      z_profile=ramp([(0, 300), (6, 80), (12, 320), (20, 320)]),
                      success=True)
    r = ge.finalise(ge.analyse_episode(ep), 20.0)
    assert r.cls == "held_at_cut"
    assert r.under_grasp is False


def test_over_grasp_fires_on_a_take_that_never_placed(tmp_path):
    """A hard squeeze with no placement still counts as an over-grasp.

    This is the case the placement-only crush flag misses: `pad_peak_n` is 0
    because there is no hold window, so the count has to come from the pinch
    peak instead.
    """
    # a hard squeeze too brief to register as a hold, so there is no hold
    # window and `pad_peak_n` stays 0 — exactly the dp cell 12 shape
    ep = make_episode(tmp_path, "ep_og1", closes=[(6.0, 1e9)],
                      contact=(6.2, 6.4, 22.0), success=False)
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.cls == "contact_no_hold"          # squeezed, never lifted it
    assert r.hold_s < ge.MIN_HOLD_S
    assert r.pad_peak_n == 0.0                 # no hold window to measure
    assert r.pad_peak_pinch_n > 16.0
    assert r.over_grasp is True
    assert r.crush is False                    # crush stays a placement flag


def test_over_grasp_ignores_a_one_sided_push(tmp_path):
    """One pad loaded to 25 N with the other at zero is not a pinch."""
    ep = make_episode(tmp_path, "ep_og2", closes=[(6.0, 1e9)],
                      area=(0, 0, 0), left_offset=0.0, success=False)
    # rewrite the left pad alone with a hard one-sided load
    g = zarr.open(str(ep / "tactile_left_wrench.zarr"), mode="r+")
    ts = np.asarray(g["ts"][:])
    data = np.asarray(g["data"][:])
    data[(ts - T0 >= 6.2) & (ts - T0 <= 14.0), 2] = -25.0
    g["data"][:] = data
    r = ge.finalise(ge.analyse_episode(ep), 16.0)
    assert r.pad_peak_take_n > 20.0            # the raw peak is there
    assert r.pad_peak_pinch_n == 0.0           # but the right pad never engaged
    assert r.over_grasp is False


def test_pooled_band_uses_every_arm(tmp_path):
    """The band is the pooled clean placements, and per-arm means come out."""
    rows = []
    for i, n in enumerate([10.0, 11.0, 12.0]):
        ep = make_episode(tmp_path, f"ep_pb_t{i}", closes=[(6.0, 1e9)],
                          contact=(6.2, 14.0, n), z_profile=PICK_PLACE,
                          success=True, tags=["label:teach", f"seed:10{i}"])
        rows.append(ge.analyse_episode(ep))
    for i, n in enumerate([14.0, 15.0, 16.0]):
        ep = make_episode(tmp_path, f"ep_pb_s{i}", closes=[(6.0, 1e9)],
                          contact=(6.2, 14.0, n), z_profile=PICK_PLACE,
                          success=True, tags=["label:stud", f"seed:11{i}"])
        rows.append(ge.analyse_episode(ep))
    peaks, mean, sd, per_arm = ge.pooled_band(rows)
    assert len(peaks) == 6
    assert mean == pytest.approx(13.0, abs=0.3)
    assert per_arm["teach"][0] == 3 and per_arm["stud"][0] == 3
    assert per_arm["teach"][1] < per_arm["stud"][1]
    # the single-arm band would be ~3 N lower and would flag the other arm
    _, _, t_mean, t_sd = ge.clean_band(rows, "teach")
    assert t_mean + 3 * t_sd < max(peaks)
    assert mean + 3 * sd > max(peaks)


def test_crush_sensitivity_table_brackets_the_chosen_value(tmp_path):
    rows = []
    for i, n in enumerate([11.0, 12.0, 13.0, 19.0]):
        ep = make_episode(tmp_path, f"ep_cs{i}", closes=[(6.0, 1e9)],
                          contact=(6.2, 14.0, n), z_profile=PICK_PLACE,
                          success=True, tags=["label:armA", f"seed:10{i}"])
        rows.append(ge.analyse_episode(ep))
    md = ge.crush_sensitivity_table(rows, 18.0)
    assert "16.0" in md and "18.0 **(chosen)**" in md and "20.0" in md
    # the rows are left classified at the chosen value, not at the last probe
    ge.apply_crush_threshold(rows, 18.0)
    assert sum(r.over_grasp for r in rows) == 1


# ---------------------------------------------------------------------------
# the 20:05 MSK retag: `crushed` -> advisory `op_crushed`
# ---------------------------------------------------------------------------

def test_op_crushed_is_advisory_not_a_verdict(tmp_path):
    """`op_crushed` leaves the success verdict alone but flags the force."""
    ep = make_episode(tmp_path, "ep_oc1", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 11.0), z_profile=PICK_PLACE,
                      success=True,
                      tags=["label:testarm", "seed:101", "op_crushed"])
    r = ge.finalise(ge.analyse_episode(ep), 20.0)
    assert r.verdict == "s"            # NOT "crushed": success still stands
    assert r.op_force_flag is True
    assert r.operator_placed is True
    # the older tag vintage still overrides the verdict
    ep2 = make_episode(tmp_path, "ep_oc2", closes=[(6.0, 1e9)],
                       contact=(6.2, 14.0, 11.0), z_profile=PICK_PLACE,
                       success=True,
                       tags=["label:testarm", "seed:102", "crushed"])
    r2 = ge.finalise(ge.analyse_episode(ep2), 20.0)
    assert r2.verdict == "crushed" and r2.op_force_flag is True


def test_the_band_is_sensor_only_by_default(tmp_path):
    """The operator's eye must not shape the threshold it is compared against.

    Default: every classified placement is in the band sample, flagged or not.
    `exclude_flagged=True` is the stricter opt-in sample.
    """
    rows = []
    for i, n in enumerate([10.0, 11.0, 12.0]):
        ep = make_episode(tmp_path, f"ep_ocb{i}", closes=[(6.0, 1e9)],
                          contact=(6.2, 14.0, n), z_profile=PICK_PLACE,
                          success=True, tags=["label:teach", f"seed:10{i}"])
        rows.append(ge.analyse_episode(ep))
    ep = make_episode(tmp_path, "ep_ocb9", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 30.0), z_profile=PICK_PLACE,
                      success=True,
                      tags=["label:teach", "seed:109", "op_crushed"])
    rows.append(ge.analyse_episode(ep))
    peaks, _, _, _ = ge.pooled_band(rows)
    assert len(peaks) == 4 and max(peaks) == 30.0     # sensor-only default
    strict, _, _, _ = ge.pooled_band(rows, exclude_flagged=True)
    assert len(strict) == 3 and max(strict) < 13.0


def test_force_flag_mismatch_is_reported_both_ways(tmp_path):
    """Flagged but under the band, and over the band but unflagged."""
    flagged_light = make_episode(tmp_path, "ep_fm1", closes=[(6.0, 1e9)],
                                 contact=(6.2, 14.0, 11.0),
                                 z_profile=PICK_PLACE, success=True,
                                 tags=["label:a", "seed:101", "op_crushed"])
    quiet_heavy = make_episode(tmp_path, "ep_fm2", closes=[(6.0, 1e9)],
                               contact=(6.2, 14.0, 24.0),
                               z_profile=PICK_PLACE, success=True,
                               tags=["label:a", "seed:102"])
    a = ge.cross_check(ge.finalise(ge.analyse_episode(flagged_light), 20.0))
    b = ge.cross_check(ge.finalise(ge.analyse_episode(quiet_heavy), 20.0))
    assert a.agree is False and "flagged excessive force" in a.disagreement
    assert b.agree is False and "operator flagged nothing" in b.disagreement


def test_flagged_take_the_threshold_also_calls_a_crush_agrees(tmp_path):
    """Operator flagged the force, threshold confirms it: not a disagreement.

    The take is still a Level B failure — it just is not a case where the
    sensors and the human tell different stories, which is what the
    disagreement list is for.
    """
    ep = make_episode(tmp_path, "ep_ag1", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 24.0), z_profile=PICK_PLACE,
                      success=True,
                      tags=["label:a", "seed:101", "op_crushed"])
    r = ge.cross_check(ge.finalise(ge.analyse_episode(ep), 20.0))
    assert r.cls == "placed_crushed" and r.over_grasp is True
    assert r.haptic_success is False
    assert r.agree is True and r.disagreement == ""


def test_robust_band_ignores_the_outlier_the_mean_band_chases(tmp_path):
    """mean + 3 sd is dragged up by a crush in its own sample; the MAD is not."""
    peaks = [10.0, 11.0, 12.0, 13.0, 14.0]
    import numpy as np
    clean_mean = float(np.mean(peaks)) + 3 * float(np.std(peaks, ddof=1))
    med, mad_sd = ge.robust_band(peaks)
    with_outlier = peaks + [40.0]
    dirty_mean = (float(np.mean(with_outlier))
                  + 3 * float(np.std(with_outlier, ddof=1)))
    med2, mad_sd2 = ge.robust_band(with_outlier)
    # the mean band moves a long way, the robust one barely at all
    assert dirty_mean - clean_mean > 20.0
    assert abs((med2 + 3 * mad_sd2) - (med + 3 * mad_sd)) < 5.0


def test_band_gap_reports_the_empty_interval(tmp_path):
    assert ge.band_gap([9.0, 12.0, 17.7, 21.4], 20.0) == (17.7, 21.4)
    assert ge.band_gap([9.0, 12.0], 20.0) is None      # nothing above
    assert ge.band_gap([25.0, 30.0], 20.0) is None     # nothing below


def test_outcome_table_conditions_over_force_on_operator_placements(tmp_path):
    """'of which over-force' counts only takes the operator scored placed."""
    rows = []
    # operator placed, sensors say over-force
    ep = make_episode(tmp_path, "ep_ot1", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 24.0), z_profile=PICK_PLACE,
                      success=True, tags=["label:a", "seed:101"])
    rows.append(ge.analyse_episode(ep))
    # operator FAILED it, sensors also say over-force: must NOT be counted
    ep = make_episode(tmp_path, "ep_ot2", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 26.0), z_profile=PICK_PLACE,
                      success=False, tags=["label:a", "seed:102"])
    rows.append(ge.analyse_episode(ep))
    # operator placed, load inside the band
    ep = make_episode(tmp_path, "ep_ot3", closes=[(6.0, 1e9)],
                      contact=(6.2, 14.0, 11.0), z_profile=PICK_PLACE,
                      success=True, tags=["label:a", "seed:103"])
    rows.append(ge.analyse_episode(ep))
    ge.apply_crush_threshold(rows, 20.0)
    md = ge.outcome_table(rows)
    assert "placed (operator)" in md and "op_crushed" in md
    body = [l for l in md.splitlines() if l.startswith("| `a`")][0]
    n, placed, over = [c.strip() for c in body.split("|")[2:5]]
    assert (n, placed, over) == ("3", "2", "1")


def test_eye_vs_sensor_table_lists_both_kinds_of_flag(tmp_path):
    rows = []
    eye_only = make_episode(tmp_path, "ep_ev1", closes=[(6.0, 1e9)],
                            contact=(6.2, 14.0, 11.0), z_profile=PICK_PLACE,
                            success=True,
                            tags=["label:a", "seed:101", "op_crushed"])
    sensor_only = make_episode(tmp_path, "ep_ev2", closes=[(6.0, 1e9)],
                               contact=(6.2, 14.0, 24.0), z_profile=PICK_PLACE,
                               success=True, tags=["label:a", "seed:102"])
    quiet = make_episode(tmp_path, "ep_ev3", closes=[(6.0, 1e9)],
                         contact=(6.2, 14.0, 11.0), z_profile=PICK_PLACE,
                         success=True, tags=["label:a", "seed:103"])
    rows = [ge.analyse_episode(e) for e in (eye_only, sensor_only, quiet)]
    ge.apply_crush_threshold(rows, 20.0)
    md = ge.eye_vs_sensor_table(rows)
    assert "ep_ev1" in md and "ep_ev2" in md
    assert "ep_ev3" not in md          # neither flagged it
    assert "agree on 0 of these 2 takes" in md
