"""`--windows`: a baseline row must land on the SAME window as a PHANTOM row.

The paper's baseline table only means something if the pi0.5 / Diffusion Policy
/ X-VLA rows were scored on the windows `tools/terminal_eval.py` scored, not on
a fresh sample of the same split. `tools/pi05_offline_eval.py --windows` maps
each (episode, t0) of a terminal_eval json onto a LeRobot frame; this pins the
map and the row identity `phantom/eval/stats.py offline` joins on.

No lerobot here -- the tool imports it inside `main()`, so the mapping and the
row schema are testable on a laptop with fakes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "pi05_offline_eval", ROOT / "tools" / "pi05_offline_eval.py")
oe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oe)

FPS = 10


def episode_meta(name="ep_A", *, index=0, n_frames=100, lo=0, t_first=1000.0,
                 dt=0.1):
    """One row of the export's phantom_episodes.json."""
    return {"episode": name, "episode_index": index, "path": f"tasks/T/{name}",
            "n_frames": n_frames, "act_index_lo": lo, "first_frame_t": t_first,
            "action_dt_mean_s": dt}


def act_ts(t_first=1000.0, dt=0.1, n=101, jitter=None):
    ts = t_first + dt * np.arange(n)
    if jitter is not None:
        ts = ts + jitter
    return ts


# --------------------------------------------------------------- the mapping
@pytest.mark.parametrize("k", [0, 1, 7, 40, 83])
def test_anchor_on_a_tick_maps_to_that_frame(k):
    """t0 exactly on action tick `act_index_lo + k` is LeRobot frame k.

    terminal_eval's chunk is the rows at/after t0 + 1/fps, i.e. rows
    lo+k+1 ..; export_lerobot gives frame k the action row lo+k+1.
    """
    ts = act_ts()
    m = episode_meta()
    t0 = float(ts[k])
    assert oe.frame_in_episode(m, t0, FPS, ts) == k
    assert oe.frame_in_episode(m, t0, FPS, None) == k      # mean-tick model agrees


def test_nonzero_act_index_lo_is_subtracted():
    """Episodes clipped to the camera span start at action row act_index_lo;
    frame 0 is that row, not row 0."""
    ts = act_ts()
    m = episode_meta(lo=12, t_first=float(ts[12]))
    assert oe.frame_in_episode(m, float(ts[12]), FPS, ts) == 0
    assert oe.frame_in_episode(m, float(ts[17]), FPS, ts) == 5


def test_anchor_between_ticks_takes_the_chunk_that_starts_after_it():
    """A t0 that falls mid-tick keeps the rule 'first row at/after t0 + 1/fps',
    so the scored chunk never starts BEFORE terminal_eval's."""
    ts = act_ts()
    m = episode_meta()
    t0 = float(ts[20]) + 0.06                 # 0.6 of a tick past frame 20
    assert oe.frame_in_episode(m, t0, FPS, ts) == 21
    assert oe.frame_in_episode(m, t0, FPS, None) == 21


def test_real_tick_rate_is_not_the_nominal_one():
    """The rig records at ~9.9 Hz, not 10. Mapping through the nominal fps
    would slide ~1 frame per 10 s; the per-episode mean tick does not."""
    dt = 0.10107
    ts = act_ts(dt=dt, n=200)
    m = episode_meta(n_frames=199, dt=dt)
    k = 150
    assert oe.frame_in_episode(m, float(ts[k]), FPS, ts) == k
    assert oe.frame_in_episode(m, float(ts[k]), FPS, None) == k
    naive = round((float(ts[k]) - m["first_frame_t"]) * FPS)
    assert naive != k                          # the nominal grid is off by now


def test_exact_map_survives_jitter_that_moves_the_mean_model():
    """With a tick that stalls mid-episode the exact map follows the recording;
    this is why --episodes-root exists."""
    dt = 0.1
    jit = np.zeros(101)
    jit[50:] = 0.05                            # one 1.5x tick at frame 50
    ts = act_ts(dt=dt, jitter=jit)
    m = episode_meta()
    assert oe.frame_in_episode(m, float(ts[80]), FPS, ts) == 80


def test_unmappable_windows_are_skipped_and_counted():
    ts = act_ts()
    eps = [episode_meta("ep_A", index=0, n_frames=100)]
    wins = [
        {"episode": "ep_A", "t0": float(ts[10])},        # fine
        {"episode": "ep_MISSING", "t0": float(ts[10])},  # not in this export
        {"episode": "ep_A", "t0": float(ts[0]) - 5.0},   # before the export
        {"episode": "ep_A", "t0": float(ts[95])},        # chunk runs off the end
    ]
    mapped, skipped = oe.map_terminal_windows(wins, eps, fps=FPS, horizon=16,
                                              act_ts={"ep_A": ts})
    assert [w["frame_in_ep"] for w in mapped] == [10]
    assert [s["episode"] for s in skipped] == ["ep_MISSING", "ep_A", "ep_A"]
    assert "not in this LeRobot export" in skipped[0]["reason"]
    assert "before" in skipped[1]["reason"]
    assert "past the episode end" in skipped[2]["reason"]
    assert len(mapped) + len(skipped) == len(wins)


def test_global_frame_index_offsets_by_the_episode_start():
    """The index handed to LeRobot is dataset_from_index[ep] + frame_in_ep."""
    ts = act_ts()
    eps = [episode_meta("ep_A", index=0), episode_meta("ep_B", index=1)]
    mapped, _ = oe.map_terminal_windows(
        [{"episode": "ep_B", "t0": float(ts[9])}], eps, fps=FPS, horizon=16,
        act_ts={"ep_B": ts})
    froms = [0, 100]                            # ds.meta.episodes["dataset_from_index"]
    assert froms[mapped[0]["episode_index"]] + mapped[0]["frame_in_ep"] == 109


# ------------------------------------------------------- the windows file
def test_load_terminal_windows_dedupes_seeds(tmp_path):
    """496 rows = 124 windows x 4 seeds -> 124 windows, seeds [0,1,2,3]."""
    rows = [{"episode": f"ep_{e}", "t0": 1000.0 + e, "seed": s,
             "endpoint_err_mm": 1.0}
            for e in range(124) for s in range(4)]
    f = tmp_path / "v6_teacher.json"
    f.write_text(json.dumps({"summary": {}, "rows": rows}))
    wins, seeds = oe.load_terminal_windows(f)
    assert len(wins) == 124 and seeds == [0, 1, 2, 3]
    assert wins[0] == {"episode": "ep_0", "t0": 1000.0}


def test_load_terminal_windows_accepts_a_bare_list(tmp_path):
    f = tmp_path / "rows.json"
    f.write_text(json.dumps([{"episode": "ep_A", "t0": 5.0, "seed": 2}]))
    wins, seeds = oe.load_terminal_windows(f)
    assert wins == [{"episode": "ep_A", "t0": 5.0}] and seeds == [2]


# ----------------------------------------------------------- the row schema
def chunk(dx=0.0, dz=0.0, grip=0.0, h=16):
    a = np.zeros((h, 7))
    a[:, 0] = dx
    a[:, 2] = dz
    a[:, 6] = grip
    return a


def test_endpoint_metrics_match_terminal_evals_definition():
    """base_row + head_metrics reproduce tools/terminal_eval.py score_window."""
    from tools.terminal_eval import score_window                       # noqa: E402

    rng = np.random.default_rng(0)
    gt, pred = rng.normal(0, 0.01, (16, 7)), rng.normal(0, 0.01, (16, 7))
    ours = {**oe.base_row(0, gt, pred), **oe.head_metrics(gt, pred, 8)}
    theirs = score_window(gt, pred, z0=0.1, head_steps=8)
    for k in ("endpoint_err_mm", "zero_endpoint_err_mm", "z_end_err_mm",
              "head_endpoint_err_mm", "head_z_err_mm", "head_steps"):
        assert ours[k] == pytest.approx(theirs[k]), k


def test_zero_endpoint_is_the_no_motion_floor():
    gt = chunk(dz=-0.002)                       # 2 mm down per step, 16 steps
    row = oe.base_row(0, gt, np.zeros_like(gt))
    assert row["zero_endpoint_err_mm"] == pytest.approx(32.0)
    assert row["endpoint_err_mm"] == pytest.approx(row["zero_endpoint_err_mm"])


def test_row_carries_the_keys_stats_offline_joins_on():
    """phantom/eval/stats.py window_key = (episode, round(t0,4), seed); a row
    missing any part is never paired."""
    from phantom.eval.stats import window_key                          # noqa: E402

    meta = {"episode": "ep_A", "t0": 30000.666897426498, "episode_index": 3,
            "frame_in_ep": 34}
    gt, pred = chunk(dz=-0.002), chunk(dz=-0.001)
    row = {"episode": meta["episode"], "t0": meta["t0"], "seed": 2,
           "episode_index": meta["episode_index"], "frame_in_ep": meta["frame_in_ep"],
           **oe.base_row(107, gt, pred), **oe.head_metrics(gt, pred, 8)}
    assert window_key(row) == ("ep_A", 30000.6669, 2)
    assert window_key(row) != (None,)
    for k in ("endpoint_err_mm", "zero_endpoint_err_mm", "head_endpoint_err_mm",
              "steps_scored", "frame"):
        assert k in row
    assert json.loads(json.dumps(row))["seed"] == 2


def test_short_chunk_policies_are_scored_over_their_own_length():
    """Diffusion Policy plays 15 of the 16 steps (n_obs_steps 2); the row says
    so, and the GT is truncated to match -- never padded."""
    gt = chunk(dz=-0.002, h=16)
    pred = chunk(dz=-0.002, h=15)
    row = oe.base_row(0, gt[:15], pred)
    assert row["steps_scored"] == 15
    assert row["endpoint_err_mm"] == pytest.approx(0.0, abs=1e-9)
    assert row["zero_endpoint_err_mm"] == pytest.approx(30.0)


def test_old_mode_row_schema_is_unchanged():
    """The non --windows row must stay byte-identical: same keys, same order."""
    gt, pred = chunk(dz=-0.002), chunk(dz=-0.001)
    assert list(oe.base_row(5, gt, pred)) == [
        "frame", "steps_scored", "endpoint_err_mm", "z_end_err_mm",
        "rot_end_err_deg", "grip_mae", "close_step_err", "zero_endpoint_err_mm"]
