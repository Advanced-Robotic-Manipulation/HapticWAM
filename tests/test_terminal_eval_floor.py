"""terminal_eval: no-motion floor and head-window metrics per scored window."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terminal_eval as TE  # noqa: E402


def _chunk(dz_per_step, close_at=8, H=16):
    a = np.zeros((H, 7))
    a[:, 2] = dz_per_step
    a[close_at:, 6] = 0.6
    return a


def test_zero_floor_is_the_gt_displacement_and_head_uses_the_executed_steps():
    gt = _chunk(-0.010)                      # descends 160 mm over 16 steps
    pr = _chunk(-0.005)                      # half the descent
    r = TE.score_window(gt, pr, z0=0.2, head_steps=10)
    assert TE.HEAD_STEPS == 8 and TE.score_window(gt, pr, 0.2)["head_steps"] == 8
    assert r["zero_endpoint_err_mm"] == 160.0          # never moving misses by the full GT travel
    assert r["endpoint_err_mm"] == 80.0
    assert r["head_steps"] == 10 and np.isclose(r["head_endpoint_err_mm"], 50.0) and np.isclose(r["head_z_err_mm"], 50.0)
    assert np.isclose(r["commit_ratio"], 0.5) and np.isclose(r["head_commit_ratio"], 0.5)
    # a still policy scores exactly the floor
    still = TE.score_window(gt, np.zeros_like(gt), z0=0.2)
    assert still["endpoint_err_mm"] == still["zero_endpoint_err_mm"] == 160.0
    # head_steps is clipped to the chunk
    assert np.isclose(TE.score_window(gt, pr, 0.2, head_steps=99)["head_endpoint_err_mm"], 80.0)


def test_floor_block_ratio_and_worse_than_zero_rate():
    rows = [{"endpoint_err_mm": 10.0, "zero_endpoint_err_mm": 40.0},
            {"endpoint_err_mm": 50.0, "zero_endpoint_err_mm": 40.0},   # worse than not moving
            {"endpoint_err_mm": float("nan"), "zero_endpoint_err_mm": 40.0}]
    fb = TE.floor_block(rows)
    assert fb["ratio_to_floor"] == 60.0 / 80.0 and fb["worse_than_zero_rate"] == 0.5
    assert np.isnan(TE.floor_block([])["ratio_to_floor"])


def test_summary_carries_floor_metrics_per_task():
    gt = _chunk(-0.010)
    rows = []
    for ep, task in (("e1", "waffles"), ("e2", "egg")):
        for seed in range(2):
            rows.append({"episode": ep, "task": task, "t0": 1.0, "seed": seed,
                         **TE.score_window(gt, _chunk(-0.005), 0.2, head_steps=10)})
    s = TE.summarize(rows, nfe=1)
    assert s["n"] == 4 and s["n_windows"] == 2 and s["n_episodes"] == 2
    assert np.isclose(s["zero_endpoint_err_mm"], 160.0) and np.isclose(s["ratio_to_floor"], 0.5)
    assert s["per_task"]["egg"]["worse_than_zero_rate"] == 0.0
    assert np.isclose(s["head_endpoint_err_mm"], 50.0)
