# F17 + E13 — start-pose stats over the FULL dataset, and the honest v4 vs v5_6 number (2026-08-30)

Two items from `VALIDATION_0830.md`: §2 P0 #7 (the STOP hitbox / z floor / joint gate were fitted to a
17–37-episode subset) and §3.5 item 4 (the quoted `20.7 → 17.4 mm` v4→v5 improvement was never reproduced
with the hardened `tools/terminal_eval.py`).

**Dataset on compute3.** Fetched from `armteam/phantom-episodes` (the packed `dataset_v3_packed` tarballs
stalled at 0.2 MB/s; the per-file path moves 110 MB instead of 97 GB):

| path | what | size |
|---|---|---|
| `~/phantom-icra-2027/data/full/tasks/<task>/ep_*` | all **1000 success episodes** (720 from `tasks/` + 280 from `archive/20260822_*`), each carrying `meta.json`, the whole `arm_tcp_pose.zarr`, and frame 0 of `arm_q.zarr` / `gripper.zarr` — everything `gen_start_poses.py` reads and **nothing else** | ~110 MB |
| `~/phantom-icra-2027/data/val124/tasks` + `manifests/all.jsonl` | the real v5 val split, 124 episodes, symlinks (78 → `data/val_eval`, 46 full episodes → `data/full/raw/archive`) | ~4.7 GB |

The per-task episode dirs under `data/full/tasks` are **derived-stats only** — they are not complete
episodes and must not be pointed at a trainer or at `terminal_eval`. `data/val124` is complete.

---

## 1. Start-pose envelope: subset (2026-08-28) vs full 250/task

`tools/gen_start_poses.py` (rewritten: one episode set behind every number, `q_n == n` emitted,
failure demos excluded even when symlinked into a success task dir) over all 250 success episodes per
task — 180 v4 + 70 `batch_20260822`, exactly the set `tcp_mean`/`tcp_std` already used.

**Sanity check on the pipeline:** every task's `tcp_mean` / `tcp_std` / `gripper_mean` / `gripper_std`
came out **bit-for-bit identical to the shipped file**. Only the blocks that had been computed on the
subset moved.

### STOP hitbox (`tcp_min` / `tcp_max`, mm) — bold = moved > 10 mm

| task | n (tcp) | q_n old → new | axis | old min/max | new min/max | Δmin | Δmax |
|---|---|---|---|---|---|---|---|
| Carton | 250 | 20 → 250 | x | −511.1 / −322.0 | −518.0 / −310.0 | −6.9 | **+12.0** |
|  |  |  | y | −336.5 / +122.9 | −365.0 / +150.7 | **−28.5** | **+27.8** |
|  |  |  | z | 76.1 / 405.2 | 76.1 / 475.5 | +0.0 | **+70.3** |
| egg | 250 | 17 → 250 | x | −432.2 / −284.0 | −440.3 / −255.7 | −8.1 | **+28.3** |
|  |  |  | y | −282.4 / −2.0 | −335.6 / +60.2 | **−53.2** | **+62.2** |
|  |  |  | z | 59.5 / 320.9 | 59.5 / 378.0 | +0.0 | **+57.1** |
| waffles | 250 | 37 → 250 | x | −468.5 / −289.6 | −468.5 / −250.6 | +0.0 | **+39.0** |
|  |  |  | y | −349.4 / +145.8 | −376.5 / +145.8 | **−27.1** | +0.0 |
|  |  |  | z | 52.0 / 410.1 | 41.5 / 472.1 | **−10.5** | **+62.0** |
| whiteboard | 250 | 21 → 250 | x | −420.0 / −262.5 | −454.6 / −229.7 | **−34.6** | **+32.8** |
|  |  |  | y | −313.9 / +81.6 | −373.4 / +165.5 | **−59.5** | **+83.9** |
|  |  |  | z | 74.6 / 364.4 | 67.5 / 412.6 | −7.1 | **+48.2** |

**Every task is flagged.** 17 of the 24 envelope faces moved more than 10 mm, and **every one of them moved
outward** — the subset hitbox was too tight on all four tasks, worst at whiteboard +y (+83.9 mm) and
Carton +z (+70.3 mm). `run_deploy` adds `--hitbox-margin 0.03` on top, so the shipped 30 mm margin was
being consumed (and in five faces exceeded) by sampling error alone: a legitimate demo-like trajectory
could have tripped the STOP.

### z no-go floor (`tcp_z_min`, mm)

| task | old | new | Δ |
|---|---|---|---|
| Carton | 76.1 | 76.1 | +0.0 |
| egg | 59.5 | 59.5 | +0.0 |
| waffles | 52.0 | 41.5 | **−10.5** |
| whiteboard | 74.6 | 67.5 | −7.1 |

`resolve_z_floor` = `tcp_z_min − 0.01`. **waffles' floor was 10.5 mm too high** and whiteboard's 7.1 mm too
high: the executor was clamping commanded z above heights the demos actually reach. On a task whose rig
failure mode is *closing 65–120 mm too high*, a floor that is itself too high is the wrong direction of error.
Carton and egg are unchanged — their subset happened to contain the deepest episode.

### start joint configuration (`q_mean` / `q_std`)

Largest mean shifts: egg q4 **+11.1°**, egg q5 **−9.6°**, waffles q4 **−8.4°**, Carton q5 **+8.2°**,
egg q3 −6.0°, Carton q1 +5.7°, waffles q3 +5.6°, egg q1 −5.3°. Every other joint moved < 5°.
`q2` (shoulder) and `q6` (wrist 3) barely moved on any task, and `q_mean[5]` stays near −π everywhere, so
the 2026-08-28 wrist-wrap diagnosis is unaffected. The std pattern is not uniform: Carton's tightened
(20 episodes had over-clustered), whiteboard's roughly doubled (q1 4.3° → 7.8°, q3 3.6° → 8.5°) — its
21-episode subset had **understated** the demo spread, so `start_sigma_report` was reporting inflated
sigma and `--max-start-sigma` would refuse legitimate starts.

`phantom/deploy/start_pose.load_start_stats` now raises `ThinStartStatsError` when a file carries a
joint/envelope/floor block whose `q_n` disagrees with `n` (`allow_thin_q=True` downgrades it to a loud
warning, for offline analysis only). The check runs after the existing shape assertions, so a
half-specified block still fails exactly the way it did.

---

## 2. E13 — v4 vs v5_6 on the REAL val split

`tools/terminal_eval.py` (hardened: manifest hard-fail, all episodes, medians, `pred_close_height_mm`),
`--seeds 4`, EMA weights, NFE 5, guidance 1.0, on compute3:

    ./.venv/bin/python tools/terminal_eval.py --ckpt runs/teacher_v4_790eps/teacher_020000.pt \
        --data ~/phantom-icra-2027/data/val124/tasks --hardware configs/hardware.nuc.yaml \
        --seeds 4 --split val --out /tmp/e13_val124_v4.json     # and the same with v5_6.pt

Two splits are reported. **val124** is the true v5 validation set — the frozen v4 78 plus the 46
`batch_20260822` holdout episodes (`intake_holdout.json`), which is exactly the set the quoted
`20.7 → 17.4` was measured on. **val78** is the frozen v4 half alone, i.e. the only episodes v5's
fine-tune data could not have leaked into by session.

### Overall (mean over all terminal windows)

| split | run | eps | windows | endpoint err mm | z-at-end err mm | commit ratio | close step err | pred close height mm |
|---|---|---|---|---|---|---|---|---|
| val124 | v4 | 124 | 496 | 20.82 | +2.76 | 1.28 | +0.19 | 117.44 |
| val124 | **v5_6** | 124 | 496 | **18.23** | +4.16 | 1.10 | −0.67 | 115.30 |
| val78 | v4 | 78 | 312 | 22.17 | +6.93 | 1.18 | +0.36 | 112.50 |
| val78 | **v5_6** | 78 | 312 | **20.86** | +8.53 | 0.94 | −0.60 | 110.80 |

### Medians (same runs; GT close height is a property of the demos)

| split | run | endpoint | z-at-end | commit | close step | pred close h | GT close h |
|---|---|---|---|---|---|---|---|
| val124 | v4 | 17.20 | −0.90 | 1.02 | +2.0 | 116.14 | 98.11 |
| val124 | v5_6 | 15.74 | +1.45 | 0.89 | +1.0 | 110.03 | 98.11 |
| val78 | v4 | 18.77 | +2.06 | 0.89 | +2.0 | 107.36 | 95.56 |
| val78 | v5_6 | 17.99 | +3.71 | 0.76 | +1.0 | 102.21 | 95.56 |

### Per task — endpoint err mm

| split | run | Carton | egg | waffles | whiteboard |
|---|---|---|---|---|---|
| val124 | v4 | 27.51 | 8.50 | 29.14 | 17.65 |
| val124 | v5_6 | 21.07 | 7.20 | 26.18 | 17.53 |
| val78 | v4 | 28.60 | 8.28 | 30.56 | 19.30 |
| val78 | v5_6 | 25.78 | 7.59 | 29.54 | 18.63 |

### Per task — close step err / commit ratio

| split | run | Carton | egg | waffles | whiteboard |
|---|---|---|---|---|---|
| val124 | v4 | −0.91 / 1.19 | −0.61 / 1.96 | +1.49 / 1.26 | +0.39 / 1.08 |
| val124 | v5_6 | −0.82 / 1.39 | −3.26 / 1.17 | +0.93 / 0.97 | −0.39 / 0.95 |
| val78 | v4 | +0.41 / 1.71 | −0.81 / 1.51 | +0.86 / 0.63 | +0.71 / 1.00 |
| val78 | v5_6 | +0.34 / 1.46 | −3.52 / 0.88 | +0.21 / 0.51 | −0.06 / 0.89 |

Mean across-seed std of endpoint err (per window, 4 seeds): val124 v4 7.04 mm / v5_6 6.01 mm;
val78 v4 6.39 mm / v5_6 5.74 mm.

---

## 3. What replaces "20.7 → 17.4 mm"

**The correct statement is `20.8 mm -> 18.2 mm` on the 124-episode val split (v4 `teacher_020000.pt`
vs v5_6, hardened `terminal_eval.py`, 4 seeds, EMA) — a 2.6 mm / 12% improvement in terminal endpoint
error, not the 3.3 mm / 16% that `20.7 -> 17.4` claimed.** The v4 side reproduces almost exactly
(20.82 vs the quoted 20.69); the difference is entirely on the v5 side (18.23 vs 17.44), i.e. the
original number was measured with 2 seeds and flattered v5_6 by ~0.8 mm. On the frozen v4-only half
(val78) the gap narrows further, to **22.2 -> 20.9 mm (1.3 mm, 5.9%)** — the larger val124 gain is
carried by the 46 `batch_20260822` holdout episodes, which are same-day, same-rig-geometry siblings of
the 280 recovery episodes v5 was fine-tuned on. The honest reading is that v5_6 is *slightly* better
than v4 offline, and most of the visible gain is in-distribution to the new batch.

Two caveats that matter more than the headline. (1) The per-window across-seed std is 6-7 mm — larger
than the entire v4→v5_6 difference — so a single-seed comparison of these checkpoints is meaningless;
the means above are over 496 (or 312) windows and are separated, but any per-episode claim is not.
(2) **Neither checkpoint solves the rig failure.** Predicted close height is 115-117 mm on val124 while
the demos close at 98 mm: both models still commit the grasp ~17-19 mm above where the demos do, and
v5_6 buys only 2 mm of that. That is the same direction and roughly the same magnitude as the 3-6 cm
closed-loop miss seen on the rig, and it is not fixed by the fine-tune. v5_6's real change is in
*timing*, not height: close-step error goes from +0.19 to −0.67 steps (median +2 → +1) and commit ratio
from 1.28 to 1.10 — it stops over-descending and closes earlier, which is the intended FT-A effect, but
the absolute height it closes at is essentially unchanged.

Practical consequence for the A/B on the rig: v5_6 vs v4 is worth running, but a 2.6 mm offline endpoint
difference should not be expected to show up as a large success-rate difference. The decision-grade
signal has to come from the closed-loop trace decomposition, not from this number.
