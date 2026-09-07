# Native demonstration check for an optional fixed wrist reference

Across ten native Aug22 waffles demonstrations, the rolling wrist subguard produces **0 debounced triggers**, and an episode-fixed reference produces **1**, at 13.952 s in `ep_waffles_1787396028_003`. Neither produces a debounced trigger during any initial 0–1 s interval. The later event is unadjudicated: **these are descriptive trigger counts, not false-positive rates or real-hardware safety qualification**.

[Numeric results and source hashes](native_demo_audit.json) · [reproducible CPU extractor](native_demo_audit.py)

Both rules use the unchanged 60 N / 15 N·m limits and 0.3-second debounce. The rolling reference uses a 2-second EMA and updates only when both deviations are below their limits. The fixed reference remains at the exact **first native `arm_ft` sample**, including its unknown pose/tare/current-estimator bias. This is not necessarily the simulator's later common-stream initialization anchor: for episode5928 those anchors differ by about 8 ms.

The audit reads all 26,610 native wrist samples over 212.82 seconds of recorded motion. It evaluates each rule at the stored master-clock timestamps, with no resampling or repeated calibration-offset correction. The largest wrist timestamp gap is 18.98 ms. It does not reconstruct the teleoperation system's actual online safety decisions, execute another motion, import a model/simulator, or change primary-study data or scores.

## Whole recorded streams

Times are relative to the first native wrist sample. Each pair below is **rolling / episode-fixed**. “Crossing” means an instantaneous force **or torque** threshold crossing; it does not itself trip the guard. Maxima inspect the entire unchanged recording, including samples after a hypothetical stop.

| Episode suffix | Duration (s) | First crossing (s) | First debounced trigger (s) | Maximum force deviation (N) | Maximum torque deviation (N·m) |
|---|---:|---|---|---|---|
| 5928_000 | 19.85 | 0.272 / 0.272 | — / — | 63.51 / 63.61 | 17.22 / 17.53 |
| 5963_001 | 22.22 | 9.745 / 9.705 | — / — | 55.62 / 60.35 | 15.30 / 17.48 |
| 6028_003 | 21.31 | 1.051 / 0.976 | — / **13.952** | 77.62 / 113.60 | 20.28 / 28.91 |
| 6060_004 | 22.15 | — / 18.675 | — / — | 50.63 / 56.47 | 13.37 / 15.17 |
| 6094_005 | 21.57 | — / 1.336 | — / — | 57.15 / 73.91 | 14.42 / 18.91 |
| 6128_006 | 19.12 | 0.432 / 0.432 | — / — | 62.35 / 67.12 | 16.15 / 18.31 |
| 6273_010 | 21.01 | 0.361 / 0.361 | — / — | 65.75 / 77.58 | 17.55 / 20.76 |
| 6314_011 | 21.14 | — / — | — / — | 53.03 / 56.88 | 14.30 / 14.99 |
| 6346_012 | 22.39 | 0.448 / 0.344 | — / — | 66.46 / 86.21 | 15.90 / 21.74 |
| 6461_000 | 22.04 | — / 9.564 | — / — | 50.34 / 64.68 | 13.38 / 17.56 |

The episode6028 fixed-reference trigger occurs at sample1744, master time5695.355631 s, after an uninterrupted over-limit interval beginning at relative13.649376 s. At the trigger, force/torque deviations are 72.1464 N / 18.4056 N·m. Its earlier initial crossing at0.976 s is transient. Later contact, unloaded-pose and acceleration contributions have not been independently adjudicated here; the one trigger cannot be classified as correct or false from this replay alone.

## Initial unloaded-window evidence

All 30 RGB previews near0,0.5,1 s were manually reviewed. They show no human hand or visible grasp; the packet remains on the mat. Both tactile SDK area streams are exactly zero throughout the first second in every episode. Measured TCP Z remains at least0.287 m. The JSON preserves exact image indices, actual timestamps, stored/decoded payload hashes, per-pad tactile summaries and robot/gripper motion.

These are **moving unloaded windows**, not an assertion of a stationary robot: the largest first-second TCP translation is43.46 mm, joint change0.0882 rad, and measured joint speed0.4144 rad/s. Several grippers also open during this interval. The combined evidence supports an initially unloaded distal gripper, while three sampled views and zero tactile area cannot exclude every occluded proximal contact.

The initial windows still contain instantaneous threshold crossings in **4/10 rolling** and **5/10 fixed** replays. None persists through the unchanged0.3-second debounce. This distinction matters: a high transient raw or baseline-relative wrench is not equivalent to a safety trigger. The result supports only these ten short, evidence-backed intervals and does not establish an unloaded-motion false-positive rate across all poses.

## Interpretation and reproduction

The fixed reference avoids incorporating sustained contact into a moving baseline, but these recordings show why a fixed threshold must remain an explicit, separately qualified option: the measured UR3 wrist estimate changes with motion and has unknown bias. Full unloaded-pose/acceleration coverage and adjudication of the later6028 event remain necessary before any real-hardware conclusion. The ten recordings are closely related Aug22 demonstrations; they are not an independent broad safety dataset.

Run the extractor on compute3, redirecting only to a new audit artifact:

```bash
ssh compute3 'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python -' \
  < native_demo_audit.py > native_demo_audit.json
```

Add `--with-preview` after the Python `-` to return the 30 JPEG previews as JSON payloads. Numerical extraction leaves visual adjudication pending; the saved audit's clearly marked manual-review annotations were added after viewing those hashed frames. Full native source streams remain unchanged at `/home/physicalai/phantom-icra-2027/data/full/tasks/waffles/`. No runtime or frozen primary-study source was edited by this task.
