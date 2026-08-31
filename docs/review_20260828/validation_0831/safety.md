# LENS: SAFETY WITH THE NEW ENVELOPE — re-validation at HEAD 820b4eb (2026-08-31)

Everything below was produced on this Mac with `.venv/bin/python` against the REAL
`configs/start_poses.yaml` + `configs/hardware.nuc.yaml`, driving the real
`SafetyMonitor.check` / `apply_hitbox` / `apply_z_floor` / `envelope_conflict` /
`ChunkExecutor` / `PlannerLoop.run` / `run_deploy.main`. Scripts:
`.../validate2/scratch/safety/{envelope,runaway,qgate,pacing,chatter,caps,cli,numberline}.py`.
Repo untouched.

Baseline: `pytest tests/test_safety.py test_control_safety.py test_rig_safety_0828.py
test_start_pose.py test_start_poses_full_0830.py test_deploy_fixnow_0830.py
test_deploy_levers.py -q` → **128 passed in 63.3 s**.

## 0. The envelope as it will actually be applied (real code, default flags)

```
workspace(nuc) x=(-0.7,0.15) y=(-0.5,0.3) z=(0.03,0.8)   # comment: "CALIBRATE to the real table"

task         zmin  clampfloor  hbfloor    gap  bandtop            hb_x                 hb_y   hb_zhi conflict
Carton       76.1        66.1     46.1   20.0    144.0 [-548.0,-280.0] [-395.0, 180.7]   505.5 NONE
egg          59.5        49.5     30.0   19.5    101.0 [-470.3,-225.7] [-365.6,  90.2]   408.0 NONE
waffles      41.5        31.5     30.0    1.5    103.0 [-498.5,-220.6] [-406.5, 175.8]   502.1 NONE
whiteboard   67.5        57.5     37.5   20.0    181.0 [-484.6,-199.7] [-403.4, 195.5]   442.6 NONE
```
`hbfloor` for waffles/egg is pinned by the workspace bound (0.03), not by the demo
envelope — `tcp_z_min − 30 mm` would be 11.5 / 29.5 mm. The floor-is-a-clamp
invariant therefore holds on waffles by **1.5 mm**.

## 1. Would each 08-28 runaway still be stopped/clamped by the NEW box?

`runaway.py` runs `SafetyMonitor.check()` (clamp-then-hitbox, the F-batch order) on
each recorded excursion, OLD envelope vs NEW, for all four tasks:

| runaway | task run on | OLD | NEW | where it stops / lands |
|---|---|---|---|---|
| y → +355 mm | all 4 | STOP `hitbox_exit` | **STOP `hitbox_exit`** | y is clamped to the workspace 300 mm, which is still outside every hitbox: first refusal at y > 175.8 (waffles) / 180.7 (Carton) / 90.2 (egg) / 195.5 (whiteboard) mm. NEW lets the arm travel **+23 to +62 mm further in +y** than OLD before stopping. |
| z → +673 mm | all 4 | STOP `hitbox_exit` | **STOP `hitbox_exit`** | first refusal at z > 502.1 (waffles) / 505.5 (Carton) / 408.0 (egg) / 442.6 (whiteboard) mm; **+48 to +70 mm higher** than OLD. |
| whiteboard z = 34 mm dive | whiteboard | CLAMP at 64.6 mm | **CLAMP at 57.5 mm** | still prevented, 7.1 mm deeper. |
| waffles z = 63 mm "table slam" | waffles | **not stopped** (floor 42) | **not stopped** (floor 31.5) | 63 mm is above both floors and inside the hitbox → `ok`, no event. The floor has never covered this event on waffles, old or new. |

Cross-task: on **waffles** a commanded z = 34 mm is now let through unclamped
(`-> ok []`), where the OLD envelope clamped it at 42 mm.

## 2. Is a 31.5 mm clamp floor above the physical table?

**No evidence that it is, and the statistic it is built from is a sampling artefact.**

`tools/gen_start_poses.py:82` computes `tcp_z_min = Z.min()` where `Z` is the
per-episode whole-trajectory z minimum, i.e. the **minimum of 250 minima**, and
`run_deploy.resolve_z_floor:353` subtracts a further 10 mm.

Per-task number line (mm) — demo distributions from
`docs/review_20260828/rig_0828/joint_ood_and_demo_close_stats.txt` (the val_eval
subset, n = 17-21, the only per-episode distributions in the repo):

```
task        hitbox  clamp  zmin250  zmin_min(n)  zmin mean±sd  close_min  close_mean  Z_MAX/band
Carton        46.1   66.1     76.1      76.1(17)  100.3±11.8       78.6      103.6       144.0
egg           30.0   49.5     59.5      59.5(17)   68.8± 4.4       61.1       71.9       101.0
waffles       30.0   31.5     41.5      52.0(19)   67.3± 9.8       54.6       71.9       103.0
whiteboard    37.5   57.5     67.5      74.6(21)   80.1± 3.4      148.7      156.2       181.0
```
The clamp floor now sits **20.5 mm below** the deepest of the 19 waffles demos and
**17.1 mm below** the deepest of the 21 whiteboard demos (Carton/egg are unchanged at
10.0 mm, the margin). The 08-28 waffles slam bottomed at **63 mm**, i.e. 31.5 mm
above the new floor.

Order-statistic check (`numberline.py`), waffles per-episode z_min ~ N(67.3, 9.8):
```
E[min of N]:  N=19 → 51.2 (observed 52.0)   N=37 → 48.3
              N=250 → 41.3 (observed 41.5)  N=1000 → 37.0 (floor 27.0 — BELOW the 30 mm workspace bound)
              N=2000 → 35.1 (floor 25.1)
```
The observed 41.5 mm matches E[min of 250] to 0.2 mm. `tcp_z_min` is therefore not a
physical bound; it is an extremum that keeps falling as the dataset grows, and
subtracting a fixed 10 mm from it makes the *protection* shrink with dataset size.

Exercised consequence (`pacing.py`, real `ChunkExecutor` + real waffles envelope, mock arm):
```
nfe1 172ms, 4x-aggressive descent   submitted=34  final_z=31.50mm  stop=None  events=['workspace_clamp']
nfe1, aggressive + --max-tcp-speed .15  final_z=31.50mm  stop=None  events=['workspace_clamp']
```
i.e. an over-committing policy is pinned at exactly 31.5 mm and **held there,
commanding into the surface, for the remainder of the episode** (`cmd_speed med=0.0000`
in `chatter.py`), with only the CB3 wrench guard (60 N / 15 Nm deviation, 0.3 s
debounce — a current-based estimate) and the uncalibrated `tactile_depth_limit 0.6`
behind it. Nothing in the repo records a measured table height in TCP z
(`hardware.nuc.yaml:253` says "CALIBRATE to the real table/collision geometry").
Upper bound from the recorded data: the 08-28 whiteboard rollout reached z = 34 mm
(17.2 s, no protective stop), so *that* station's surface is ≤ 34 mm — nothing says
the waffles station is.

**Recommended floor margin for ≥5 mm true clearance.** Stop deriving the floor from a
whole-trajectory extremum; derive it from the two demo distributions that bracket it:
`floor = min(tcp_z_min + 5 mm, min(demo z_close) − 5 mm)`.

| task | current | recommended | ≥5 mm above deepest demo? | ≥5 mm below shallowest demo close? |
|---|---|---|---|---|
| Carton | 66.1 | **76.1** (`--z-floor-margin 0`; the 2.5 mm Carton gap has no room for both) | at it | 2.5 mm |
| egg | 49.5 | **59.5** (`--z-floor-margin 0`) | at it | 1.6 mm |
| waffles | 31.5 | **46.5** (`--z-floor 0.0465`) | +5.0 | 8.1 |
| whiteboard | 57.5 | **72.5** (`--z-floor 0.0725`) | +5.0 | 76.2 |

All four pass `envelope_conflict` (hitbox floors are 46.1/30/30/37.5). For Carton and
egg the deepest demo trajectory point *is* the grasp (2.5 / 1.6 mm apart), so
`--z-floor-margin 0` is the only defensible setting there — the current 10 mm margin
commands below every demo. And measure the table once per station (jog down to
fingertip contact, read `tcp_pose[2]`) — that is the number that should set the floor.

## 3. Does the widened joint gate still reject the 08-28 configurations?

`qgate.py` feeds the 26 recorded 08-28 start joint vectors through the real
`start_pose.start_sigma_report` with the NEW `q_mean`/`q_std` and gates at
`--max-start-sigma 2.5`:

```
22/26 of the recorded 08-28 starts are refused by the joint gate at 2.5 sigma
```
All 16 wrapped-wrist / flipped-branch episodes are refused with 31-73 sigma, every one
of them printing the `!!! FULL-TURN` line. Isolated per task:

```
Carton      wrist3=+183deg   worst= 72.2 sigma on q6 (raw +360.8 deg / std 5.0 deg) fullturn=True  -> GATED OUT
Carton      base q1=-112deg  worst= 24.5 sigma on q1 (raw -122.3 deg / std 5.0 deg) fullturn=False -> GATED OUT
egg         wrist3=+183deg   worst= 63.3 sigma on q6 (raw +358.2 / std 5.7)                        -> GATED OUT
egg         base q1=-112deg  worst= 21.1 sigma on q1 (raw -126.1 / std 6.0)                        -> GATED OUT
waffles     wrist3=+183deg   worst= 72.5 sigma on q6 (raw +362.4 / std 5.0)                        -> GATED OUT
waffles     base q1=-112deg  worst= 24.7 sigma on q1 (raw -123.3 / std 5.0)                        -> GATED OUT
whiteboard  wrist3=+183deg   worst= 67.2 sigma on q6 (raw +360.2 / std 5.4)                        -> GATED OUT
whiteboard  base q1=-112deg  worst= 16.4 sigma on q1 (raw -128.4 / std 7.8)                        -> GATED OUT
```
The widened stds cost 1.4-2.3× of the sigma (whiteboard q1 4.3° → 7.8°) and the gate
still refuses by 6.6-29× the threshold. `Q_STD_FLOOR_RAD = 5°` dominates on q1/q2/q6
everywhere, so the wrist-wrap detection is structurally immune to the regeneration.
**No regression.** (The four that now pass — 1787923361, 1787923675, 1787930224,
1787937561 — are exactly the pre-17:19 episodes the 08-28 postmortem called valid.)

Residual, not a regression: `--home-joints` moveJ's to the new `q_mean`, which moved by
up to 11.1° (egg q4), −8.4° (waffles q4), +8.2° (Carton q5). There is still no FK
cross-check that `q_mean` is consistent with `tcp_mean`, no path check on the moveJ,
and `URArm.move_j` still ignores moveJ's return (deploy-safety lens §4, P3 #39). The
joint spread is unimodal (largest new q_std is Carton q4 15.2°, far from a branch flip),
so the mean is not landing between two IK branches.

## 4. Veto z-band vs floor ordering

`veto_z_margin` = `max(15 mm, Z_MAX_MM[task] − tcp_z_min)` and `at_floor` is measured
from `z_ref = stats.tcp_z_min` (`planner.py:477-478`), so the band TOP is pinned to
`Z_MAX_MM` **regardless of how far tcp_z_min moves** — verified: band top came out
exactly 144.0 / 101.0 / 103.0 / 181.0 mm. F4 is therefore invariant under the F17
regeneration. Ordering per task, no inversion anywhere:

```
Carton      hbfloor 46.1 < clamp 66.1 < zmin 76.1 < close_min 78.6 < close_mean 103.6 < band top 144.0
egg         hbfloor 30.0 < clamp 49.5 < zmin 59.5 < close_min 61.1 < close_mean  71.9 < band top 101.0
waffles     hbfloor 30.0 < clamp 31.5 < zmin 41.5 < close_min 54.6 < close_mean  71.9 < band top 103.0
whiteboard  hbfloor 37.5 < clamp 57.5 < zmin 67.5 < close_min 148.7< close_mean 156.2 < band top 181.0
```
Sanity on the mask itself: every recorded 08-28 phantom close (waffles z@close 140/161/
190/298/353/381/437/673 mm; Carton 153/341/356/368; whiteboard 290/291/315/334/336/343)
is ABOVE its task band top, so the close-mask engages on all of them — the widened band
does not weaken the veto on the failure it exists for.

## 5. Pacing / chatter under `--nfe 1`

`chatter.py` drives the real `ChunkExecutor` (nuc 125 Hz) with plans arriving every
865 / 172 / 60 ms and spies every `servo_l`:

```
nfe5 865ms, demo descent            ticks= 411 tick_dt=10.88ms  v med=0.0500 p99=0.0504 max=0.0505 (cap .25)  z-dir reversals=0/410
nfe1 172ms, demo descent            ticks= 451 tick_dt=11.13ms  v med=0.0500 p99=0.0504 max=0.0506 (cap .25)  z-dir reversals=0/450
nfe1 172ms, --max-tcp-speed 0.15    ticks= 455                  v max=0.0505 (cap .15)                        z-dir reversals=0/454
nfe1 172ms, 4x descent (hits floor) ticks= 507                  v p99=0.2010 max=0.2017 (cap .25)             z-dir reversals=0/144
nfe1 172ms, 4x + vmax 0.15          ticks= 516                  v p99=0.1508 max=0.1570 (cap .15)             z-dir reversals=0/211
60ms replans, 4x descent            ticks= 524                  v p99=0.2009 max=0.2018 (cap .25)             z-dir reversals=0/161
```
Zero plan rejections, zero commanded-direction reversals at every period down to 60 ms.
`submit()`'s rebase (`plan.t0_pose = _last_cmd − c0`) plus the 0.1 s cross-fade keeps the
command continuous even when the replan period (172 ms) is under 2× the blend window.
The speed cap holds to ~4.7% (0.157 vs 0.150) — `dt_eff = clip(dt, period, 2*period)`
licenses up to 2× the per-tick displacement when a tick runs long; on this Mac the tick
ran 10-11 ms against a nominal 8 ms. **No resonance/chatter risk. READY on this point.**

What is NOT ready under `--nfe 1` is the episode length — see finding 1.

## 6. What the fix batch got right (checked, no finding)

- `envelope_conflict` fires exactly as advertised: `--hitbox-margin 0.005` → rc=2 with
  "UNSAFE ENVELOPE" on both waffles and Carton (`cli.py`).
- `apply_z_floor` never lowers below the workspace bound, so `--z-floor-margin 0.04`
  no longer drops the floor to 1.5 mm (row 3 of the 08-30 lens's table); on waffles it
  clamps to 30 mm.
- F6 gripper release: `_halt` reaches `gripper.move(open_aperture)` for `tactile_*`,
  `wrench_limit`, `hitbox_exit`, `veto_retry_cap` and not for `arm_stale`,
  `camera_scene_stale`, `workspace_clamp`, `protective_stop` (exercised per kind).
- The `RobotiqGripper` socket race I suspected in F6 (servo/planner thread issuing
  `move` while `_grip_worker` polls) is **not** a bug: `_cmd` holds `self._lock` across
  send+recv, so each request/response pair is atomic; and the worker cannot re-close
  after the release because `_grip_target` is unchanged and the deadband suppresses it.
- `load_start_stats` now refuses `q_n != n` (`ThinStartStatsError`); the shipped file is
  `q_n: 250` on all four tasks and `tcp_mean/tcp_std/gripper_*` are bit-identical to the
  08-28 file (`git diff bafa519^ bafa519`), matching the E13 claim.
- `zfloor:`/`hitbox:`/`vmax:` tags are honest (`zfloor:none` under `--no-z-floor`).

## 7. Findings

### F-1 (high) Arm B still ends at 7.0 s: `--max-episode-s 35` is raised, `--max-replans 40` is not
`docs/rig_session_v5.md:74`, `phantom/deploy/planner.py:637-647`. The replan cap is
checked FIRST and its default is still 40.
```
arm A  nfe5  --max-replans 40 (default) --max-episode-s 35        -> replans= 40 wall= 34.8s stop_reason=replan_cap
arm B  nfe1  --max-replans 40 (default) --max-episode-s 35        -> replans= 40 wall=  7.0s stop_reason=replan_cap
arm B  nfe1 x k-seeds4 (~0.7 s)  40 / 35                          -> replans= 40 wall= 28.2s stop_reason=replan_cap
arm B  nfe1  --max-replans 200 (P0#4 stopgap) --max-episode-s 35  -> replans=199 wall= 35.1s stop_reason=episode_time_cap
```
Waffles demos run 16.0-22.9 s and close at 4.7-9.7 s (mean 6.2). VALIDATION_0830 P0 #4's
own stopgap ("`--max-replans 200` whenever `--nfe < 5`") is in neither the doc nor the
default. Fix: append `--max-replans 200` to the Arm-B `EXTRA` line, or derive the replan
cap from `max_episode_s / measured latency` in `run_deploy`.

### F-2 (high) waffles/whiteboard lost 10.5 / 7.1 mm of floor to an unstable order statistic
See §2. Fix: `--z-floor-margin 0` for all four tasks and `--z-floor 0.0465` (waffles) /
`0.0725` (whiteboard); replace `Z.min()` in `gen_start_poses.py:82` with a low percentile
of the per-episode z_min and emit it alongside; measure the table once per station.

### F-3 (medium) `--no-z-floor` is now a hard refusal on Carton and whiteboard, with the wrong advice
`phantom/scripts/run_deploy.py:421`. With no task floor the workspace bound is 30 mm while
the hitbox floor is `tcp_z_min − 30` = 46.1 (Carton) / 37.5 (whiteboard), so
`envelope_conflict` fires and `main` returns 2. Exercised through `run_deploy.main`:
```
### waffles    --no-z-floor    rc=0 floor=  30.0mm hb_z=[30.0, 502.1] tags=['zfloor:none', ...]
### Carton     --no-z-floor    rc=2  (refused before the runtime)
### Carton     zfloor-margin 40mm  rc=2  (refused before the runtime)
```
The message says "Raise --hitbox-margin above --z-floor-margin (or lower
--z-floor-margin)" — but under `--no-z-floor` there is no z-floor-margin; the only fix is
`--hitbox-margin ≥ 0.046`. The repo's own test exercises `--no-z-floor` with
`hitbox=False` (`tests/test_deploy_fixnow_0830.py:448`), so the combination is uncovered.

### F-4 (medium) a 0.3 s DM-Tac dropout now opens the gripper and drops the object
`phantom/deploy/executor.py:38-43`. `is_letgo_reason` matches the whole `tactile_` prefix,
which includes `tactile_<name>_stale` — a *sensor-freshness* event
(`_ring_stale_s = 3/min(field_ds_rate_hz, fps)`), not an over-force. Exercised per kind:
```
event tactile_left_stale   is_letgo=True  gripper.move -> [0.232]
event tactile_fz           is_letgo=True  gripper.move -> [0.232]
```
0.232 is the waffles demo START aperture, well open of a 0.5-0.6 grasp plateau, so this is
a release. `hitbox_exit` is in the same set and the 08-28 hitbox excursions were at
z = 290-673 mm, i.e. the object is dropped from transport height. On egg that is D15's
"damage %" metric writing itself. Fix: match the over-force kinds explicitly
(`tactile_fz`, `tactile_depth`) instead of the prefix, and gate the `hitbox_exit` release
on `tcp z < some transport threshold` (or lower the arm first).

### F-5 (medium) the rig doc states the OLD floors
`docs/rig_session_v5.md:119-120`: "commanded TCP z >= task demo `tcp_z_min` - 10 mm
(waffles 42, Carton 66, whiteboard 65, egg 50 mm)". The shipped floors are 31.5 / 66.1 /
57.5 / 49.5 mm. The operator's only written statement of the safety envelope is 10.5 mm
wrong on the A/B task.

### F-6 (medium) the rig doc still quotes 20.7 → 17.4 mm
`docs/rig_session_v5.md:154-155`, contradicted by `docs/review_20260828/E13_rescore.md`
(val124 20.82 → 18.23; val78 22.17 → 20.86). Mikhail decision #4 / P3 #50: E13 was run, the
doc was not updated.

## 8. Verdict

**NOT READY** — the physical envelope and the joint gate are sound and the pacing is clean,
but the waffles clamp floor is now 20.5 mm below the deepest demo we have a distribution
for with no measured table height behind it, and the documented Arm-B command line still
truncates every episode at 7 s.
