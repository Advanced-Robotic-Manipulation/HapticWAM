# LENS: RIG RECIPE END-TO-END — re-validation of HEAD 820b4eb (2026-08-31)

Everything below was executed on the Mac with `.venv/bin/python`, mock drivers and `--tiny`,
through `run_deploy.main` / the real `PlannerLoop` / the real `ChunkExecutor`, plus three
read-only probes of the 08-28 rig traces on compute3 (`~/phantom-icra-2027`, HEAD `820b4eb`).
Scratch: `.../scratchpad/validate2/scratch/rig_recipe/`.

## 0. What I confirmed WORKS (fix list, exercised not read)

| item | evidence |
|---|---|
| suite green | `pytest tests/test_deploy_fixnow_0830.py test_deploy_levers.py test_deploy_parity_fixes.py test_rig_safety_0828.py test_start_pose.py test_start_poses_full_0830.py test_rig_recovery.py -q` → **147 passed** |
| Arm A runs one process, N episodes | `run_deploy --episodes 2 ... --nfe 5 --ema --persistent-noise` → rc 0, two episode dirs |
| Arm B bundle runs | `--nfe 1 --terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 35` → rc 0; tags `['nfe1','g1.0','pnoise','ckpt:','git:820b4eb','zfloor:32mm','hitbox:60mm','vmax:0.25','parity:on','veto:pc0.50/pn0.90/r3','kseeds:4','seed:<n>']` |
| F9 trace fields | trace row keys `['accepted','actions','diag','gate','latency_s','p_evt','sigma','t','tcp_pose','terminal_veto']`; `diag = {'nfe':1,'guidance':1.0,'k_seeds':4,'head_dz_mm':[-1447.01,-1147.88,-173.38,-2671.43],'head_dz_spread_mm':2498.05,'k_rejected':0,'k_pick':0}`; `actions_pre_veto` present on exactly the rewritten rows (S1: rows [3,4,5]) |
| per-episode seeds differ in one process | `seed:2664893036` / `seed:1838461769` (unseeded, same process) |
| `--seed` increments per episode | `seed:4242` / `seed:4243`, identical across two processes |
| F5 named stop reasons | `stop=replan_cap` and `stop=episode_time_cap` both surface through `EpisodeResult` end-to-end |
| F4 band | `build_veto(waffles)` → `z_ref=0.0415, z_margin=0.0615` → band top **103.0 mm** = `Z_MAX_MM['waffles']` |
| F6 gripper release | real `ChunkExecutor` + real `PlannerLoop`: `veto_retry_cap` → `gripper moves [0.9,0.23,0.9,0.23,0.9,0.23,0.9,0.23]`, last = **0.23** |
| F7 envelope assert | `--hitbox-margin 0.005 --z-floor-margin 0.010` → rc 2 `UNSAFE ENVELOPE` |
| F17 start_poses | every task `n: 250`, `q_n: 250` |
| K-seed × veto ordering | `policy.replan` selects seed `j` and carries `_cpk_row(pred.cpk, j)` **before** the veto rewrites; `_invalidate_cpk` then drops it only for the rewritten replan (S2 cpks `['CPK',None,'CPK','CPK','CPK']`). Correct. |
| parity `prev_chunk` after a rewrite | parity ON builds `prev_chunk` from the measured arm ring + `executor.gripper_cmd_at` (the **executed**, post-veto command); parity OFF carries `prev_plan = plan` post-rewrite. Both see the executed chunk. Correct. |
| mid-descent stop → `recover_control` | `veto_retry_cap` → `executor.stop()` calls `arm.servo_stop()` once (`servo_stop calls: 1, servo_active now: False`) → a subsequent `move_l` is accepted |

## 1. HIGH — the terminal veto self-cancels on the rig's own gate statistics

`p_none` read out of the **26 recorded 08-28 planner traces on compute3**:

```
ep_teacher_waffles_1787922904_000  n=20  p_none min=0.785 mean=0.962 max=0.998 frac>0.9=0.90
ep_teacher_waffles_1787923361_000  n=20  p_none min=0.885 mean=0.967 max=0.997 frac>0.9=0.95
ep_teacher_Carton_1787938156_001   n= 8  p_none min=0.934 mean=0.976 max=0.997 frac>0.9=1.00
... (waffles frac>0.9 = 0.10-1.00, median 0.85; Carton 0.92-1.00)
```
and at the FIRST commanded close (`max a[:,6] > 0.45`) plus the next three replans:
```
ep                                  i_close p_none@close  next 3
waffles_1787922904_000                  9    0.998        0.984 0.972 0.979
waffles_1787923361_000                  8    0.928        0.997 0.991 0.989
Carton_1787938156_001                   1    0.992        0.985 0.990 0.982
```
`TerminalVeto.p_none = 0.9` and `p_close = 0.5` are read off the **same** number:
`p_contact = 1 - p_none <= 0.1` everywhere, so a close only ever survives the mask through
the `at_floor` hatch F4 opened (waffles band top 103 mm; the rig closes at 65-120 mm) — and
that same `p_none > 0.9` fires the recovery on the very next replan.

Replaying the recorded `p_evt` + recorded gripper command sequences through the **real**
`PlannerLoop._apply_veto` with the arm inside the demo close band
(`scratch/rig_recipe/replay_veto_on_rig.py`):

```
episode                                        n closes reopen  outcome
ep_teacher_waffles_1787922904_000             20      4      3  VETO_RETRY_CAP
ep_teacher_waffles_1787923361_000             20      4      3  VETO_RETRY_CAP
ep_teacher_waffles_1787923675_000             23      4      3  VETO_RETRY_CAP
ep_teacher_waffles_1787945153_000              9      4      3  VETO_RETRY_CAP
ep_teacher_Carton_1787938064_000              13      4      3  VETO_RETRY_CAP
...
5/18 of the recorded 08-28 waffles/Carton episodes end in veto_retry_cap under --terminal-veto
```
50 `close_allowed` → 27 `recovery_open`. Five episodes cap; four more reach retry 3/3 and only
escape because the recorded trace runs out (they were recorded at `--max-replans 20`; arm B gets
40). This is a **lower bound**.

A grasp that HELD is dropped the same way (`scratch/rig_recipe/held.py`, p_none 0.96 = the rig's
measured value during the lift):
```
  r  z(mm)  measured_g  policy_cmd  ->  EXECUTED_cmd   veto           sum_dz(mm)
  2   70.0      0.62        0.90    ->     0.90        close_allowed     -80.0
  3   90.0      0.62        0.90    ->     0.23        recovery_open       0.0   <-- object released mid-lift
  4  120.0      0.62        0.90    ->     0.62        close_masked      160.0
  5  160.0      0.62        0.90    ->     0.23        recovery_open       0.0
  7  240.0      0.62        0.90    ->     0.23        recovery_open  (retry 3/3)
```
Before the fixes the veto was inert (P0 #2/#5: the mask never fired, the latch never armed).
F2+F3+F4 made it live, and on the rig's measured gate it is now an **anti-grasp**: G2
("Arm B >= 3/16 tactile-confirmed grasps") is 0/16 by construction, and every arm-B episode
ends in the operator's `veto_retry_cap` abort-rule check.

Minimal fix (either): (a) do not fire the recovery on a close whose `rec["at_floor"]` was the
only reason it was allowed — record `allowed_by` on the latch and require `p_contact` to have
been the reason; or (b) calibrate `--veto-p-none` from the recorded traces (>= 0.999 would leave
~1 firing per episode) and print the per-episode `p_none` histogram in the preflight. (a) is 4
lines and is the semantically right one.

## 2. HIGH — arm B is still a ~7 s episode: `--max-replans` is never raised

`GO_ANY.sh` on compute3 (read live) launches
`run_deploy ... --episodes $EPS --device cuda --persistent-noise --nfe $NFE --guidance $G $EXTRA`
with **no `--max-replans`**, so it stays at `run_deploy.py:198` `default=40`.
`PlannerLoop.run` checks the count cap **before** the wall-clock budget, so whichever fires first wins:

```
arm B: --nfe 1, measured 172 ms: 40 replans, 7.2 s wall, stop_reason='replan_cap'
arm A: --nfe 5, measured 865 ms: 40 replans, 34.8 s wall, stop_reason='replan_cap'
```
`--max-episode-s 35` in the documented arm-B EXTRA is dead. VALIDATION_0830 P0 #4's own stopgap
("`--max-replans 200` whenever `--nfe < 5`") was never written into the recipe, and
`rig_session_v5.md:86` asserts the opposite ("`--max-episode-s 35` is the wall-clock budget").
K=4 latency was never profiled, so the only way the budget binds is if K=4 happens to push the
NFE-1 replan past 875 ms.

Fix: `EXTRA="--nfe 1 --terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 35 --max-replans 200"`,
or make `--max-replans` default to `None` when `--max-episode-s > 0`.

## 3. MEDIUM — the close-mask's own HOLD command arms the phantom-grasp recovery

`_apply_veto` rewrites a masked chunk to `a[:,6] = grip_now` (the measured aperture) and the
executor faithfully enters those steps; `_note_executed_close` (`planner.py:410`) then reads them
back as an EXECUTED close, because the running minimum `g_min` is the episode's first aperture,
not the pre-mask one.

`scratch/rig_recipe/veto_scen.py` S1 — the recorded ramp 0.31→0.52, p_contact 0.01, z = 110 mm
(above the 103 mm band), **no `close_allowed` anywhere**:
```
  actions: ['none','none','none','close_masked','close_masked','recovery_open']
  submitted grip[0]: [0.35, 0.39, 0.43, 0.44, 0.48, 0.23]
```
The 0.48 at index 4 is the veto's own hold; it crosses `close_pos 0.45` with rise 0.17 > 0.15,
latches `closed_idx`, and the next replan opens the gripper to 0.23, zeroes the lift and burns
retry 1/3 — on a close the veto itself prevented. The trace records `recovery_open`
("phantom grasp"), so the session's veto-fires-N-times number is wrong too.

Fix: skip the steps the veto wrote — set a `state["masked_until_t"]` when `close_masked` rewrites
a chunk and ignore `entered_grip_after` steps before it, or latch only on steps whose command
exceeds the pre-veto proposal.

## 4. MEDIUM — F6's gripper release races the gripper worker it does not own

`executor._halt` (`:163-167`) calls `_release_gripper()` **before** `self._grip_target = None`
and `self._stop.set()`, from the executor thread on `safety_stop` and from the **planner** thread
on `veto_retry_cap`. `_grip_worker` documents itself as the "single owner of gripper I/O" and
re-checks `_stop` only *before* `gripper.move`; a target published but not yet sent survives the
whole duration of the release's blocking socket move and lands after it.

`scratch/rig_recipe/release_race.py` (30 ms round-trip, a Robotiq-like single-socket lock):
```
 trial 0: moves=[0.23, 0.9] last=0.9
 ...
LAST COMMAND WAS A CLOSE in 40/40 trials
```
`scratch/rig_recipe/release_race2.py` (full `ChunkExecutor`, 2 ms per SET/GET, close ramp
0.23→0.90, stop at a random moment): **1/30** trials left the gripper commanded closed. The real
`RobotiqGripper.get_state()` is two GETs under one lock with a 2 s socket timeout, so the window
is wider on the rig than 2 ms.

Fix: `self._stop.set()` and `self._grip_target = None` **first**, then release; and guard the
worker's `gripper.move` and `_release_gripper` with one shared I/O lock so the release is ordered
last.

## 5. MEDIUM — the doc Ilya runs from still quotes the struck 17.4 / 14.9 mm

`docs/rig_session_v5.md:154-155`:
```
endpoint error v4 20.7 mm -> v5_6 17.4 mm; ... new-batch holdout 20.7 -> 14.9 mm
```
`E13_rescore.md:141` : "The correct statement is `20.8 mm -> 18.2 mm` on the 124-episode val split
... 2.6 mm / 12% ... not the 3.3 mm / 16% that `20.7 -> 17.4` claimed." VALIDATION_0830 §3.5 #4
said "Re-score both arms (E13) or strike the number from the docs — it cannot stay as is."
E13 landed in a NEW doc (c422467); the rig doc was not touched.

## 6. LOW — the stop reason F5 added is never persisted

`meta.json` of a finished episode:
```
tags = [... 'seed:2664893036', 'unlabeled']   status = aborted   (no stopped_reason field)
```
`stopped_reason` exists only on `EpisodeResult`, a console line, and `trial_runner`'s ledger —
which the GO-script path does not use. So after the session nothing on disk distinguishes
`replan_cap` / `episode_time_cap` / `safety_stop` / `veto_retry_cap`, which the abort rules, the
"unmeasurable vs failure" call and the per-arm truncation rate all turn on.
Fix: append `stop:<reason>` to `ep_tags` (or write `meta.stopped_reason`) in `run_deploy` after
`run_episode` returns — the recorder's `relabel()` is already called on the same path.

## 7. LOW — F17 lowered the waffles z clamp to 1.5 mm above the raw workspace floor; the doc still says 42 mm

`configs/start_poses.yaml` waffles `tcp_z_min: 0.0415` (was 0.052) → `run_deploy` logs
`z no-go floor: commanded TCP z clamped to >= 32 mm (workspace z now [0.0315, 0.8])`, while
`hardware.nuc.yaml` `workspace_m.z[0] = 0.03`. The task-specific floor now buys 1.5 mm over the
global floor (whiteboard 65 → 57.5 mm). `rig_session_v5.md:119` still says
"waffles 42, Carton 66, whiteboard 65, egg 50". Honest number, stale doc, weaker table protection.

## 8. LOW — `veto_retry_cap` masks a `servo_stop` failure from `recover_control`

`_set_reason` is first-writer-wins, so after `request_stop("veto_retry_cap")` a later
`servo_stop failed` cannot set `servo_stop_failed`; `veto_retry_cap` is not in
`_CONTROL_DEAD_REASONS` (`run_deploy.py:146`), so the next episode skips `recover_control()`,
`move_l` is refused by the servo guard and the operator silently continues from a hand-jogged
start — the 2026-08-27 failure the code comment describes. Also the `retry_cap` trace row is
written with `"diag": {}`, dropping that replan's `head_dz_mm`.
