# LENS: SAFETY LAYER FINAL — PHANTOM @ d9c40f2

Scope: `phantom/deploy/{safety,executor,planner,start_pose,runtime}.py`,
`phantom/scripts/{run_deploy,gripper_ctl}.py`, `configs/start_poses.yaml`,
`tests/test_rig_safety_0828.py`, `tests/test_deploy_levers.py`.
Everything below was RUN, not just read. Scratch:
`scratchpad/validate/scratch/safety-final/`.

## 0. What is genuinely fixed (verified)

- `tests/test_rig_safety_0828.py` + `tests/test_deploy_levers.py`: **54 passed**.
  `tests/test_safety.py test_control_safety.py test_deploy_fixes.py
  test_deploy_parity_fixes.py test_start_pose.py`: **67 passed**.
- **z floor is a clamp, not a stop.** `safety.py:211` evaluates the hitbox on
  `self.clamp_target(...)`. Reproduced with the shipped waffles envelope: a target
  at `tcp_z_min - 20 mm` returns `CLAMP` with **no** `hitbox_exit`, and
  `clamp_target[2] == 42 mm`. A lateral exit at the same z still returns
  `STOP_EPISODE`/`hitbox_exit`.
- **`--max-tcp-speed` really caps the executor.** Measured commanded speed over
  ~210 servo ticks with a chunk demanding 0.5 m/s: median 0.250 / 0.150 / 0.050 m/s
  at caps 0.25 / 0.15 / 0.05.
- **Joint gate.** `+2π` on wrist 3 → 72σ with the 5° std floor, `!!! FULL-TURN`
  printed, `sig[:7] == 0` (TCP passes). Flipped elbow + base −127° → >2.5σ.
  `--home-joints` orders `move_j` then `move_l`.
- **`gripper_ctl`** open/close/status/reset all exit 0 against `MockGripper`;
  `close` respects the 0.9 pad ceiling.
- **Seeded sampler fix is live.** A mock episode records
  `seed:3232723731` in `meta.tags`; `episode_seed(7, 3) == 10`.
- **Provenance.** Every lever is tagged on or off
  (`parity:off veto:off kseeds:1 hitbox:none vmax:0.25`), `deploy_overrides`
  carries the envelope, and `run_episode` files every deploy episode
  `status='aborted' + tag 'unlabeled'` (P9) — confirmed in a real mock run.
- **`safety_stop` IS in `_CONTROL_DEAD_REASONS`** (`run_deploy.py:~140`), so a
  hitbox/stale/tactile stop triggers `recover_control()` before the next episode.
  `veto_retry_cap` correctly is not.
- **Parity `prev_chunk` after a veto rewrite is CORRECT.** With
  `--parity-fixes --terminal-veto`, the executor's `_grip_hist` and
  `gripper_cmd_at()` return **0.30** (the vetoed hold), not the policy's 0.80
  proposal — i.e. the executed command, which is what training feeds. No bug.
- **Veto → hitbox interaction is safe.** The recovery only lowers commanded z and
  the clamp pins it at the floor, which is above the hitbox lower edge; it can
  never manufacture a `hitbox_exit`.

## 1. The replan loop has no pacing — `--max-replans` is a COUNT, and arm B is `--nfe 1`

`PlannerLoop.run` (planner.py:422) is a tight `while` with no sleep and no
reference to `control.model_tick_hz` (declared in all three hardware yamls,
**consumed nowhere** — `grep model_tick_hz phantom/ tools/ configs/` returns only
the schema and the yamls). Episode wall-time == `max_replans × inference latency`.

Measured (`probe_budget.py`, stubbed latency):

    latency 865 ms (--nfe 5, the 08-28 rig number): 10 replans in 8.72 s -> 40 replans = 34.9 s
    latency 172 ms (--nfe 1, the Session-4 arm-B recipe): 10 replans in 1.81 s -> 40 replans = 7.2 s

And with the tiny model end-to-end: `--max-replans 40` produced a **1.7 s**
episode (`replans=40 stop=None`).

`docs/rig_session_v5.md` raised the default 20 → 40 precisely because "the
20-replan cap (~19 s) cut every retry short; demos run 16-31 s" — that arithmetic
assumed ~0.9 s replans. The same doc's Session-4 arm B is
`EXTRA="--nfe 1 --terminal-veto"`, whose own headline is "172 ms replans instead
of 865". So arm B gets a **~7 s** episode against 16-31 s demos, and the terminal
veto's open → re-descend retries (up to 3) have nowhere to run. Every arm-B
episode terminates by cap, is labelled `f`, and the A/B measures episode length.

Fix: pace the loop (or budget it in seconds). Minimal: add
`--max-episode-s` (default ~35) checked in `run()` alongside `max_replans`, and
have `GO_ANY.sh`/the Session-4 recipe pass `--max-replans 200` with `--nfe 1`.

## 2. The terminal veto's `closed_at` latch never clears — one high-p_none replan drops a real grasp

`planner.py:379`: `if state["closed_at"] is not None and p_none > v.p_none:`.
`closed_at` is set at `:485` on an accepted `close_allowed` and cleared **only**
inside the recovery branch. The docstring at `:308-310` says the rule reacts to
"the very next replan"; nothing in the code bounds it — `closed_at` is stored as a
timestamp and used only as a boolean.

Reproduced (`probe_veto_state2.py`) with the gripper actually following the
command (so `closing` is False on every later replan and the latch is never
re-armed):

    replan 0: p_contact=0.80 action= close_allowed grip_sent=0.80 retries=0
    replan 1: p_contact=0.80 action=          none grip_sent=0.80 retries=0
    replan 2: p_contact=0.80 action=          none grip_sent=0.80 retries=0
    replan 3: p_contact=0.05 action= recovery_open grip_sent=0.25 retries=1   <-- object dropped
    replan 4: p_contact=0.05 action=  close_masked grip_sent=0.25 retries=1   <-- cannot re-close

So: a grasp succeeds, the arm lifts, and the FIRST replan for the rest of the
episode whose `p_evt[none] > 0.9` commands the demo open aperture and forbids the
lift — and the close-mask then refuses every attempt to re-close, because
`p_contact` is still low. The tactile grasp rule (P8: hold ≥ 2 s + lift ≥ 50 mm
while in contact) scores that as a failure. That is the arm the D5/G2 gate is
supposed to bless.

Nobody can say how likely this is: there are **zero** successful policy grasps in
the corpus, so the post-grasp `p_evt` distribution has never been observed, and
`--veto-p-close`'s own help says "calibrate on the 08-20 gate trace" — no such
calibration exists anywhere in `docs/`.

Fix (3 lines): store the replan index with `closed_at` and only fire recovery
while `n - closed_at_idx <= 1` (what the docstring promises); clear `closed_at`
unconditionally after that window.

## 3. The STOP hitbox and the z floor are fitted to 17–37 episodes while `n: 250`

`configs/start_poses.yaml` (header, and measured):

    Carton      n= 250   q_n=  20
    egg         n= 250   q_n=  17
    waffles     n= 250   q_n=  37
    whiteboard  n= 250   q_n=  21

`tcp_min`/`tcp_max` (→ the STOP hitbox that ENDS an episode) and `tcp_z_min`
(→ the z clamp, the last backstop over the table) come from the `q_n` subset; only
`tcp_mean`/`tcp_std` come from 250. `phantom/deploy/start_pose.py:67-105`
(`load_start_stats`) never reads `q_n` — verified: `"q_n" not in
open("phantom/deploy/start_pose.py").read()` is `True`. `tools/gen_start_poses.py`
asserts only `len(P) >= 20` (`:47`) and emits `n` and `q_n` from separately-grown
lists with no equality check — the D2 requirement "`gen_start_poses` must refuse
when `q_n != n`" was not implemented, and `REVIEW_SYNTHESIS.md §1.11` made
regeneration blocking: "Land this before any rig episode is counted."

Consequence: the envelope that ends episodes is an **extremum** statistic taken
over ≤15% of the demos. Under-covered by ~a few cm it converts legitimate demo-band
motion into `safety_stop` (which also forces a full RTDE control-script rebuild);
over-tight in z it clamps the descent above where demos actually grasp.

## 4. Every STOP_EPISODE stops the arm and leaves the gripper commanded closed

`executor.py:252-255`: `arm.stop(2.0)` then `_halt("safety_stop")`. `_halt`
(`:131`) sets `self._grip_target = None` and the stop event; `_grip_worker`
(`:334`) exits on `self._stop.is_set()` **without sending anything**. The only
other `gripper.move` in the deploy path is `start_pose.py:208` — the *next*
episode's homing.

Reproduced (`probe_estop_gripper.py`) with a chunk commanding a 0.90 close and a
`SafetyMonitor` stub that trips STOP at tick 41:

    stopped_reason: safety_stop
    arm.stop() calls: 1  arm.servo_stop(): 1
    gripper.move() targets, in order: [0.9]
    => gripper's LAST commanded target after the fingertip e-stop: 0.9

The `tactile_fz` / `tactile_depth` guard (`safety.py:157-182`) exists *specifically*
to protect the gel fingertips. It stops the arm and leaves the fingers squeezing at
the pad ceiling through the label prompt and the "clear the arm's path" prompt —
minutes, unattended. On this rig `force_calibrated` is false, so the active limit
is `tactile_depth_limit: 0.6` on `abs(depth).max()`, a threshold with no recorded
per-session baseline (the review's completeness critic flagged gel drift as
nobody's lens). Same gap after a hitbox stop and a veto retry-cap stop.

Fix: in `_halt`, before setting `_stop`, issue one
`gripper.move(open_aperture, speed, force)` on the reasons that mean "let go"
(`tactile_*`, `wrench_limit`, `veto_retry_cap`, `hitbox_exit`), or add a
`release_on_stop` executor hook. Also add `GRIPPER_OPEN.sh` to the post-stop
prompt text.

## 5. The floor-is-a-clamp fix only holds while `--hitbox-margin > --z-floor-margin`

`run_deploy.py:454` applies the hitbox (intersected with the workspace) and `:472`
then raises the workspace floor. The invariant that makes the fix work is
`hitbox.z[0] < workspace.z[0]`, i.e. `hitbox_margin > z_floor_margin`. Nothing
asserts or warns about it. Reproduced (`probe_floor_vs_hitbox.py`, shipped waffles
envelope, target at `tcp_z_min - 20 mm`):

    hitbox_margin=30mm z_floor_margin=10mm | hitbox z=[22,440] floor=42mm | -> CLAMP ['workspace_clamp']
    hitbox_margin= 5mm z_floor_margin=10mm | hitbox z=[47,415] floor=42mm | -> STOP  ['workspace_clamp','hitbox_exit']
    hitbox_margin=30mm z_floor_margin=40mm | hitbox z=[22,440] floor=12mm | -> OK    []

Row 2 is the exact 08-28 bug, silently restored by an operator tightening the
hitbox — a knob `docs/rig_session_v5.md` advertises. Row 3 shows the mirror: a
generous `--z-floor-margin` puts the floor 40 mm below the lowest demo (the
table-slam the floor exists to prevent) with no warning.
Fix: after both `model_copy`s, assert/warn `hitbox_m.z[0] <= workspace_m.z[0]`.

## 6. The veto's "already low enough to close" band sits below the whole demo close distribution

`planner.py:407`: `at_floor = z_now <= v.z_floor + v.z_margin`, where
`v.z_floor = tcp_z_min - z_floor_margin`. REVIEW_SYNTHESIS P3 specified
`z ≤ tcp_z_min + 15 mm`; measuring from the already-lowered floor makes the band
10 mm tighter. Measured against the P8 demo close-height p95 and E9's own
`pred_close_height_mm = 110.8`:

    Carton      tcp_z_min  76.1  code band <=  81.1 mm   spec <=  91.1   demo close p95 ~ 129 mm
    egg         tcp_z_min  59.5  code band <=  64.5 mm   spec <=  74.5   demo close p95 ~  86 mm
    waffles     tcp_z_min  52.0  code band <=  57.0 mm   spec <=  67.0   demo close p95 ~  88 mm
    whiteboard  tcp_z_min  74.6  code band <=  79.6 mm   spec <=  89.6   demo close p95 ~ 166 mm

The geometric escape hatch is 30–90 mm below where demos close and ~50 mm below
where the model itself predicts it will close (E9, 110.8 mm). It will essentially
never fire, so **every** close in arm B hangs on `p_contact > 0.5` — a threshold
that has never been calibrated and that read 0.01 (`p_none 0.99`) at the rig's
observed closes. Expected outcome: arm B commands zero closes, G2 reads 0/16 for
both arms, and the gate cannot distinguish "the veto worked" from "the veto
silenced the policy".

Fix: `at_floor = z_now <= (stats.tcp_z_min if stats else v.z_floor) + v.z_margin`
and raise `z_margin` to the per-task demo close p95 (i.e. make the hatch
`z ≤ Z_MAX[task]`), so a descent that reaches the demo grasp band is always
allowed to close.

## 7. The veto rewrites `plan.actions` in place and the trace keeps only the rewrite

`planner.py:371` `a = plan.actions` then mutates; `:490` writes
`"actions": plan.actions.tolist()`. The pre-veto proposal is nowhere in
`planner_trace.json` — confirmed: with the close masked, `trace['actions'][0][6]`
is `0.3`, the policy's `0.8` is gone. `terminal_veto` records only
`{p_contact, action, retries}`.

`tools/replay_rig.py:343` reads `r["actions"]` as *the policy's own chunk*: it is
both the `prev_chunk="proposal"` feed and the reference for `trace_head_dz` /
`trace_in_spread`, which is the **G0 validity test**. It warns only on
nfe/guidance drift (`:300-304`) and never looks at `terminal_veto`. So every
arm-B episode replays against a chunk the model never produced (the recovery
zeroes z outright), and D5's trace decomposition — the thing that picks D6–D9 —
silently reads garbage on the arm that matters.
Fix: record `"actions_pre_veto"` in the trace when the veto fires, and have
`replay_rig` prefer it (and warn when `terminal_veto` is present).

## 8. `URArm.move_j` has no servo guard and ignores moveJ's return

`phantom/drivers/real/ur.py:341-344`: no `_servo_active` check, no `ok is False`
raise — while `move_l` (`:346-360`) has both, and `MockArm.move_j`
(`mock/ur.py:102-104`) *does* raise. `start_pose.move_to_start:220` issues the
moveJ **before** the guarded moveL, so `--home-joints` performs the dangerous
motion first (a base rotation "sweeps the bin", per the docstring) and only then
hits the check. A mock dry run is strictly stricter than the rig here, so nothing
in the test suite can catch it. Already noted in
`docs/review_20260828/lenses/deploy-safety.md:110-117`; unfixed.

## 9. Smaller

- `run_deploy.py:524` tags `zfloor:{workspace.z[0]}` unconditionally, so a
  `--no-z-floor` run records `zfloor:30mm` (the raw workspace bound) and is
  indistinguishable in the tags from a task floor that happened to land at 30 mm.
  Verified in a real `--no-z-floor` mock run. `deploy_overrides` is the only
  honest record.
- `SafetyMonitor.recovered()` (`safety.py:255`) is dead in the deploy path — the
  executor never resumes after a STOP; only `record_episodes` / `data_collect`
  use their own guards. It is tested (`test_safety.py:139`) but unreachable.
- `test_rig_safety_0828.py:267` reads `phantom/scripts/run_deploy.py` via a
  cwd-relative `open()`; the assertion silently depends on pytest's rootdir.
- With a checkpoint whose ACC head is absent, `plan.p_evt` is `zeros(5)` →
  `p_none = 0.0` → `p_contact = 1.0`, so both the close-mask and the K-seed
  head-descent rejection silently become no-ops that log `close_allowed`. Fails
  open, with no assertion that the deployed checkpoint has an ACC head.

## Verdict

**NOT READY** to hand to Ilya/Petr unattended. The static envelope (floor clamp,
hitbox, speed cap, joint gate, gripper tool) is sound and verified, but three
things would break the session or its interpretation: the replan loop is unpaced
so the Session-4 `--nfe 1` arm gets ~7 s episodes; the terminal veto's `closed_at`
latch can reopen the gripper on a real grasp at any point in the episode; and the
hitbox/floor that gate every episode are fitted to 17–37 demos while the file
claims 250. Add the gripper release on a stop and the
`hitbox_margin > z_floor_margin` assert before anyone runs it without you.
