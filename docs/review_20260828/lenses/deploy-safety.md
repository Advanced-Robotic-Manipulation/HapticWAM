# PHANTOM deploy safety + robustness review — after the 08-28 safety batch (HEAD 3539988)

Scope: `phantom/deploy/*`, `phantom/drivers/real/*`, `phantom/scripts/run_deploy.py`,
`phantom/scripts/gripper_ctl.py`, `tests/test_rig_safety_0828.py`, `configs/start_poses.yaml`,
`tools/gen_start_poses.py`. No repo files modified. `tests/test_rig_safety_0828.py`: 16 passed.

Confidence labels: **certain** = read in code and/or reproduced with a python check here;
**believe** = read in code, not executed against hardware; **guess** = inference about UR/ur_rtde
behaviour I could not verify from this machine (ur_rtde is not installed in the Mac venv).

Context that matters for severity: the commit is stamped 17:10 -0400 on 08-28 and the doc says
every rig episode from 17:19 (rig-local) on was invalid, i.e. **the whole safety batch has not
run on the arm yet**. Everything below is therefore "what will happen on first contact".

---

## 1. The z no-go floor is not a clamp on the rig — it is an episode STOP (certain)

`run_deploy.main` applies `apply_z_floor` first (run_deploy.py:264-269), then `apply_hitbox`
(run_deploy.py:276-278). `apply_hitbox` intersects the box with the (already raised) workspace
(safety.py:327-334), so the hitbox's lower z face is exactly the floor. In
`SafetyMonitor.check` the workspace test and the hitbox test both run on the **unclamped**
target (safety.py:198-210), the hitbox verdict is `STOP_EPISODE`, and `_max(CLAMP, STOP)` is
STOP. The executor then calls `arm.stop` and halts (executor.py:218-221) before the clamp on
line 223 is ever reached.

Reproduced with the shipped waffles stats + `configs/hardware.nuc.yaml`:

```
floor (0.042, 0.8)  hitbox z (0.042, 0.440)
target z = floor+0.000 -> ok
target z = floor-0.001 -> stop  events=['workspace_clamp', 'hitbox_exit']
target z = floor-0.005 -> stop  events=['workspace_clamp', 'hitbox_exit']
```

So under the default flags (floor ON, hitbox ON — both default-on per the doc) a commanded
TCP 1 mm below `tcp_z_min - 10 mm` ends the episode with `safety_stop`, which then also
triggers a full RTDE control rebuild before the next episode (`safety_stop` ∈
`_CONTROL_DEAD_REASONS`, run_deploy.py:114). The documented behaviour — "clamp: stops the
descent there and lets x/y continue" (docs/rig_session_v5.md:66-68, safety.py:309-313) —
never happens on the rig. The tests never exercise the combination: `test_z_floor_..._clamps`
builds a floor without a hitbox, `test_hitbox_exit_...` a hitbox without a floor.

Why it matters for the A/B: the floor sits 10 mm under the minimum of a 17-37-episode subset
(see §2). A policy that actually reaches the object (v5 is the one that gets closer) and
dips a few mm under that line loses the episode outright — no close, no `z at close` in the
trace, a forced `f` label. That biases exactly the comparison the session is for, in the
direction of punishing the model that descends further.

**Fix (one line + one test):** evaluate the hitbox on the clamped target, e.g. in `check()`
`hb.contains(self.clamp_target(tcp_target)[:3])`. The floor face then clamps (documented
semantics), every other face still stops (the hitbox never coincides with the workspace box
elsewhere — waffles: x (-0.499,-0.260) vs ws (-0.7,0.15), y (-0.379,0.176) vs (-0.5,0.3),
z-hi 0.44 vs 0.8). Add a test that applies floor+hitbox (the shipped default) and asserts
CLAMP for `z = floor - 5 mm` and STOP for `y = hb.y[1] + 5 mm`. Update the doc line if you
decide you actually want STOP at the floor (defensible for the whiteboard slam) — but then
say so and drop the "clamp" wording.

## 2. Floor / hitbox / joint stats come from 7-15 % of the demos (certain that it is a subset; the effect size is a guess)

`configs/start_poses.yaml:13-16` says it: q_mean/q_std, tcp_z_min, tcp_min/tcp_max were
computed "from the compute3 subset (val_eval + batch_20260822 successes, n=q_n per task)" —
q_n = 20 (Carton), 17 (egg), 37 (waffles), 21 (whiteboard) — while the TCP start stats use
n = 250. Min/max envelopes are order statistics: the min of 250 episodes is ≤ the min of 37,
typically by more than the 10 mm floor margin and quite possibly more than the 30 mm hitbox
margin for the transport arc (z_max) and the lateral approach spread. Combined with §1 this is
the mechanism for false episode ends.

Also: the shipped `tools/gen_start_poses.py` cannot reproduce the shipped yaml — it computes
everything from one root and asserts `len(P) >= 20` (gen_start_poses.py:47), yet the yaml has
`q_n: 17` for egg. The file was hand-merged; the tool is not the source of truth it claims to
be.

**Fix:** regenerate the joint + envelope fields over the full 250/task success set (the hub
has the data; the header already says to do this) before the A/B. Until then run with
`--z-floor-margin 0.02 --hitbox-margin 0.05` and say so in the tags. Make `gen_start_poses`
write `q_n == n` or refuse.

## 3. Nothing looks at the measured joint vector after the start gate (certain about the code; UR3 wrist-3 range is from memory)

The start gate (start_pose.py:160-178) is the only joint-space check in the whole deploy
path. `SafetyMonitor.check` reads only `ft` and `protective_stop` from the arm ring
(safety.py:117-133) although the ring carries `q`, `qd`, `tcp_pose` (workers.py:255-262).
`URArm.servo_l` runs `getInverseKinematics` on the controller and servoJ's the result
(ur.py:330-331) with no comparison against the actual q — a branch jump near a wrist
singularity (the wrapped-wrist start configs of 08-28 are close to one) becomes a
full-speed joint whip → protective stop, or worse, unnoticed cable wrap. The UR3's wrist 3 is
infinite-rotation (no joint-limit backstop), and the DM-Tac USB + gripper cables ride across
it. The executor's rotational rate limit (executor.py:247-253) is on the TCP rotation
vector, not on joints, so wrist-3 turns accumulate silently over an episode.

Ditto the measured TCP: the hitbox is on the command only. Tracking lag at 0.25 m/s with
lookahead 0.1 s is of the order of the 30 mm margin; on the floor face the descent is slow so
it is fine, but a measured-pose check costs nothing since the ring already has it.

**Fix:** (a) add `q_min/q_max` (all-frames joint envelope) to `start_poses.yaml` (the tool
already opens `arm_q.zarr`), and in `check()` STOP with `joint_envelope_exit` when
`arm["q"][0]` leaves it by more than ~15°; this catches branch flips, cable wrap and
singular configs in one check. (b) in `URArm.servo_l`, compare the IK result with
`self._recv.getActualQ()` and raise (→ `executor_crash`, already a control-dead reason) if
any joint differs by > 0.5 rad. (c) STOP on measured `tcp_pose` outside hitbox + 20 mm.

## 4. `--home-joints` moveJ has no path check (certain about the code)

`move_to_start` calls `arm.move_j(q_mean, 0.2, 0.5)` from wherever the arm is
(start_pose.py:220). Joint interpolation from a flipped elbow branch (base −127°, the Carton
case) swings the TCP through an arc that can pass through the bin or the table; the doc
delegates this to the operator's E-stop. `URArm.move_j` also ignores moveJ's bool return
(ur.py:341-344; `move_l` raises on False at ur.py:358-360), so a rejected moveJ on a dead
script is silent and the following `move_l` is what fails.

**Fix:** before the moveJ, moveL straight up to `hitbox z_max` in the current branch (a pure
+z move is branch-preserving), then moveJ, then moveL to the jittered start; or sample the
joint path with `ctrl.getForwardKinematics(q_i)` (ur_rtde has it) and refuse if any sample
is below the floor or outside the hitbox. Check moveJ's return value like move_l.

## 5. E-stop / safeguard stop / fault are invisible to the safety layer (certain about the code; RTDE bit semantics from memory)

The only robot-state flag in the arm ring is `protective_stop = isProtectiveStopped()`
(workers.py:262). That RTDE bit is set for a *protective* stop only; an emergency stop,
safeguard stop, violation or fault leaves it false. What actually happens on an E-stop
mid-episode: the control script dies, `servoJ` returns False, `servo_l` raises
(ur.py:332-335), `_run_guarded` logs "executor thread crashed" with a traceback
(executor.py:281-284) and the episode ends as `executor_crash`. Recovery works because
`executor_crash` ∈ `_CONTROL_DEAD_REASONS`, and `is_ready_for_control` correctly blocks on
robot mode 7 / safety mode 1 (ur.py:265-269). But the operator reads a crash traceback at
the moment they pressed the button, `stopped_reason` cannot distinguish "E-stop" from "bug"
in the campaign bookkeeping, and if servoJ ever does *not* return False promptly the
backstop is the stall watchdog (2 replans ≈ 2 s).

Related recovery-path gaps, all in run_deploy.py: (a) a homing move that fails with no
preceding recovery (E-stop pressed *during* the moveL/moveJ) takes the tolerant branch
(run_deploy.py:398-401) — no `program_running()` check, no rebuild — so the next episode
starts on a dead control script and dies at its first plan; (b) `recover_control` never
touches the gripper; on a CB3 an E-stop drops tool power, the Robotiq loses activation, and
the gate then loops on "gripper not settled" until the operator runs `GRIPPER_RESET.sh` from
another shell. `Rig.connect_all` activates once at session start only (base.py:296-301).

**Fix:** push `safety_mode` (getSafetyMode) and `robot_mode` into the arm ring and map
safety_mode ∉ {1 NORMAL, 2 REDUCED} to the `PROTECTIVE_STOP` verdict with the mode in the
event kind; in stage 1's except branch call `arm.program_running()` and jump to
`recover_control` when it is False; have `recover_control` (or stage 1) call
`gripper.activate()` (idempotent) and fall back to ACT 0→ACT 1 when STA ≠ 3.

## 6. Deploy episodes are stamped with a per-run hardware hash (certain)

`run_deploy` builds the runtime and recorder from the *modified* hw copy (run_deploy.py:357 →
runtime.py:144 → `EpisodeWriter`), and `EpisodeWriter` writes `meta.config_hash =
hw.config_hash()` (episode_store.py:47). `config_hash` hashes the full `model_dump`
(hardware.py:522-527), so the floor, hitbox and speed cap all change it (checked: all three
differ from the base hash; shape fields are unchanged). `WindowSampler` counts such episodes
as config drift (windows.py:147-155) and `train_teacher` hard-fails on them
(train_teacher.py:288-296). Every rollout recorded from now on — the DAgger / self-improvement
data the next-phase plan depends on — will be rejected by training unless
`--allow-config-drift`, which then also hides real drift. Checkpoint load is unaffected
(hash mismatch there is a warning, common.py:253-255).

**Fix:** stamp provenance with the base hw (keep the loaded `hw` and pass it to
`DeploymentRuntime` for the recorder only), or exclude runtime-only overrides
(`safety.hitbox_m`, and the overridden workspace/limits) from `snapshot_yaml()` for hashing.
The envelope is already in the tags.

## 7. What happens when the hitbox stops mid-chunk (believe; ur_rtde script fetched and summarised, not read line by line)

Sequence: tick N verdict STOP → `arm.stop(2.0)` = `stopL(2.0)` under `_ctrl_lock`
(executor.py:219, ur.py:362-367) → `_halt` → servo loop exits, gripper worker exits on the
next poll. In ur_rtde's control script `stopl` runs in the command handler without killing
the servo thread (only the `servo_stop` handler does `kill servo_thrd` then `stopl`), so
either the servo thread keeps re-asserting the last target and the arm holds (harmless), or
the controller objects to motion commands from two script threads and the script dies
(then `servoStop` fails, the guard is cleared, and the `safety_stop` reason rebuilds control
anyway). The actual deceleration happens in `executor.stop()` → `servo_stop()`, which the
runtime reaches only after the planner notices `_executor_stopped()` — up to one replan
(~0.9 s) later. Meanwhile `submit()` has no `_stop` check (executor.py:81-112), so the plan
that lands during that window is "accepted" and traced although nothing executes it;
`rig_trace_decompose` will read it as an executed window.

Also: a close already sent to the Robotiq completes after the stop (position command);
`_halt` only invalidates the mailbox. That is the "pad-on-pad and nobody opens it" scenario
gripper_ctl was written for; the next episode's `move_to_start` re-opens it.

**Fix:** make the STOP path call `self.arm.servo_stop()` (ur_rtde: kill servo thread +
decelerate) instead of `stop()`, and have `submit()` return False once `_stop` is set.

## 8. gripper_ctl (certain)

Works as advertised on the mock. On the rig: no mutual exclusion with a running deploy
process (the URCap socket happily takes a second client; `_grip_worker` and the tool would
interleave `SET` lines — the docstring says "between runs", nothing enforces it); `reset`
does ACT 0 → sleep 0.5 → ACT 1 with no STA/FLT read, so a gripper in fault after an E-stop
mid-motion may need a second reset; `close` defaults to 1.0 and is silently clamped to 0.9
by the driver. `reset` on the mock is just `open` (no `_set`), so the test exercises none of
the reset logic. Low priority; a pidfile/flock and a STA check are cheap.

On force/speed: force is clamped to `max_force_cmd` = (30−20)/215 → FOR 10 ≈ 28.6 N (the pad
ceiling); the close *speed* (SPE 127) is not bounded by FOR on an empty close, which is what
"slams into itself" is. Do not lower the speed blindly — it changes gripper dynamics vs the
demos; if the pads are marking, trim `max_close_cmd` (0.9 → 0.88) instead.

## 9. Lift speed after a close, speed cap

`--max-tcp-speed` lowers `arm.limits.tcp_speed_m_s`, which the executor applies per tick on
the measured dt (executor.py:240-246). There is no separate post-close lift cap; the governor
scales playback by predictive sigma only. Given the failure mode (closes on air, lifts) this is
not a safety issue. The cap is only tested as a config copy — nothing checks the executor
honours it.

## 10. RTDE reconnect

No reconnect anywhere by design: the arm-state worker dies on `isConnected()==False`
(workers.py:250-252) → `worker_died` → rc 5; the parent's `_recv` raises in `get_state`
(ur.py:294-298), which `_gate` calls unguarded → rc-1 traceback out of `main` (the `with`
still tears down). Acceptable for a 3-week horizon; a robot reboot means restart the
process. `SafetyMonitor.recovered()` (safety.py:250-287) is dead code on the deploy path.

## 11. Smaller correctness notes

- `hitbox_exit` / `workspace_clamp` events store `norm(target)` as the value
  (safety.py:199-209) — useless for the operator; store the offending axis and the overshoot.
- `zfloor:` tag reports the workspace floor even when no task floor was applied
  (run_deploy.py:329).
- `URArm.move_j` has no `_servo_active` guard while `MockArm.move_j` does (mock/ur.py:102-104);
  cannot bite today because every path that leaves the guard up is a control-dead reason.
- `snapshot_yaml()` round-trips the modified hw with `hitbox_m` (checked here; the real-arm
  worker processes are spawned from that YAML and no test covers it).
- Joint gate + jittered homing: 1-σ independent per-axis TCP jitter (rotation σ up to 0.16
  rad) can land 1.5-2 joint-σ on wrist joints for whiteboard; the 5° floor keeps it under
  2.5 but expect occasional "re-run homing" refusals — not a bug, just know why.

## 12. Missing tests (concrete)

1. Floor + hitbox together (the shipped default) at the SafetyMonitor and executor level:
   z below floor → CLAMP + continue (fails today), y beyond envelope → STOP.
2. Executor honours a lowered `tcp_speed_m_s` (per-tick displacement ≤ v·dt_eff).
3. `run_deploy._gate` passes `q` — a fake rig whose arm reports `q_mean + 2π` on wrist 3
   must refuse (the existing tests only call `start_sigma_report` directly).
4. `submit()` after `_stop` → rejected; STOP path ends with `servo_stop` called.
5. `snapshot_yaml` round-trip with hitbox (one-liner).
6. Deploy episode `meta.config_hash` equals the base hw hash (after §6 is fixed).
7. Once §3/§5 land: joint-envelope STOP, safety_mode ≠ 1 → PROTECTIVE_STOP verdict.
8. gripper_ctl `reset` against a fake socket gripper (ACT 0 → ACT 1 → STA poll → POS 0).

## Priority for the next rig session

Fix §1 (one line) and re-run §2's generator on the full dataset before any A/B episode is
counted; §3(b) and §7 are each a few lines and remove the two ways a mid-episode stop can
still go wrong; §5/§6 are half-day items that pay off the moment rollouts are reused as data.
