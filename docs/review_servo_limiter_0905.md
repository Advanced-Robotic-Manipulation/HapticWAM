# Servo-level limiter (rig 2026-09-04) — review before enabling

Scope: the box-local patch written on compute3 on 2026-09-04 19:01–20:07 (servo
limiter in `phantom/drivers/real/ur.py`, `stop.json` persistence, policy-server
accept robustness, `--lift-complete-z 0` default, wrist stop 0.468 in the NUC yaml).
Merged to main on 09-05 with the limiter **OFF by default** (`elbow_min_rad: null`,
`servo_joint_speed_max_rad_s: null` in `SafetyConfig`); everything else ships as
written. Reviewer: adversarial Opus pass that executed the limiter against its own
test harness. Reproduced items are marked.

## Why it is off
Correction (Ilya, issue #4): the limiter DID run on the arm for the last block of 09-04
(19:50–19:56, six episodes, `elbow_min_rad=0.40 / 1.0 rad/s`); `ur.py` received the
limiter at ~19:35 and the 20:05 edit only added the counter persistence, which is why
those `stop.json` files carry no `servo_limiter` field. In that block the apex dropped to
0.34–0.36 m, max joint speed 0.4 rad/s (vs 1.0–1.3 before) and no guard fired — but six
episodes cannot separate "limiter engaged" from seed variance. It stays off because
three defects reproduce in its own harness, one of which (item 2) is a live crash path
at the parking distance the 0.468 m stop now allows.

## Blocking (fix before enabling)
1. **IndexError on an unreachable target** (`servo_l` log line) — *fixed on main 09-05*:
   `np.degrees(q[2])` on an empty IK result; this is precisely the tick the limiter
   exists for. The empty-IK case also no longer enters the limiter when it is disabled.
2. **Deadlock when the anchor already violates `elbow_min_rad`** (reproduced): bisection
   assumes the previous streamed pose is feasible. Parked at elbow 0.30 rad, 50
   consecutive retract commands all HOLD; holds never update `_last_qsol`, so the
   branch guard reaches 25 rejects and raises `RuntimeError` → `executor_crash`. The
   entry condition is live: the 0.468 m stop allows parking at 0.464 m = elbow 0.344 rad
   < 0.40. Fix: when `prev` is infeasible accept any candidate that strictly increases
   `|q[2]|`.
3. **The executor never learns a step was shortened**: `ChunkExecutor._last_cmd`
   advances by the full rate-limited step while the driver streams a fraction.
   `record_action` then logs actions that were never executed (the same corruption
   class the aperture-latch fix addressed), `plan.t0_pose = _last_cmd - c0` re-anchors
   every replan to a phantom pose, and `safety.check` / `clamp_target` / hitbox judge a
   pose the arm is not at. Fix: `servo_l` returns the streamed pose; the executor adopts
   it as `_last_cmd`.

## Should-fix
4. Step vs slide fractions are compared as if commensurable: a large fraction of a
   near-zero tangential step beats a small fraction of the real step. Measured: 20 ticks
   of a sustained outward request at the boundary → 0.005 mm radial progress, mode
   `slide 88%` every tick. The stop is replaced by a silent stall, not boundary-hugging.
   Compare by achieved displacement.
5. Up to 7 `getInverseKinematics` calls per 8 ms tick (1 + 3 bisection + 3 slide) inside
   `_ctrl_lock`; each is an RTDE round trip on a controller whose own cycle is 8 ms, and
   it engages on most carry ticks (demo max reach 0.461 m = elbow 0.416 rad, only
   0.016 rad above 0.40). The overrun is masked because `dt_eff` clips at `2*period`.
   Measure the per-tick time on the rig; cut to 2 bisection levels or solve the reach
   sphere analytically.
6. Policy-server accept loop retried a broken *listener* with no backoff (100% CPU +
   log flood) — *fixed on main 09-05* (backoff + 20-failure cap).
7. `DeploymentRuntime._arm_state_now` samples after `executor.stop()`/`stopJ`, so
   `stop_state` records the arm after the stop (qd ≈ 0), not at it. Capture at `_halt`.

## Notes
8. `str(SafetyAction.STOP_EPISODE)` is `"SafetyAction.stop"` — *fixed* (`.value`).
9. Double `stop:` tag (`stop:none` + `stop:None`) — *fixed* (runtime tags, run_deploy
   only adds `evt:` tags).
10. Tests: the six pass and are not tautologies, but the "shortens step towards
    boundary" assertion measures a 1.7 µm second-order artefact of the tangent chord and
    passes while item 4 is active. Untested: the hold path, empty IK, an infeasible
    anchor, a negative elbow, missing `hw.safety`.
11. Checks out: wd(0.40 rad) = 0.4617 m (author: 0.4616), wd(0) = 0.4705, 0.468 m ↔
    elbow 0.213 rad so the clamp sits 6.3 mm inside the stop; `|q[2]|` is the right
    quantity; `self.hw` is always set (`drivers/base.py`); a hold sends no servoJ and
    the UR script tolerates that by holding the last setpoint.

## Enabling later
Set in `configs/hardware.nuc.yaml` under `safety:` — `elbow_min_rad: 0.40` and
`servo_joint_speed_max_rad_s: 1.0` — after items 2–5 are addressed and one bench run
confirms the tick time. Counters land in `stop.json` (`servo_limiter`).
