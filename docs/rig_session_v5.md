# Rig session — how to run inference on compute3 (Session 6: teachers vs sensor-free students)

## 🎯 SESSION 6 (2026-09-07): teachers vs SENSOR-FREE students — read this first

Seven models are on the box and on the `./PICK.sh` / `./SERVE.sh` menu. Ports follow
the menu order (slot 1 → 7777, slot 2 → 7778, ... slot 7 → 7783); one SERVE.sh
terminal per model you want warm. Everything is also on the hub
(`armteam/phantom-checkpoints`), so nothing on the box is precious.

| menu | what it is | inputs | offline val124 (mean / median mm) | expect on the rig |
|---|---|---|---|---|
| `v5_6` | teacher (lead until 09-04) | camera + tactile pads | 18.2 / 15.7 | the reference arm of the v5_6 pair |
| `ftA` | teacher, fine-tuned (all 09-01/09-04 grasps) | camera + tactile pads | 19.6 / 13.7 (waffles 24.4) | the reference arm of the ftA pair |
| `stu_ftA_r1` | **student, round 2** — distilled from ftA, teacher at full NFE, trained sigma head | camera + proprio ONLY (no pads) | **22.7 / 16.4** (best student) | the paper's student: should grasp, plausibly less often than ftA; watch for closes on air |
| `stu_ftA` | student, round 1 — distilled from ftA | no pads | 23.6 / 17.9 | slightly worse than r1; run only if r1 disappoints or time allows |
| `stu_v5_6` | student, round 1 — distilled from v5_6 | no pads | 27.3 / 18.6 | weaker on waffles (39.9 offline); the v5_6 pair's student |
| `ctl_ftA` | CONTROL: same inputs/init/steps as stu_ftA, NO distillation | no pads | 24.6 / 17.0 | ablation for the table; expect ≈ stu_ftA or worse |
| `ctl_v5_6` | CONTROL for stu_v5_6, NO distillation | no pads | 37.7 / 24.4 | ablation only; expect the worst grasp rate of the day |

The students and controls print `[STUDENT: sensor-free]` in the menu. They still run
with the pads plugged in (the safety guards, the aperture latch and the labels use the
pads) — the MODEL just never sees them.

### Order of the day (each pair = 16 paired episodes, alternating arms per cell)
1. **Warm up three servers** (terminal A ×3): `./SERVE.sh` → `v5_6`, `./SERVE.sh` → `ftA`,
   `./SERVE.sh` → `stu_ftA_r1`. ~3 min each, then every launch attaches in seconds.
2. **Pair 1 — `ftA` vs `stu_ftA_r1`** (the headline number). Same object placement for
   both arms of a pair, same seed, same start; alternate which arm goes first.
3. **Pair 2 — `v5_6` vs `stu_v5_6`** (warm `stu_v5_6` while pair 1 runs).
4. **Pair 3 (if time) — `stu_ftA_r1` vs `ctl_ftA`**: this is the "does distillation
   matter" row; the control should be visibly worse. Skip `ctl_v5_6` unless the day is
   long — its offline number says it will mostly fail.
5. Optional Block A at the start if the rig behaves oddly: 5 demo replays through the
   real deploy path (`tools/replay_rig.py`) to confirm the arm follows a known-good plan.

### Operator rules that changed since Session 5
- **A bare Enter no longer stops an episode.** Type `x` + Enter to end a carry cleanly
  (gripper stays closed); stray newlines (wireless receiver) ended 5 episodes on 09-04.
- **Label every episode** at the prompt: `s` (grasped and carried), `f`, `c`. Unlabeled
  episodes are excluded from the A/B — the 09-04 set had to be rule-labelled after the
  fact. Take the object OUT of the gripper before confirming the next homing.
- **Auto-success is OFF** (`lift_complete` cut carries short on 09-04); the wrist stop
  is 0.468 m, the servo limiter is OFF, the aperture latch is ON (a held object cannot
  be commanded open mid-carry).
- **One robot owner per host.** A second `run_deploy` — including one Ctrl-Z'd in
  another terminal — is refused by name before it touches the robot. If a launch says
  "rig owned by pid …", `fg`/kill that process; do not fight it. If it says the server
  is BUSY, another run_deploy is still attached to that model's server.
- Free disk is checked at launch (abort < 5 GB, warn < 20 GB); the box had 105 GB on 09-06.

### What "expected" looks like, honestly
Offline mm has not predicted rig behaviour well in either direction. What the numbers do
say: the round-2 student is 3 mm from its teacher offline and its egg/whiteboard numbers
are as good as the teacher's; waffles (our rig task) is its weakest task. A student that
grasps waffles at a usable rate is the paper's result; a student that closes on air is
the known failure mode (the tactile phantom recovery only sees pad load, which the model
does not). Record per episode: label, whether the close was on the object, and whether
a guard or `x` ended it. That plus the pair structure is the whole statistics block.

## ⚡ Morning quickstart (Session 5) — the whole flow

```
# terminal A (once, ~3 min):        # terminal B (every run, seconds):
cd ~/phantom-icra-2027              cd ~/phantom-icra-2027
./SERVE.sh                          ./PICK.sh
```
1. `SERVE.sh` picks + holds the model on the GPU. `PICK.sh` attaches to it automatically.
2. A failed start gate now FIXES ITSELF: 3 s countdown, slow joint-space auto-home,
   re-check. You never jog or restart for a start-pose problem.
3. Episodes end on a guard stop, the 150 s budget — or type `x` + Enter anytime for a
   clean manual end (gripper stays as-is). A bare Enter is IGNORED since 09-05: on
   09-04 stray newlines (wireless receiver + primary-paste) ended 5 of the first 10
   episodes within seconds. The `lift_complete` auto-success is OFF since
   09-04 (it fired at z 0.32 m, below the demo apex, and cut every grasp before the
   carry; `--lift-complete-z 0.32` restores it). Label honestly at the prompt.
   Every episode now writes `stop.json` (reason, safety events, arm state at stop).
4. If a safety stop prints the "label this episode a SUCCESS" NOTE, believe it: the
   guard beat the success detector on a good grasp.
5. TAKE THE OBJECT OUT OF THE GRIPPER before confirming the next episode's homing —
   homing opens the fingers to the start aperture (this is what dropped the 09-01 grab).
6. A/B protocol for the session: see "Session-5 A/B protocol" below. Interleave
   v5_6 / ftA_1500 per cell, same config, let episodes terminate, >= 8 pairs.


State as of 2026-08-28. Everything below was verified live on compute3 (repo `b0e87b5`, torch 2.13 + CUDA OK).

## What is on the box
```
~/phantom-icra-2027/phantom/runs/teacher_v5_batch0822/DEMO.pt -> v5_6.pt   # v5 fine-tune, final (best) build
~/phantom-icra-2027/phantom/runs/teacher_v5_batch0822/v5_1.pt              # first v5 build (for ablation only)
~/phantom-icra-2027/phantom/runs/teacher_v4_790eps/DEMO.pt   -> teacher_020000.pt   # v4 CONTROL (unchanged)
```
`v5_6` is byte-identical (sha256) to hub `armteam/phantom-checkpoints/teacher_v5_batch0822/teacher_003000.pt`.
All six v5 builds, the training log and every offline eval JSON are on the hub in that folder.
Build labels on the box hide step numbers on purpose; the mapping is `runs/teacher_v5_batch0822/.stage_map`.

## Preflight (once per session)
1. Robot ON (pendant: Enable robot, Start), gripper powered, RealSense on the **USB3** hub port, gel sensors connected.
2. `ping 192.168.88.56` must answer (UR). `GO_ANY.sh` checks the arm and the camera USB3 mode and refuses otherwise.
3. Nobody else on the GPU: `nvidia-smi` (a few GB from other users is fine; a training job is not).
4. Place the taped 3x3 grid (40 mm spacing) centred on the demo object position.

## Launch — two arms, same flow, same flags
```
cd ~/phantom-icra-2027
./GO_waffles.sh     [episodes=3]   # v4 control  (runs/teacher_v4_790eps/DEMO.pt)
./GO_v5_waffles.sh  [episodes=3]   # v5 build    (runs/teacher_v5_batch0822/DEMO.pt)
```
(same for `Carton`, `egg`, `whiteboard`). Both go through `GO_ANY.sh` -> `run_deploy.py` with
`--ema --persistent-noise --nfe 5 --guidance 1.0`, `configs/hardware.nuc.yaml`, EMA weights (deploy default).
Per-episode flow (unchanged from LAUNCH.txt): wait ~3 min for model load + warmup; **Enter #1** homes the arm
slowly to the task's demo start pose (clear its path, e-stop in hand); the sigma gate prints live-vs-demo per axis
and refuses > 2.5 sigma (jog and Enter to re-check); **Enter #2** starts the episode; at the end answer the
outcome prompt: `s` success / `f` failure / `c` contaminated (a hand in frame etc.).

## Fast launches: hold the model warm with SERVE.sh (terminal A)

```
cd ~/phantom-icra-2027 && ./SERVE.sh     # pick the model once; it stays on the GPU
```
Then every `./PICK.sh` launch in terminal B attaches to it in seconds instead of the
~3 min in-process load (PICK auto-detects; if the server holds a different model it
says so and loads locally — restart SERVE.sh to switch). The robot process can crash,
be Ctrl-C'd and relaunched freely; the model never reloads.

## Session-5.1 changes (2026-09-05, issues #4 #7 #8 #9)
- **One robot owner per host**: `URArm.connect(control=True)` takes an `flock` lease
  (`/tmp/phantom-rig-<arm-ip>.lock`) before its first control call. A second process —
  including a Ctrl-Z'd `run_deploy` that still holds the RTDE registers — is refused
  with the owner's pid, state (`T (stopped)`) and command line, and sends NOTHING to
  the robot. `fg` / kill the owner, then relaunch.
- **Busy ≠ absent**: the policy server answers `info` to every connection, even while a
  client owns it. PICK/`--policy-server auto` now abort on a BUSY or non-answering
  server instead of loading a second model in-process and fighting for the arm.
  Probe by hand: `.venv/bin/python -m phantom.scripts.policy_server --probe --port 7778`
  → `<ckpt> idle|busy <sha>`.
- **The executor anchors on what the arm received**: `servo_l` returns the streamed
  pose (or a held/rejected reason); a held tick no longer advances `_last_cmd`, the
  recorded action, or the next replan's anchor. Empty/NaN IK never reaches servoJ.
- **Provenance**: `ckpt_sha:<12>` (digest of the loaded file — `BEST.pt` is a moving
  name), `meta.deploy_overrides.safety_effective` (every guard/latch/lift value in
  force), `stop.json.stop_state.at_halt` (arm state AT the halt, before stopJ).

## Session-4.1 behavior changes (2026-09-01, post-session fixes)
- **Start gate self-corrects**: a failed joint gate now AUTO-HOMES (slow joint-space
  move, 3 s countdown — e-stop if the path is not clear) and re-checks, twice, before
  ever asking the operator. Same on the re-gate after "place the object". A wrapped
  wrist unwinds itself.
- **`lift_complete` success auto-stop — OFF by default since 09-04** (Ilya's call from
  the second half of session 5: at z 0.32 m it fired below the demo apex and cut every
  good grasp before the carry; the operator ends carries with Enter). Opt in with
  `--lift-complete-z 0.32`: both pads loaded (trailing 0.6 s max of |fz| > 4.0 N over
  the per-episode baseline — the left pad carries a drifting 0.7-2.2 N zero offset)
  with the TCP above z, held 0.4 s -> the episode ends cleanly HOLDING the object,
  reason `lift_complete`. Knobs `--lift-complete-z/-fz/-hold`.
- **Singularity guards, layered** (root cause = IK branch flip at the elbow-straight
  boundary, wrist-centre 470.5 mm): (1) `servo_l` seeds IK with the previous solution
  and REFUSES branch-flipped solutions (driver-level, the actual fix); (2)
  `wrist_extension` stop at wd > 0.462 m — pure elbow geometry, above every demo
  ever recorded (max 461 mm) and below the whip zone (465+), non-release; (3) chunk-tail cap — playback holds after
  10 action steps (`--max-play-steps`; steps past HEAD_STEPS are never validated by a
  replan and all four whips began there); (4) measured joint speed > 1.2 rad/s stop
  (last-ditch — max whip-free qd all session was 0.73). The 0.62 m TCP-radius clamp is
  RETIRED (0/4 whips; it fought one lift) — the mechanism remains, default off.
  (5) **servo-level limiter** (Ilya, 09-04 evening — written AFTER the last episode of
  the day, never ran on the arm; **OFF by default** pending the fixes listed in
  `docs/review_servo_limiter_0905.md`, enable via `elbow_min_rad` /
  `servo_joint_speed_max_rad_s` in the NUC yaml): per 8 ms tick, if the IK solution would fold the
  elbow below `elbow_min_rad` (0.40 rad = wrist centre 0.4616 m) or move a joint faster
  than `servo_joint_speed_max_rad_s` (1.0), the driver shortens the step (bisection,
  then a tangential slide along the reach sphere) instead of streaming it, so the arm
  hugs the reach boundary rather than tripping the stop. Each extra IK probe is one
  RTDE round trip (~8 ms), so expect a visible slow-down near the boundary. The NUC
  yaml raises `wrist_extension_stop_m` to 0.468 as the net behind it. Counters land in
  `stop.json` (`servo_limiter`); `null` for both fields = off.
- **`stop.json` per episode** (Ilya, 09-04): reason, every safety event, arm state at the
  stop (q, wrist-centre distance, qd), limiter/IK-guard counters — diagnose from disk.
- **Aperture latch + tactile recovery** (09-04 analysis): once both pads carry >2.5 N
  the commanded closure can only increase (4/5 objects "lost" were policy-commanded
  releases; `--no-grip-latch` disables); a close on air that lifts >30 mm with both pads
  quiet for 1 s triggers `recovery_tactile` (open + re-descend) instead of a phantom carry.

  Known labeling quirks to watch for in logs: a sustained IK-reject hold surfaces as
  `executor_crash` (the crash net wins over motion_stall); a wrist_extension stop
  during a confirmed hold prints a NOTE to label the episode a SUCCESS.

## Session-5 A/B protocol (from the 09-01 run analysis — do it THIS way)
1. Same inference config both arms: nfe1 / k-seeds 4 / veto on / parity on (LEVERS).
2. **Paired seeds AND paired start poses**, alternating v5_6 / ftA_1500 per cell,
   >= 8 pairs. (09-01 spanned 4 configs x 3 git revisions x all-different seeds and
   starts — no within-day comparison was valid.)
3. Let episodes reach a TERMINAL condition (a guard / the 150 s budget), or end a
   completed carry with Enter and LABEL it `s` — with `lift_complete` off, the label
   is the only success record (the 09-04 set is unlabeled for exactly this reason).
4. Primary metric: tactile-confirmed grasp rate (label + `tools/label_grasps` from the
   pad forces); secondary: apex z and carry duration. Do not chase offline
   endpoint-mm on the rig.
5. The one v5_6 episode of 09-01 (3.7 s, 4 replans) says NOTHING about v5_6 — and its
   "sideways press" start-pose region made ftA drift even harder. Not a finding.

## Or: the interactive launcher (PICK.sh) — pick model x preset from a menu

```
cd ~/phantom-icra-2027 && ./PICK.sh
```
Asks four questions and launches:
1. **Model** — the curated list from `MODELS.tsv` (only builds worth running): `v5_6` (LEAD, val124
   endpoint 18.23 mm), `v4` (CONTROL, 20.82 mm), `ftA` (EXPERIMENTAL: best median 13.4 mm but 21.0 mean,
   ends ~12 mm high and under-commits — try only after arms A/B read out).
2. **Preset** — `LEVERS` (recommended arm-B stack: nfe 1 + terminal-veto + parity-fixes + k-seeds 4 +
   max-episode-s 150 + max-replans 200), `PLAIN` (nfe 5, pre-fix style, attribution control only),
   `VETO` (nfe 5 + veto + parity), or `CUSTOM`.
3. Task + episode count, plus optional extra flags.
It prints the exact command and waits for Enter. **During an episode, pressing Enter ends it cleanly** (stop_reason `operator_stop`: motion stops, the gripper is NOT opened — a held object stays held); the 150 s budget is the backstop, the operator is the normal end. To add a new model to the menu: put the concrete `.pt`
under `phantom/runs/...` and add a `label<TAB>path<TAB>note` row to `MODELS.tsv` (no DEMO symlinks there).
Tracked copies of PICK.sh / MODELS.tsv / the GO family live in the repo at `tools/rig/`.

## Protocol for a comparable A/B
- One task first (waffles). Per grid cell: one v4 episode, then one v5 episode (interleaved), same placement.
- >= 10 episodes per arm per task (20 to see anything smaller than a night-and-day effect).
- Primary number = miss distance / z at close from `planner_trace.json` (not just grasp count). Note the cell id in
  the episode notes prompt. Film 2 successes + 2 misses per arm.
- Episodes land under `runs/deploy_*/ep_teacher_<task>_<epoch>_<seq>/` with `planner_trace.json`, condition tags
  in `meta.json` (`nfe5`, `g1.0`, `pnoise`, `ckpt:<file>`, ...).

## Staging a different build (needs the rental to exist; otherwise pull from the hub)
```
# from the hub (any time):
cd ~/phantom-icra-2027/phantom && .venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil
p = hf_hub_download("armteam/phantom-checkpoints", "teacher_v5_batch0822/teacher_002500.pt", repo_type="model")
shutil.copy(p, "runs/teacher_v5_batch0822/v5_5.pt")
PY
ln -sfn v5_5.pt runs/teacher_v5_batch0822/DEMO.pt
```

## Session 4 recipe (2026-08-29) — the sampler was seeded to a constant; every earlier session ran ONE noise draw

Found 08-29 by replaying the rig's own recorded states: `phantom/model/rf.py` seeded the flow sampler's noise
generator to a constant and `run_deploy` never reseeded it, so the warm-up replan consumed a fixed number of draws
and **episode 0 of every rig session sampled the same persistent-noise tensor** — a 6th-percentile "slow" draw
(the rig's chunk equals the slowest of 16 seeds at every replan; replaying the deploy RNG sequence reproduces every
rig chunk to 0.3 mm, v4 and v5 alike). The model's actual seed spread at those states is −93…−26 mm per replan;
demos descend ≈ −45. v4 vs v5 on the rig was never a model comparison.

Fixed in `ba61354`: every episode now draws a fresh seed and records it (`seed:<n>` in meta tags; pass `--seed N`
for reproducible arms). New deploy levers, all flag-gated and tagged in meta: `--nfe 1` (deterministic conditional
mean, the most committed descent in the sweep and 172 ms replans instead of 865), `--terminal-veto` (no close unless
the contact gate agrees or the gripper is at the demo grasp band; a close on air reopens and re-descends, 3 tries),
`--parity-fixes` (train/deploy input parity), `--k-seeds K` (multi-seed selection; only affordable at NFE ≤ 3).

Arms for the first clean session (waffles first — `q_n=37`, the best-covered envelope; then Carton).
**One process per arm block**, ≥16 episodes per arm so G2's 3/16 is readable:
```
# arm A — v5_6 baseline, NFE 5, no levers.  ONE process for the whole block:
EXTRA=""                                                    ./GO_v5_waffles.sh 10
# arm B — the lever bundle:
EXTRA="--nfe 1 --terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 150 --max-replans 200" \
                                                            ./GO_v5_waffles.sh 10
```
(`EXTRA` is appended after the GO script's own flags, so its `--nfe` wins.)

**Do NOT pass `--seed` to a single-episode process.** `episode_seed(base, i) = base + i` uses the *within-process*
episode index, so `EXTRA="--seed 4242" ./GO_v5_waffles.sh 1` repeated ten times gives seed 4242 ten times — with
`--persistent-noise` that is one identical noise tensor for the whole arm, i.e. exactly the constant-seed bug this
session exists to escape (VALIDATION_0830 P0 #6). If a seeded arm is wanted, run the whole block in ONE process; the
unseeded path already records `seed:<n>` per episode (and since 08-30 the homing jitter is drawn from that same
per-episode seed, with the realised start pose tagged `start:<x,y,z>mm/g<aperture>`).

`--max-episode-s 150` is the wall-clock budget, and **`--max-replans 200` is what lets arm B reach it**.
`PlannerLoop.run` checks the replan COUNT before the wall clock, so the default 40 ends an `--nfe 1` episode
(172 ms replans) at `replan_cap` after **7.2 s** — measured through the real loop — against a 35 s arm A and
16-31 s demos. Without the raise the two arms are not the same experiment and arm B never reaches the terminal
phase. Arm A at NFE 5 (~0.9 s replans) hits 40 replans at 34.8 s and does not need it. The stop reason is logged
as `episode_time_cap` / `replan_cap` instead of a silent `None`.

Interleave A/B per placement cell on the taped 3×3 grid and alternate which arm goes first per cell; note the
cell id in the verdict prompt (`s`/`f`/`c`, optionally `d` for damage). The z floor, hitbox and joint gate are on by
default; if the joint gate refuses after a protective stop, unwind wrist 3 on the pendant. Judge with
`tools/label_grasps.py` (tactile hold + lift) in addition to the operator verdict, and treat any `hold_truncated`
episode as unmeasurable rather than as a failure.

**Abort rules.**
- Two consecutive `safety_stop`s that need an RTDE control rebuild → stop and inspect the envelope before continuing.
- Any protective stop or manual jog → re-run the joint gate before the next episode (the 08-28 session lost 16 of 26
  episodes to a wrist wrapped 360° / a flipped IK branch that the TCP-only gate passed).
- Any `veto_retry_cap` → check the gripper before touching the rig. A let-go stop (`tactile_*`, `wrench_limit`,
  `hitbox_exit`, `veto_retry_cap`) now commands the fingers open itself; if they are still closed, `./GRIPPER_OPEN.sh`.
- `--hitbox-margin` must stay above `--z-floor-margin` — `run_deploy` refuses the run otherwise (the floor-is-a-clamp
  fix inverts below it and every deep descent becomes a `hitbox_exit`).

## Safety batch (2026-08-28 evening) — what changed after the first v5 session

Root cause of the "later" 08-28 episodes (16 of 26): after the protective stop / manual jogging the arm was
left in a **different joint configuration** — wrist 3 wrapped by a full turn (+183 deg instead of -179 deg) or a
flipped elbow branch (base rotated -127 deg). The TCP pose passed the 2.5-sigma gate, but the policy consumes the
**raw joint vector** (`ur_state = [q, qd, tcp_pose, tcp_speed, gripper]`), so it ran ~60-150 sigma out of
distribution and closed the gripper at the start pose / wandered. Every episode from 17:19 on is invalid data.

New in `run_deploy` (all default-on, tags land in `meta.json` as `zfloor:`, `hitbox:`, `vmax:`):
- **Joint-space start gate**: the sigma table now lists `q1..q6` against the demo start configuration
  (`start_poses.yaml q_mean/q_std`, 5-deg std floor). A wrapped/flipped joint prints a `!!! FULL-TURN` line and
  gates out. Fix it on the pendant (joint jog wrist 3 by -360 deg, or move back to the demo branch), then Enter.
  `--home-joints` does a slow moveJ to the demo joint configuration BEFORE the moveL homing — path must be clear,
  a base rotation sweeps the bin; E-stop in hand.
- **z no-go floor** (clamp): commanded TCP z >= task demo `tcp_z_min` - 10 mm. Regenerated from
  `configs/start_poses.yaml` after F17 re-fitted the envelope over all 250 episodes/task (E13_rescore.md §1):

  | task | `tcp_z_min` mm | floor mm |
  |---|---|---|
  | waffles | 41.5 | **31.5** |
  | egg | 59.5 | **49.5** |
  | whiteboard | 67.5 | **57.5** |
  | Carton | 76.1 | **66.1** |

  A live run prints these as `zfloor:32mm` etc. in the meta tags — if the tag and this table disagree, the
  binary is not the one this page documents. (The pre-F17 subset floors were 10.5 mm too HIGH on waffles and
  7.1 mm too high on whiteboard — the wrong direction of error on a task whose failure mode is closing too
  high.) Override `--z-floor <m>`, margin `--z-floor-margin`, off `--no-z-floor`.
- **STOP hitbox**: the task's demo TCP envelope (`tcp_min/tcp_max`, all frames) +/- 30 mm; a commanded target
  outside ENDS the episode (stop reason `safety_stop`, event `hitbox_exit`). `--hitbox-margin`, `--no-hitbox`.
- **Speed cap**: `--max-tcp-speed <m/s>` lowers the executor's commanded-TCP cap below hardware.yaml's 0.25.
  (Demos peak at 0.24-0.34 m/s during transport, so 0.25 is already conservative; 0.15 is a sane "careful" value.)
- **`--max-replans` default 20 -> 40**: the 20-replan cap (~19 s) cut every retry short; demos run 16-31 s.

Gripper from the shell (between runs, when no deploy process owns it):
```
cd ~/phantom-icra-2027 && ./GRIPPER_OPEN.sh      # open (activates first if needed)
cd ~/phantom-icra-2027 && ./GRIPPER_RESET.sh     # ACT 0 -> ACT 1 calibration stroke -> open (after e-stop / power cycle)
.venv/bin/python -m phantom.scripts.gripper_ctl status --hardware configs/hardware.nuc.yaml
```

After an E-stop / protective stop, in this order: (1) clear the stop on the pendant, (2) if the arm was moved by
hand or jogged, expect the joint gate to complain — unwind wrist 3 / return to the demo branch on the pendant,
(3) `GRIPPER_OPEN.sh` if the gripper is stuck closed, (4) relaunch; the first Enter homes with moveL (or moveJ+moveL
with `--home-joints`), the gate re-checks before anything runs.

Pass extra flags through the GO scripts with `EXTRA`, e.g. `EXTRA="--home-joints --max-tcp-speed 0.15" ./GO_v5_waffles.sh 3`.

**Electrical**: a shock from the arm after the E-stop is NOT a software condition and nothing here addresses it —
stop touching the arm while the controller is in a fault state and have the control-box earth / the USB chain
between the NUC, the tactile sensors and the robot checked before the next session.

## If something refuses to start
- `arm NOT reachable` -> robot off / pendant not started / wrong subnet.
- `NO CAMERA` or USB type not 3.x -> replug the RealSense into the back USB3 hub port.
- Gate refuses with a sigma table -> jog the arm toward the demo start (or place the object mid-mat), Enter to re-check.
- After a protective stop: clear it on the pendant when prompted; `run_deploy` rebuilds RTDE control itself
  between episodes and refuses the next episode if the script did not come back (exit code 4/5 = restart the process).
- Stale camera / stale arm stream now end the episode cleanly (stop reason in the log) instead of crashing.

## Offline numbers behind v5_6 (terminal_eval, val124, 4 seeds, EMA — re-scored 2026-08-30)
Endpoint error v4 **20.82 mm -> v5_6 18.23 mm** (2.6 mm / 12%); z-at-end **+2.76 -> +4.16 mm**; commit ratio
**1.28 -> 1.10**; close-step error **+0.19 -> -0.67 steps**; new-batch holdout (the 46 `batch_20260822`
episodes) **18.54 -> 13.78 mm**. The per-window across-seed std is 6-7 mm — larger than the whole v4->v5_6
difference — so no per-episode claim follows from these, and most of the val124 gain is in-distribution to the
new batch (on the frozen v4-only half it is 22.17 -> 20.86 mm). Full derivation, caveats and the retracted
close-height statistic: **`docs/review_20260828/E13_rescore.md`**. Offline != rig: the rig decides.

## Audit trail (2026-09-01)
Every command execution and every file deletion/rename by ANY user (local or ssh)
is logged by auditd on compute3 (kernel-level, cannot be bypassed from a shell).
- `who-deleted [path-substring]` / `who-ran [string]` — helpers in /usr/local/bin
- raw: `sudo ausearch -k delete -i` / `-k exec`, log at /var/log/audit/
- rotation: hard 800 MB ring (8 x 100 MB), oldest overwritten — never bloats.

## Rig helper scripts (tracked in `tools/rig/`, box copies may be wiped from /tmp on reboot)
- `probe_rig.py` — pre-session wiring probe (arm RTDE + Robotiq + RealSense):
  `.venv/bin/python tools/rig/probe_rig.py`
- `gate_check.py` — print the live joint-gate sigma table without launching deploy
- `WRIST.sh` — live wrist-3 internal-angle readout while jogging (the pendant
  always displays 0-360; the internal value is what the gate reads)
