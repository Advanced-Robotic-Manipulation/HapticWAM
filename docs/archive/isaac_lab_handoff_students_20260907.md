> Archived student-scope handoff, superseded by the user's teacher-only request. See the [current handoff](../isaac_lab_handoff_20260907.md). Historical claims below describe the earlier revision.

# Waffles lab handoff — 7 September 2026

Start with a matched **pick-and-hold** pilot: revised ftA student and v5_6 student, each compared with the pinned ftA teacher. Test placement only after the physical release controller has been reviewed and validated. The simulator's placement-aware release is currently **simulator-only**; the native aperture latch can block an opening requested by the policy. A held packet, `lift_complete`, or an operator success label for pickup is not a completed placement.

This handoff is based on a read-only compute3 audit on 6 September, approximately 22:00 UTC, at live repository commit `0259ed3355f2ad3078778e7cc347b100775e06d3`. No hardware connection, robot/gripper command, model inference or server launch was performed for this audit. Eight existing fake-RTDE servo-limiter tests passed on that live source with bytecode and pytest cache disabled; this does not establish physical timing or safety. Source inspection takes precedence over older session notes where they disagree. Native command and behavior references: [run_deploy](../../phantom/scripts/run_deploy.py), [start-pose handling](../../phantom/deploy/start_pose.py), [executor](../../phantom/deploy/executor.py), [safety monitor](../../phantom/deploy/safety.py), [UR driver](../../phantom/drivers/real/ur.py) and [rig launchers](../../tools/rig/README.md).

## Models and a fixed inference recipe

Paths are relative to `/home/physicalai/phantom-icra-2027/phantom`.

| Role | Concrete checkpoint | Native system | SHA256 |
|---|---|---|---|
| Reference teacher |`runs/teacher_v5_ftA/teacher_001500.pt`|`teacher`|`67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e`|
| Revised ftA student |`runs/student_ftA_r1/student_001200.pt`|`student`|`b13362472843c4b09d9c7a091c0e97ff34a58ea45411961886f5d72ebf1703b3`|
| v5_6 student |`runs/student_v5_6/student_001200.pt`|`student`|`aee98c1cc01f06ba5b507422f433a7676c174603d92decf2e2c25e52dd857eca`|

Use the checkpoint's own architecture, normalization and EMA weights. The [checkpoint inventory](../results/student_readiness_20260906/student_inventory.md) confirms both requested students contain 630 finite EMA tensors and normalized-input statistics exactly matching the audited ftA teacher; their canonical normalizer SHA256 is `42153b79ceb6a6c711296323c6b171e42929eb5dfe4018198d63d2cd3ace8c48`. Both retain wrist F/T (`mask_wrist=False`): native `--system student` removes fingertip tactile model inputs, while camera, robot state and wrist F/T remain. Pads still run for safety, latch, recovery and recording. These are **fingertip-tactile-free students**, not fully sensor-free policies. Do not switch to `vision_only` or `drop_tactile` to run this comparison.

The matched runtime recipe is the source-supported LEVERS sampling/controller combination: task text `waffles`, EMA, NFE 1, guidance 1.0, persistent episode noise, K 4, parity fixes, live terminal veto, maximum 10 played action steps. K 4 means four candidate chunks per replan; it is independent of the episode sampling seed. Saved model NFE 5 is overridden explicitly by this established runtime recipe. Preserve checkpoint-specific noise semantics: revised ftA uses per-strip noise; v5_6 student does not. Do not rewrite either model configuration to make them superficially identical.

This first lab pilot explicitly shortens the menu's 150 s/200-replan budget to 35 s and uses the historical pickup stop at TCP Z 0.32 m. These are declared pickup-protocol choices applied equally to every arm, not a claim to reproduce the 60 s simulator placement campaign. The 35 s wall-clock budget is the sole budget because no explicit replan count is supplied. Keep the native 0.25 m/s TCP cap unless the lab selects and records a lower common cap before the first trial.

The subsequent [four-trial student simulator smoke](../isaac_student_smoke.md) completed with valid logs but **no acquisition, lift or placement for either student**. Revised ftA seeds 4242/4243 stopped on `wrench_limit` at 7.596/8.740 s, with estimated packet–pad normal peaks of 109.4/118 N; v5_6 stopped on `hitbox_exit_top` at 13.180 s and `wrist_extension` at 15.196 s. There were no IK rejects, stale-plan events or inference errors. This checks that the models can run through the adapter and records task failures under the current uncalibrated contact model; it does not establish nominal readiness, a physical force prediction or a model ranking. Begin the lab block with the matched teacher reference and pause for diagnosis if its pickup cannot be reproduced. Placement remains gated separately.

## Checks that do not control hardware

Run these on compute3 before allocating a model or connecting to the rig. They inspect files, processes and host resources only:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
git rev-parse HEAD
git status --short
sha256sum configs/hardware.nuc.yaml configs/start_poses.yaml
sha256sum runs/teacher_v5_ftA/teacher_001500.pt \
  runs/student_ftA_r1/student_001200.pt \
  runs/student_v5_6/student_001200.pt
ps -eo pid,stat,args | rg 'run_deploy|policy_server|record_episodes|teleop'
ss -ltnp
nvidia-smi
df -h /home/physicalai/phantom-icra-2027/data
```

The audited hardware YAML SHA256 is `2867897e49b3bf3fddb7ece6dd69e6f38be38f17dc0fb26a8887a7e804c55524`; start-pose YAML SHA256 is `13498363d7b3bac018ed41b248c4a933103a52555264e76c30fbc68fc4fbeb4c`. Record any reviewed revision before proceeding; do not silently mix configurations. The live repo has unrelated untracked files, including `configs/hardware.nuc.nosafety.yaml`; do not use that configuration or overwrite unrelated work.

The root-level `~/phantom-icra-2027/PICK.sh` is older than tracked `tools/rig/PICK.sh`: it still prints the obsolete instruction to omit `--seed`. The tracked version at 0259ed3 correctly requires paired seeds. Root-level and tracked `GO_ANY.sh`/`SERVE.sh` matched during this audit; the two `MODELS.tsv` copies differed in student description text. Use the explicit commands below. An operator may later reconcile launcher copies through a reviewed update, but this handoff does not modify them.

Do not infer model identity from a menu label, `BEST.pt`, a basename or a listening port. Native `run_deploy --policy-server` only **warns** if the requested checkpoint differs from the server's checkpoint and then runs the server model. The native probe reports a 12-character digest, not a full hash. Verify the full file hash above, the exact server launch command/system and matching probe digest. A busy or non-answering endpoint is a blocked launch, not a reason to start another robot process. Preserve unrelated resident servers 7777/7778 and any other user's processes.

The native disk check aborts below 5 GB and warns below 20 GB; free space is needed for every attempted episode. No live-source safety flags or configuration changes are authorized by this document.

## Human bench checks and a matched start

The checked configuration explicitly says `meta.bench_verified: false`. Review the relevant procedures in [hardware_bench_day1.md](../hardware_bench_day1.md) and verify the current physical rig before running policy control; do not mark the flag true without measurements. Source values describe the configured rig, not a certified collision-free motion path.

| Item | Audited configuration / required observation |
|---|---|
| Robot | UR3 **CB3**, previously identified as PolyScope 3.14.3/SN 2018333547; YAML declares e-series to accommodate the current-based wrist estimate. Reconfirm on the pendant. Retain 125 Hz control/receive/executor rates; do not switch to 500 Hz based on that YAML label. |
| Tool | TCP offset `[0,0,0.18,0,0,0]` m/rad, configured payload 1.2 kg; verify the mounted gripper/pads and payload. |
| Gripper | Robotiq 2F85, 85 mm nominal stroke, maximum normalized close 0.9, default force 0.04 with 30 N command ceiling. Closure 0=open, 1=closed; it is not a jaw-gap measurement. Verify pad clearance without squeezing an empty gel pair. |
| Sensors | Left `L26050098`, right `L26190169`; compare mounted sides, no-load baseline and live contact response. RealSense D435 serial `944622074411`, 640×480 color 15 Hz; depth disabled, wrist camera disabled. Keep learned text `waffles`. |
| Native geometry | Hardware workspace X [−0.7,0.15], Y [−0.5,0.3], Z [0.03,0.8] m. Waffles CLI derives TCP floor 0.0315 m and STOP hitbox X [−0.4985,−0.2206], Y [−0.4065,0.1758], Z [0.0115,0.5021] m. These are TCP limits, not complete robot collision geometry. |
| Guards | Wrist-center extension stop 0.468 m, measured joint-speed stop 1.2 rad/s, wrench deviation 60 N/15 Nm, tactile depth limit 0.6, stale-plan timeout 1 s, native governor, branch guard and 10-step playback cap remain active. Distributed-force calibration is unset (`dist_force_unit_to_N=0`), so verify the actual depth-based tactile safety path; do not interpret the 5 N configuration field as the only protection. |

The current-based CB3 wrist estimate has pose-dependent bias; it is not a calibrated external six-axis load cell. Simulator wrist/contact proxies and the fitted table/bin/pad dimensions are not calibration values to copy onto the physical rig. Recheck object dimensions, table height, box position, pad alignment, camera view and the full arm sweep on the bench. In particular, do not transfer the simulated release volume or fitted table-height correction into native safety limits.

For a **read-only device probe**, a bench operator may run the following after confirming device ownership. These connect to RTDE receive/gripper GET sockets/camera enumeration but issue no motion commands; they are separate from the file-only checks above:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python tools/rig/probe_rig.py
.venv/bin/python tools/rig/gate_check.py
```

The probe does not validate tactile streams or the complete start gate. Native deployment checks TCP, unwrapped joint coordinates, closure and settled gripper `OBJ=3`, and rechecks after scene placement. Keep the 2.5-sigma start gate; do not use `--allow-ood-start`.

Choose one observed start before the pilot, physically review its path and stage it with the trained operator using the pendant/approved bench procedure. Reproduce that state and object placement between paired arms. The [retained Sep4 teacher start](../results/teacher_pick_place_v1/sept4_policy_initial_state.json) supplies a measured reference, **not an automatically safe waypoint**:

```text
q(rad, RTDE order) = [0.23474513, -1.51981479, 1.07542038,
                     0.72334540, 1.17474580, -3.00465829]
TCP(m + rotation-vector rad) = [-0.35399663, -0.30305326, 0.33813600,
                               -1.22251299, -1.83900539, 1.60320911]
measured closure = 0.07843138; gripper OBJ = 3
```

The configured waffles distribution has TCP mean `[-0.3652,-0.2859,0.3346,-1.1053,-1.8268,1.5079]`, joint mean `[0.1966,-1.5291,1.1159,0.6794,1.1853,-3.1303]` and mean closure 0.232. A small TCP error does not excuse a full-turn joint mismatch. Record the achieved state rather than assuming the requested waypoint was attained.

The launch below uses `--no-home` so the native runtime keeps both start gates but does not perform its randomized homing or automatic re-homing. Default homing samples up to one standard deviation of TCP/closure jitter; even `--home-joints` first moves to joint mean and then performs a jittered moveL. There is no zero-jitter homing CLI flag. Never run a shell loop that homes from arbitrary arm configurations. Entering `run_deploy` still opens real device/control sessions;`--no-home` does not turn it into a read-only command.

## Operator-only server and physical pickup commands

These are reviewable commands for an attended lab session. They were **not executed** during this audit. Keep terminal input attached: do not use unattended execution, pipe input, or suspend a robot owner with Ctrl-Z. One model server at a time is sufficient; inspect GPU memory before loading. Port 7796 is an example dedicated lab port: verify it is free first and terminate only the server the operator starts in that terminal after its client exits.

In terminal A, select exactly one row below; the shown selection is the revised ftA student:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
MODEL_CKPT=runs/student_ftA_r1/student_001200.pt
MODEL_SYSTEM=student
MODEL_PORT=7796
# Teacher selection: MODEL_CKPT=runs/teacher_v5_ftA/teacher_001500.pt; MODEL_SYSTEM=teacher
# v5_6 student selection: MODEL_CKPT=runs/student_v5_6/student_001200.pt; MODEL_SYSTEM=student
.venv/bin/python -m phantom.scripts.policy_server \
  --ckpt "$MODEL_CKPT" --system "$MODEL_SYSTEM" \
  --hardware configs/hardware.nuc.yaml --port "$MODEL_PORT" \
  --device cuda --nfe 1 --guidance 1.0
```

This inference-only server loads EMA through the native builder and warms up without constructing a robot runtime; the server CLI does not accept `--ema`. In terminal B, inspect its ready log and probe it:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python -m phantom.scripts.policy_server --probe --port 7796
```

Require `idle`, the selected concrete checkpoint and digest prefix `b13362472843`(revised ftA student), `aee98c1cc01f`(v5_6 student), or `67c93287123e`(teacher). The server does not expose its system mode in the probe: preserve its exact terminal-A command and confirm the build log. Do not assume a warm student can be converted into a teacher by changing client flags.

After manual staging, clear-scene checks and operator agreement on the pickup criterion, terminal B may run this **physical** command. Change `--system` and `--ckpt` together to the selected model; do not change them without changing/verifying the server:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python -m phantom.scripts.run_deploy \
  --system student --ckpt runs/student_ftA_r1/student_001200.pt --ema \
  --policy-server 127.0.0.1:7796 --hardware configs/hardware.nuc.yaml \
  --task waffles --text waffles --episodes 1 --seed 101 \
  --nfe 1 --guidance 1.0 --persistent-noise --k-seeds 4 \
  --parity-fixes --terminal-veto --max-play-steps 10 \
  --no-home --max-start-sigma 2.5 --max-episode-s 35 \
  --lift-complete-z 0.32 --lift-complete-fz 4.0 --lift-complete-hold 0.4 \
  --out /home/physicalai/phantom-icra-2027/data/episodes/deploy/20260907_matched_pick
```

The proposed attended pilot is **five matched pairs per student: ten pairs, twenty episodes**, after the bench checks pass and if operator capacity permits. The primary pickup criterion is the packet visibly clearing its support and remaining held through the pickup stop; record lift/hold duration and failure stage separately. This small pilot establishes feasibility, not a reliable ranking or hardware success probability.

| Block | Paired episode seeds | Order within each pair |
|---|---|---|
| ftA teacher versus revised ftA student |101–105| Teacher first for 101/103/105; student first for 102/104 |
| ftA teacher versus v5_6 student |201–205| Student first for 201/203/205; teacher first for 202/204 |

Use one episode per process. Both members of a pair use the same seed, measured start and object placement. A reused seed matches sampling initialization, not execution or hardware sensor noise. Preserve first attempts, refusals, safety stops and contaminated trials; do not repeat failures until they succeed. If the lab stops early, retain the planned twenty-episode denominator and report the completed/invalid/missing counts explicitly. Any changed conditions or repaired controller start a separately identified block.

During a run, `x` or `stop`+Enter requests a clean operator stop; **bare Enter is ignored**, despite stale help text claiming otherwise. This request is polled by the planner and is not an emergency-stop substitute. A trained operator remains responsible for the physical stop. Normal operator stop and `lift_complete` preserve the current grip. Tactile-limit, wrench-limit, hitbox-exit and veto-retry-cap paths can open it; keep that possibility clear of people and record whether a guard caused a release. Support/remove a held packet through the approved bench procedure before resetting the rig.

## Placement is a separate release-validation gate

Native `ChunkExecutor._latched_grip` arms when both baseline-corrected trailing pad loads exceed 2.5 N and then preserves a running maximum of commanded closure. It does not unlock simply because the TCP reaches the box or the teacher requests opening. The live terminal veto can clear it during recovery and can mask a whole gripper chunk; it also includes tactile phantom-grasp recovery added after the historical `fd4a032` reference. Thus this live controller is not identical to the historical-veto simulator variant.

The opt-in [simulation release controller](../../phantom/sim/release_controller.py) admits sustained original policy opening inside an explicit measured-TCP release volume after a loaded latch, suppresses immediate re-latching until measured unload/open/reclose, and lets safety preempt release. Its hooks exist in [the simulation adapter](../../phantom/sim/policy_adapter.py) and [simulation veto bridge](../../tools/sim/deployment_filters.py), not native `phantom/deploy/executor.py`/`run_deploy.py`. Native `run_deploy` has no `--placement-release-config` flag. This release controller never reads the simulated object state.

Before any physical placement comparison, an independently reviewed native port must demonstrate that actual policy opening over the measured box is delivered, safety stops dominate, recovery cannot trigger an unintended release, and the gripper cannot re-latch during opening. Bench-test it first with verified clear motion and supervised low-height releases. The supported native `--no-grip-latch` flag disables retention everywhere; it is **not equivalent** to this release gate and is not the proposed shortcut. Do not loosen reach/hitbox/force guards to force a placement result.

Only after that gate passes should the lab register a separate placement controller/version, set `--lift-complete-z 0`, choose a fixed placement horizon, and compare the same teacher/student identities and paired starts. Score visible acquisition, sustained lift, retained transport, policy-commanded release, empty-gripper retreat and a packet settled inside the box. A safety stop after true placement is still recorded separately. The existing 40-case simulator result is 5/40 strict placements under an estimated scene and uncalibrated tactile/wrist proxies; it is neither a physical success-rate estimate nor evidence of reliable native release. The [post-experiment tactile diagnostic](../results/teacher_pick_place_v1/post_experiment_tactile_diagnostic.md) also documents a sparse-manifold pressure-mapping limitation. Keep those frozen results unchanged.

## Servo limiter and records to retain

Keep `elbow_min_rad=None` and `servo_joint_speed_max_rad_s=None`. The [old limiter review](../review_servo_limiter_0905.md) is partly stale: current live source contains the infeasible-anchor escape, achieved-displacement comparison, accepted `ServoResult` feedback and at-halt state capture. Its eight existing fake-RTDE tests pass; do not repeat repaired defects as current findings. However, up to three bisection probes per candidate remain, and each real IK request incurs a controller round trip. Current physical 125 Hz timing and behavior have not been qualified here. Simulator drive success does not qualify this driver. Keep the independent branch, joint-speed and wrist-extension guards active, and treat any future limiter enablement as a separately tested controller change.

For every attempted launch retain a session record with cell/stage/model/system, full checkpoint SHA, EMA, full CLI, commit and dirty status, base YAML hash plus effective overrides, server command/PID/port/digest, seed, intended and achieved q/TCP/closure, camera/object/bin setup photo and measured offsets, sensor baseline/rates, timestamps, refusal/stop reason and operator identity. Do not claim the runtime already records a full checkpoint hash: its `ckpt_sha` is normally a 12-character prefix, so retain the full pre-session hash separately.

Each generated episode under `/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260907_matched_pick/` should contain:

- `meta.json`: task/text/tags, seed, git revision, base config hash, model identity, `deploy_overrides` including hitbox/floor, effective safety, grip latch and max-play settings, plus operator verdict/notes/damage.
- `planner_trace.json`: proposal/actions, timing, uncertainty/event/gate diagnostics, terminal-veto action and `actions_pre_veto` where rewritten.
- `stop.json`: reason, safety events, `stop_state.at_halt`, measured joints/speeds and available IK/limiter counters. The outer stop state may be captured after stopping; use the at-halt record when present.
- Timestamped native Zarr streams: `arm_q`, `arm_qd`, `arm_tcp_pose`, `arm_tcp_speed`, `arm_ft`, `gripper`, `camera_scene_color`, tactile infer images/fields/keyframes/wrenches and action streams actually present. Preserve timestamps and recorded clock offsets.

Use the label prompt; for the pickup block include explicit notes such as `stage=pick held_after_lift=yes placement=not_attempted`. Native `s` means the operator's selected task criterion, not a verified full pick/place. Record slips, empty closure, post-stop release, damage and hand contamination separately. Unlabelled takes remain excluded by native training conventions but still belong in the attempted-trial denominator. Keep proposal actions distinct from measured motion: older/rederived episodes differ in action provenance. Do not run rederivation in place on source evidence.

`tools/replay_rig.py` and `tools/replay_deploy_path.py` are **offline policy replay**, not physical demonstration playback and not a physics simulator. The latter currently builds teacher mode with K 1; do not use it as an exact student/LEVERS compatibility claim. A future measured-trajectory hardware acceptance test needs its own reviewed controller and motion path. Review this handoff alongside [the current native runtime](../deployment_runtime.md), [the paired-seed session notes](../rig_session_v5.md), [the simulator protocol](../isaac_teacher_pick_place.md) and its [frozen 40-trial report](../results/teacher_pick_place_v1/README.md).
