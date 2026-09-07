# Teacher-only waffles lab handoff — 7 September 2026

Begin with dimensions, sensor checks and an attended **teacher pick-and-hold** gate. Proceed to pick-and-place only after the native release/FINISH bridge and physically measured release region have been reviewed and bench-tested. The current request is teacher-only; the [earlier student handoff is archived](archive/isaac_lab_handoff_students_20260907.md).

**The completed screen advanced v5_6 and ftA3000, both at NFE 1/K 4; confirmation is pending and neither is a winner.** They acquired the packet in 2/8 and 1/8 screening trials respectively, with no sustained lifts or placements. See the [complete screening results](results/teacher_robustness_v2_delivery/README.md). FtA1500 remains an identified historical reference, not the selected final-experiment recommendation. Fill the confirmed recommendation and limitations here after the reserved trials. Controller completion, `lift_complete`, a pickup label or a simulated grasp does not establish a packet settled in the box or a physical success probability.

The active immutable study is `teacher_robustness_v2_delivery`: 32 valid completed discovery trials followed by 24 paired confirmation trials on six reserved starts, **56 planned trials in total**. Its source is `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_v2_delivery`; outputs are under the sibling `runs/teacher_robustness_v2_delivery/`. Confirmation and candidate acceptance remain pending. The ten completed trials in the earlier halted `teacher_robustness_v2` screen remain separate diagnostic evidence; they are not replacement-study scores. See the [current protocol and amendment](../configs/sim/teacher_v2_delivery_protocol.json).

This handoff combines the read-only audit of live compute3 commit `0259ed3355f2ad3078778e7cc347b100775e06d3` with the isolated v2 worktree. No hardware connection, robot/gripper command or GPU inference was performed for this audit. **The new shared release/FINISH hooks are in v2 source; they are not an automatically installed change to the live rig repository.** Native commands below are for trained operators after source/configuration review, not unattended execution. Source references: [run_deploy](../phantom/scripts/run_deploy.py), [start gate](../phantom/deploy/start_pose.py), [executor](../phantom/deploy/executor.py), [safety](../phantom/deploy/safety.py), [UR driver](../phantom/drivers/real/ur.py) and [rig launchers](../tools/rig/README.md).

## Teacher identities and registered recipes

Paths below are absolute or relative to `/home/physicalai/phantom-icra-2027/phantom`.

| Teacher | Concrete checkpoint | SHA-256 |
|---|---|---|
| ftA1500 reference | `runs/teacher_v5_ftA/teacher_001500.pt` | `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e` |
| Later ftA3000 | `/home/physicalai/phantom-icra-2027/sim/waffles/checkpoints/teacher_v5_ftA/teacher_003000.pt` | `ee0a448c00da2fe047b4bcba5036a8b79e9ae33e3a6ec47ea69d27466b4ff1dc` |
| Earlier v5_6 teacher | `runs/teacher_v5_batch0822/v5_6.pt` | `7edcb8335681e19bead5ad39a2a80fe3fd8c5893b1761d2dd812c304881fe65c` |

Each audited payload declares teacher architecture and contains 676 finite EMA tensors. Normalizers match; canonical SHA-256 is `42153b79ceb6a6c711296323c6b171e42929eb5dfe4018198d63d2cd3ace8c48`. Load each checkpoint's own saved configuration and EMA. FtA uses per-strip noise; v5_6 retains its older noise/training settings. See the [teacher inventory](results/teacher_v2_design/README.md) and [payload audit](results/teacher_v2_design/teacher_payloads.json).

| Registered simulator configuration | NFE | K candidate chunks per replan |
|---|---:|---:|
| ftA1500, LEVERS sampling recipe | 1 | 4 |
| ftA3000, same recipe | 1 | 4 |
| v5_6 teacher, same recipe | 1 | 4 |
| ftA1500, native VETO sampling recipe | 5 | 1 |

Common settings are `--system teacher`, task/text `waffles`, EMA, guidance 1, persistent episode noise, parity fixes, **live** terminal veto and maximum 10 played action steps. K is independent of episode sampling seed. NFE 1/K 4 versus NFE 5/K 1 changes two inference factors; it is not an isolated NFE ablation. The [registered protocol](../configs/sim/teacher_v2_delivery_protocol.json) is authoritative for freeze status, eligible starts, source/input hashes, discovery/confirmation order and timing. A draft or preflight is not a scored finding.

V2 fixes the canonical August green packet and transfers authentic measured arm/gripper/wrist starts. It does not impose a recorded trajectory as policy control or silently transfer each recording's object jitter. The [Aug22 static tactile baseline](results/teacher_v2_design/aug22_tactile_baseline.md) freezes complete measured pad captures from the initial 0.305–0.413 s with residuals intact. Those keyframes were not all available at the earlier robot-start anchor; this is declared sensor calibration. The deformation law, rigid packet, pressure mapping, wrist transfer and fitted geometry remain uncalibrated. The corrected simulator wrist profile includes external normal contacts on the housing and both pads, with contact-point moments about measured TCP. It preserves recorded initial bias and still omits tangential loads and changing gravity/inertia; it is not a substitute for the physical UR3 current-based reading. A housing/bin collision exposed the previous pad-only omission. **Do not supply this NPZ to a physical teacher in place of live tactile sensors.**

`--wrist gripper_contact_proxy` is a **simulator-only** option. It sums signed PhysX normal forces on exactly three gripper bodies and their contact-point moments in world/base axes, excluding self contacts and proximal arm loads. Native `phantom.scripts.run_deploy` has no such option: it reads the configured real `arm_ft` stream. The [command-replay preflight](results/teacher_robustness_v2/nominal5928_diagnostic/gripper_wrist_preflight_audit.md) verifies arithmetic, causal sampling and unchanged simulated motion; it does not verify the physical wrist's axes, torque origin, gravity compensation or response to a known applied load. Those checks belong on the measurement sheet before comparing physical and simulated wrench thresholds.

## Checks that do not control hardware

Run these on compute3 before allocating a model or connecting to the rig. They inspect files, processes and host resources only:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
git rev-parse HEAD
git status --short
sha256sum configs/hardware.nuc.yaml configs/start_poses.yaml
sha256sum runs/teacher_v5_ftA/teacher_001500.pt \
  runs/teacher_v5_batch0822/v5_6.pt \
  /home/physicalai/phantom-icra-2027/sim/waffles/checkpoints/teacher_v5_ftA/teacher_003000.pt
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

Complete the [teacher measurement sheet](isaac_teacher_measurements_20260907.md), starting with table/base datum, pad/TCP transform, actual gap versus gripper feedback, packet dimensions/mass and box position. Record instrument resolution and uncertainty.

The checked configuration explicitly says `meta.bench_verified: false`. Review the relevant procedures in [hardware_bench_day1.md](hardware_bench_day1.md) and verify the current physical rig before running policy control; do not mark the flag true without measurements. Source values describe the configured rig, not a certified collision-free motion path.

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

Choose one observed start before the pilot, physically review its path and stage it with the trained operator using the pendant/approved bench procedure. Reproduce that state and fixed packet/bin placement between paired trials. The [canonical Aug22 5928 start](../configs/sim/initial_states/waffles_aug22_1787395928_000.json) is an evidence reference, **not an automatically safe waypoint**:

```text
q(rad, RTDE order) = [0.24457107, -1.51014740, 0.92519188,
                     0.89983332, 1.18592298, -3.20375735]
TCP(m + rotation-vector rad) = [-0.33538233, -0.29982739, 0.35634588,
                               -1.03112878, -1.92408417, 1.47232898]
measured closure = 0.42745098; gripper OBJ = 3
```

Begin with that canonical start. After safe reproduction, the first varied-start pilot can use the measured starts `5928`, `5963`, `6028`, `6273`, then repeat `5928`, in that order. Review each full path and its [initial-state JSON](../configs/sim/initial_states/) before staging. Keep the packet at the same marked canonical position: the recorded wrapper reversal or object jitter is not an arm-start factor. Do not use random joint configurations or later near-grasp states as substitutes for demonstration starts.

A file-only check with the native `start_sigma_report` accepts these four recorded states at maximum deviations 1.432, 1.107, 1.261 and 1.388 sigma respectively, including measured aperture and unwrapped joints. This only checks the recorded numbers against the pinned start distribution. The live achieved state must independently pass both native 2.5-sigma gates and settled `OBJ=3`; this calculation authorizes no motion and validates no swept path. Preserve each start's own measured closure rather than resetting every start to the mean 0.232 or converting the simulated 70 mm travel into a physical command.

The configured waffles distribution has TCP mean `[-0.3652,-0.2859,0.3346,-1.1053,-1.8268,1.5079]`, joint mean `[0.1966,-1.5291,1.1159,0.6794,1.1853,-3.1303]` and mean closure 0.232. A small TCP error does not excuse a full-turn joint mismatch. Record the achieved state rather than assuming the requested waypoint was attained.

The launch below uses `--no-home` so the native runtime keeps both start gates but does not perform its randomized homing or automatic re-homing. Default homing samples up to one standard deviation of TCP/closure jitter; even `--home-joints` first moves to joint mean and then performs a jittered moveL. There is no zero-jitter homing CLI flag. Never run a shell loop that homes from arbitrary arm configurations. Entering `run_deploy` still opens real device/control sessions; `--no-home` does not turn it into a read-only command.

## Operator-only server and physical pickup commands

These are commands for an attended lab session after bench gates pass. They were **not executed** during this audit. Keep terminal input attached: do not pipe input, run unattended, or suspend a robot owner with Ctrl-Z. Use one owned model server at a time; verify GPU memory and that example port 7796 is free. Terminate only the server the operator starts after its client exits.

Terminal A shows the pinned ftA1500 reference. For another audited teacher change the concrete checkpoint only after candidate/recipe review:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
MODEL_CKPT=runs/teacher_v5_ftA/teacher_001500.pt
MODEL_NFE=1
MODEL_PORT=7796
.venv/bin/python -m phantom.scripts.policy_server \
  --ckpt "$MODEL_CKPT" --system teacher \
  --hardware configs/hardware.nuc.yaml --port "$MODEL_PORT" \
  --device cuda --nfe "$MODEL_NFE" --guidance 1.0
```

This native inference-only server loads EMA through its builder and warms up without constructing a robot runtime; its CLI has no `--ema` flag. In terminal B inspect the ready/build log and probe:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python -m phantom.scripts.policy_server --probe --port 7796
```

Require `idle`, the selected concrete checkpoint and digest prefix `67c93287123e` (ftA1500), `ee0a448c00da` (ftA3000), or `7edcb8335681` (v5_6 teacher). Preserve the full SHA separately. The probe omits system mode: confirm `teacher` from the exact server command/build log. Do not assume client flags can change the loaded architecture.

The server's warm-up defaults are K 1 and parity off; the physical client configures K, parity, persistent noise, guidance and task text when it connects. Check the client's effective-setting log for the chosen recipe instead of treating the warm-up log or probe as proof of those settings. Use a new episode reset and the explicit paired seed for each launch. Native terminal veto and parity fixes are opt-in flags; naming a recipe in a notebook does not enable them.

After manual staging and attended clear-scene checks, the following command **controls physical hardware**:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python -m phantom.scripts.run_deploy \
  --system teacher --ckpt runs/teacher_v5_ftA/teacher_001500.pt --ema \
  --policy-server 127.0.0.1:7796 --hardware configs/hardware.nuc.yaml \
  --task waffles --text waffles --episodes 1 --seed 101 \
  --nfe 1 --guidance 1.0 --persistent-noise --k-seeds 4 \
  --parity-fixes --terminal-veto --max-play-steps 10 \
  --no-home --max-start-sigma 2.5 --max-episode-s 35 \
  --lift-complete-z 0.32 --lift-complete-fz 4.0 --lift-complete-hold 0.4 \
  --out /home/physicalai/phantom-icra-2027/data/episodes/deploy/20260907_teacher_pick
```

For the registered ftA1500 NFE 5/K 1 recipe, replace **both** `--nfe 1` and `--k-seeds 4` with `--nfe 5 --k-seeds 1`, and launch/verify the corresponding server. For ftA3000 use its audited absolute path; for v5_6 use `runs/teacher_v5_batch0822/v5_6.pt`. All systems remain `teacher`; no student ablation belongs to this plan.

This pickup gate declares a 35 s wall-clock budget and lift stop at TCP Z 0.32 m. It is not the 60 s simulator placement protocol. Retain the native 0.25 m/s TCP cap unless a lower common cap is reviewed and recorded before the block. The pickup criterion is visible support clearance and retained hold through the pickup stop, with actual lift/hold duration recorded. If the reference cannot reproduce pickup, pause for sensor/geometry/controller diagnosis before comparing candidates.

`--lift-complete-z` defaults to **0**, disabled, in the audited native CLI; the example deliberately enables 0.32 only for the attended pickup gate. The canonical start is already above that TCP height. Confirm unloaded pads before launch and judge the actual object lift visually: a height/load flag alone cannot prove the object left the table.

After the gate passes, a bounded pilot can use **five matched reference/candidate pairs: ten episodes**, seeds 101–105 and the five reviewed starts listed above. Reference first for 101/103/105; candidate first for 102/104. Fill the candidate and recipe from reviewed confirmation evidence before starting; do not choose one from a convenient single simulator success. Matched members share seed, achieved start, fixed scene and controller. Reusing a seed aligns sampling initialization, not hardware execution or sensor noise.

Preserve first attempts, refusals, safety stops, contamination and missing cases in the planned denominator. A changed controller or physical setup starts a separate block. During control, `x` or `stop` plus Enter requests an operator stop; bare Enter is ignored. This planner-polled request is not an emergency stop. Operator stop and `lift_complete` retain grip. Tactile/wrench limits, lateral hitbox exit and veto retry cap can open it. Keep that possibility clear of people; record guard-triggered release and support/remove held packets through the approved bench procedure before reset.

## Placement release and FINISH validation gate

Default native aperture latching remains unchanged: once both baseline-corrected trailing pad loads exceed the configured 2.5 threshold, closure preserves a running maximum until recovery/episode handling clears it. Reaching the box or a policy opening alone does not unlock it. Do not disable retention globally with `--no-grip-latch` to force placement. Live terminal veto retains close masking, floor recovery and tactile phantom-grasp recovery; historical `fd4a032` is not the new study's native profile.

The v2 [shared controller](../phantom/deploy/release_controller.py), [native executor](../phantom/deploy/executor.py), [native planner](../phantom/deploy/planner.py), [runtime](../phantom/deploy/runtime.py) and [simulation adapter](../phantom/sim/policy_adapter.py) now implement an explicit opt-in bridge. The simulator re-exports the shared API. It uses measured TCP/gripper/tactile load, previous loaded latch and **original policy opening intent**; never packet pose, object success or bin containment. Only original opening rows inside the reviewed release volume pass a whole-chunk close mask; arm guards/recovery remain active and rewritten contact-package feedback is invalidated.

With this release profile enabled, the corrected simulator and new native bridge evaluate veto rules using **current measured delivery TCP/aperture**, while retaining the original request observation for model conditioning. Default native operation without the profile retains its historical request-snapshot feedback. Check simulator `policy_info.json:terminal_veto_feedback_source=current_delivery` and native `meta.json:deploy_overrides.placement_veto_feedback="current measured delivery feedback"`. This documents the controller feedback choice; it does not equate simulated sensors or asynchronous hardware timing. The retained waffles close hatch is TCP Z ≤ 0.103 m (`z_ref=0.0415`, margin 0.0615), not a pad-clearance or full-robot collision test.

Release requires sustained policy opening after a loaded latch. Immediate re-latching is suppressed until measured unload/open and a subsequent policy close, or reset. With `finish_after_release: true`, bilateral measured unload/open dwell plus an actually sent open command triggers FINISH: hold the achieved measured TCP and accepted open aperture, reject new plans and stop inference. It generates no arm trajectory, forced opening or object-success decision. Workspace/reach clamps, IK/branch checks, measured speed/wrench/tactile/freshness guards and operator stops remain authoritative.

These hooks are **new isolated v2 source, not a claim that the live rig has been updated**. Before physical use, review/install the intended version through a controlled update, record its commit/dirty status and verify it contains the interface:

```bash
rg -n 'placement-release-config|placement_release_variant' phantom/scripts/run_deploy.py
rg -n 'completed_reason|release_config' phantom/deploy/executor.py phantom/deploy/planner.py
sha256sum phantom/deploy/release_controller.py phantom/deploy/executor.py \
  phantom/deploy/planner.py phantom/deploy/runtime.py phantom/scripts/run_deploy.py
```

The JSON requires explicit `tcp_min_m`/`tcp_max_m` in the actual robot base frame. Measure box/pad/TCP geometry, drop height and clearance; do not paste the simulator's fitted bounds. Other fields are `open_command_max`, `opening_hold_s`, `unloaded_force_max_n`, `unloaded_hold_s`, `rearm_close_command_min`, `finish_after_release` and `finish_observation_s`. The `_n` suffix follows the existing load convention; verify sensor units for physical use. Tested v2 intent uses opening threshold 0.45, reclose threshold 0.5, 0.2 s opening/unload dwells, unload threshold 0.5, and 2 s native post-FINISH observation. These are controller settings, not sensor calibration.

The release profile defaults to **absent**. Supplying a JSON enables release, but `finish_after_release` defaults to **false**: set it explicitly to `true` for the FINISH profile. Bounds have no defaults and must form a positive volume. The native load latch must remain enabled, and `rearm_close_command_min` must not exceed the hardware maximum closure. Those parser checks do not verify that the volume lies in the reachable/safe workspace, that it clears the rim, or that the packet lands inside the box. Review the usable intersection with the native hitbox and complete gripper/arm geometry; never widen safety bounds just to make the proposed release volume reachable.

After an operator has written and reviewed the measured JSON, this **file-only** check validates its schema and hardware latch compatibility without constructing a driver or moving anything:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
.venv/bin/python - /absolute/path/to/reviewed_lab_release_finish.json <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from phantom.config.hardware import load_hardware
from phantom.deploy.release_controller import make_release_controller

path = Path(sys.argv[1])
config = json.loads(path.read_text())
assert config.get("finish_after_release") is True, "FINISH must be explicit"
controller = make_release_controller(config, load_hardware("configs/hardware.nuc.yaml", quiet=True))
print(controller.variant)
print(json.dumps(controller.config.to_dict(), indent=2))
print("release_config_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

Expected variant is `placement_policy_release_finish_v2`. Preserve that printed configuration and hash with the physical session record. A successful schema check does not satisfy the measurement and attended transition tests below.

CPU tests exercise retention/release/rearm, stale feedback, queued gripper closes, actual accepted opening, measured FINISH hold and sensor/operator/safety preemption. They do not qualify real RTDE/gripper timing. Bench-test those transitions with reviewed clear motion and supervised low-height releases before physical policy placement. Only after measured release-config and controller review may an attended placement command replace the pickup-only stop with:

```text
--placement-release-config /absolute/path/to/reviewed_lab_release_finish.json
--lift-complete-z 0
--max-episode-s 60
```

Keep all teacher, inference and safety flags explicit. Native FINISH observes for the configured 2 s before normal teardown, subject to the time cap and stronger stops. The scored simulator intentionally keeps sensors/safety/physics active through its common 60 s horizon after FINISH; that observation-duration difference is declared. `completed_reason=placement_release_finished` is separate from subsequent safety stops and independent object success.

Score actual acquisition, sustained lift, retained transport, policy-commanded release and a packet settled inside the measured box. Record empty-gripper behavior and later guards separately. A held packet above the box is not placement. Frozen v1 remains 5/40 strict simulated placements under estimated geometry and uncalibrated sensor/contact transfer; its [diagnostic](results/teacher_pick_place_v1/post_experiment_tactile_diagnostic.md) and [report](results/teacher_pick_place_v1/README.md) remain unchanged. V2 mapper/controller changes need separate frozen evidence and do not rewrite v1 scores.

## Servo limiter and records to retain

Keep `elbow_min_rad=None` and `servo_joint_speed_max_rad_s=None`. The [old limiter review](review_servo_limiter_0905.md) is partly stale: current live source contains the infeasible-anchor escape, achieved-displacement comparison, accepted `ServoResult` feedback and at-halt state capture. Its eight existing fake-RTDE tests pass; do not repeat repaired defects as current findings. However, up to three bisection probes per candidate remain, and each real IK request incurs a controller round trip. Current physical 125 Hz timing and behavior have not been qualified here. Simulator drive success does not qualify this driver. Keep the independent branch, joint-speed and wrist-extension guards active, and treat any future limiter enablement as a separately tested controller change.

For every attempted launch retain a session record with cell/stage/model/system, full checkpoint SHA, EMA, full CLI, commit and dirty status, base YAML hash plus effective overrides, server command/PID/port/digest, seed, intended and achieved q/TCP/closure, camera/object/bin setup photo and measured offsets, sensor baseline/rates, timestamps, refusal/stop reason and operator identity. Do not claim the runtime already records a full checkpoint hash: its `ckpt_sha` is normally a 12-character prefix, so retain the full pre-session hash separately.

Each generated episode under `/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260907_teacher_pick/` should contain:

- `meta.json`: task/text/tags, seed, git revision, base config hash, model identity, `deploy_overrides` including hitbox/floor, effective safety, grip latch and max-play settings, release-config hash/variant when enabled, plus operator verdict/notes/damage.
- `planner_trace.json`: proposal/actions, timing, uncertainty/event/gate diagnostics, terminal-veto action, release state, and `actions_pre_veto` where rewritten.
- `stop.json`: reason, separate `completed_reason`/`completed_at_s` when present, safety events, `stop_state.at_halt`, measured joints/speeds and available IK/limiter counters. The outer stop state may be captured after stopping; use the at-halt record when present. Native completion timestamps use the monotonic runtime clock; do not compare them directly with simulator elapsed seconds without the recorded clock alignment.
- Timestamped native Zarr streams: `arm_q`, `arm_qd`, `arm_tcp_pose`, `arm_tcp_speed`, `arm_ft`, `gripper`, `camera_scene_color`, tactile infer images/fields/keyframes/wrenches and action streams actually present. Preserve timestamps and recorded clock offsets.

Use the label prompt; for the pickup block include explicit notes such as `stage=pick held_after_lift=yes placement=not_attempted`. Native `s` means the operator's selected task criterion, not a verified full pick/place. Record slips, empty closure, post-stop release, damage and hand contamination separately. Unlabelled takes remain excluded by native training conventions but still belong in the attempted-trial denominator. Keep proposal actions distinct from measured motion: older/rederived episodes differ in action provenance. Do not run rederivation in place on source evidence.

`tools/replay_rig.py` and `tools/replay_deploy_path.py` are **offline policy replay**, not physical demonstration playback and not a physics simulator. The latter currently builds teacher mode with K 1; do not use it as an exact teacher/LEVERS compatibility claim. A future measured-trajectory hardware acceptance test needs its own reviewed controller and motion path. Review this handoff alongside [the current native runtime](deployment_runtime.md), [the paired-seed session notes](rig_session_v5.md), [the active teacher experiment](isaac_teacher_v2_experiment.md) and the separate [frozen v1 40-trial report](results/teacher_pick_place_v1/README.md).
