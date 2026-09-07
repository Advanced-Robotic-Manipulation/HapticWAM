# Teacher-only waffles lab handoff

**Recommend ftA1500 EMA, NFE 1, K 4, guidance 1 as the next attended lab candidate. There is no validated reliable winner.** The completed v5 reserved confirmation produced the following results at one fixed simulator start and fixed waffle pose:

| Candidate | Acquired | Sustained lift | Retained carry | Strict physical placement | Clean FINISH | Drops | Safety stops |
|---|---:|---:|---:|---:|---:|---:|---:|
| ftA1500 / NFE1 / K4 |12/12|7/12|6/12|1/12|1/12|1/12|11/12|
| ftA3000 / NFE1 / K4 |8/12|4/12|4/12|0/12|0/12|0/12|12/12|

These are confirmation counts only. Do not pool the 24 screening cases, earlier v1/v2 campaigns, or development trials into them. The physical-placement difference is +8.33 percentage points, paired bootstrap 95% interval [0,+25], exact two-sided McNemar p=1. The declared ≥8/12 placements, positive lower bound, significance and no-extra-drops gates fail. FtA1500 is the conditional ranked leader; its one success does not establish robust placement or a real-world success rate. [Authoritative result](results/teacher_success_anchor_v5/confirmation/report.md) · [Independent raw audit](results/teacher_success_anchor_v5/confirmation/independent_audit.md).

The successful seed 904510 physically released and settled in the box at 24.268 s, reached controller FINISH at 25.076 s, and remained unloaded/bin-supported through the 60 s horizon without a stop. The teacher supplied the original sustained opening; historical recovery contributed to the final accepted aperture after physical placement. That distinction is preserved in the [selected event audit](results/teacher_success_anchor_v5/selected_success_audit.md). FINISH itself never observes object state and is not the task-success criterion.

## Exact model and software

| Item | Recommended candidate |
|---|---|
| Checkpoint | `/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_ftA/teacher_001500.pt` |
| SHA256 | `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e` |
| Architecture/weights | `teacher`, EMA; retain the checkpoint's saved configuration and normalizers |
| Inference | NFE1, K4, guidance1, persistent episode noise, parity fixes; task and text `waffles` |
| Execution | 125 Hz native control, 10 Hz action grid, maximum 10 played steps; existing governor, aperture latch and safety guards |
| Native opt-in | `--placement-controller-profile minimal_v5`, `--terminal-veto`, explicit `--placement-release-config` |
| Reference simulator source | `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_anchor_minimal_v5` |

The [full model settings table](results/teacher_success_anchor_v5/model_settings_table.md) retains all four tested recipes and checkpoint identities. The NFE5 comparator in v5 was **NFE5/K4**, not the older NFE5/K1 experiment. No student configuration belongs to this handoff.

The native opt-in port is in [PR14](https://github.com/Advanced-Robotic-Manipulation/phantom/pull/14), branch `sim/teacher-success-anchor-20260907`; it was introduced by commit `51457a36a8f3af3e4dbdc0100a8d6121aae70804`. [minimal_v5.py](../phantom/deploy/minimal_v5.py) SHA256 is `aaf9c620e199b16bf04f3ab5aba7b11ea83b788e46a95bdbf14e16085a112b6f`. Record the actual reviewed checkout commit/dirty status and file hashes before a session; the PR is not evidence that the live rig checkout has been updated. No update or hardware action was performed by this handoff.

The [native controller audit](results/teacher_success_anchor_v5/native_controller_port.md) passed 32 focused CPU checks, including all 37 archived delivered action arrays. Explicit minimal_v5 retains historical `fd4a032` veto rules and **request-snapshot TCP/aperture feedback**, plus policy-authorized release/FINISH. Current-native live-veto/current-delivery behavior is a different profile. Native sensor construction and checkpoint preprocessing stay unchanged: consume live RGB, tactile images/fields/loads and the real UR3 wrist stream. Do not feed a simulator baseline NPZ, synthetic gel image or PhysX wrench into the physical teacher.

The scored simulator used gel coverage `manifold_patch_v2`, a fixed September measured tactile baseline and the original pad-only `contact_proxy` wrist. Its immutable source digest is `e3d2aa0881a7f59bd5f0a98161abf19ff9be87775c2a0d40aaca48fe2d755df8`; see [source manifest](results/teacher_success_anchor_v5/minimal_profile_source_manifest.json) and [frozen profile](results/teacher_success_anchor_v5/protocol.json). This is source/profile reproducibility, not calibrated sensor transfer.

## Measure before the physical pilot

Use the [measurement guide](isaac_teacher_measurements_20260907.md) and [37-row fillable sheet](measurements/setup_20260907_template.csv). Keep its dated filename as provenance; enter the actual date of the next bench session. Record units, datum, instrument resolution, repeated readings and photos.

1. Reconfirm UR3 CB3 identity and mounted payload. Measure robot base-to-table datum, flange/TCP-to-active-pad faces, backing/housing dimensions and pad yaw. Verify the configured 180 mm TCP and actual hardware; an image fit is not a measurement.
2. Measure inner-pad gap versus feedback at several clear apertures. Normalized closure is not a jaw-gap measurement; bare 85 mm stroke and fitted 70 mm custom-pad travel are different quantities.
3. Measure the exact packet dimensions/mass, wrapper orientation, fixed reset marks and box inner/rim geometry in the same frame. Fit a reachable release volume that clears the rim and remains within existing safety bounds.
4. Check camera calibration and left/right sensor identity. Record no-contact wrist/tactile traces at every chosen start and through attended unloaded task motion; then review known-load response, sign, units and timestamps. Do not apply a second arbitrary tare to the teacher inputs.

The simulated rigid packet, compliant overlap and contact-to-gel mapping remain uncertain. One failed approach had sampled one-sided packet normal force above 120 N and large compliant overlap; the rolling wrist reference absorbed gradual load, leaving its residual below the 60 N threshold. This demonstrates an unresolved force/contact and baseline limitation. It is not an argument to relax limits, route peripheral loads into gel, or treat the force as hardware-calibrated. The [contact audit](results/teacher_success_anchor_v5/contact_force_audit.md) separates pad/packet, environmental, gel and rolling-baseline evidence.

Keep the actual workspace, reach, joint-speed, branch, wrench/tactile, stale-input and operator guards. Retain the 2.5-sigma native start gate and settled gripper feedback. A saved initial-state JSON or a successful schema check does not certify a safe physical path. Use the operator's established staging procedure; `--no-home` avoids automatic randomized homing but still opens real control sessions.

## Release configuration and native invocation

Prepare a new, reviewed lab JSON; do not install the campaign hardware YAML or paste the simulator's estimated TCP bounds. The native release schema is in [release_controller.py](../phantom/deploy/release_controller.py). Required bounds are `tcp_min_m` and `tcp_max_m`, measured in the real robot base frame. The tested logical parameters were:

| Field | Value to qualify on the bench |
|---|---:|
| `open_command_max` |0.45|
| `opening_hold_s` |0.2|
| `unloaded_force_max_n` |0.5|
| `unloaded_hold_s` |0.2|
| `rearm_close_command_min` |0.5|
| `finish_after_release` |true|
| `finish_observation_s` |2.0|

The load suffix follows the native signal convention; its physical units require checking. Release requires a previously loaded latch, sustained original policy opening while measured TCP is inside the reviewed volume, and measured open/unloaded dwell. FINISH additionally requires accepted opening; native mailbox/stale-feedback checks remain active. Native FINISH observes for the explicit 2 s then tears down normally, unlike the scored simulator's 60 s hold. Keep the final packet visible and score its actual release/support independently.

The profile also checks explicit finite veto values and recovery aperture against the hardware closure limit before constructing drivers. The reference task defaults were `z_ref=.0415`, `z_floor=.0315`, `z_margin=.0615`, `open_aperture=.232`, `p_close=.5`, `p_none=.9`, retry cap 3. Native task statistics supply the reference height/aperture. Preserve and review the actual effective values; a different rig datum or task-stat revision starts a new declared configuration.

File-only checks for an operator's reviewed checkout:

```bash
cd /home/physicalai/phantom-icra-2027/phantom
git rev-parse HEAD
git status --short
sha256sum runs/teacher_v5_ftA/teacher_001500.pt   phantom/deploy/minimal_v5.py configs/hardware.nuc.yaml configs/start_poses.yaml
rg -n 'placement-controller-profile|placement-release-config' phantom/scripts/run_deploy.py
```

The following is the explicit **setting fragment for an attended native run**, after measured release-volume/bench qualification. It is not executed by this document. Use one operator-owned inference server, verify the actual loaded checkpoint/EMA/system and full hash, and record its effective recipe. A client checkpoint warning does not replace the model already in a server. Preserve unrelated resident processes.

```text
python -m phantom.scripts.run_deploy
  --system teacher --ckpt runs/teacher_v5_ftA/teacher_001500.pt --ema
  --policy-server 127.0.0.1:7796 --hardware configs/hardware.nuc.yaml
  --task waffles --text waffles --episodes 1 --seed 907101
  --nfe 1 --k-seeds 4 --guidance 1 --persistent-noise --parity-fixes
  --terminal-veto --veto-p-close 0.5 --veto-p-none 0.9 --veto-max-retries 3
  --veto-z-margin 0.0615 --max-play-steps 10
  --placement-controller-profile minimal_v5
  --placement-release-config /absolute/path/to/reviewed_lab_release_finish.json
  --no-home --max-start-sigma 2.5 --lift-complete-z 0 --max-episode-s 60
  --out /absolute/path/to/new_lab_episode_directory
```

The reviewed release file and its hash, output path, date and free owned port must be filled in. Keep the trained operator present with the established stop procedure. Qualify pickup/retention and low-height release first; do not globally disable the load latch to obtain placement. A FINISH flag or `lift_complete` event alone is never an object-success label.

## Fixed waffle, measured variable-start pilot

The next physical study should reflect collection practice: **fix the waffle/box/camera and vary measured initial arm/gripper states**. The trajectory is generated by the policy. Record intended and achieved q/TCP/closure and no-contact sensor state; do not impose recorded later trajectory phases or arbitrary joint perturbations.

After the single-start bench gate, a concrete proposed pilot is six operator-reviewed starts × two fresh episode seeds 907101/907102 (12 planned attempts), with the second-seed start order reversed. Candidate evidence starts are Sept4 anchor, Aug22 5928, 6273, 6128, 5963 and 6461; their exact bundles/hashes are listed in the [measured-state inventory](results/teacher_success_anchor_v5/arm_extension/amendment_v5.json). They are evidence references, not robot motion commands. Freeze the physically reviewed/achieved start targets and order before the first scored pilot; report the two anchor attempts separately from the ten varied-start attempts.

This is a new proposed **lab calibration/transfer pilot**, not the unexecuted simulator winner extension. That extension remains gated off because v5 has no clear winner. Keep ftA1500 and its controller/recipe fixed within the pilot, retain refusals and every failure in the planned denominator, and record any excluded setup trial before scoring. Do not replace unsuccessful starts. Changing the controller, safety reference, geometry or recipe creates a separate block. Object-reset jitter is measured and tested afterward as its own sensitivity factor.

For every attempt retain full checkpoint/source/config/release hashes, seed, achieved start, reset measurements/photo, synchronized raw streams, proposed and accepted commands, native/full-client timing, latch/release/FINISH events and all stop reasons. Independently annotate acquisition, visible sustained lift, carry retention, release, settled box support, drops, collisions and later stops. Preserve source recordings unchanged.

## Follow-ups remain separate

The completed V6 limiter diagnostic is not promoted: both cases lifted/carried but stopped after25 constraint holds before release. [Exact hold/timing audit](results/teacher_success_anchor_v6/diagnostic_results/recommendation.md). It does not establish that a reach limiter prevents low-height contact or rolling-baseline absorption.

The completed [V7 full-client timing diagnostic](results/teacher_success_anchor_v7/README.md) supports retaining K4. On two reused development seeds, K4 lifted/carried2/2 and placed1/2; K1 acquired2/2 but lifted0/2 despite mean full-client duration0.279s versus0.848s. Both had one wrist-extension stop and no drop. These counts are not pooled into V5 confirmation and do not validate hardware transfer. V7 preserves native action/CPK clocks and changes only the declared simulated delivery timing; actual native deployment uses its own real request timing.

The optional fixed wrist-reference work in [PR13](https://github.com/Advanced-Robotic-Manipulation/phantom/pull/13) likewise needs unloaded-motion/known-load qualification. No new limiter, wrist reference, force threshold, live-veto profile or delivery-clock switch is silently enabled here.

[One-page card](isaac_teacher_lab_card_20260907.md) · [Final v5 results](results/teacher_success_anchor_v5/confirmation/independent_audit.md) · [Native port evidence](results/teacher_success_anchor_v5/native_controller_port.md) · [Measurements](isaac_teacher_measurements_20260907.md).
