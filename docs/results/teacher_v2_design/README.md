# Teacher v2 design evidence

Three teacher checkpoints are ready for a bounded, conditional simulator study. Ten complete August 22 teleoperation recordings provide genuine recording-start states with synchronized robot, gripper and wrist evidence. They support a **fixed green-waffle scene with varied recorded arm/gripper/wrist starts**. They do not establish robustness over the entire training distribution or calibrated hardware success.

This audit used CPU reads only. It did not launch inference, simulation, training or hardware, and did not change recordings or existing weights. The parent staged the new ftA3000 checkpoint in a separate cache, after which this audit verified its payload.

## Teacher identities and readiness

| Candidate | Exact file on compute3 | Step | SHA-256 |
|---|---|---:|---|
| Deployed reference ftA1500 | `/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_ftA/teacher_001500.pt` | 1500 | `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e` |
| Complete later ftA3000 | `/home/physicalai/phantom-icra-2027/sim/waffles/checkpoints/teacher_v5_ftA/teacher_003000.pt` | 3000 | `ee0a448c00da2fe047b4bcba5036a8b79e9ae33e3a6ec47ea69d27466b4ff1dc` |
| Earlier baseline v5_6 | `/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_batch0822/v5_6.pt` | 3000 | `7edcb8335681e19bead5ad39a2a80fe3fd8c5893b1761d2dd812c304881fe65c` |

All three CPU-loaded payloads declare teacher architecture and contain 676 finite EMA tensors, totaling 27,889,205 values. Their normalization mean/std are identical; canonical JSON SHA-256 is `42153b79ceb6a6c711296323c6b171e42929eb5dfe4018198d63d2cd3ace8c48`. See [payload verification](teacher_payloads.json), including saved model/training configurations and normalization values. No checkpoint was constructed as a model or loaded on a GPU during this audit.

`BEST.pt` aliases ftA1500; `DEMO.pt` aliases v5_6. They are not additional candidates. The refreshed [inventory](teacher_inventory.json) found no newer named teacher family. The archive contains complete ftA500/1000/1500/2000/2500/3000 weights at revision `17e2821b7bfbcab54e08cd2f0bbb9859b9b88729` of `armteam/phantom-checkpoints`. Later steps are later training snapshots, not established improvements. FtA3000 is now locally staged and audited; ftA2000/2500 remain archive-only in this inventory. No active teacher-training process was found at the inventory time.

The [preserved project review](../../project_review_20260905_remote.md) reports both v5 and ftA training reaching 3000 steps with finite logged metrics. The selected ftA1500 had lower waffles EMA endpoint error than v5_6 (24.42 versus 26.18 mm), but higher overall mean error; this offline metric does not rank closed-loop task performance. FtA3000 readiness is established by complete weights and finite EMA, not by an assumed validation win.

## Native settings and the proposed screen

Saved checkpoints use NFE5, action horizon16, text conditioning including exactly `waffles`, time-based RoPE and two-pass ACC. FtA uses per-strip action noise and EMA decay .995; v5_6 has older noise/training settings and EMA decay .999. Load each checkpoint's own saved configuration and its EMA. Do not impose ftA model fields on v5_6.

The native [PICK.sh](../../../tools/rig/PICK.sh) LEVERS preset specifies NFE1, K4, terminal veto and parity fixes; [GO_ANY.sh](../../../tools/rig/GO_ANY.sh) supplies EMA, persistent noise and guidance1. The native VETO preset uses NFE5 and K1 with veto/parity. Thus NFE1/K4 versus NFE5/K1 is a **two-factor inference-recipe comparison**, not an isolated NFE ablation. Keep task `waffles`, EMA, guidance1, persistent noise, parity, controller, placement-release logic, safety and observation mappings common and explicitly recorded. The simulator's historical-veto or corrected-veto choice must be named; neither recipe alone establishes native-executor parity.

A tractable candidate design is 56 scored episodes:

- Discovery: four measured starts × two fresh matched episode seeds × four configurations = 32. Configurations are ftA1500, ftA3000 and v5_6 at NFE1/K4, plus ftA1500 at NFE5/K1.
- Confirmation: six unused starts × two new matched seeds × two configurations selected by a rule frozen before discovery = 24.

The parent owns the final freeze, selection rule, source hashes and scheduling. Retain ties and every failed or invalid attempt. Rank on support-verified physical full-task outcomes, with stage counts reported separately; do not promote a success flag or controller stop reason into physical success. Confirmation intervals should resample starts as clusters containing both seeds and matched candidates. Six start clusters give weak precision; ties or sparse successes do not establish equivalence or a winner. Shared GPU latency is measured or controlled as a declared condition, not silently assumed identical.

## Recording coverage and authentic starts

The data root contains 250 successful waffles teleoperation episodes. Of these, 180 belong to the original collection and 70 carry `batch_20260822` tags. **240 retained mirrors contain only q/TCP/gripper arrays and have no `ts` arrays at all**. They cannot support same-clock wrist starts, settling analysis, camera comparison or observation replay. The remaining ten have native timestamped arm q/qd/TCP, gripper, wrist `arm_ft`, scene RGB and tactile streams. The complete [raw audit](demo_starts_raw.json) preserves every episode's available stream status, metadata, source-array hashes and manifest membership.

Among the 240 thin mirrors, the original manifest identifies 160 train and 20 validation episodes; 60 augmented episodes have no entry in that original manifest. All ten complete recordings are explicitly in expanded `val124`, nine in session `20260822_135140_waffles` and one (`6461`) in `20260822_140049_waffles`. The complete expanded training manifest is absent from the inspected evidence, so these labels do not prove absence from every training pool. The proposed split is **simulator-study discovery/confirmation**, not a claim of newly held-out training data or session generalization.

| First recorded TCP distribution | N | X range (m) | Y range (m) | Z range (m) | Closure range |
|---|---:|---|---|---|---|
| Older thin mirrors; unsynchronized first rows | 240 | −.43394 to −.30762 | −.35221 to −.19305 | .21604 to .39281 | .0118–.5373 |
| Complete August recordings; first rows | 10 | −.34956 to −.32935 | −.32069 to −.27513 | .29698 to .35709 | .2549–.4275 |

The complete cohort covers a narrower region and is shifted in X relative to the older median (−.33880 versus −.36532 m). Missing timestamps, wrist signals and images in the other 240 prevent validating broader restarts today. Their numerical first rows are descriptive only; no arbitrary timing was assigned to them.

Each [initial-state JSON](../../../configs/sim/initial_states/) uses the first native arm-q timestamp at or after q, TCP, gripper and wrist streams all begin. q and TCP timestamps match exactly in all ten. Gripper/wrist values are the latest causal measured sample, with maximum age 7.815 ms. Anchors are 0–23.640 ms after the first arm sample. Timestamps are already MasterClock seconds: metadata clock offsets are preserved, never applied again. Camera frames nearest the anchor are 8–49 ms later and are labeled with their actual times. No initial state was taken from a grasp, lifted phase, deployment policy trajectory or a conveniently settled later time.

Start drift is small but nonzero: over the first .25 seconds, maximum joint change is .00052–.00686 rad and maximum measured joint speed is .0106–.0724 rad/s. The simulation must separately record its settled initial error/velocity and fail an invalid initialization; do not silently substitute a later real pose. Raw `arm_ft` samples include unknown tare/gravity/calibration bias and vary substantially between recordings. They are exact evidence, not a calibrated contact-free wrench. Any runtime bias convention must be separately declared rather than replacing these values invisibly. All closures initially exceed .25, making an absolute “reach before closure>.25” diagnostic already exceeded at t0; the existing first-additional-closure diagnostic is the informative one.

The [candidate manifest](cohort_candidate.json) and [CSV](cohort_starts.csv) propose discovery `5928`, `5963`, `6028`, `6273`: the already fitted example, opposite Y extremes and lowest Z. Confirmation uses `6060`, `6094`, `6128`, `6314`, `6346`, `6461`. This uses recording inputs only. It reserves all other complete starts without examining any new policy outcome.

## Fixed object, camera and human handling

The [first-frame contact sheet](initial_frames_contact_sheet.jpg) and [0/.5/1-second sheet](start_settling_contact_sheet.jpg) show the same camera/table/bin and green wrapper. No human hand is visible in these 30 sampled start frames; this is not an exhaustive claim about each full episode. Static-table translation registration across starts has correlation at least .99798 and maximum translation magnitude .0493 pixels. That supports a stable camera within this recording cohort, not calibrated extrinsics. The [image checks](image_session_checks.json) retain the image-space diagnostics.

The packet is kept in the same working region, with small placement/orientation differences and gripper occlusion. The visible green-label centroid varies by several pixels and is not the whole-packet center. Episode `6273` visibly reverses the wrapper by roughly 180 degrees. Do not treat that reversal as a new arm-start variable: transfer its measured robot start into the same canonical packet pose used in every trial. This standardization is explicitly a conditional arm-start study, not per-episode scene replay.

Use the August reference geometry in [waffles_pick_place.json](../../../configs/sim/waffles_pick_place.json): packet center `[-.366715,-.259618,.07]` m, yaw `.44229` rad and actual green packet texture. Those are image-fitted estimates, not physical measurements. Its existing camera was fitted from September evidence and subsequently checked against August images; the within-August stability check does not make that camera an optical calibration. The September post-hand beige packet pose differs and must not be mixed into this cohort without an explicit transfer label.

The previously useful September4 teacher5016 deployment includes a human packet adjustment at 4.4–5.6 seconds. Its post-hand object fit, deployment initial state and later near-grasp states are excluded from this demo-start cohort. A static post-hand reconstruction cannot claim to reproduce that intervention. Tomorrow's ruler/CAD measurements, camera calibration and unloaded wrist/pad tare remain necessary to resolve contact geometry and force realism.

## Reproduce this evidence audit

From the worktree, run the two read-only scripts on compute3 using the live recording environment. The first emits native numeric evidence plus JPEG payloads; retain a fresh filename because the published `demo_starts_raw.json` has its JPEG payloads extracted into [frames/](frames/) and replaced by frame links.

```sh
ssh compute3 'cd /home/physicalai/phantom-icra-2027/phantom && .venv/bin/python -' \
  < docs/results/teacher_v2_design/audit_demo_starts.py > /tmp/teacher_v2_demo_starts_fresh.json
ssh compute3 'cd /home/physicalai/phantom-icra-2027/phantom && .venv/bin/python -' \
  < docs/results/teacher_v2_design/audit_checkpoint_payloads.py > /tmp/teacher_v2_payloads_fresh.json
```

The scripts need the existing NumPy/Zarr/OpenCV/PyTorch recording environment; no installation into a running inference environment is required. The initialization files retain every selected sample value, index and timestamp, allowing direct comparison against the native audit without regenerating or modifying recordings.
