# Teacher-only lab card — 7 September 2026

**There is no validated final pick-and-place winner.** The completed simulator confirmation gave ftA3000 and v5_6 each **3/12 acquisitions, 1/12 lifts/carries, 0/12 placements**. Both carries stopped at wrist extension while holding, above the release region. The models remain paired diagnostic pickup candidates. These estimated sensor/contact results do not predict real success rates.

1. **Measure and record the rig.** Prioritize base/table height, TCP-to-pad faces/backing, actual aperture, packet/box dimensions and reset marks, then camera calibration and unloaded wrist/tactile readings at rest and through reviewed motion. Use the [37-row sheet](measurements/setup_20260907_template.csv) and [measurement guide](isaac_teacher_measurements_20260907.md).
2. **Reproduce the existing native pickup path.** The [full handoff](isaac_lab_handoff_20260907.md) contains exact read-only checks, server identity checks, reviewed starts and attended commands. Its first pickup gate uses the historical ftA1500 reference. Score visible object clearance and retained hold; a `lift_complete` flag alone is insufficient.
3. **Run a bounded paired pickup pilot after that gate.** Compare ftA3000 versus v5_6 for five reviewed starts, seeds 101–105, ten episodes. Both use **teacher, EMA, NFE1, K4, guidance 1, persistent noise, parity, live terminal veto and max-play 10**. Alternate which candidate runs first as specified in the handoff. Preserve achieved starting q/TCP/aperture, all attempts, proposals, accepted commands, videos and stop reasons. Keep packet/camera/box fixed and verify the loaded server checkpoint for every change.
4. **Qualify transfer and release separately.** Both simulated carries remained outside the release volume; no eligible opening was played after the latch. Nominal in-volume IK solutions exist, but their paths and collision clearance are unvalidated. Measure the real release region and bench-test the opt-in release/FINISH bridge before policy placement. Keep wrist-extension, force, speed and workspace limits unchanged.

| Candidate | Checkpoint on compute3 | SHA256 prefix; full digest in handoff |
|---|---|---|
| ftA3000 | `/home/physicalai/phantom-icra-2027/sim/waffles/checkpoints/teacher_v5_ftA/teacher_003000.pt` | `ee0a448c00da` |
| v5_6 | `/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_batch0822/v5_6.pt` | `7edcb8335681` |
| ftA1500 historical reference | `/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_ftA/teacher_001500.pt` | `67c93287123e` |

The new [fixed wrist-reference mode](https://github.com/Advanced-Robotic-Manipulation/phantom/pull/13) is a separate opt-in experiment. It detects a saved gradual overload earlier, but changing real wrist bias can also create stops; one later recorded-demo trigger remains unadjudicated. It is not enabled by the study or this card. No hardware has been controlled or updated by this work.

[Complete results and failure analysis](results/teacher_robustness_v2_delivery/README.md) · [Tactile comparison video](results/teacher_robustness_v2_delivery/media/paired_pickup_attempts_60s.mp4) · [Full operational handoff](isaac_lab_handoff_20260907.md).

After calibration, reuse the completed simulator cases as regression checks and reserve fresh measured starts/seeds for another teacher selection. Test measured small waffle-reset jitter afterward as a separate factor; avoid simultaneous arbitrary arm/object randomization.
