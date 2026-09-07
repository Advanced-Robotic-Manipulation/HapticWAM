# Fixed-waffle teacher comparison — corrected delivery profile

The 32-trial discovery screen is complete and valid. **No teacher achieved a sustained lift or placement.** The frozen selector advanced `v5_6_nfe1_k4` and `fta3000_nfe1_k4` to 24 confirmation trials on six reserved measured starts with fresh seeds. Confirmation is running; no final winner is declared.

| Teacher / recipe | Trials | Bilateral acquisition | Sustained lift | Strict placement | Clean placement | Safety stops |
|---|---:|---:|---:|---:|---:|---:|
| ftA1500, NFE 1 / K 4 | 8 | 0 | 0 | 0 | 0 | 4 |
| ftA3000, NFE 1 / K 4 | 8 | 1 | 0 | 0 | 0 | 6 |
| v5_6, NFE 1 / K 4 | 8 | 2 | 0 | 0 | 0 | 8 |
| ftA1500, NFE 5 / K 1 | 8 | 1 | 0 | 0 | 0 | 7 |

Acquisition is the frozen bilateral physical-contact criterion; it does not establish lifting or a stable grasp. The NFE 5/K 1 configuration also has one non-safety `veto_retry_cap` stop. See the authoritative [screen selection and gates](screen/selection.json), [per-trial CSV](screen/trials.csv), and [screen report](screen/report.md). The campaign analysis `summary.json` contains an index of score files; it does not embed the per-trial outcomes. Aggregates must use the selector rows or the referenced score files.

The experiment follows collection practice: one fixed canonical green waffle, common camera/materials/lighting and authentic measured arm/gripper/wrist starts. Four discovery starts and two seeds per configuration precede six reserved starts and two fresh seeds for each selected teacher. These are simulator-study holdouts; training-unseen status is not established. The [protocol](../../../configs/sim/teacher_v2_delivery_protocol.json), [execution guide](../../isaac_teacher_v2_experiment.md) and [derived confirmation configuration](teacher_v2_delivery_confirmation.json) define the design and unchanged physical success thresholds.

The [first-start contact audit](first_start_diagnostic/README.md) independently reconstructs six wrist stops and two horizon failures, with no acquisition in that start block. It identifies housing/box, pad/environment and pad/packet contacts. Substantial contact overlap in some cases limits physical force interpretation. [Checkpoint and input preprocessing](../teacher_v2_design/preprocessing_audit.md) passed for all three teachers; [the first corrected trial's audit](../teacher_v2_design/corrected_first_policy_case_audit.md) verifies checkpoint/input/source identity, wrist arithmetic and causal clock behavior. These are implementation checks, not calibration of tactile or wrist transfer.

Two implementation gaps were corrected before this study: the release-enabled terminal guard now uses measured delivery feedback, and the wrist proxy includes external contact on the housing as well as both pads. Shared policy-commanded release/FINISH behavior and corrected gel manifold coverage remain active for every candidate. The ten earlier v2 trials are preserved as superseded diagnostics and excluded from these scores. No confirmation outcome was observed before this corrected freeze. This comparison changes several inputs from the previous teacher campaign and is not a paired estimate of the effect of any one correction.

Exact compute3 source: `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_v2_delivery`. Raw results: sibling `runs/teacher_robustness_v2_delivery/{screen,confirmation}/rollouts/`. [Execution arguments and source archive hash](execution_plan.json), [frozen inputs](prelaunch_frozen_inputs.json), and [published-source audit](published_source_audit.json) are retained here. All 105 executed source files match the published executable code; one file differs only in a corrected historical comment. Full raw observations, states, proposals, accepted commands and all tactile review videos remain on compute3.

For the September 7 laboratory session, use the [teacher handoff](../../isaac_lab_handoff_20260907.md) and [measurement guide](../../isaac_teacher_measurements_20260907.md). Table/base and pad/TCP geometry, housing/backing dimensions, aperture, packet/box geometry, camera calibration and sensor transfer remain unresolved. Measured values must produce a new scene revision before simulator findings are carried over to physical final experiments.
