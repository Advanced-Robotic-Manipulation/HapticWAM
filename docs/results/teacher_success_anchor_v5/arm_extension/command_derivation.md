# Mechanical derivation after a confirmed v5 winner

There is no executable winner campaign yet. This is a plan-only derivation
contract; do not fill a candidate ID from interim screen counts or a ranked
leader. Do not reuse the core selector to treat this extension as another
confirmation stage.

1. Read the completed v5 confirmation campaign and selection with their SHA256
   identities. Require `stage=confirmation`,
   `status=complete_valid_matched_stage`, 24 expected and available trials,
   empty missing/invalid lists, every `winner_gates` value true and a non-null
   `clear_simulator_winner`. Recompute the existing selection from its saved
   rows using `tools/sim/teacher_anchor_compare.py`; require the same result.
2. Verify this amendment's parent hash, the original six state hashes and all
   unchanged-field digests. Copy the one winning policy dictionary verbatim
   from the confirmed campaign, including checkpoint hash, EMA and all inference
   settings. Retain its frozen runtime/input/source contract, nominal scene,
   hardware, thresholds, sensor/controller profile, timing and horizon.
3. Stage unchanged copies of the six measured-state JSONs under a **new**
   extension input directory and verify each SHA256. Do not add them to a frozen
   runtime source. Create six conditions in the amendment's order by copying
   the original anchor condition, preserving zero object offset, friction scale
   1, no camera override and zero injected delays. Set each condition's `id`
   to its start ID and `initial_state` to its staged `{path, sha256}`. Keep the
   shared tactile baseline in `adapter_profile` unchanged. The first condition
   remains family `nominal`; label the other five `measured_arm_start`.
4. Derive a **new** campaign `teacher_success_anchor_v5_arm_extension`, with the
   winning policy only, seeds `[904601, 904602]`, six conditions and counts
   `{conditions: 6, seeds_per_condition: 2, per_policy: 12, primary: 12,
   secondary: 0, total: 12}`. Use two execution phases: first seed 904601 with
   forward condition order, then seed 904602 with reversed condition order.
   Clear obsolete core `case_manifest`/execution-order summaries and replace
   them with these 12 exact keys; preserve the original input contract separately.
   Bind the new amendment, completed confirmation and selection hashes in
   `extension_provenance`. Freeze this derived JSON and the external controller
   helper hash before any inference.
5. Generate a plan with the existing controller **without `--execute`**, then
   verify `planned_trials=12`, exact block order and every
   `initial_state_by_condition` path/hash. The controller's `server_command`
   and `simulation_command` functions supply the final argument arrays;
   materialize them in the new output manifest before execution. Confirm the
   only per-start runtime difference is `--policy-initial-state`, plus the
   already declared seed/output/condition paths. All six condition scene JSONs
   must be semantically identical. Ordinary settling and start guards still
   apply; a rejected start is retained, not changed or replaced.

The simulation-only plan command has this fixed argument derivation:

```text
<live-repo>/.venv/bin/python <hash-pinned-driver>/tools/sim/run_policy_campaign.py
  --campaign <new frozen derived extension JSON>
  --source /home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_anchor_minimal_v5
  --live-repo /home/physicalai/phantom-icra-2027/phantom
  --evidence /home/physicalai/phantom-icra-2027/sim/waffles/evidence
  --hardware-config /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml
  --robot-usd /home/physicalai/phantom-icra-2027/sim/waffles/runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda
  --output <new extension output directory>
  --port <verified free owned inference port outside reserved rig ports>
```

The command above is deliberately not a hardware launch, and omits `--execute`.
The experiment owner controls subsequent process scheduling and cleanup. There
is no fallback execution if the winner gate is false. Extension scoring remains
descriptive and cannot alter the original winner or denominator.
