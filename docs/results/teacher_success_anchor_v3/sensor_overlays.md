# Separate sensor overlays for the successful v1 reference

The [preparation tool](../../../tools/sim/prepare_teacher_anchor_sensors.py) creates **one isolated variant per output directory**. It verifies every file listed in the original frozen-input manifest, the exact successful v1 runner hash and the selected helper-module hashes. It refuses changed input files, an existing output directory or a combined variant. It runs no model, simulator or hardware.

| Variant | Changed or added files | Required rollout flag |
| --- | --- | --- |
| `gel_v2` | `tools/sim/run_waffles.py` adds one coverage choice; `tools/sim/gel_contact.py` uses the unchanged vetted v2 helper | `--gel-contact-coverage manifold_patch_v2` |
| `gripper_wrist` | Minimal wrist setup/cache/logging changes in `tools/sim/run_waffles.py`; unchanged `gripper_wrist.py` and `robot_environment_contacts.py` helpers added | `--wrist gripper_contact_proxy` |

The gel variant leaves the main runner's runtime AST unchanged, including mechanics, and changes only its argument parser. The wrist variant preserves tactile synthesis, all articulation/object writes, world stepping/reset calls and existing controller modules. Both preserve scene settings, initial state, September tactile baseline, pressure synthesis, historical veto, placement release and safety thresholds. The wrist variant intentionally changes policy and safety feedback; it is not an observational ablation. Defaults remain v1 until the explicit variant flag is selected.

[Module/runner hashes](sensor_overlay_pins.json), the [gel patch](gel_v2_runner.patch) and the [wrist patch](gripper_wrist_runner.patch) make the changes reviewable. The existing helper implementations are copied without modification. The gel variant retains a fixed2 mm geometric support envelope, the explicit single-convex-body whitelist and missing-separation fallback/diagnostics; it does not invent forces. The wrist variant reports signed normal force and true contact-point moments for housing and two pads against table, mat, five bin bodies and packet. The recorded six-component initial bias is unchanged. It omits friction, gravity, inertia, self/proximal-arm contact and calibrated CB3 estimator behavior.

The wrist input is sampled once on each125 Hz control tick, cached for both `adapter.observe` and the execution log, with an initial sample and continued read-only sampling during the post-stop observation period. `wrist_contact_trace.jsonl` preserves actor/filter labels, contact points, signed separations/impulses, converted forces, point moments, bias and sample time. `wrist_capture_t` links each execution input to its cached sample. No contact force is clipped to force a successful outcome.

For the parent-owned remote preparation, upload only the preparation tool; the pinned sensor helpers already exist in `source_teacher_v2_delivery`. Example, with `BASE=/home/physicalai/phantom-icra-2027/sim/waffles` and the project Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python \
  "$BASE/review_tools/prepare_teacher_anchor_sensors.py" \
  --base-source "$BASE/source_teacher_pick_place_v1" \
  --frozen-inputs "$BASE/runs/teacher_pick_place_v1/campaign/frozen_inputs.json" \
  --module-source "$BASE/source_teacher_v2_delivery" \
  --variant gel_v2 \
  --output "$BASE/source_teacher_anchor_gel_v2"
```

Run the preparation command separately with `--variant gripper_wrist` and a different output, for example `source_teacher_anchor_gripper_wrist`. No preparation or rollout command was launched remotely by this task. Use each output's copied launcher, original reference runtime configuration and only its declared sensor flag. The tool records every unchanged file's hash and every changed file's base/output hash in `sensor_overlay_manifest.json`.

Unchanged files are symlinks into the immutable base to conserve disk; the launcher is copied unchanged so its repository-root calculation selects the overlay. **Never edit the overlay symlinks.** Set `PYTHONDONTWRITEBYTECODE=1` during execution and verify the manifest before a diagnostic. The outputs depend on the continued presence of the original frozen source; these are overlays, not standalone archives.

Nine overlay tests pass against a63 KB exact frozen-runner fixture; the overlay and existing contact-helper suites pass81 tests in total. They check rejection of altered source, no overwritten output/base files, precise variant file isolation, unchanged physical write calls, unchanged gel synthesis and a defensive-copy wrist cache whose repeated reads do not resample. Both patched runners compile. These tests do not validate the added PhysX view at runtime. Before a policy comparison, the wrist observer still needs a separate command-replay invariance check; fresh gel traces should confirm populated separation evidence and force conservation. Neither overlay establishes calibrated gel pressure or hardware wrist loads.
