# Executable fixed-anchor comparison

The accepted [64-trial protocol](protocol.json) remains unchanged. The separate [stage-gate amendment](stage_gate_amendment.json) permits support-verified command replay to establish mechanics without calling it a policy success. The [passed gate audit](stage_gate_audit.json) preserves all four failed historical-seed controls, verifies120 historical source/input hashes, and records exact saved-input proposal reproduction. Accepted-command replay places the packet at25.2s with free dynamic physics, zero attachments and zero post-initialization object pose writes. It does not establish stochastic closed-loop repeatability.

The screen config is [teacher_success_anchor_v3_screen.json](../../../configs/sim/teacher_success_anchor_v3_screen.json): four teachers/recipes × six fresh matched seeds, at the same already-offset physical scene. Condition offset is zero, so the historical −10mmX/−10mmY offset is applied once. The main controller remains the original v1 profile; component diagnostics cannot choose the comparison profile.

Keep the executable controller/analyzer in a **separate full source copy**, with `phantom/`, `tools/`, `configs/` and this v3 documentation directory. Do not modify the frozen v1 source. The external driver hashes its own analysis/controller files separately from the actual simulator/server runtime and verifies the passed gate before launching anything. Both per-policy NFE settings reach the v1 server and runtime inference JSON. The generated v1 simulator flags are checked against the original parser fixture.

On compute3, substitute the actual external driver directory below. The first command plans only; add `--execute` to the same invocation after review. The driver refuses reserved rig ports and preserves completed outcomes.

```bash
ANCHOR_DRIVER=/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_anchor_driver_v3
ANCHOR_BASE=/home/physicalai/phantom-icra-2027/sim/waffles
ANCHOR_PY=/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python
ANCHOR_OUT=$ANCHOR_BASE/runs/teacher_success_anchor_v3/screen

"$ANCHOR_PY" "$ANCHOR_DRIVER/tools/sim/run_policy_campaign.py" \
  --campaign "$ANCHOR_DRIVER/configs/sim/teacher_success_anchor_v3_screen.json" \
  --source "$ANCHOR_BASE/source_teacher_pick_place_v1" \
  --live-repo /home/physicalai/phantom-icra-2027/phantom \
  --server-python "$ANCHOR_PY" --evidence "$ANCHOR_BASE/evidence" \
  --hardware-config "$ANCHOR_BASE/runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml" \
  --robot-usd "$ANCHOR_BASE/runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda" \
  --stage-gate-audit "$ANCHOR_DRIVER/docs/results/teacher_success_anchor_v3/stage_gate_audit.json" \
  --output "$ANCHOR_OUT" --port 7799
```

After all24 screen trials finish, run CPU selection and mechanically derive the24 reserved confirmation cells. The builder independently rescans raw inputs and scores; it refuses missing/invalid cells, altered selection, changed inputs and an existing output file.

```bash
"$ANCHOR_PY" "$ANCHOR_DRIVER/tools/sim/select_teacher_anchor.py" \
  --campaign "$ANCHOR_OUT/campaign_snapshot.json" \
  --runs "$ANCHOR_OUT/rollouts" --out "$ANCHOR_OUT/selection" --stage screen

"$ANCHOR_PY" "$ANCHOR_DRIVER/tools/sim/freeze_teacher_anchor_confirmation.py" \
  --protocol "$ANCHOR_DRIVER/docs/results/teacher_success_anchor_v3/protocol.json" \
  --screen "$ANCHOR_OUT/campaign_snapshot.json" \
  --selection "$ANCHOR_OUT/selection/selection.json" \
  --out "$ANCHOR_BASE/runs/teacher_success_anchor_v3/confirmation_campaign.json"
```

Run the same external driver with the derived confirmation campaign and a new output directory. After its24 cells finish, call `select_teacher_anchor.py --stage confirmation` with that campaign, rollout directory and selection output directory.

Physical support-verified placement is primary. A later controller stop remains separate and does not erase placement; final containment/support and clean FINISH are reported independently. Confirmation uses twelve **matched seed pairs at one fixed state**, not arm-start clusters. A winner requires at least8/12 physical placements, a positive paired95% bootstrap lower bound, exact two-sided McNemar p≤.05, and no extra drops or pre-placement wrench/tactile-force/depth stops. Degenerate intervals do not establish equivalence. The union force-stop gate counts a trial once if either wrench or tactile force/depth ends it before placement; individual causes remain available in the CSV.

The [component design audit](component_design_audit.json) verifies that all six diagnostic profiles retain the exact effective scene, thresholds, two seeds and recipe. Delivery-feedback isolation is implemented by its separate source overlay, so its campaign profile equals baseline. Source-overlay correctness is documented separately by the controller/sensor overlay audits. No diagnostic outcome enters model ranking.

Validation:88 focused tests pass (v3selection/gates, current campaign driver and existing selection/confirmation regression checks); Ruff passes on changed code. No inference, simulator or hardware process was launched by the report/selection work. Screen and confirmation outcomes are pending.
