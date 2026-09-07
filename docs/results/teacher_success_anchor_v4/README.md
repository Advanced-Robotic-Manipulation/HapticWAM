# Corrected profile at the successful physical anchor

This is a **development-informed amendment**, frozen before its two combined-profile bridge outcomes and every prospective model score. Gel-only diagnostics succeeded on both reused development seeds while baseline failed both. That motivates testing the combined correction; it does not prove the combination works. The [v3 protocol](../teacher_success_anchor_v3/protocol.json), its unexecuted legacy model screen and its frozen source remain preserved.

The [frozen v4 protocol](protocol.json) pins the exact [bridge campaign](bridge_campaign.json). It retains the historical successful scene, packet position, arm state and September4 baseline. All models use the same immutable `source_teacher_v2_delivery`: corrected gel coverage, gripper-body wrist proxy, live veto with current-delivery feedback, and FINISH. The [source audit](source_audit_before_bridge.json) verifies all105 files against that source's earlier frozen manifest; the independent [controller audit](controller_source_audit.md) explains the remaining proxy limitations and native timing substitution.

**The bridge gate requires both development trials to pass.** Each must be valid, physically place the packet with verified bin support, finish with the packet contained and the robot/pads unloaded, report `placement_release_finished`, have no actual stop, and retain a complete60s physics trace. Source inspection confirms that FINISH suppresses replanning but does not shorten this Isaac runner: only `command.stopped` shortens the loop. The gate accepts one terminal physics step plus the frozen clock tolerance; it does not infer coverage from a duration header.

Failure of either bridge stops model ranking at diagnosis. There is no automatic fallback to legacy or gel-only, replacement of failed bridge trials, or search for favorable bridge seeds. The bridge remains development evidence and contributes no model-selection successes.

If the gate passes, [the screen](../../../configs/sim/teacher_success_anchor_v4_screen.json) runs four teacher/recipe candidates at this same fixed physical state, with six fresh matched seeds each. Two leaders advance to twelve disjoint matched confirmation seeds each:24+24 prospective model trials. The candidates, seeds, ranking tiers and winner tests are unchanged from the v3 plan. Physical placement remains primary; later safety stops and clean completion remain separate. A clear fixed-anchor simulator winner needs at least8/12 physical placements, a positive paired95% bootstrap lower bound, exact two-sided McNemar p≤.05, and no additional drops or pre-placement wrench/tactile-force/depth stops. Zero or degenerate intervals do not show equivalence. No hardware or varied-arm-start winner follows from this fixed-state study.

The [separate arm extension amendment](../teacher_success_anchor_v3/arm_extension/amendment_v4.md) can run only after a passed bridge and a non-null v4 `clear_simulator_winner`; it preserves the six reserved arm starts and fresh seeds without reranking.

## CPU audit and execution

Keep the v4 external driver separate from the immutable simulator source. Its full package needs `phantom/`, `tools/`, `configs/`, this v4 directory and the preserved v3 design dependencies. Substitute the actual external driver path and completed bridge output directory below.

```bash
ANCHOR_V4_BASE=/home/physicalai/phantom-icra-2027/sim/waffles
ANCHOR_V4_DRIVER=$ANCHOR_V4_BASE/source_teacher_anchor_driver_v4
ANCHOR_V4_PY=/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python
ANCHOR_V4_BRIDGE=$ANCHOR_V4_BASE/runs/teacher_success_anchor_v3/corrected_profile_bridge
ANCHOR_V4_ROOT=$ANCHOR_V4_BASE/runs/teacher_success_anchor_v4

"$ANCHOR_V4_PY" "$ANCHOR_V4_DRIVER/tools/sim/audit_teacher_anchor_v4_bridge.py" \
  --runs "$ANCHOR_V4_BRIDGE/rollouts" --out "$ANCHOR_V4_ROOT/bridge_gate.json"

"$ANCHOR_V4_PY" "$ANCHOR_V4_DRIVER/tools/sim/run_teacher_anchor_v4.py" \
  --campaign "$ANCHOR_V4_DRIVER/configs/sim/teacher_success_anchor_v4_screen.json" \
  --source "$ANCHOR_V4_BASE/source_teacher_v2_delivery" \
  --live-repo /home/physicalai/phantom-icra-2027/phantom \
  --server-python "$ANCHOR_V4_PY" --evidence "$ANCHOR_V4_BASE/evidence" \
  --hardware-config "$ANCHOR_V4_BASE/runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml" \
  --robot-usd "$ANCHOR_V4_BASE/runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda" \
  --stage-gate-audit "$ANCHOR_V4_ROOT/bridge_gate.json" \
  --output "$ANCHOR_V4_ROOT/screen" --port 7799
```

The runner invocation above plans only. Add `--execute` to run after the bridge audit passes. Before launching any child process, it re-audits both bridge raw traces and verifies the corrected source/input hashes. It invokes its own current analyzer, independently from the simulator source, so the NFE5/K4 override is audited correctly. Per-policy model settings and actual external-controller/runtime source hashes are recorded separately.

After all24 screen rows complete:

```bash
"$ANCHOR_V4_PY" "$ANCHOR_V4_DRIVER/tools/sim/select_teacher_anchor_v4.py" \
  --campaign "$ANCHOR_V4_ROOT/screen/campaign_snapshot.json" \
  --runs "$ANCHOR_V4_ROOT/screen/rollouts" --out "$ANCHOR_V4_ROOT/screen/selection" --stage screen

"$ANCHOR_V4_PY" "$ANCHOR_V4_DRIVER/tools/sim/freeze_teacher_anchor_v4_confirmation.py" \
  --protocol "$ANCHOR_V4_DRIVER/docs/results/teacher_success_anchor_v4/protocol.json" \
  --screen "$ANCHOR_V4_ROOT/screen/campaign_snapshot.json" \
  --selection "$ANCHOR_V4_ROOT/screen/selection/selection.json" \
  --out "$ANCHOR_V4_ROOT/confirmation_campaign.json"
```

Run the same v4 driver with the derived confirmation campaign and a new output directory. After completion, call `select_teacher_anchor_v4.py --stage confirmation` against those24 rows. The builder refuses overwritten outputs, changed raw inputs, missing/invalid matches and forged selection. The paired uncertainty unit is a sampling-seed pair at one fixed physical state, not an arm-start cluster.

Validation:110 focused CPU tests pass across the new v4 bridge/selection and prior campaign/selection regression modules; Ruff is clean. Tests cover full60s bridge coverage, later stops, missing support, immutable profile/seed/threshold bindings, raw-gate/source changes, per-policy NFE, exact matched grids, reserved confirmation, and the exact small-sample guard. No GPU or hardware process was launched by these tools during preparation. Combined bridge and prospective model outcomes are pending.
