# Minimal-profile teacher comparison

V5 is a **new development-informed amendment**, not a relabeling of v3 or v4. The combined v4 bridge failed both valid trials: wrist-extension stop at16.824s and tactile-depth stop at19.624s, with no physical placement. Its [authoritative failed gate](../teacher_success_anchor_v4/failed_gate.json) remains preserved and prohibits v4 model ranking. V3 and v4 prospective model screens were never executed.

The [frozen v5 protocol](protocol.json) compares all four teachers/recipes under the minimal gel-v2 plus FINISH profile, regardless of reference ftA1500 development performance. Both minimal-profile development trials also failed; they remain excluded and do not prevent testing other candidates. Environment admission comes from reproducible mechanics, saved-input inference validation and verified source/input integrity, not requiring a teacher win in advance.

The [admission audit](admission_audit.json) passes120 immutable source/native/input checks, including all four declared replacements. The [minimal source manifest](minimal_profile_source_manifest.json) pins the independent source copy and its exact4file delta. Its assembly inventory contains bytecode caches; admission verifies executable source, configuration, assets and required inputs rather than volatile cache bytes. [The source review](minimal_profile_audit.md) explains the gel correction and FINISH behavior.

All candidates retain the exact historical successful physical scene, packet position, arm state and September4 tactile baseline. The screen uses an already-offset scene with zero additional condition offset. Historical request-time fd4a032 veto, pad-only wrist proxy and original safety thresholds remain explicit benchmark limitations. This does not claim an identical current native deployment executor. FINISH suppresses replanning but the minimal runner continues physics and safety monitoring through60s unless an actual stop occurs.

The authoritative v5 counts are **2 excluded development diagnostics +24 prospective screen +24 reserved confirmation =50**. The inherited protocol sentence mentioning a64-trial core describes the earlier plan; `planned_counts` defines this amendment. No arm-start trials are included in these50.

The four candidates and recipes, prospective seeds904401–904406, reserved seeds904501–904512, physical-primary ranking, paired95% seed bootstrap, exact McNemar p≤.05 and winner gates remain unchanged. The reference teacher's development failure does not remove it or admit a fallback profile. Physical placement is determined only from ordered object/contact motion and verified bin support. Later stops do not erase it. Clean FINISH remains secondary.

## Execution and analysis

The screen uses the **existing frozen external generic driver**, so no new runtime code was needed:

```bash
ANCHOR_V5_BASE=/home/physicalai/phantom-icra-2027/sim/waffles
ANCHOR_V5_DRIVER=$ANCHOR_V5_BASE/source_teacher_anchor_driver_v5
ANCHOR_V5_PY=/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python
ANCHOR_V5_ROOT=$ANCHOR_V5_BASE/runs/teacher_success_anchor_v5

"$ANCHOR_V5_PY" "$ANCHOR_V5_DRIVER/tools/sim/run_policy_campaign.py" \
  --campaign "$ANCHOR_V5_DRIVER/configs/sim/teacher_success_anchor_v5_screen.json" \
  --source "$ANCHOR_V5_BASE/source_teacher_anchor_minimal_v5" \
  --live-repo /home/physicalai/phantom-icra-2027/phantom \
  --server-python "$ANCHOR_V5_PY" --evidence "$ANCHOR_V5_BASE/evidence" \
  --hardware-config "$ANCHOR_V5_BASE/runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml" \
  --robot-usd "$ANCHOR_V5_BASE/runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda" \
  --output "$ANCHOR_V5_ROOT/screen" --port 7799 --execute
```

This is the campaign invocation, not an instruction to start a duplicate while it is active. There is no `--stage-gate-audit` policy-success gate in v5. The source/input admission audit and the driver's frozen manifests provide provenance.

CPU selection uses one new parameterized helper, `tools/sim/teacher_anchor_compare.py`, in a separate review layer. **Do not add it to the running frozen driver.** The review layer needs the helper plus byte-identical shared modules `analyze_policy_campaign.py`, `select_teacher_anchor.py`, `select_teacher_candidate.py` and `teacher_anchor_design.py`; link `phantom/`, `configs/` and `docs/` read-only to the frozen driver. Use `PYTHONDONTWRITEBYTECODE=1` to avoid writes through those links.

After all24 screen cases finish:

```bash
ANCHOR_V5_REVIEW=$ANCHOR_V5_BASE/review_teacher_anchor_v5
PYTHONDONTWRITEBYTECODE=1 "$ANCHOR_V5_PY" "$ANCHOR_V5_REVIEW/tools/sim/teacher_anchor_compare.py" score \
  --protocol "$ANCHOR_V5_DRIVER/docs/results/teacher_success_anchor_v5/protocol.json" \
  --campaign "$ANCHOR_V5_ROOT/screen/campaign_snapshot.json" \
  --runs "$ANCHOR_V5_ROOT/screen/rollouts" --out "$ANCHOR_V5_ROOT/screen/selection" --stage screen

PYTHONDONTWRITEBYTECODE=1 "$ANCHOR_V5_PY" "$ANCHOR_V5_REVIEW/tools/sim/teacher_anchor_compare.py" confirm \
  --protocol "$ANCHOR_V5_DRIVER/docs/results/teacher_success_anchor_v5/protocol.json" \
  --campaign "$ANCHOR_V5_ROOT/screen/campaign_snapshot.json" \
  --selection "$ANCHOR_V5_ROOT/screen/selection/selection.json" \
  --out "$ANCHOR_V5_ROOT/confirmation_campaign.json"
```

The confirmer re-audits the raw screen and rejects changed inputs, incomplete matches, altered rankings and overwritten outputs. Run the derived24 confirmation cases with the same generic driver/source and a new output directory; then use `teacher_anchor_compare.py score --stage confirmation`. The paired uncertainty unit is a sampling seed at this one physical state, not varied-arm-start clusters or a hardware population.

The helper supports old FINISH telemetry explicitly: when `run.json` omits its completion field, consistent execution-trace completion diagnostics supply that controller metadata, with `completion_reason_source=execution_only`. Explicit conflicts remain invalid. No raw recording is edited, and completion cannot substitute for physical success. Ten focused CPU tests cover that schema distinction, profile/seed/recipe preservation, confirmation and the exact statistical guard; Ruff passes.

Prospective model results are pending. These diagnostics and changes do not themselves establish a clear model winner.
