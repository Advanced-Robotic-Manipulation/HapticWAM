# Independent controller and mechanics diagnostics

The reference is the immutable `source_teacher_pick_place_v1` tree under
`/home/physicalai/phantom-icra-2027/sim/waffles/`. Current code with legacy CLI
flags is not an exact substitute: its placement-enabled terminal filter reads
current delivery TCP/aperture, whereas v1 reads the captured request snapshot.
This source audit does not establish which variant performs better.

The reference case is
`runs/teacher_pick_place_v1/campaign/rollouts/teacher__placement_xm10_ym10mm__seed4242`.
Reuse its exact effective scene, measured initial state, robot USD, hardware and
inference settings. Its prepared image/episode source is Aug22 5928; the v1
measured startup and tactile baseline are the separately saved Sept4 artifacts.
Changing either is a separate input factor.

| Factor | Reference setting | Isolated change and dependency |
|---|---|---|
| Gel coverage | `--gel-contact-coverage manifold_patch` | `manifold_patch_v2` needs the newer `tools/sim/gel_contact.py` and a runner CLI choice extension. The measured `phantom/sim/tactile_proxy.py` itself is byte-identical across v1/current. |
| Wrist | `--wrist contact_proxy` (pad-only) | `gripper_contact_proxy` needs its gripper-contact observer and runner initialization/control sampling. This changes wrist input/guard feedback, so it is not a rendering-only factor. |
| Native veto version | JSON `implementation: fd4a032` | Set only the copied JSON to `live`. V1 already contains both implementations; no source overlay needed. Preserve historical request feedback for this factor. |
| Veto feedback timing | Captured request TCP/aperture | `prepare_teacher_anchor_feedback.py` emits one filter module. Keep `fd4a032`, v1 local opening passthrough, and every other setting. Model observations remain captured at request. |
| Post-release FINISH | Absent | `prepare_teacher_anchor_finish.py` emits only adapter and local release classes; add only `finish_after_release: true` in copied release JSON. Keep v1 filter, sensors, safety and 60 s scored horizon. |

Each overlay helper refuses an unreviewed base hash and requires a new output
directory outside its source trees. Apply emitted files only to a **new copy**
of v1. Do not merge overlays when measuring a single factor. Their manifests
record base, donor and output hashes. The old and amended studies remain intact.

FINISH uses policy-commanded opening, a previously loaded latch, measured open
aperture and unloaded tactile dwell. It freezes the achieved measured TCP and
accepted open command, stops requesting plans and keeps safety active. It does
not read object state or imply physical task success. The old runner continues
physics to the declared horizon; completion remains in execution diagnostics.

Two separate diagnosis helpers isolate timing and mechanics:

```bash
python tools/sim/prepare_teacher_anchor_diagnostic.py \
  --base-source /path/to/source_teacher_pick_place_v1 \
  --donor-source /path/to/reviewed/current/source \
  --output-overlay /new/path/latency_overlay --variant latency_schedule

python tools/sim/prepare_teacher_anchor_diagnostic.py \
  --base-source /path/to/source_teacher_pick_place_v1 \
  --donor-source /path/to/reviewed/current/source \
  --output-overlay /new/path/commands_overlay --variant accepted_commands
```

For the latency overlay, retain the original policy command and add
`--anchor-latency-trace ORIGINAL/planner_trace.json`. Omit other latency overrides.
It replays each recorded `latency_s` by replan ordinal into the adapter delivery
time/action grid, records actual-versus-original request times, and fails before
any additional inference if the schedule ends. Live model inputs, physics and
safety remain active. Native K4 selection still uses its newly measured compute
time before the adapter override; the next CPK offset uses the overridden prior
latency. Therefore this is **simulator delivery timing replay**, not complete
native timing or selected-action replay.

For the accepted-command overlay, use `--mode policy` solely to preserve the
exact v1 initialization/settling path and add
`--anchor-command-trace ORIGINAL/execution_trace.jsonl` with the original
`--policy-initial-state`. No server or adapter is constructed. The output records
`mode: command_replay` and the input command hash. At each physics tick it holds
the latest actually submitted joint/finger targets, never samples future targets,
and never writes object state after initialization. It replays the recorded
safety decisions rather than making new ones; it is a mechanics diagnostic and
must not enter policy-score denominators.

The reference command trace was checked read-only: all **3,923 rows** have
`status: drive_submitted`, finite six-joint/two-finger targets and strictly
increasing timestamps from **0.004 to 31.380 s**. Finger targets range from
0.0098620911 to 0.035 m. SHA-256:
`c0eb541f2574130b4cef64ca781e1be62e1fb798e0265cfbee8c17d93c50a70d`.
The final stopped row has valid held targets. Set the original actual observation
horizon to **33.376 s** to include the held post-stop physics tail. Compare
initialization, command-grid alignment and measured physics traces before
attributing any difference to contact dynamics.

CPU verification includes frozen-v1 command equality with FINISH disabled;
equality until release completion with FINISH enabled; achieved-pose hold and
safety override; latency-grid override without mutation of the raw proposal;
schedule validation/exhaustion; causal drive targets; and delivery-only versus
request-only close masks with preserved policy observations, historical opening
passthrough and retry-stop priority. These checks do not substitute for Isaac or
physical qualification.
