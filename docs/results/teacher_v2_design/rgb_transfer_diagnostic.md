# Paired RGB transfer diagnostic

The [client](../../../tools/sim/diagnose_rgb_transfer.py) is ready, with eight CPU tests passing. CPU preparation succeeded without contacting any model server; [the preparation manifest](rgb_transfer_preparation.json) pins every input and array. **Inference has not run.** This helper is excluded from the corrected primary study's frozen source and its 56 scored rollouts.

This diagnostic uses the first actual teacher request from the halted original case `fta1500_nfe1_k4__start_1787395928__seed903101`. That request has one latest RGB frame and eight other observation fields. Only the RGB tensor changes. Joint/TCP state, wrist history, fingertip arrays, contact state, reactive input, previous-chunk array, task text and request timestamps remain identical; each non-RGB array is checked by dtype, shape and byte hash. Every call passes no previous plan and resets the native server's episode/noise state to its specified seed.

The real replacement is canonical demo5928 camera frame 3, at native time `5581.950773053104`. It is 10.148 ms after the requested camera timestamp and 49.852 ms before the model request. The paired real q/TCP sample shares exactly one native timestamp; the complete streams have different lengths, so samples are joined by timestamp rather than row number. The real arm moved 1.032 mm from its recorded initial TCP. Its pose differs from the saved simulator request by 5.130 mm and 0.799°. Full-image RGB MAE is 36.383 on a 0–255 scale; this measures appearance difference, not camera calibration. The saved request also retains the halted study's earlier non-RGB proxy limitations.

The six calls are fixed:

| Sampling seed | Call order |
|---|---|
| 903101 | Rendered RGB, real RGB, rendered RGB repeat |
| 903102 | Real RGB, rendered RGB, rendered RGB repeat |

All use ftA1500 EMA (`67c93287…543893e`), NFE 1, K 4, guidance 1, parity and persistent noise enabled, and task text `waffles`. A dedicated owned server must match the saved checkpoint, normalizers, backbone, text cache, hardware configuration and six native inference-source hashes. The client refuses rig ports 7777–7783, checks the supplied PID against the server helper's process arguments and ready marker, and requires both corrected stages to be complete and their controller/server processes to have exited. It starts or stops no process and changes no inference settings.

After the primary study, the parent operator starts a fresh dedicated server with this command on compute3, retaining ownership of its process. `RGB_PORT` is an unused dedicated port chosen by that owner, outside 7777–7783. The helper performs its normal warmup and publishes the ready marker; EMA is its default and is checked by the client.

```bash
SIM_BASE=/home/physicalai/phantom-icra-2027/sim/waffles
/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python \
  "$SIM_BASE/source_teacher_v2_delivery/tools/sim/policy_server.py" \
  --repo /home/physicalai/phantom-icra-2027/phantom \
  --ckpt /home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_ftA/teacher_001500.pt \
  --expected-sha256 67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e \
  --hardware "$SIM_BASE/runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml" \
  --system teacher --nfe 1 --k-seeds 4 --guidance 1 \
  --parity-fixes --persistent-noise --task-text waffles \
  --port "$RGB_PORT" --out "$SIM_BASE/runs/teacher_rgb_transfer_v1/server"
```

Then run the client from a separate owned command. `READY_JSON` is the new server's `server/ready.json`; `OWNED_PID` is its actual PID from that file. `execution` must be a new directory; partial attempts are preserved and never silently retried.

```bash
SIM_BASE=/home/physicalai/phantom-icra-2027/sim/waffles
PYTHONPATH="$SIM_BASE/source_teacher_v2" \
  /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python \
  "$SIM_BASE/review_tools/tools/sim/diagnose_rgb_transfer.py" \
  --observation "$SIM_BASE/runs/teacher_robustness_v2/screen/rollouts/fta1500_nfe1_k4__start_1787395928__seed903101/observations/0000.npz" \
  --initial-state "$SIM_BASE/source_teacher_v2/configs/sim/initial_states/waffles_aug22_1787395928_000.json" \
  --episode /home/physicalai/phantom-icra-2027/data/full/tasks/waffles/ep_waffles_1787395928_000 \
  --out "$SIM_BASE/runs/teacher_rgb_transfer_v1/execution" \
  --execute \
  --completed-study "$SIM_BASE/runs/teacher_robustness_v2_delivery" \
  --server-ready "$READY_JSON" --owned-server-pid "$OWNED_PID"
```

Omit `--execute` and the final three guard arguments for CPU preparation only, using a different new output directory. The existing CPU preview is `/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_rgb_transfer_v1/cpu_preview/`.

Each call saves the native unfiltered 16-step proposal, inference latency, action timestamps, event/contact diagnostics and cumulative 10/16-step endpoint. The deltas are added without a second `dt` multiplication. The report compares the paired RGB change against the repeated-render control for each seed; latency-dependent action timestamps are reported separately from action values. These are proposals before governor, veto, IK or physical execution. They cannot establish a pickup success rate, a checkpoint winner, or a causal explanation of hardware failures.
