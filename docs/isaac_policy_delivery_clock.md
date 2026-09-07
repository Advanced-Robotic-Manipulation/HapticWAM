# Policy delivery clocks

This opt-in simulator change is prospective. Frozen v5/v6 sources, runs and
scores retain their original `native` timing; no new model or physics study is
implied by the CPU qualification below.

The default `native` mode preserves the prior behavior: delivery uses the
policy's reported `Plan.latency_s`, with an optional explicitly recorded latency
override. The new `rpc_wall` mode uses the wall duration of the complete
synchronous client `policy.replan(...)` call. For a remote client this includes
request serialization, transport, server execution, response deserialization
and client plan reconstruction. The timer runs after snapshot preparation and
the observation-saving callback, and stops before adapter copying, validation
and planner-trace bookkeeping. It uses `time.perf_counter`; a callable clock can
be injected for CPU tests. The default mode does not sample this new clock.

Enable it in a new campaign's `adapter_profile`:

```json
{"policy_delivery_clock": "rpc_wall"}
```

The campaign forwards `--policy-delivery-clock rpc_wall` to `run_waffles.py`.
Missing declarations mean `native`, including historical metadata. The option
is valid only for policy mode. It cannot be combined with `--policy-latency` or
the campaign's `delivery_latency_s`, because such an override would replace the
measured timing. Independently declared `inference_delay_add_s` remains an
additional post-response transport delay.

For request time `t`, reported native inference duration `L`, measured client
duration `R` and additional delay `D`, delivery is scheduled for `t + R + D`.
The original action grid `t + L + arange(H) / action_rate_hz`, `Plan.latency_s`,
and remote CPK token remain unchanged. Seed selection has already happened
inside the native policy with its original temporal conditioning. Delivery uses
the existing elapsed-head skip and rebase path; an entirely expired chunk is
rejected. Safety can cancel a pending delivery. This change does not replace
native temporal conditioning with a different estimate of inference duration.

Both modes still execute synchronously while simulator time is paused, then
advance physics to the scheduled delivery. This represents measured response
delay; it is not a reproduction of native asynchronous scheduling, all client
overheads outside `policy.replan`, or real-time GPU contention across concurrently
advancing physics and inference. It does not change the real deployment CLI,
controller, sensor inputs, force model or safety limits.

Audit fields are deliberately separate:

| Recorded field | Meaning |
| --- | --- |
| `policy_info.policy_delivery_clock` and `run.policy_delivery_clock` | Effective selected mode |
| Planner `latency_s` and `diagnostics.sim_native_inference_latency_s` | Native policy duration, retained for action-grid and CPK semantics |
| `diagnostics.sim_policy_replan_wall_time_s` | Measured complete client policy-call duration, present for `rpc_wall` |
| Planner `inference_wall_time_s` | Existing outer runner duration, including snapshot/callback and adapter overhead |
| `diagnostics.sim_policy_delivery_base_s` | Measured client duration before injected delay |
| `diagnostics.sim_inference_delay_add_s` | Separately configured post-response delay |
| `diagnostics.sim_effective_delivery_delay_s` | Client duration plus injected delay |
| `diagnostics.sim_rpc_minus_native_latency_s` | Difference between the two measured durations |
| `activated_at` | Actual discrete executor activation, when accepted |

For historical compatibility, `sim_effective_inference_latency_s` continues to
record the total scheduled delivery delay. Its name should not be interpreted
as a new native inference measurement. Nonfinite, decreasing or negative clock
durations, and client wall time below reported native duration beyond numerical
tolerance, raise `PolicyTimingError` before submission. Planner failures carry
`error_category=instrumentation_or_input_invalid`; the campaign audit excludes
these cases from valid scoring. The audit also checks the declared mode,
required durations, unchanged native grid, injected delay and causal activation.
This distinguishes an instrumentation failure from a valid task failure.

CPU tests cover the unchanged default and latency override, timer boundaries,
late-head skip/rebase with original CPK state, fully expired plans, safety
cancellation, invalid clocks, configuration forwarding and corrupted audit
metadata. Run the focused qualification without Isaac or hardware:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_sim_rpc_delivery.py tests/test_sim_policy_adapter.py \
  tests/test_sim_policy_campaign.py tests/test_sim_release_finish.py
```

Any future comparison using this option needs a new frozen profile and fresh
study provenance. The effect on pickup, carry or placement is unmeasured.
