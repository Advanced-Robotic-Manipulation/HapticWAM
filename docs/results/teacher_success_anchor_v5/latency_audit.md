# Native latency and held-command diagnosis

This read-only audit uses ftA1500 screen seeds **904401–904405**, which were
complete when inspected. It compares timing at two declared recipes; it neither
replaces the final screen analysis nor attributes placement outcomes to latency.
The [numeric audit](latency_audit.json) records source, server and all ten input
identities, definitions and per-case values. No inference, source change or
hardware action was performed for this audit.

| Declared recipe | Calls | Median saved native latency | Median observed total RPC | Median RPC minus native |
|---|---:|---:|---:|---:|
| ftA1500 NFE1/K4 | 99 | 0.7816 s | 0.9786 s | 0.1930 s |
| ftA1500 NFE5/K4 | 79 | 3.8360 s | 4.0478 s | 0.2123 s |

K4 is **already batched**, not four sequential replans. Native
`phantom/model/rf.py:544–560` repeats the same conditioning into batch four;
the sequential loop is `range(nfe)` at line 598. NFE5 performs five denoising
steps. Guidance 1 does not evaluate a second unconditional branch. Server
metadata confirms EMA, BF16, K4 and the declared settings; the server already
merges LoRA and the sampler computes invariant text conditioning once per call.

The first call and calls following an invalidated CPK can also pay the native
two-pass anticipation cost. When a previous CPK exists, the sampler already
skips that work. Across these cases, later-call native medians following logged
CPK invalidation were 1.1643 s versus 0.7776 s without invalidation for NFE1,
and 4.1375 s versus 3.7783 s for NFE5. This is consistent with that explicit
code branch; the audit does not inspect GPU execution or hidden CPK tensors.

For seed 904401, NFE5 records **18.944 s of explicit stale-plan holds** and a
separate **22.072 s at the playback cap**, excluding stale, FINISH and stopped
rows. NFE1 records zero in both categories. Its first activation is 1.500 s,
versus 4.404 s for NFE5. Durations integrate actual forward executor intervals
without extrapolating the final row.

The executor plays at most ten 10-Hz steps, then holds that capped command.
Its unchanged stale rule begins after the full 16-step horizon plus the
1-second timeout: 2.6 s after activation. A roughly 3.8-second replan therefore
leaves substantial cap dwell and then stale holding. Saved action-grid starts
equal request time plus native latency exactly; delivery differs only by the
125-Hz tick, at most 8 ms in these cases. This explains the measured waiting
intervals under the declared rules, not which teacher would place better.

There are two different clocks in the records. Native `policy.py:239–253`
measures a host `perf_counter` duration around observation batching and sampler
execution, ending **before** CPU materialization and selection. It contains no
explicit CUDA synchronization at that endpoint. The saved
`inference_wall_time_s` brackets the complete client replan call, including
transport and remaining work. It is not a pure GPU timer either. **V5 simulation
uses native L; remote physical deployment pays the full RPC duration** and must
handle the resulting later delivery. The approximately 0.2-second difference
cannot be erased when judging deployable response time.

CPK indices below are **reconstructed from native code and saved timings**, not
logged CPK telemetry. The native latent period is 4/4 = 1 s. It computes
`round(previous_plan.latency_s / latent_dt)` and clamps to the last of three
contact frames if a previous package exists. Seed 904401 therefore gives index
1 for NFE1, versus raw index 4, clipped to 2, for NFE5. A logged invalidation
means the package is absent, so that calculated index selects no actual CPK.
Separately, K-selection's previous-action reference computes a time offset and
clamps it into 0–15: reconstructed offsets are 8–12 for NFE1 and 37–43, all
clipped to 15, for NFE5. Prior accepted activation was verified before each
current request. This is an additional horizon limitation of the slow recipe.

Keep stale thresholds, playback bounds and measured native timing unchanged.
K1 may reduce batch compute, but it changes candidate selection and noise draws;
it is not a missing batching optimization. The focused
[NFE1/K1 versus NFE1/K4 notes](nfe1_k1_followup_notes.md) describe a possible
future settings comparison without modifying or launching the current study.
