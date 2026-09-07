# Possible ftA1500 NFE1/K1 versus NFE1/K4 follow-up

**Timing-label correction, 2026-09-07:** the earlier approximately 0.2 s
outer-minus-native difference includes observation-saving callbacks and adapter
bookkeeping; it is not measured RPC transport overhead. The old numbers and
frozen v5 outcomes are preserved in the [corrected latency audit](latency_audit.md).
V7 now measures the client call separately; its first sample adds 5.886 ms over
native inference, while the outer runner adds another 186.602 ms. This one
sample does not establish typical latency.

**Historical planning notes, not a model recommendation.** These notes preceded
the separately declared four-case v7 development diagnostic, which uses full
client-call delivery and reused development seeds. That diagnostic does not
constitute fresh model-ranking confirmation. The original proposal below did
not itself change a setting or launch an experiment. Resolve the separate limiter/controller diagnosis first.
Then freeze one identical runtime profile for both arms of any comparison.
Do not mix K changes with a controller, geometry, force threshold or sensor-map
change, and do not use this note to alter v5 results.

| Setting | K1 candidate | K4 reference |
|---|---|---|
| Checkpoint | ftA teacher 001500, EMA | Same exact payload and EMA |
| Checkpoint SHA256 | `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e` | Same |
| NFE | 1 | 1 |
| K | 1 | 4 |
| Guidance / parity / persistent noise | 1 / enabled / enabled | Same |
| Task / max playback | `waffles` / 10 | Same |
| Timing | Actual saved native L, no override | Same convention |

K4 already executes one batch. K1 reduces batch activation storage and FLOPs,
but the wall-time gain is unknown until measured on the deployment GPU. It may
reduce response delay and observation age. No carry or placement improvement
follows from that prediction: NFE1/K4 already had zero cap/stale dwell in the
five audited ftA1500 cases, so stale holding is not established as the current
NFE1 carry limitation.

K1 also removes the four-candidate descent and temporal-consistency selection.
Persistent noise tensor shape changes, and action-strip noise consumes the
generator after a differently sized initial draw. The same episode seed is
therefore a reproducible paired label, not proof that K1 uses exactly K4's first
candidate noise. Preserve the native sampler; do not manufacture equal outputs
by copying a chosen K4 proposal or setting the latency artificially low.

Before any closed-loop comparison, a separately authorized bounded inference
check can use identical saved observation tensors at initialization, approach
and carry/release contexts. Reset native episode state identically for each
condition and distinguish cold/first-call anticipation from steady calls with
an actual preceding CPK. Measure both native L and full client RPC, plus selected
proposal deltas, gripper intent, K selection and CPK provenance. These are speed
and behavior diagnostics, not task successes. Do not replay one unrelated CPK
as though it were a causal predecessor.

If the question becomes model/recipe ranking, freeze **new**, disjoint sampling
seeds and a counterbalanced order before observing those outcomes. Check them
against the existing screen, confirmation, diagnostic and arm-extension seed
registries; seed numbers and sample size remain unset here. Reusing v5's known
outcome seeds gives a paired diagnostic only, not fresh confirmation. Keep the
exact fixed packet/reset, measured robot state, controller profile, tactile
baseline, safety limits and 60-second horizon common. Record carry, physical
release/support, FINISH, all stops and actual timing independently, preserving
every planned failure and invalid case. Do not choose additional seeds until a
preferred recipe wins.

For any new comparison, explicitly freeze the same delivery-clock convention
for both recipes and publish native inference, measured client call and outer
runner duration separately. The historical table above used native-L delivery;
the separate v7 diagnostic instead uses measured client-call delivery while
preserving native internal action/CPK clocks. Real remote deployment pays its
actual client/server delay, but the earlier outer-minus-native difference
cannot be treated as a measured 0.2 s transport component.
Do not extend the stale timeout or maximum playback to conceal slow inference.
This note proposes a bounded single-setting question, not a new broad model study.
