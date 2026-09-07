# Sustained contact-load audit

The 122.146 N packet-force peak in `fta1500_nfe1_k4__successful_anchor__seed904401` is **a sustained one-sided contact problem, not a brief isolated impact or a 122 N bilateral pinch**. It is a secondary diagnosis of the frozen run. Its physical outcome score and thresholds remain unchanged.

| Recorded case | Largest sampled pad→packet normal sum | Bilateral minimum-force peak | Samples above 15 N |
|---|---:|---:|---|
| V5 failed carry, 904401 | 122.146 N | 5.933 N | 39 consecutive samples, 6.200–8.736 s |
| Original placement, 4242 | 9.692 N | 6.962 N | none |
| Gel-v2 placement, 904301 | 5.774 N | 4.926 N | none |

These are approximately 15 Hz recorded samples; they do not bound unsampled peaks or prove uninterrupted physics-rate contact. In the V5 case, consecutive samples exceed 60 N from **6.936–8.736 s** (1.800 s span) and 100 N from 8.468–8.736 s (0.268 s span). The opposite pad has zero packet contact during this buildup. Both successful references use the same recorded physical geometry/material/drive configuration and avoid this heavy sampled load. That implicates the executed contact path as well as model compliance; it does not establish a universal force ceiling or a causal comparison across unmatched inference timing.

At the 8.736 s peak, the left pad→packet vector is `[-8.052, 24.586, 95.978] N`, with norm **99.403 N**, versus the sum of positive normal magnitudes **122.146 N**. Different contact normals partly cancel within the pad; there is no opposite-pad cancellation at this sample. Whole-pad net force equals packet-filtered net force. In pad coordinates the net is `[32.573, -79.493, -50.009] N`: **93.915 N is transverse to the prismatic X axis**. The 15 N setting is the maximum of the finger's linear drive, not a cap on the rigid body's total contact reaction. The along-axis reaction also reaches 32.573 N; the available arrays do not contain joint reaction/drive effort or individual finger acceleration needed to close that force balance. The arm's driven motion, inertia and constraints remain relevant.

The nearest detailed contact sample, **8.756 s**, explicitly identifies only `/World/Waffle` as the loaded counterpart: left normal sum **124.230 N**, right zero. All of it is excluded from the active gel: 27.117 N fails the inner-face test and 97.113 N has an oblique normal. The dominant contact is at pad-local `[-5.973, 9.155, 19.432] mm`, with normal `[0.081, -0.887, -0.454]` and signed separation **−11.527 mm**. This supports a peripheral/oblique contact interpretation. The API log identifies the pad rigid body and packet, but lacks a collider/shape ID, so it cannot conclusively name the pad/backing submesh. The separation is geometric overlap in the compliant contact model, not a measured gel indentation. It is large compared with the modeled 12 mm pad thickness; pad compliance (12,000 N/m, 30 N·s/m), rigid packet geometry and 4 ms integration remain uncalibrated assumptions.

The packet load then falls away, but environmental load continues: at 9.252 s the right pad's explicit filters report **48.338 N on `/World/Mat/Base`** and **2.994 N on `/World/Bench/Slab`**; at 10.380 s those loads are **48.856 N** and **2.858 N**. These also fail the gel face/normal selection. This explains why wrist load remains elevated after packet-only force drops. The detailed rows and input hashes are preserved in [the numeric audit](contact_force_audit.json).

## Why the wrist/depth guards did not stop it

The recorded wrist proxy did contain the load. It uses the summed whole-pad force plus the measured initial wrist bias. Replaying the exact frozen rolling-wrench logic on the logged 125 Hz inputs gives a maximum force residual of **53.718 N**, below 60 N, and maximum torque residual **2.338 N·m**, below 15 N·m. There is **no over-threshold interval**, so the 0.3 s debounce never starts.

At 8.756 s, wrist force has changed **100.834 N** from its initial value. The rolling baseline has already moved **47.177 N**, leaving the 53.718 N residual. Baseline drift later reaches **54.564 N** at 10.348 s. Here “calm” means below the current deviation thresholds; it does not require an unloaded or stationary arm. The gradual load is therefore absorbed. The active-gel force is zero at the peak, so the calibrated-proxy interface adds no gel compression beyond its unloaded baseline; routing these peripheral loads into gel would invent a different sensor. The actual first stop is the separate measured wrist-extension guard at **18.748 s**.

The successful references' maximum rolling-wrist residuals are only **4.794 N** (original) and **1.310 N** (gel-v2); neither has a comparable sampled contact-load episode. The original recording lacks signed separations, so its penetration cannot be compared quantitatively. The gel-v2 success's largest recorded loaded overlap is **0.725 mm**, with individual loaded-contact forces below 6 N.

## Implications

This is material evidence against treating the simulation's force magnitudes and adaptive safety behavior as validated hardware behavior. It does not invalidate or retrospectively relabel the frozen object-state scores. Earlier [identical-command timestep diagnostics](../teacher_v2_design/command_force_convergence_audit.md) also retained high oblique packet loads at smaller timesteps; reducing timestep alone is not a calibrated remedy.

The wrist radius calculated from measured joints at this force peak is **0.305669 m**, far inside the prospective 0.461711 m command envelope. The future elbow limiter is aimed at later reach/extension behavior and cannot be presumed to prevent this low-height contact path or rolling-baseline absorption. After the running comparison, qualify geometry, pad/packet compliance and environmental contact against real measurements, retain load/deformation diagnostics, and evaluate any wrist-reference change as a separately declared setting. Do not relax force thresholds, mask environmental forces or turn non-gel contacts into gel signal to obtain success.
