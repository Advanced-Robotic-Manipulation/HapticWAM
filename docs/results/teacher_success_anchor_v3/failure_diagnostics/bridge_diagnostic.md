# Combined-profile bridge diagnosis

Both completed bridge trials have valid strict scores. The failure stages differ; neither reaches the placement-release window. The matched gel-only cases placed at 23.736/22.136 s, but these live comparisons change several profile features plus image/timing histories.

| Seed | Strict physical stages | First executor stop | Measured TCP XYZ, m | Object/contact at stop |
|---|---|---|---|---|
| 904301 | acquisition, lift, carry; no release/drop | wrist_extension, 14.828 s | [−.46915, .01537, .38259] | Retained, packet 5.75/5.88 N; gel 4.94/5.44 N |
| 904302 | acquisition only; maximum rise 3.94 mm | tactile_depth, 17.628 s | [−.41298, −.10339, .16290] | Packet on mat, zero packet contact; left pad presses bin front |

## Retained transport: seed 904301

The native latch arms at 9.252 s and holds closure command .63947 while measured closure is .55531 at stop. No original-eligible played opening or raw post-latch opening ≤.45 exists. The TCP remains 108.6 mm above the unchanged release ceiling. FINISH has not reached its release/terminal phase.

The first stop is the measured elbow-extension guard: radius .468056763 m exceeds .468 m. Requested-versus-measured TCP error is 5.89 mm and .87°. Over the preceding .496 s the measured TCP moves [+2.2,+61.8,−18.3] mm: it advances toward the bin while descending. At essentially the same transport Y in the gel-only success (Y mismatch .017 mm), the bridge is 14.1 mm farther in −X and 25.8 mm higher; the successful trajectory radius is .465275 m. This supports a coupled position/orientation reach-boundary diagnosis, not a failed squeeze or blocked opening.

All 17 delivered plans have action arrays identical to their raw proposals. The live veto reports only `none`/`close_allowed`; delivery-feedback and veto changes therefore do not directly rewrite this trajectory. The changed wrist proxy and observation history remain candidate influences, not established causes. The new wrist view records a right-pad/mat load up to 53.71 N during approach, but at the final guard only the two packet contacts remain; the distal net wrench at that point is small because opposing pad normals largely cancel. The stop is not a wrist-force stop.

## Failed grasp then environment contact: seed 904302

Bilateral packet contact last appears at scene t10.136 s; the packet rises only 3.94 mm and never satisfies sustained lift. Maximum simultaneous weaker gel load is 1.310 N, below the unchanged 2.5 N native latch criterion. The latch never arms.

At 11.892 s, the live veto makes an explicit `recovery_tactile` rewrite using current feedback with zero loads, TCP Z .23745 m and measured closure .57620. This changes Z increments and grip in raw plan 13 (original closures .5527–.5810). At 13.060 s plan 14 receives `close_masked`, changing its original .5209–.5480 closures. These are two rewrites among 20 delivered plans. They occur after retention has failed, so cannot explain the initial failed grasp; they can affect the later empty recovery trajectory.

Left-pad contact with `/World/Bin/Front` starts at 16.860 s. At the 17.628 s stop, its physical normal magnitude is 69.703 N and the active-gel selection admits 26.750 N; the right pad and packet have no contact. The mapper correctly uses point fallback for this undeclared support body, rather than extending the packet-only convex-patch assumption to the bin. Contact points lie near the front rim at Z .189–.193 m. The maximum pad/front normal load before stop is 74.154 N. Thus the tactile-depth stop has identified environmental contact, not imagined packet retention. Force/compliance magnitudes remain uncalibrated simulation values.

## Native wrist guard and missing command screening

Frozen `source_teacher_v2_delivery/phantom/deploy/safety.py:151` reads **measured** joint index 2 and computes

```text
radius = sqrt(a2² + a3² + 2*a2*a3*cos(q[2]) + d4²)
a2=.24365 m, a3=.21325 m, d4=.11235 m
stop when radius > .468 m
```

The same computation in resume checks is recovery hysteresis, not predictive command screening. `clamp_target` at line 410 implements optional TCP-origin radial scaling and Cartesian workspace limits; it does not evaluate predicted elbow/wrist radius. The campaign hardware explicitly sets `reach_clamp_m: null`. Even when enabled, a TCP-origin sphere would not be equivalent to the orientation-dependent wrist-center constraint.

The simulation runner at line 1320 accepts nominal IK subject to a joint-branch delta bound, then clips per-joint increments to the velocity limit. Neither the IK solver nor that acceptance path rejects/projects a target based on predicted wrist radius. Native `SafetyMonitor` likewise has no such predictive projection/hold.

The saved bridge trace demonstrates this gap without a new simulation: at **14.764 s**, an accepted target has radius **.468004468 m** while measured radius is still **.467497525 m**. The measured guard fires **64 ms later**, at 14.828 s. A future opt-in command screen could preserve the measured threshold while screening candidate joint targets and holding/projecting before entry. Its dynamic margin, bounded runtime, branch continuity and safe behavior under tracking/inertia would require independent tests and a separate profile; this audit neither implements it nor claims that a simple clamp prevents stops.

## Attribution limits and evidence

The combined profile changes distal-wrist normal-contact sensing, live veto, current-delivery feedback and FINISH relative to gel-only, while keeping the same gel v2 mapper. Scene configuration bytes and physical initialization fields match. Initialization JSON differs only by an added zero-contact support-audit record. All eight non-RGB first-observation arrays are bitwise identical; RGB RMS differs by 1.551/0.993 uint8 levels. First activation matches at 1.460 s for seed 904301, while seed 904302 changes from 1.436 to 1.508 s. Later timing and observations remain unmatched. No single changed input can be assigned causal responsibility for both failures from these data.

[Strict case/stage audit](bridge_cases.json), [signal/contact/source audit](bridge_signals.json), and [CPU helper](audit_bridge_signals.py) preserve input and frozen source hashes. All inspection was CPU-only and read-only. No model ranking, safety changes, hardware actions or frozen-source mutations were performed.
