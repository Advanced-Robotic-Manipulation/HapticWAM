# Initial v3 failure-stage audit

Eight completed diagnostic rollouts were checked against the original successful v1 placement. All eight scores are valid; none completed placement. These are observed trajectories with varying rendered pixels and measured native inference timing, not a causal or model-ranking experiment.

Reach is the last 15 Hz pad-midpoint distance to the packet OBB before measured closure crosses .25. A/L/C are the unchanged acquisition/sustained-lift/carry gates. TCP and first guard are from the 125 Hz executor.

| Case | Reach error mm | A/L/C | First guard at s | TCP XYZ at stop, m | Object state |
|---|---:|---|---|---|---|
| repeat1 / 4242 | 2.2 | Y/Y/Y | wrist_extension 17.044 | -0.451, -0.024, 0.346 | Held through stop |
| repeat1 / 4243 | 20.4 | Y/N/N | hitbox_exit_top 15.156 | -0.450, -0.107, 0.433 | Empty; max rise 8 mm |
| repeat2 / 4242 | 4.7 | Y/Y/N | wrist_extension 16.892 | -0.461, -0.043, 0.385 | Dropped 3.692 s before stop |
| repeat2 / 4243 | 2.6 | Y/Y/Y | hitbox_exit 18.532 | -0.429, 0.174, 0.286 | Held before safety-open; drop +.068 s |
| baseline / 904301 | 0.0 | Y/Y/Y | wrist_extension 14.780 | -0.469, -0.006, 0.380 | Held through stop |
| baseline / 904302 | 6.0 | Y/Y/Y | wrist_extension 15.692 | -0.474, 0.030, 0.393 | Held through stop |
| FINISH / 904301 | 0.0 | Y/Y/Y | wrist_extension 16.060 | -0.461, -0.077, 0.400 | Held through stop |
| FINISH / 904302 | 7.1 | Y/N/N | wrist_extension 15.700 | -0.489, -0.007, 0.405 | Transient rise 46 mm; no sustained lift |

The reference completed acquisition at 8.000 s, lift at 12.668 s, carry at 15.336 s, release at 24.400 s and support-verified placement at 25.200 s. Its wrist-extension stop at 31.380 s occurred after placement.

## What differs before the stops

The six wrist-extension cases advance in +Y toward the bin while their last-half-second TCP Z change is negative (−6 to −26 mm). Their tracking errors at the guard are small; the guard radius crosses the unchanged .468 m elbow-extension limit. This is not evidence that the arm was prevented from following a far-away requested trajectory. Neither “wrong direction” nor continued upward motion explains all six. The successful reference also lifts high (carry-stage TCP Z .376 m and packet peak rise .342 m).

At acquisition and carry, orientation differences from the corresponding reference stage are mostly a few degrees. Later, small coupled position/orientation deviations matter near the reach boundary. Matching the successful transport leg by Y progress gives:

| Held wrist-stop case | ΔX / ΔZ vs reference, mm | Orientation difference | Reference radius / case radius, mm |
|---|---:|---:|
| repeat1 / 4242 | +3.0 / -43.3 | 10.15° | 465.59 / 468.02 |
| baseline / 904301 | -16.5 / -2.1 | 1.70° | 465.63 / 468.01 |
| baseline / 904302 | -27.4 / +31.6 | 7.36° | 461.47 / 468.03 |
| FINISH / 904301 | -6.1 / -3.0 | 3.44° | 463.81 / 468.05 |

The matched-Y error is below .32 mm for these four. This comparison diagnoses the observed path, not a causal decomposition into translation versus orientation. Repeat1/4242 is **43 mm lower** than the successful path at the same Y yet reaches the extension guard; baseline904301 is 16.5 mm farther in −X at almost the same height. Baseline904302 combines −27.4 mm X with +31.6 mm Z. Orientation remains another coupled difference. A blanket reduction of lift height is therefore not justified by this sample.

## Opening, latch and grip loss

Five new cases armed the native load latch. Across all five there are **zero original-eligible played openings**, zero raw post-latch predicted closure rows ≤.45 (including unused proposal tails), and no active release window before the stop. Eligible played closures remain ≥.509. The held wrist-stop cases are 72–126 mm above the release ceiling at the guard; FINISH904301 is also 8.8 mm short of the release Y boundary. Opening passthrough was never exercised in these failures. The FINISH-only controller never reached its post-placement function, so it cannot by itself explain their earlier trajectory differences.

The original reference first entered the release window at 19.036 s and first played an eligible opening at 23.372 s. It then committed and placed. These times are descriptive, not deadlines applied to the other trials.

Repeat2/4242 never armed the latch: the maximum simultaneous weaker gel load was 1.896 N, below the native 2.5 N bilateral criterion. Physical bilateral contact persisted until scene t13.000 s (~2.16/2.22 N), then the packet dropped at 13.200 s while the arm continued. FINISH904302 likewise never armed (maximum weaker gel load 2.350 N); its 46 mm transient rise fails the fixed .5 s lift requirement. Repeat1/4243 acquired briefly but had no simultaneous positive active-gel load and rose only 8 mm. These data expose grip/sensor hypotheses; they do not prove that replacing the mapper alone would rescue these trajectories.

Repeat2/4243 is different: at 18.468 s it still has 3.72/3.84 N bilateral packet load. At the 18.532 s workspace stop, the command changes to the safety-open value .232; the first detected drop is 18.600 s. Calling this a spontaneous pre-stop retention failure would be misleading. The TCP has overshot the release Y ceiling by 32.2 mm and remains 11.8 mm above its Z ceiling. The nearest reference Y is 59.7 mm away, so the matched-Y comparison is not a close match for this case.

## Controlled next diagnostic

Complete the frozen component/model study before choosing another setting. The most informative next single dimension is an explicit fixed delivery-latency control on the selected profile, with scene, initial state, renderer seed, sampling seed, policy settings, controllers and safety unchanged. This addresses the known timing confound; it is not evidence that slower or faster delivery improves task performance. The existing scalar latency override should be recorded explicitly, and source/campaign wiring verified before use. Do not tune a height cap or raise the wrist-extension threshold from this small sample: the failures include both higher and lower paths than the successful reference.

The separately completed [gel-only comparison](gel_mapper_comparison.md) supplies a more specific mapper mechanism and two physical placements. A subsequent gel-plus-FINISH validation would change only the already implemented post-placement behavior relative to gel-only; it should be reported as a new profile and cannot be used to reinterpret these eight pre-release failures.

## Reproduction and limits

[Numeric audit](eight_cases.json) contains stage poses, frozen outcomes, closure eligibility counts, requested/measured stop poses, pre-stop versus nearest scene contact samples, gel peaks and SHA-256 of every source input. [CPU helper](audit_failures.py) reads only completed groups and prints JSON. Run it with the compute3 project venv; no simulator or model is invoked. Event/contact timing is limited by 15 Hz scene and 8 Hz tactile sampling. Forces, geometry and tactile pressure remain uncalibrated simulation estimates. No source, score, threshold or trial was changed.
