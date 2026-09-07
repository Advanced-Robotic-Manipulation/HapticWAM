# Why the four screening acquisitions did not become sustained lifts

These are the **four acquisitions identified by the authoritative selector after all 32 corrected screening trials completed**. They are not counts from an interim monitor. All four are valid under the unchanged scorer; none reaches its 30 mm / 0.5 s lift gate. Acquisition means bilateral packet contact above 0.1 N for 0.15 s, not that the packet is suspended or securely retained. No candidate ranking, score changes, source edits, or new simulations are included.

[Full numeric audit](audit.json) · [acquisition traces](acquisition_traces.png) / [PDF](acquisition_traces.pdf) · [rigid-packet height/tilt check](peak_height_geometry.json)

| Completed case | Acquisition (s) | Largest packet-center rise | Active-gel peak L/R | Latch | Diagnosis |
|---|---:|---:|---:|---|---|
| ftA3000 NFE1/K4, start5963, seed903101 | 12.136 | 17.52 mm | 0.52 / 1.08 N | Never | Brief low grasp; loses retention before the main lift |
| v5_6 NFE1/K4, start6028, seed903101 | 12.668 | 3.71 mm | 1.94 / 1.82 N | Never | Contact lost before the upward motion; later bin collision |
| v5_6 NFE1/K4, start6273, seed903101 | 16.200 | 4.55 mm | 8.92 / 7.50 N | From 17.500 s | Prolonged pressing, late latch, low loss, then baseline-relative stop |
| ftA1500 NFE5/K1, start6273, seed903102 | 11.336 | 1.39 mm | 2.14 / 7.66 N | Never | Further descent while finger backing loads mat/bench |

The gel peaks are separate per-pad maxima over the executed interval, not necessarily simultaneous. In the last case the right peak occurs before the sustained acquisition. Scene contact/pose samples are about 15 Hz, gel samples 8 Hz, and wrist/execution samples 125 Hz. The JSON preserves these clocks and uses causal gel samples; differing instantaneous force values across clocks are not treated as a conservation error.

## Two lost grasps before the main lift

**ftA3000/start5963.** Bilateral contact occurs in three short bouts: 12.136–12.400, 13.000–13.336, and 13.536–14.200 s. There is a small real geometric clearance: at the maximum center height, the rigid packet's lowest corner has risen 15.97 mm, so this is not merely a packet tipping on the table. It is still below the fixed sustained-lift threshold. All packet contact is gone by the first commanded 30 mm TCP rise at 14.428 s; measured TCP subsequently rises 256.5 mm while the packet returns to the table. The full 60-second run has no safety stop.

The first contact is near the modeled toe: the left contact-support polygon lies at local Z 19.5–27.5 mm, outside the active gel's ±18 mm Z extent. Mapper v2 includes the populated zero-impulse vertices and correctly returns zero overlap for that polygon. At the largest combined gel load, left/right overlap fractions are about 0.227/0.481; gel loads remain below the native 2.5 N bilateral latch threshold. Executed closure reduces between the first contact bouts; the later loss also occurs during an upward move without an armed latch. This identifies weak retention and limited gel overlap, without proving a physically calibrated contact-pressure distribution.

**v5_6/start6028.** Bilateral contact lasts approximately 12.668–13.336 s. The active patches overlap the gel substantially, but measured proxy loads never exceed 1.94/1.82 N, and the latch remains unarmed. Closure rises then reduces around contact loss. The first commanded/measured 30 mm TCP rises occur at 14.100/14.196 s, after bilateral contact has ended. A later empty transfer ends at 21.660 s with housing/bin-front contact of 129.59 N. That later collision did not cause the earlier grasp loss. The maximum packet lowest-corner rise is only 1.87 mm.

## Two pressing failures at the lower start

**v5_6/start6273.** Before acquisition the left finger has already pressed the packet with a peak modeled normal load of 115.43 N. During acquisition, the right backing loads the mat, peaking at 114.03 N. The accepted TCP target initially continues downward while measured TCP stays near 70 mm. Measured closure exceeds the opening target substantially during this contact-constrained motion; the resulting bilateral-contact event is not evidence of a clean free-space pinch.

The latch does work once both active gel loads become large: it arms at 17.500 s and rises from closure 0.5212 to 0.5314. At 17.628 s the left/right support polygons overlap 92.3%/76.8% of their respective projected patches, producing gel loads 8.92/7.50 N. Thus this failure is not uniformly missing gel coverage. After acquisition, accepted and measured TCP rise at most 18.79/8.35 mm before termination; packet center rises 4.55 mm and its lowest corner only 1.61 mm. Bilateral contact ends after 17.736 s. Latching alone did not turn this constrained pose into a retained lift.

**ftA1500 NFE5/start6273.** After acquisition, the accepted TCP continues downward by 7.39 mm and measured TCP by 2.76 mm. No 30 mm upward command occurs. The left gel never crosses 2.5 N, so the bilateral latch does not arm, even though some right-side gel loads are higher. At the 11.932 s wrench stop, right-finger/mat and right-finger/bench loads are 166.49/20.59 N, while left/right packet loads are only 4.06/3.40 N at the same wrist sample. The apparent 1.39 mm center rise is a brief tilt before the sustained acquisition; the lowest rigid corner never appreciably clears its initial plane. This is a descent/contact failure rather than a failed main lift command.

## Baseline drift must be separated from direct-load stops

[Actual temporal wrist/contact/baseline plot](wrist_baseline_drift.png) / [PDF](wrist_baseline_drift.pdf) · [125 Hz scalar evidence](wrist_baseline_trace.json)

The v5_6/start6273 stop occurs **after unloading**, not at the largest current contact. The native rule adapts its rolling baseline whenever deviation is below 60 N / 15 N·m; it does not independently require stationarity or absence of contact. Gradually increasing contact from about 7 s onward is therefore incorporated into the baseline.

At 17.972 s the rolling baseline Fz is 114.9616 N, but actual proxy wrist Fz is 13.5511 N, close to its measured initial bias of 13.5283 N. The summed positive current contact load is only 0.09362 N, so this is not merely cancellation of large opposing forces. Force deviation from the shifted baseline is 101.5394 N, with the final over-limit interval starting at 17.668 s. The unchanged 0.3-second debounce then produces the logged stop exactly. A short earlier excursion at 16.012 s does not survive its debounce window.

This is distinct from the direct loaded housing/bin stop in start6028 and the direct mat/bench stop in the NFE5 case. It does not make the earlier 100 N pressing harmless. The [independent native-rule reconstruction](extract_wrist_baseline.py) changes neither safety nor feedback. Any alternative reference policy is a separate diagnostic proposal and must not be counted as a result of these frozen trials.

## Limits and reproduction

Force magnitudes, contact compliance, active-pad extent and uniform patch pressure remain uncalibrated; no real UR3 wrench equivalence is established. The rigid-corner calculation is an additional geometry diagnostic and does not change the frozen center-height scorer. Commanded closure here means the executed target after the deployment stack; raw policy proposals and veto attribution are separate questions.

The audit reads only completed corrected-screen files. Its input and selector hashes are in [audit.json](audit.json). On compute3, keep [audit_first_start.py](../first_start_diagnostic/audit_first_start.py) on `PYTHONPATH` and run [audit_acquisitions.py](audit_acquisitions.py) with the existing Python environment under `nice -n 19`, `OMP_NUM_THREADS=1`, and `PYTHONDONTWRITEBYTECODE=1`. [Plot scripts](plot_acquisitions.py) and [baseline plot script](plot_wrist_baseline.py) use only the saved JSON. Temporary CPU analysis scripts reside at `/tmp/teacher_delivery_cpu_audit`; no process or waiter remains active after extraction.
