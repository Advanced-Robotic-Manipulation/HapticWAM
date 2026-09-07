# Housing/bin blockage identified without changing the replay

**The gripper housing strikes the top of the bin's front wall.** This supplies the major load during the second reference-teacher trial's physical tracking divergence. The optional observer records it without changing any of the original trajectory or contact arrays.

The separate `first_seed903102_full_robot_contacts` diagnostic replays the exact saved joint and finger targets. All **21 shared trace arrays, 182 frames, and timestamps through 12.016 s are bitwise identical** to the completed original seed903102 case. The scene configuration is identical. This is command replay with no policy inference and no new safety decisions; its contact records are diagnostic evidence, not replacement policy inputs.

The [numeric audit](observer_replay_audit.json) verifies all 13 exact robot actors and eight explicit environment filters. All 1,061 populated contacts retain signed impulse, world point/normal and signed separation, including 952 zero-impulse geometric entries. The impulse-to-force and pair/actor/global force-budget errors are exactly zero. The largest per-actor contact count remains below capacity. Initial total robot/environment load is zero. The [CPU auditor](audit_observer_replay.py) includes source/input hashes and every scene-sampled pair-force time series.

## Contact sequence

- **5.136 s:** first sampled housing/bin-front load above 0.1 N.
- **5.200 s:** housing/bin-front peak **175.196 N**. The loaded point is `[-.295917, -.080538, .194000]` m, on the bin's top front edge, with world normal `[0, 0, 1]`. Its signed separation is −1.237 µm.
- **5.300 s:** the original execution trace first exceeds 20 mm accepted-to-measured TCP error. The error subsequently grows beyond 200 mm while commands continue downward.
- **9.936 s:** later left-pad/bin-front peak **4.492 N**, much smaller than the housing load.
- **10.020 s:** the original joint-speed safety stop occurs, followed by its two-second observation tail.

The housing contact precedes the sharp path-tracking divergence. It explains why accepted Cartesian commands can continue downward while the physical arm is displaced sideways and upward. The original pad-only wrist proxy omits this housing load; its wrist-force guard therefore cannot observe this modeled tool-body collision through that proxy. The existing fingertip gel mapper also correctly does not paint a housing contact into the gel. Adding this load to a future sensor model would be a separate validated change, not an observation-only amendment to these results.

The controller audit separately finds no raw-to-delivered action rewrite in these first two runs; see [controller evidence](../../teacher_v2_design/control_first_cases.md). This contact diagnosis therefore should not be described as the terminal veto refusing an otherwise successful reach.

## Limits

Exact actor attribution is not physical calibration. Housing collision-box dimensions, bin pose, contact stiffness and force magnitudes remain estimates. Scene-rate sampling can miss faster peaks; each impulse belongs to the current 4 ms physics step, not the full interval between rendered frames. Robot/environment contacts are now observed, but robot self-contact is not.

The fixed robot base also overlaps the raised table in the current mounting approximation. The observer retains these deeply overlapping geometric constraints, all with zero impulse; they do not account for the moving housing's positive rim load. This remains a separate geometry/measurement issue.

No threshold, collision pair, material, actuator setting or policy input was changed. The observer remains optional and disabled in the primary study. This successful runtime audit confirms its body labeling, force units and lack of measured physics effect for this diagnostic replay.
