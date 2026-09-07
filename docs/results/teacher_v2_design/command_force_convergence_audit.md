# Identical-command contact-force sensitivity

**The high packet load persists at smaller timesteps. The complete 4 ms command replay reproduces the original run exactly.** These are mechanics diagnostics, without inference or new safety decisions; they are not additional policy trials.

The source is the previously completed teacher case `teacher__placement_xm10_yp10mm__seed4242`. Its 2,280 drive-submitted commands span 0.004–18.236 s. Every replay reads the exact same command JSONL, SHA `f02dd3373473e0b99cb7406f7cf68e22c3609d1f3a841a867e86da08ce6b28cf`, through a causal zero-order hold. No future command interpolation or object-pose replay is used.

The [numeric audit](command_force_convergence_audit.json) verifies that only physics timestep differs between the effective scene configurations; solver iterations, drive settings, collision geometry, material coefficients and initial packet/arm state are retained. All three runs cover the commands and two-second tail, ending one timestep before the requested 20.236 s horizon as documented by the simulation loop.

| Physics step | Last saved time | Peak packet/robot normal load | Sampled normal impulse | Peak gel force |
|---|---:|---:|---:|---:|
| 4 ms | 20.232 s | 116.714 N | 467.754 N·s | 0 N |
| 2 ms | 20.234 s | 136.178 N | 523.700 N·s | 0 N |
| 1 ms | 20.235 s | 138.819 N | 547.885 N·s | 0 N |

For the 4 ms replay, all 305 saved samples match the original run's timestamps. **Every one of the 21 saved state/contact arrays is bitwise identical**, including q, qd, actual TCP, gripper closure/OBJ, packet pose, all pad forces and positive packet support loads. Each replay's sampled `target_q` also equals the independently reconstructed causal command target with zero error. Finger target submissions are present in the hashed source command file; the output trace separately records measured closure and estimated OBJ, not finger targets. The exact 4 ms measured gripper match is included in the state comparison.

The 2 ms and 1 ms results differ by 1.902% in peak load and 4.414% in sampled impulse, with a maximum packet translation difference of 0.709 mm after interpolation onto the 2 ms run's saved frame times. These meet the proposed diagnostic limits of 10%, 5% and 2 mm. **They are sampled diagnostics:** the approximately 15 Hz contact records can miss short peaks and do not replace physics-rate impulse integration or formal convergence testing.

The largest loads remain packet contacts on backing/toe geometry or near-inner positions with normals tangent to the active gel. V2 therefore reports zero gel force throughout all three replays while preserving the physical load. Recorded drive commands press against a supported rigid packet; the high load is not explained away by V1's sparse manifold interpolation bug. Deep signed contact penetrations remain a model limitation and require measurements of actual geometry, compliance and packet deformation.

No safety threshold was relaxed, no contact was disabled, and no force was clipped. The controller study retains the previous 4 ms physics so its changed controller/mapper behavior is identifiable; its force magnitudes and transfer to hardware remain explicitly uncalibrated and timestep-sensitive. A lower timestep alone is not a calibrated physical fix.

The earlier `teacher_commands_dt{4,2,1}ms` diagnostics were unintentionally capped at the supplied recording's 14.666 s duration. They remain preserved and cover the force peak, but were not accepted as full-horizon evidence. This report uses only the separately named `teacher_commands_full_dt{4,2,1}ms` runs, made after the command-replay horizon handling was corrected.

To reproduce, run [audit_command_force_replay.py](audit_command_force_replay.py) on compute3 with:

```sh
--root /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_v2_preflights \
--reference /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_pick_place_v1/campaign/rollouts/teacher__placement_xm10_yp10mm__seed4242
```

The helper reads numeric artifacts only and passes Ruff checks. Full input hashes are in the JSON audit.
