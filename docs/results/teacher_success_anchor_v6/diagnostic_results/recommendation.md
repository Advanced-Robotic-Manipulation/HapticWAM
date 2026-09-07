# V6 development decision

**Do not promote the current v6 profile as a full-task solution or a hardware-ready setting.** Retain the optional command limiter as a development component: it enforced its declared bounds in both trials, but its 25-reject rule stopped both while carrying, before release. The next useful isolated change is to distinguish a verified constrained hold from an IK fault; extending the timeout alone is insufficient for native servo parity. No setting was changed in this audit.

| Result | 904301 | 904302 |
|---|---:|---:|
| Acquired / sustained lift / carry | yes / yes / yes | yes / yes / yes |
| Release / support-verified full task | no / no | no / no |
| Maximum packet rise | 340.3 mm | 297.5 mm |
| First limiter intervention | 12.420 s | 13.220 s |
| Final consecutive holds | 25 | 25 |
| Ordinary controller stop | `servo_limiter_stall`, 13.076 s | `servo_limiter_stall`, 14.188 s |
| Minimum commanded elbow magnitude | 0.400079 rad | 0.400195 rad |
| Maximum commanded joint step speed | 0.996181 rad/s | 0.999416 rad/s |
| Maximum radius from measured joints | 0.461548 m | 0.461545 m |
| Maximum measured joint speed before stop | 0.898883 rad/s | 0.927720 rad/s |

Both command sequences stayed inside the declared 0.40 rad elbow / 1.0 rad/s step limits, with no branch/envelope audit violation. Their maximum radius calculated from measured joints remained about **6.45 mm inside the unchanged 0.468 m measured stop**. There were no measured safety events at the limiter stops. Both retained bilateral packet contact through the observed tail, with no scored drop, no bin load and no FINISH. Their sampled packet normal-force peaks were 10.658 N and 8.587 N; these two trajectories do not establish a general contact-force bound.

The original minimal-profile baselines both ended on measured wrist-extension stops, at 15.988 s and 15.868 s. Baseline 904301 acquired/lifted/carried; baseline 904302 acquired but did not sustain a lift. All four cases have zero full placements. These are two previously observed development seeds with different inference timing and resulting observations; the stage differences are descriptive, not a controlled estimate that the limiter improved pickup or that a longer hold would permit placement.

## The stops were verified constraint holds

An additional CPU replay of the first and last held ticks found **all seven IK queries converged on every inspected tick**. The existing anchors satisfied the limits; every candidate was rejected for elbow magnitude below 0.40 rad. Thus these inspected stalls were caused by the configured constraint and bounded search, rather than missing/invalid IK. This does not prove an alternative feasible route exists.

The timing concern from [the prospective review](../hold_stop_review.md) occurred in both actual traces:

- **904301:** holds ran 12.876–13.068 s; the next tick stopped at 13.076 s. A pending plan captured at 12.780 s, before hold onset, was due at 13.551826 s and never activated. No observation taken after hold onset produced a delivered plan.
- **904302:** holds ran 13.988–14.180 s; the next tick stopped at 14.188 s. A new observation was captured at **13.996 s, after the first hold**. Its plan was due at **14.810136 s**, but the stop discarded it. The policy had no opportunity to act on that post-hold response.

The held TCPs were outside the declared release region: at the stops their Y coordinates were −0.217154/−0.232609 m versus minimum −0.068538 m, and Z was 0.409497/0.392329 m versus maximum 0.274 m. The geometry and current command directions therefore did not reach the release gate in these prefixes. No conclusion about all raw opening proposals or global reachability follows.

## Provenance and command feedback passed

The prepared, hash-pinned audit independently rescored both v6 trials and both baselines using their respective frozen designs. **All four were valid, and all four original outcome/event records agreed exactly.** The v6 design was `55a5ad7852b1f08b6cce6a6d7f136454cac69504aed11ad1b441c36945aac9b6`. Runtime metadata matched the declared limiter algorithm, 0.40 rad / 1.0 rad/s settings, 0.468 m measured stop, 25-reject count and `tracking_guarantee=false`; the baseline declared no limiter. A separate post-run check verified **155 pinned source, asset, hardware, episode and profile inputs with zero mismatches**. All four effective scene configurations were byte-identical, and their configured initial arm joints agreed.

Every one of the **3,357 accepted v6 feedback poses** equaled nominal FK of its submitted `target_q` exactly. Every held joint target remained bitwise equal to the preceding target, and every held row correctly reported `accepted_tcp=null`. At each stop, gripper command and latch retained the final held values, 0.6237486005 and 0.6112303138. There was no fictitious accepted motion or automatic opening caused by the limiter stop. The full source/input verification is separate from this command-bound result; neither certifies physical calibration or hardware tracking.

Evidence is preserved in [the independent numerical audit](diagnostic_audit.json), [feedback and pending-plan timing](feedback_hold_timing.json), and [post-run provenance verification](post_run_provenance.json). Four independent strict scorer records are retained under `minimal_baseline/trials/` and `limiter_v6/trials/`. Raw observations, video and original source remain on compute3; only compact audit products were mirrored locally.

A future bounded-hold diagnostic should keep all measured guards active, deliberately stream a verified stationary native setpoint, retain meaningful hold/fault telemetry, and use a predeclared elapsed-time cap that can accommodate a fresh request. It must fail closed on actual invalid IK/branch/anchor faults, and must not erase held commands, reset indefinitely on plan arrivals or relabel a controller timeout as infrastructure. Test that isolated change separately; preserve these two v6 failures unchanged.
