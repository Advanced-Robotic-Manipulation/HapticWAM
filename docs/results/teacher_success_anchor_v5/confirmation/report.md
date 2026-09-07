# Teacher confirmation comparison

Status: **complete_valid_matched_stage**.

Expected 24; missing 0; invalid 0.

| Candidate | Physical place | Final support | Clean FINISH | Lift | Acquire | Pre-place safety | Later safety | Drops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fta1500_nfe1_k4 | 1 | 1 | 1 | 7 | 12 | 11 | 0 | 1 |
| fta3000_nfe1_k4 | 0 | 1 | 0 | 4 | 8 | 12 | 0 | 0 |

Conclusion: conditional_ranked_leader_without_validated_winner; clear simulator winner: None.

[All gates and rows](selection.json) · [CSV](trials.csv). Completion may come from consistent execution diagnostics when the legacy run header omits it. Controller flags never supply physical object success.

- Development-informed profile amendment; only prospective model seeds enter selection and reserved confirmation.
- One fixed physical simulator state, not unseen arm-start or hardware success probability.
- Degenerate intervals do not establish equivalence; the exact paired small-sample guard remains required.
- Physical placement is primary and survives later stops. FINISH is secondary and never supplies object-state success.
- Historical request-time veto and pad-only wrist proxy are intentional minimal-profile limitations; this is not identical to current native deployment.
- Native inference latency and shared-compute/render variation remain part of the measured closed-loop behavior.
