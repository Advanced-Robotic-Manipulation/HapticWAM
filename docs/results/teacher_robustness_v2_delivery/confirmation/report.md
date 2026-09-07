# Teacher confirmation selection

Status: **complete_valid_matched_stage**.

Valid matched coverage: 24/24; missing 0, invalid 0.

Clean placement requires strict physical task success, final bin containment, unloaded robot/pads, positive bin support, controller completion and no actual stop. Controller completion alone earns nothing.

| Candidate | Clean | Strict place | Lift | Acquire | Safety stops | Drops |
|---|---:|---:|---:|---:|---:|---:|
| fta3000_nfe1_k4 | 0 | 0 | 1 | 3 | 9 | 0 |
| v5_6_nfe1_k4 | 0 | 0 | 1 | 3 | 12 | 0 |

Conclusion: inconclusive; pickup_candidate_only.

Selected screen IDs: []; clear simulator winner: None.

See [full selection and gates](selection.json) and [per-trial CSV](trials.csv). Native inference, effective delivery and activation latencies are distinct. Logged safety events include nonterminal clamps; terminal causes are separate.

- Operational simulator selection gates, not hardware success probabilities.
- Discovery ranking is selection-biased; only reserved-start confirmation can declare a clear simulator winner.
- Bootstrap resamples measured starts with both matched seeds intact; six start clusters give limited precision.
- A zero or degenerate interval does not establish equivalence; no selection from representative videos.
- Contact geometry, wrist/tactile proxies, support telemetry and camera estimates remain conditional on the frozen simulator.
- NFE1/K4 versus NFE5/K1 changes two recipe factors; it is not an isolated NFE effect.
