The current evaluation helpers compare host-clock planner times directly with master-clock sensor times, count bilateral onset more than once, and use an event-matching path inconsistent with its time/horizon semantics. A probe with unchanged physical events but a shifted clock origin changes a true 0.5-second lead to 1234.5 seconds and F1 from 1 to 0. The binary bootstrap returns [0,0] for zero successes in sixteen attempts.

The archived E13 close-height correction also uses different finite-close subsets for v4 and v5. The actual shared 194-row comparison is +4.82 versus +7.21 mm; report missing predicted closes alongside conditional height error. Offline terminal-window closure is not robot grasp success.

Related: #3's metric work. This issue gates claims made with these metrics; it need not block an engineering pilot using a separate explicit human outcome table.

Acceptance criteria:

- Convert clocks once using the recorded calibration and define one canonical event timeline. Changing clock origin without changing physical timing leaves scores invariant.
- Define the event unit and matching horizon explicitly; a bilateral physical onset cannot become two independent successes by accident.
- Empty/no-event/missing-stream cases are explicit, not silently scored as success or perfect precision.
- Use an appropriate interval for binary task outcomes with nonzero uncertainty at all-zero/all-one samples; respect paired and session structure when estimating comparisons.
- Match episode/window/seed keys before paired comparisons. Report shared-row counts and outcome-dependent missingness, including predicted-close frequency.
- Emit metric version, source identities, inclusion/exclusion counts and settings with results.
- Regression fixtures cover clock shifts, bilateral onset, unmatched/duplicate events, no predicted close, and zero/all successes. Recompute archived E13 rows to reproduce the review's explicitly defined matched statistics.

Sources: [metrics](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/phantom/eval/metrics.py#L75), [aggregation](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/phantom/eval/aggregate.py#L32). The review includes a standalone E13 recomputation and input-file digests; those JSON digests are not model-weight digests.
