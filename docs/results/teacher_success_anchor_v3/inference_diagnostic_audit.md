# First-plan and native timing audit

The four-call input control establishes a bounded causal result: substituting
only the saved first RGB image reproduces each corresponding archived first
**16×7 action array exactly**. The historical image and its repeat also return
identical actions. All eight other stored fields are identical across the three
inputs; every call resets seed 4242 and supplies no previous plan. All select K0.

Checkpoint/EMA, model configuration, bfloat16 dtype, normalizers, backbone,
text cache, hardware, effective inference settings and the six logged native
inference source hashes match the historical run. The diagnostic client/helper
hashes match the reviewed local files. All four outputs are finite. The owned
diagnostic server PID 2177196 is absent and its saved status is `stopped`; the
parent has since reused its port for separate experiments.

| First RGB input | Difference from historical: 10-step XYZ endpoint | Closure RMSE |
|---|---:|---:|
| Repeat 1 frame | 0.195246 mm | 0.000334488 |
| Repeat 2 frame | 0.156826 mm | 0.001615611 |
| Historical frame repeated | 0 mm | 0 |

These small first-proposal changes precede later controller/contact divergence.
The control identifies their RGB origin; it does **not** establish that RGB is
the sole cause of the later failed pickup. No closed-loop frozen-RGB experiment
is part of this audit. Full identities and independently recomputed archived
action equalities are in [input_first_plan_audit.json](input_first_plan_audit.json).

The original has 38 completed plans; the repeated runs have 21 and 19. Matched
K4 choices first differ at replan **10** for repeat 1 (original K1, repeat K2;
repeat request 9.012 s), and **11** for repeat 2 (original K3, repeat K2;
repeat request 9.860 s). Their first raw actions already differ at replan 0,
where K0 is common.

There is **no discrete requested CPK-offset difference** at any matched replan.
The loaded backbone defaults have temporal compression 4 and FPS 4, giving
`latent_dt=1 s`; each prior latency rounds to one latent step. The reconstructed
K-selection reference offset is also identical throughout repeat 1. Repeat 2
changes that offset from 8 to 12 only at replans 15–18, after its first selected
K difference. A requested CPK offset does not mean a package survived veto;
the audit separately records previous-plan CPK invalidation.

The arithmetic uses actual previous active-plan IDs from execution records,
not merely the preceding proposal. Source references, all rows, input hashes
and missing original tail counts are in
[native_selection_audit.json](native_selection_audit.json). These facts exclude
an initial discrete CPK-index change as the explanation; they do not exclude
continuous timing or later observation differences.

The adapter-latency diagnostic has now completed under parent scheduling. It
consumed 21 of the 38 recorded delays before a `wrist_extension` safety stop at
17.348 s; the final observation is 19.344 s. It acquired at 8.000 s, lifted at
12.668 s and carried at 15.336 s, without a drop, release or placement. The trace
is valid under the frozen scorer and remains excluded from model selection.

All 21 request timestamps, delays and action grids match the original exactly;
all 20 activated plans also match original activation times. First selected-K
divergence moves to replan 14 (K0 to K3), with no requested CPK-index difference.
The native K-selection reference offset nevertheless changes at replan 8
because fresh compute latency is still used inside native selection. The
adapter override does not constitute full native timing replay.

Initial non-RGB inputs remain bitwise equal; RGB RMS differs by 0.917832 uint8
levels. Both owned processes exited with code zero and are absent. Matching the
adapter schedule alone therefore did not restore historical placement; the
experiment leaves native selection timing and subsequent observations live.
Full evidence, limitations and a read-only CPU reproduction script are in the
[latency diagnostic audit](latency_diagnostic_audit.md) and
[compact JSON](latency_diagnostic_audit.json). The original preparation plan and
source checks remain preserved as
[latency_diagnostic_plan.json](latency_diagnostic_plan.json) and
[latency_source_audit.json](latency_source_audit.json).
