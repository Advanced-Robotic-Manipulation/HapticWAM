# First scored teacher-v2 case integrity audit

The first completed screen trial passed independent CPU integrity/scoring checks: `fta1500_nfe1_k4__start_1787395928__seed903101`. All112 source/core/evidence/initial/baseline hashes matched; the actual checkpoint file, EMA declaration, effective NFE1/K4 recipe and parent/screen identities matched. No frozen source or raw trial file was changed.

The901-frame trace covers0–59.996 simulation seconds, with maximum physics-clock mismatch2.85 microseconds. Native and effective delivery latency averaged .786874seconds; inference wall time averaged .969029seconds. These are distinct clocks.

This was a **valid task failure**: no acquisition, lift, carry or placement, no controller stop, and no completion flag. Final packet-to-robot and packet-to-bin forces were both0N. The independent physical scorer and clean-placement selector agreed. This single-case audit establishes integrity, not a candidate ranking or winner. Selection remains withheld until the full32-case matched screen.

[Machine-readable audit](first_policy_case_audit.json) and [read-only reproduction helper](first_policy_case_audit.py). Run the helper by stdin from the frozen remote `source_teacher_v2` directory using the live PHANTOM Python environment; it reads source/weights/traces without constructing models or changing files.
