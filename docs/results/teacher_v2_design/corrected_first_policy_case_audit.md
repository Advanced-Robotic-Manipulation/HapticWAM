# Corrected study: first-case integrity audit

**Passed.** This is an integrity check of `fta1500_nfe1_k4__start_1787395928__seed903101`, not a screen ranking or a winner claim. The [machine-readable audit](corrected_first_policy_case_audit.json) and [read-only reproduction helper](corrected_first_policy_case_audit.py) belong to the corrected `teacher_robustness_v2_delivery` study; the original ten halted trials remain separate diagnostic evidence.

The amended screen/parent hashes, actual ftA1500 checkpoint, EMA/settings, effective scene, recorded start and tactile baseline match. All 116 frozen source/core/input files checked match their expected hashes. The evaluator and selector both accept this case as valid. Its physics clock differs from recorded simulation time by at most 2.85 μs over the 59.996-second trace.

Current-delivery veto feedback is recorded. The wrist input includes housing and both pad bodies against the eight declared environment bodies. Independent reconstruction of all 7,501 wrist samples produces exactly zero error for signed impulse conversion, contact-point moments, actor aggregation, recorded-bias addition and cached scene-frame values/timestamps. An initial sample is taken at t=0; the executor then samples at 125 Hz from t=0.004, with numerical clock error below 8e-15 seconds. This remains an idealized normal-contact wrench, with the omitted force terms and uncalibrated UR3 transfer explicitly recorded in the audit.

This one rollout has no acquisition, sustained lift, carry or placement; it reaches the horizon without a controller stop or IK rejection. It does have weak unilateral packet contact, with peak pad/packet normal force 0.694 N, so “no acquisition” does not mean “no contact.” The reach error before first additional closure is 56.2 mm. Mean native/delivery latency is 0.785 seconds, mean activation delay 0.790 seconds, and mean inference wall time 0.969 seconds. No threshold or configuration changed after inspecting this result.

Reproduce on compute3, reading frozen source and existing outputs only:

```bash
cd /home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_v2_delivery
PYTHONPATH=. /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python - < /path/to/corrected_first_policy_case_audit.py
```
