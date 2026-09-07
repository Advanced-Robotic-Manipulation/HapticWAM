# Prepared v6 diagnostic audit

[audit_diagnostic.py](audit_diagnostic.py) is a CPU-only analysis helper for the two planned limiter rollouts and their existing minimal-profile baselines. It has **not been run on v6 trials**, and no watcher is running.

The default input groups on compute3 are:

- `sim/waffles/runs/teacher_success_anchor_v6/diagnostic`
- `sim/waffles/runs/teacher_success_anchor_v3/minimal_profile_diagnostic`

Both groups must have controller progress records showing completed, exit-zero cases named `teacher__fixed_anchor__seed904301` and `teacher__fixed_anchor__seed904302`. Otherwise the helper exits immediately. It does not wait or modify a recording. Existing output directories are refused. Analysis output must be outside raw case folders and the frozen driver.

After root confirms completion, run with the compute3 Python environment and no CUDA visibility:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python audit_diagnostic.py \
  --out /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v6/review_cpu/limiter_diagnostic
```

The helper sets CPU priority to nice 19. It pins the v6 external analyzer (`d47b4fa18fb093cbfd2aa751be908e53209a861aba9a1a49d4d71974aeaf4abc`) and strict physical scorer (`b8460fc530acd6f11ea166f3fbff0ddc2ee74f4c73d956752abab0435975969f`). Each trial is independently scored under its own frozen campaign snapshot, so the baseline remains limiter-disabled. Original outcome/event results are compared when the authoritative score already exists. No threshold is changed.

Outputs are `diagnostic_audit.json`, `diagnostic_audit.md` and four separately stored scorer records. They retain the seven raw score-input hashes, initialization hashes, full physical metrics and explicit validity failures. The summary reports:

- Limiter step/slide/hold counts, violation and rejection reasons, first intervention, maximum consecutive rejects and bounded IK-call count.
- Commanded elbow margin, wrist radius calculated from submitted `target_q`, and maximum consecutive commanded joint speed. The first command lacks a preceding submitted command and is excluded from the speed statistic. The switch to a measured-position safety hold is also excluded; unchanged targets during normal limiter holds contribute zero speed.
- Actual PhysX measured q/qd, with wrist radius **calculated from measured joints**, and requested-to-measured TCP position error. Pre-stop maxima and scene-sampled maxima including the observation tail are separate.
- Acquisition, sustained lift, carry, release, support-verified full task and drops from the frozen object-state scorer. Execution FINISH is reported separately and never treated as physical task success.
- Actual measured-safety stops, limiter stalls, other controller stops and full-horizon behavior as separate categories. Configured, settled and first-execution elbow joints outside the intended principal branch produce an explicit audit error.

A small local synthetic integration check passed: it distinguished a 0.5 rad/s commanded step from a 0.7 rad/s measured velocity, excluded a large safety-hold target jump, classified `servo_limiter_stall` as an executed controller stop and rejected a startup elbow shifted by a full revolution. Ruff passes. This validates the analysis plumbing, not the future physics outcome.

The comparison is descriptive: matching sampling seeds does not match rendered inputs, inference latency or later closed-loop observations. A bounded command is not a measured tracking or stopping-distance guarantee. This helper does not identify contact actors, certify sensor calibration or rank models.
