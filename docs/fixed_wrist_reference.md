# Experimental fixed episode wrist reference

This isolated change adds `safety.wrench_baseline_mode: episode_fixed` to the deployment `SafetyMonitor`. It prevents a slowly applied external load from entering that monitor's reference. The default remains `rolling_calm`. No hardware configuration, frozen simulator study, scored trial, teacher observation, model weight or data-collection `ArmGuard` is changed.

The correction is an **opt-in safety experiment, not a calibrated UR3 force model or a pickup improvement**. It has not been enabled on physical hardware. The corrected 56-trial teacher study used its frozen rolling reference and retains its original results.

## Why a separate mode

The existing rolling reference updates whenever force and torque **deviations** are below their limits. “Calm” does not mean unloaded or stationary. A slowly rising external load can therefore become part of the reference; freezing only after a threshold crossing cannot undo that drift. This was observed in the simulated `v5_6_nfe1_k4__start_1787396273__seed903101` trace: its reference tracked a sustained backing/mat contact, then exceeded the deviation limit after unloading.

The fixed mode captures the first finite six-component arm-wrist sample available to the new episode's `SafetyMonitor` and retains it for that monitor's lifetime. It requires a fresh sample in the existing arm timestamp tolerance; missing, nonfinite, malformed or excessively future-dated samples stop the episode with `wrench_invalid`. Small receive-thread timestamp races use the same bounded tolerance as arm freshness. A new episode creates a new monitor and reference. Recovery hysteresis, event auto-clear, latch resets and release FINISH do not re-tare it.

On the native path, capture occurs at the first `SafetyMonitor.check()` with a usable arm sample; the executor can wait for the first plan before that check. It is not guaranteed to use native stream row 0 or the simulator initialization anchor. The logged capture timestamp identifies the actual reference.

The operator must establish an unloaded start. Software does not infer that the first sample is unloaded. The captured value includes whatever pose-dependent bias, tare and current-estimator error were present. No zeros are invented and no reference is subtracted from model inputs.

Both modes retain the configured force/torque norms, strict `>` comparison, time-based debounce (`wrench_debounce_ticks / action_rate_hz`), recovery fraction, protective-stop priority, tactile/kinematic/workspace guards and existing stop/let-go semantics. Fixed mode changes only reference adaptation and rejects invalid wrist feedback. Safety still preempts FINISH; the achieved-pose hold cannot conceal a later force violation.

## Explicit configuration and provenance

For a separately reviewed deployment, select the mode in a **new** hardware YAML:

```yaml
safety:
  wrench_baseline_mode: episode_fixed
```

This is a fragment to merge into a complete configuration. The native deployment entry point also accepts the explicit override `--wrench-baseline-mode episode_fixed`. Omitting that option respects the loaded YAML, whose absent-field default is `rolling_calm`; `--wrench-baseline-mode rolling_calm` explicitly selects the old rule. These are configuration instructions, not authorization to run the physical launcher. Geometry/start checks, release-volume review and existing operator safety procedures remain required.

The override is validated before policy/hardware initialization. Native metadata records it in `deploy_overrides.wrench_baseline_mode` when supplied on the CLI and always in `deploy_overrides.safety_effective.wrench_baseline_mode`, alongside the unchanged tau/debounce/threshold settings. The simulator reads the same hardware configuration through its existing adapter; no new simulator default is enabled.

`wrist_guard` diagnostics record `mode`, the initial reference and its **sensor-clock** `capture_t_s`, current `sample_t_s`/`checked_at_s`, measured wrench, reference before/after the tick, six-component deviation, separate force/torque norms, over-limit start time and whether adaptation occurred. They are included in native planner rows, final stop state and the executor's `at_halt` state, and in each simulator execution command's diagnostics. Invalid samples preserve the earlier reference and report invalidity. Native planner snapshots are sparse; `at_halt` preserves the final guard decision. Values retain their input coordinate axes and nominal N / N·m units; this metadata does not certify physical calibration.

To keep existing checkpoint/config provenance stable, canonical YAML omits **only the new field when it has its old default value**. Both existing config hashes remain exact: `hardware.yaml = 6ab949e3fea83bd3`, `hardware.nuc.yaml = 7ecb586f6666f291`. Full `model_dump()` metadata exposes the effective mode; fixed mode is retained in canonical YAML and changes the hash. An explicit CLI override continues the repository's established base-hash-plus-`deploy_overrides` recording convention.

**Scope:** `phantom/data_collect/safeguard.py:ArmGuard` remains rolling and does not consume this new field. Setting it does not change data collection or teleoperation. The raw wrist window supplied to the teacher remains unchanged, including its normal model normalization. The checkpoint's tactile `wrench_baseline_rows` option is a separate preprocessing setting and is unaffected.

## Saved-input qualification

The [6273 replay](results/wrist_baseline_20260907/6273_replay_audit.json) uses 2,247 measured simulator wrist rows through the recorded stop. The [compressed input](results/wrist_baseline_20260907/6273_wrist_trace.npz) and [provenance](results/wrist_baseline_20260907/6273_wrist_trace_provenance.json) preserve source and extracted hashes. Its original execution-trace SHA-256 is `45149781082aab9bd27a0676a92ef0ecd97d0a3e2f527635d90690eff8da0884`.

| Wrist subguard replay | First debounced stop | Deviation at the recorded 17.972 s stop |
|---|---:|---:|
| Frozen default rolling | 17.972 s | 101.539 N |
| Opt-in fixed | 10.460 s | 0.0936 N |

All 2,247 default decisions, references before/after update, six-component deviations, norms and debounce states equal the independent frozen `16567f3` arithmetic exactly. The fixed reference never moves. Its earlier stop precedes the recorded pickup: replaying the saved remainder after that hypothetical stop is descriptive, not a prediction of the resulting policy or object trajectory.

The independent [ten-native-demo audit](results/wrist_baseline_20260907/native_demo_audit.md) covers 26,610 wrist samples and 212.82 seconds. Rolling mode has 0 debounced triggers; fixed mode has 1, at 13.952 s in episode `1787396028_003`. The later trigger remains unadjudicated and is not labeled a false positive or a confirmed collision. Both modes have 0 debounced triggers in the initial 0–1 s intervals supported by reviewed RGB, tactile area and robot-state evidence. Those are moving unloaded intervals, not stationary calibration, and instantaneous crossings still occur. First-native-sample timing differs from the simulator's common-stream anchor in some episodes.

Reproduce the local guard replay and focused CPU checks without drivers or inference:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python docs/results/wrist_baseline_20260907/replay_6273.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q \
  tests/test_wrist_reference.py tests/test_safety.py tests/test_config.py \
  tests/test_sim_release_finish.py tests/test_sim_placement_release.py \
  tests/test_sim_policy_adapter.py tests/test_control_safety.py
```

Validation: 114 focused CPU safety/configuration/release/fake-RTDE tests passed. A further 150 native deployment tests passed: 145 directly and 5 existing fully stubbed `main()` tests with the [test-only disk fixture](results/wrist_baseline_20260907/stub_disk_fixture.py), since the checkout filesystem has less than the unchanged 5 GB launcher minimum. Production disk checks were preserved. New files and the safety/adapter modules pass Ruff; a [comparison with the parent source](results/wrist_baseline_20260907/lint_delta.json) finds zero introduced lint findings and 22 inherited findings in other edited legacy files. See the [validation record](results/wrist_baseline_20260907/validation.json).

Before any physical enablement, independently adjudicate the later native-demo trigger with synchronized footage/tactile evidence, measure unloaded wrist estimates across the intended poses and accelerations, and check the reference-capture/tare procedure, frame conventions and load response against the real UR3. These steps must establish the consequences of pose-dependent bias at the unchanged thresholds. No threshold tuning, physical execution, changed study score, or claim of calibrated safety is part of this patch.
