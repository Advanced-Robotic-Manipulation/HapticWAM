# Tactile prediction error (TPE)

*Added 2026-09-11. The metric that makes HapticWAM a world-action model in the
evaluation, not only in the architecture.*

Every `PhantomPolicy.replan` generates a **contact package**: the model's
imagined tactile future on the latent grid, `Tc` steps of `latent_dt`
(`temporal_comp / fps` = 1.0 s at the deployed 4 fps / temporal_comp 4).
Until now only its readouts (gate, `p_evt`, sigma) reached
`planner_trace.json`; the package itself was the next replan's ACC input and
was then dropped. It is now kept, denormalized, and scored offline against
what the pads actually measured at the predicted times.

## What is recorded

`phantom/deploy/cpk_trace.py`, wired through `PlannerLoop(cpk_log=...)` by
`DeploymentRuntime`. Zero configuration: every deploy episode whose policy
returns a package gets

- **`planner_cpk.npz`** — arrays stacked over replans, recorder units:

  | key | shape | unit / meaning |
  |---|---|---|
  | `t_host`, `latency_s`, `trace_index` | (N,) | host time of the snapshot (same clock as the trace `t`), replan latency, index of the matching trace row |
  | `d_fz` | (N, Tc, F, cph, cpw) | Δ distributed-force-z per latent step, field units (`dist_force_unit_to_N` in the file converts to N when calibrated) |
  | `d_disp` | (N, Tc, F, cph, cpw, 3) | Δ displacement field |
  | `mask` | (N, Tc, F, cph, cpw) | predicted contact mask in [0, 1] |
  | `cop` | (N, Tc, F, 2) | centre of pressure in [-1, 1], NaN = undefined |
  | `slip` | (N, Tc, F) | slip score |
  | `event` | (N, Tc, E) | event probabilities over `{none, onset, hold, slip, release}` |
  | `wrench` | (N, Tc, F, 6) | pad wrench, baseline-subtracted like the training target |
  | `wrist` | (N, Tc, 6) | wrist F/T |
  | `latent_dt`, `sensors`, `denormalized`, `version` | scalars | grid step, pad names, whether norm stats were available |

- **`cpk_pred`** in each trace row: a JSON digest (per step, per pad
  `mask_frac`, `dfz_mean`, `dfz_absmax`, `slip`, `cop`, `event` argmax,
  `p_event`) so a trace alone shows what the model expected; `cpk_index` is
  the row of the npz.

Time convention (WindowSampler's target grid): predicted step *s* (0-based)
describes the interval `(t0 + s·dt, t0 + (s+1)·dt]` with `t0` the snapshot
host time; `d_*` are deltas across the interval, `mask`/`cop`/`slip`/`event`
the state at its end.

Cost: ~0.1 MB per replan at the nuc geometry (cpk 36×48, Tc 3, two pads),
i.e. a few MB per episode, compressed. Recording never ends an episode: a
logging failure is logged and skipped.

## How it is scored

`phantom/eval/tactile_prediction.py` rebuilds the **observed** package from
`tactile_<pad>_fields_ds` on the same grid, with the same construction as
`WindowSampler.sample` (nearest frame, `derive_timestep`, bilinear resize to
`cpk_shape`, depth-threshold mask, any-pad event rule), minus normalization.
Replans whose horizon leaves the recording are not scored (counted in
`tpe_n_replans_total` vs `tpe_n_replans_scored`), never padded.

Per replan and step:

| score | model | persistence baseline |
|---|---|---|
| `dfz_se` | MSE of predicted vs observed Δfz over pads × cells | Δfz = 0 |
| `mask_iou` | IoU of thresholded predicted mask vs observed | mask at t0 |
| `mask_frac_err` | \|predicted − observed contact fraction\| | fraction at t0 |
| `cop_err` | CoP L2 where both defined | CoP at t0 |
| `slip_err` | \|Δ slip\| | slip at t0 |
| `event_acc` | argmax event == derived label | `hold` if in contact at t0 else `none` |
| `wrench_se` | pad wrench MSE (when the wrench stream exists) | wrench at t0 |

Episode scalars (`tpe_*`, pooled over scored steps): the RMSE / mean of each
score for model and persistence, **skill = 1 − err_model / err_persistence**
(`tpe_dfz_skill`, `tpe_mask_iou_skill`, `tpe_event_acc_skill`), the
contact-conditioned variant `tpe_dfz_skill_contact` (steps where a pad is
actually in contact — the regime where tactile foresight matters), and the
horizon-resolved `tpe_dfz_rmse_s{k}` / `tpe_mask_iou_s{k}` /
`tpe_event_acc_s{k}` for k steps ahead.

Skill is the paper number: 0 means the model predicts the pads no better
than assuming they stay as they are; 1 is perfect; negative is worse than
persistence. A vision-only ablation should sit near 0 during contact; a
tactile-conditioned model should not.

## Where it shows up

- `phantom.eval.metrics.trial_metrics` — every `tpe_*` scalar is appended
  when `planner_cpk.npz` exists ({} otherwise), so `run_eval --episode-metrics`
  averages them per system and `report.md` gains a **Tactile prediction
  error** table (pooled per system, bootstrap CI over episodes) plus
  `tactile_prediction.json` with the horizon curves.
- `tools/tactile_prediction_eval.py` — standalone over episode dirs / roots:
  `episodes.csv`, `summary.{md,json}` per group (ledger `system`, `meta.policy`
  or a tag prefix), `horizon.json`, `--pair A B` for a matched-episode
  comparison (mean difference, bootstrap CI, exact sign test), `--png` for
  the horizon figure.

```bash
python tools/tactile_prediction_eval.py data/episodes/deploy/20260912 \
    --hardware configs/hardware.nuc.yaml --ledger runs/eval/ledger.csv \
    --pair teacher vision_only --png --out-dir out/tpe
```

## Caveats

- Episodes recorded before this change have no `planner_cpk.npz`; they are
  reported as skipped, not scored as zero.
- A policy served without norm stats (e.g. a remote adapter that returns a
  package but no `norm`) stores model-space values; the file says so
  (`denormalized = 0`) and scoring refuses it rather than mixing units.
- `d_fz` skill is dominated by the onset and the force ramp; away from
  contact both model and persistence predict zero change and the step
  contributes nothing to the difference — report `tpe_dfz_skill_contact`
  alongside the pooled value.
- The observed package uses the recorded `fields_ds` stream, not the
  keyframes; make sure the stream ran at the full sensor rate on the box.
