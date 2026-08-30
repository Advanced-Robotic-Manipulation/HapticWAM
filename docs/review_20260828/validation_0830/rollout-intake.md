# Lens: ROLLOUT → TRAINING INTAKE (PHANTOM @ d9c40f2, 2026-08-30)

Scope: generate a mock deploy episode set (labelled `s`, one `f`, one unlabeled, one
contaminated) and push it through `tools/rederive_rollout_actions.py`,
`phantom/eval/grasp_label.py`, `tools/intake_recovery.py manifest()`,
`phantom/train/common.py manifest_split` + `WindowDataset`/`build_index`, and a tiny
`train_teacher` step. Then check the DAgger driver path.

Everything below was **executed**. Scratch fixtures:
`/private/tmp/claude-501/-Users-sannikov/a5af36c2-8450-43d1-9a34-fa0cdb820706/scratchpad/validate/scratch/rollout-intake/`.
Nothing under `~/GitHub/phantom` was modified (two gitignored dirs were created by the
smoke runs: `runs/teacher/val_rollout/`, `runs/hid/dag_r1_r1/`; `git status --porcelain`
is empty).

---

## 0. Fixture

`make_fixture.py` builds 6 episodes under `<S>/ds/tasks/waffles/` from
`SyntheticEpisodeGenerator` (small-hw values from `tests/phantom_test_utils.make_small_hw`,
`rate_scale 0.25`), then shapes them:

| episode | policy | success | status | tags | z profile |
|---|---|---|---|---|---|
| `ep_rollout_s` | teacher | True | finalized | deploy | descend → close → 160 mm lift |
| `ep_rollout_f` | teacher | False | finalized | deploy | descend → close → no lift |
| `ep_rollout_unlab` | teacher | None | aborted | deploy, unlabeled | no lift |
| `ep_rollout_contam` | teacher | None | aborted | deploy, contaminated | lift |
| `ep_demo_a`, `ep_demo_b` | teleop | True | finalized | — | lift |

The four rollouts get an **executor-style `actions` stream**: 7.3 Hz (governor-warped)
cadence, 2× the true Δ-pose, +2 mm z bias — i.e. the raw pre-clamp proposal
`phantom/deploy/executor.py:232-241` actually records.

Two extra probes were added later: `ep_rollout_contam_final` (contaminated but
`status=finalized`) and `ep_rollout_unjudged` (policy=teacher, success=None,
finalized) — the two cases the runtime guard would miss.

---

## 1. `tools/rederive_rollout_actions.py` — WORKS, one semantic gap

```
$ .venv/bin/python tools/rederive_rollout_actions.py <S>/ds/tasks --hardware <S>/hw_small.yaml
INFO: ep_rollout_s: actions re-derived (139 rows on the 10.0 Hz grid); proposal kept as actions_plan.zarr
INFO: 6 episodes: not-a-rollout=2, rederived=4
$ (re-run)  INFO: 6 episodes: not-a-rollout=2, skipped=4          # idempotent
```

Numeric check (`check_rederive.py`): integrating the re-derived Δ-xyz over the whole
episode reproduces the measured TCP displacement exactly.

```
actions rows 139  dt median 0.1000        # 10 Hz, uniform
plan rows    100  dt median 0.1370        # governor-warped, kept as actions_plan
integrated re-derived dxyz (mm) [ 12.75 -88.53 -60.  ]
measured   dxyz          (mm) [ 12.75 -88.53 -60.  ]
residual (mm) [0. 0. 0.]
```

**Gap: the gripper channel changes meaning.** `rederive_actions()` writes
`out[k-1, 6] = grips[k]` — `gripper.zarr[:,0]`, i.e. `GripperState.position`, the
**measured** aperture (`phantom/drivers/base.py:69`). A teleop demo records
`grip_cmd = pilot.last_sent` — the **commanded** aperture
(`phantom/data_collect/session.py:735-745`, `data_collect/gripper.py:275`), and the
deploy executor's own `prev_chunk` history stores the commanded `a[6]`
(`executor.py:241`). On the fixture:

```
re-derived actions[:,6] unique: [0.05 0.75]     # measured position
demo       actions[:,6] unique: [0.   1.  ]     # commanded aperture
```

On the rig the difference is exactly the grasp outcome: a command of 1.0 that closes on a
waffle pack reads back as whatever the object width is; a close on air reads ~1.0. So
re-derived rollout actions (a) leak the grasp outcome into the ACTION target and
(b) disagree with demos on the single channel the terminal commit is about.
`tests/test_small_fixes_0829.py:320` asserts only `act[:, :6]` — the gripper channel is
untested.

**Nothing enforces that this tool ran.** `grep -rn "actions_rederived|actions_plan"
--include=*.py phantom/ tools/` outside the tool itself returns only the schema comment.
Demonstrated: restoring the proposal as `actions` and dropping the tag, the episode is
still admitted by both gates —

```
NON-rederived rollout indexed: 8 windows -> action_weight 1.0
manifest: appended 1 rows ...
```

---

## 2. `phantom/eval/grasp_label.py` — runs; one rig-session-critical interaction

`tools/label_grasps.py <S>/ds/tasks/waffles --hw <S>/hw_small.yaml --confusion` produced a
clean dual-label table and the rule-vs-operator contingency; `--include-unfinalized` picks
up the aborted ones. Z_MAX lookup, `never_closed`, `no_tactile_stream`,
`missing_stream:` and the `obj==2` stall FLAG (never in `grasp_ok`) all behave as
documented. Aborted (unlabeled/contaminated) episodes are excluded by default.

**But the rule reads the FIRST close, and `--terminal-veto` reopens.**
`grasp_label.py:263` takes `close_index(pos)` (first close) and `:279-280` sets
`t_release` at the first drop of 0.10 below the plateau. `PlannerLoop._apply_veto`'s
recovery rule commands `open_aperture` (0.0) and re-descends, up to `max_retries=3`
(`planner.py:394-403`). So an episode whose *second* close grasped and lifted is scored on
the *first*, vetoed close.

Constructed `ep_veto_retry_success`: close at t=6, veto reopen at t=7, re-descend, real
close at t=8.5, then a 160 mm lift.

```
$ .venv/bin/python tools/label_grasps.py .../ep_veto_retry_success --hw ... --min-c-hold 0.0
episode                  task     ok  op   t_cl  z_cl  hold c_hold  lift  reasons
ep_veto_retry_success    waffles  ..  s     6.0    80   0.5   0.00     0  hold 0.5s < 2.0s; lift 0mm < 50mm
```

`grasp_ok=False` on a successful grasp. G2 ("Arm B ≥ 3/16 tactile-confirmed grasps") is
measured on exactly the arm that runs the veto — the gate is biased to fail.

Also: `label_grasps.py` has **no write-back** (args are
`--json-out/--confusion/--contact-rate/--min-*/--z-max/--include-unfinalized/--limit/--quiet`).
Nothing turns `grasp_ok` into `meta.success` or a tag, so D6's "label" step and D8's
"auto-labelled rollouts" are operator-verdict-only in practice.

---

## 3. `tools/intake_recovery.py manifest()` — the P9 gate HOLDS

```
manifest: skipping  ep_rollout_contam (status='aborted')
manifest: skipping  ep_rollout_unlab  (status='aborted')
manifest: REFUSING  ep_rollout_contam_final — contaminated
manifest: REFUSING  ep_rollout_unjudged   — unjudged rollout (policy='teacher', success=None)
manifest: appended 4 rows ... (ep_demo_a, ep_demo_b, ep_rollout_f, ep_rollout_s)
manifest: REFUSED 2 episodes (unlabeled / contaminated / unjudged policy rollouts)
```

`place()` with symlinks also works end to end (verified separately: a symlinked
`ep_teacher_waffles_..._000` under `tasks/waffles/` is manifested, `manifest_split`
resolves it, `build_index` indexes it).

**`--val-min-eps` is a trap for a mixed intake.** `holdout_sessions` sorts session names
*reverse-alphabetically* and has no notion of time:

```python
rows = 60 demos in sessions 20260822_1XXXXX_waffles + 40 in deploy_20260902 + 40 in deploy_20260904
holdout_sessions(rows, 10) -> {('waffles', 'deploy_20260904')}
```

The newest deploy-rollout session (the whole on-policy pool D8 exists to consume) is held
out as `val`. Related: `session` is `ep.resolve().parent.name`, so an episode placed as a
real directory (not a symlink) directly under `tasks/<task>/` reports session
`"<task>"` — all such episodes collapse into one pseudo-session (observed: holdout
`['waffles/waffles']`).

---

## 4. `manifest_split` + `build_index` — both gates HOLD

```
manifest_split train -> ['ep_demo_a', 'ep_demo_b', 'ep_rollout_f', 'ep_rollout_s']
windows from manifest: 32   episodes: same 4
--- build_index with NO manifest (the DAgger --extra-data path) ---
WARNING skipping ep_rollout_contam_final: not training-ready (status='finalized' tags=['deploy','contaminated'] ...)
WARNING skipping ep_rollout_unjudged:     not training-ready (status='finalized' tags=['deploy'] policy='teacher' success=None)
episodes indexed: ['ep_demo_a', 'ep_demo_b', 'ep_rollout_f', 'ep_rollout_s']
n_config_drift: 0
action_weight rollout_s: 1.0   rollout_f: 0.0   demo_a: 1.0
```

Unlabeled and contaminated never enter the index at either layer. Good.

CONFIG DRIFT: `run_deploy` stamps the BASE hash (`runtime.py:142,190`), and I confirmed
the safety overrides change the hash but not the shape fields:

```
base hash    : 72a4dd674113afd0
override hash: fd0477563fe0bc7f  changed: True     # apply_hitbox + apply_z_floor + apply_tcp_speed_limit
shape_relevant diffs under safety overrides: {}
```

`configs/hardware.nuc.yaml` has not changed since the demos were collected
(`git log -- configs/hardware.nuc.yaml` → last touch `54b7943`), so rollouts recorded with
`--hardware configs/hardware.nuc.yaml` will pass `train_teacher`'s CONFIG DRIFT check.

---

## 5. Tiny `train_teacher` on the mixed set — WORKS

```
$ .venv/bin/python -m phantom.train.train_teacher --data <S>/ds/tasks --hardware <S>/hw_small.yaml \
      --tiny --device cpu --max-steps 2 --batch-size 1 --run-name val_rollout --acc-two-pass
INFO phantom.train.common: manifest split 'train': 4 episodes
INFO train_teacher: dataset: 32 windows ... (split=train)
INFO train_teacher: episodes indexed: 4/4
step 2/2  action_v_mse=1.9344  contact_nll=8.7869  ... total=16.3670
checkpoint saved: runs/teacher/val_rollout/teacher_000002.pt
```

`--student` and `--mask-wrist` are present in the argparse block (P10A landed).
Full suite: `571 passed` in 327 s.

---

## 6. Per-episode weights (plan D8) — NOT IMPLEMENTED

The plan says: "per-episode weights (`EpisodeMeta.weight`, honoured at `windows.py:403`;
w=1 demos, 0 failures, 1 recoveries, `clip(1/(p_task+0.2),1,3)` successful rollouts, ×2
window multiplier in the commit band, ×3 oversampling of the small rollout pool)".

Reality:

```
meta.json on disk has weight: 3.0
EpisodeMeta fields: ['clock_calibration','config_hash','dagger_round','damage',
 'deploy_overrides','driver_modes','hardware_shapes','notes','operator','policy',
 'status','success','tags','task','text']
hasattr(EpisodeMeta,'weight'): False
sampled action_weight: 1.0
```

* `EpisodeMeta` has **no `weight` field**, and `from_dict` is a tolerant loader that
  silently drops unknown keys — so a `weight` written by intake vanishes without a warning.
* `windows.py:423` is the only writer of `action_weight` and it is binary:
  `0.0 if is_failure_demo(meta) else 1.0`.
* `train_teacher` never exposes `windows_per_episode` (hardcoded 8 in `WindowDataset`), so
  there is no ×3 oversampling knob for the rollout pool. ~40 rollouts against 1115 demos =
  3.5% of windows; the D8 "self-improvement" round is a rounding error unless this lands.
* `grasp_frac` (commit-band anchoring) exists and is a different mechanism from the "×2
  window multiplier"; it applies to the whole dataset, not per episode.

What *does* work out of D8's list: w=1 demos ✓, w=0 failures ✓ (an operator-`f` rollout
gets `action_weight 0.0` via `is_failure_demo`'s `success is False` branch — verified),
w=1 recoveries ✓ (default).

---

## 7. DAgger driver path

`phantom/dagger/relabel.py` **works** with the new mc-from-checkpoint code — driven
directly in tiny mode against the checkpoint from §5:

```
INFO phantom.dagger.relabel: relabeled ep_demo_a: 6 replans
... ep_rollout_contam_final: 6 replans ... ep_rollout_unjudged: 6 replans
relabeled episodes: 6
  ep_rollout_s records: 6  action shape (16, 7)
```

Three problems:

1. **The relabel output is orphaned.** `grep -rn relabels --include=*.py phantom/ tools/`
   → only the writer. `HIDConfig.teacher_mode` defaults to `"cached"` and `relabel_dir`
   exists (`config/training.py:70-71`), but neither is read by `distill_hid`, and
   `precompute_relabels` (named in `distill_hid.py:17`) **does not exist**. So
   `dagger_driver` pays a full teacher forward pass per anchor per episode and then
   `distill_hid` recomputes the teacher online anyway. It also relabels contaminated and
   unjudged rollouts (`relabel_root` uses `list_episodes`, which only filters
   non-finalized).
   `dagger/manifest.py`'s `demo_weight`/`rollout_weight` are likewise never read
   (`read_manifest` has one caller: a review-lens scratch script).

2. **`dagger_driver` does not forward `--hardware`.** `dagger_driver.py:55-63` builds
   `distill_args` with `--teacher-ckpt/--data/--extra-data/--dagger-round/--device`
   (+`--max-steps`, `--compute`) — no `--hardware`, so `distill_hid` falls back to
   `configs/hardware.yaml`. That config differs from the rig's in a shape field:

   ```
   hash default: 43cec1ae049cd61d   nuc: 72a4dd674113afd0
   shape diffs: {'wrist_window_len': (125, 31)}
   ```

   `load_phantom_checkpoint`'s shape assert fires — loudly, after the relabel pass has
   already been paid for. There is also no `--tiny`/`--synthetic` on `dagger_driver`, so
   program (3b) cannot be smoke-run at all, contradicting the playbook's "every one
   smoke-runs anywhere with `--tiny --synthetic --max-steps 2`":

   ```
   $ .venv/bin/python -m phantom.train.dagger_driver --round 1 ... --device cpu
   FileNotFoundError: .../cosmos-predict2.5-2b/robot/action-cond/38c6c645-..._ema_bf16.pt
   ```

3. **`distill_hid` and `finetune_hids` ignore the manifest split.**
   `distill_hid.py:226` and `finetune_hids.py:174` call `C.WindowDataset(data_root,
   sampler)` with no `episodes=`. This is the D9 item ("Fix distill_hid/finetune_hids to
   load `manifest_split(root,"train")` — both currently train on the held-out val split")
   and it has **not** been done. Demonstrated:

   ```
   distill_hid/finetune_hids index episodes: ['ep_demo_a','ep_demo_b','ep_rollout_f','ep_rollout_s']
   manifest val episodes:                    ['ep_demo_b']
   ```

   and in the real program:

   ```
   $ .venv/bin/python -m phantom.train.distill_hid --data <S>/ds/tasks ... --tiny --max-steps 1
   INFO distill_hid: HID dataset: 32 windows (round 0)      # 4 episodes x 8 — includes val
   ```

   Every student number and every DAgger round is trained on its own validation set.

A full tiny DAgger-shaped round nonetheless runs:

```
$ .venv/bin/python -m phantom.train.distill_hid --data <S>/ds/tasks --extra-data <S>/ds/tasks/waffles \
      --hardware <S>/hw_small.yaml --tiny --device cpu --max-steps 2 --teacher-ckpt runs/teacher/val_rollout/teacher_000002.pt --dagger-round 1
INFO phantom.train.common: partial init (teacher->student): 30 checkpoint keys dropped, 0 model keys left fresh
INFO distill_hid: HID dataset: 80 windows (round 1)
step 2/2  behavior_match=2.2313  traj_distill=3.9225  event_distill=0.0775  total=11.7457
checkpoint saved: runs/hid/dag_r1_r1/student_000002.pt
```

80 = 40 + 40: `--extra-data` is *added* to the full-root index with no dedup, so any
rollout that has been `place`d under `tasks/` (which is what `intake_recovery place` does)
is counted twice if the rollout root is also passed to `--extra-data`.

Minor: the intake manifest row for `ep_rollout_f` says `"failure_demo": false` while
training treats it as a failure demo (`action_weight 0.0`) — the row and the behaviour
disagree.

---

## Verdict

**NOT READY** for a DAgger-lite / self-improvement round after the first rig session.

The P9 gates are real and hold at every layer I could reach (deploy verdict → intake →
manifest → index), the re-derivation is numerically exact, the base-hash fix makes CONFIG
DRIFT pass, and a mixed demo+rollout set trains. But the *round itself* is not runnable as
planned: the per-episode weighting D8 depends on does not exist and fails silently, the
distillation programs train on their own val split, `dagger_driver` cannot hand
`distill_hid` the right hardware config, and the relabel pass it pays for is dead code.
Separately, the P8 auto-label will score a veto-retry success as a failure, which is the
number G2 is decided on.
