# Lens: EXPERIMENT PROTOCOL + PAPER READINESS — validation at d9c40f2 (2026-08-30)

Scope: what still has to exist to produce the paper's numbers, given the deploy levers, the
seeded sampler, `replay_rig`, `grasp_label` and the FT-A training knobs all landed.
Everything below was exercised, not only read. Scratch under `scratch/protocol/`.

## What I ran

1. `pytest -k "eval or grasp or label or aggregate or metric"` → **58 passed**. Note *what* passes:
   `test_grasp_label.py`, `test_terminal_eval_*`, `test_replay_rig.py`. There is **no test anywhere**
   that touches `phantom/eval/trial_runner.py`, `aggregate.py` or `metrics.py` except
   `test_p9_p10_fixes.py::test_eval_trial_writes_its_verdict_onto_the_episode`, which stubs the runtime
   out entirely. The three modules that must produce the paper's tables are effectively untested.
2. **A real 3-arm eval campaign through the real `DeploymentRuntime` with mock drivers**
   (`scratch/protocol/camp2.py`, 12 episodes, `make_small_hw()`, crawl policy):
   ```
   EXECUTION ORDER (system,seed,trial):
   [('teacher',0,0), ('teacher',0,1), ('teacher',1,0), ('teacher',1,1),
    ('student',0,0), ('student',0,1), ('student',1,0), ('student',1,1),
    ('no_distill',0,0), ('no_distill',0,1), ('no_distill',1,0), ('no_distill',1,1)]

   n episodes: 12
     no_distill ep_no_distill_waffles_1788097343_000 policy= no_distill tags= [] success= True
        notes= 'eval main_eval/no_distill: cell C2' trace= True
   HEADLINE: {('waffles','False'): {'recovery_ratio': nan, 'retention': 1.0}}
   ```
   Three separate defects in one output: strictly arm-major execution; `meta.tags == []`
   (no seed, no nfe/veto/kseeds/parity/ckpt/git provenance — those tags are built in
   `run_deploy.main`, never in the runtime); and a `nan` headline because the plan-of-record
   arm set (teacher/student/no_distill, §4.4) has no `vision_only` denominator.
3. **`aggregate` on a D15/D16-shaped ledger** (waffles, 3 arms × 25, `scratch/protocol/mk_ledger.py`):
   report emits success, a bootstrap CI, damage, `recovery_ratio=nan`, `retention`. No Fisher,
   no Wilson, no paired per-placement difference, no arm-vs-arm comparison of any kind.
4. **CI degeneracy** (`scratch/protocol/degen.py`):
   ```
   0/25 (the observed arm)    succ=0.000  boot95=[0.000,0.000]
   25/25                      succ=1.000  boot95=[1.000,1.000]
   13/25                      succ=0.520  boot95=[0.320,0.720]
   wilson  0/25 -> [0.000, 0.133]     wilson 25/25 -> [0.867, 1.000]
   ```
   The percentile bootstrap collapses to a zero-width "95% CI" at exactly the outcome the
   last three rig sessions produced (0/n). Also: with the plan's "single checkpoint per arm",
   there is one seed group, so the advertised cluster-over-seeds bootstrap silently degenerates
   to a trial-level bootstrap while the docstring still claims seed clustering.
5. **ACC lead-time money plot** (`scratch/protocol/lead.py`): a synthetic trace whose gate latches
   at 0.95 on the second replan (t=0.9 s) and never falls, with the physical onset at t=15 s:
   ```
   ACC lead times reported (s): [14.1, 14.1]
   ```
   `metrics.acc_lead_times` takes the *last rising crossing before* the onset with no cap, so a gate
   that simply latched high reports 14.1 s of "anticipation" — on a system whose replan period is
   0.9 s and whose gate lookahead is 0.5 s. It also emits one value **per sensor**, so every onset is
   double-counted with a perfectly correlated pair. `event_f1` (metrics.py:126-128) compares the
   *next-step* prediction against the event at the replan time itself (off by one) and uses only
   `sensors[0]`.
6. **FT-A / baseline launch lines** (`scratch/protocol/args.sh`): the FT-A bundle parses and composes:
   `--student --contact-nll-beta 0.5 --contact-self-forcing --action-noise-per-strip
   --no-action-t-max-of-two --ema-decay 0.995 --cond-dropout 0 --acc-two-pass` reaches dataset
   construction; `--student --mask-wrist` likewise; `--contact-self-forcing` without `--acc-two-pass`
   hard-fails with the right message. **The flags work. The launch lines are not written down**:
   `docs/training_playbook.md` has exactly one `python -m phantom.train...` line (the teacher, :159)
   and its comparative-systems table (:146-147) still describes `vision_only` as needing a "custom
   `PhantomModelConfig` variant". D2 = today = "launch the `no_distill` insurance run ($90, 35 h)"
   and there is nothing to copy-paste.
7. **The student's data split**: `phantom/train/distill_hid.py:226` is
   `ds = C.WindowDataset(data_root, sampler_s)` with `episodes=None`, which
   `windows.build_index` (:189) resolves as `list_episodes(root)` — every trainable episode,
   including the manifest `val` split. `manifest_split` appears only in `train_teacher.py:379,421`
   (verified by grep across `phantom/train/` and `phantom/dagger/`); `distill_hid` has no `--split`.
   The D9 line "Fix `distill_hid`/`finetune_hids` to load `manifest_split(root,'train')` — both
   currently train on the held-out `val` split" is **not done**. RQ3's headline number would be
   scored on data the student trained on.
8. **Deploy-time ablation arms**: `--terminal-veto` ✓, `--drop-video` ✓, `--k-seeds` ✓, `--parity-fixes` ✓
   — all tagged into `meta.json` by `run_deploy.py:519-532`. "Tactile on/off at deploy" (D17) has
   **no implementation**: the only route is `--system drop_tactile`, which sets `student=True`
   (`run_deploy.py:47`) i.e. a different *layout*, and `run_deploy` loads without `allow_missing`
   (`common.py:311-317`), so a teacher checkpoint under that system raises
   "checkpoint keys unknown to this model". A trained `drop_tactile` arm is also unproducible:
   `distill_hid` has no `mask_wrist` (grep: zero hits).
9. **"Miss distance"** — the plan's stated *primary continuous number* for D11 and D15 — occurs
   exactly once in the whole repo, in prose: `docs/rig_session_v5.md:37`. No code computes it, and
   nothing records an object pose (grep for `object_pose|aruco|april`: zero hits), so it is not
   derivable from any recorded stream. `planner_trace.json` rows (`planner.py:486-497`) carry
   `t / latency_s / gate / p_evt / sigma / accepted / actions / terminal_veto / diag` and **no TCP pose**,
   so even "z-at-close from `planner_trace.json`" needs a clock-calibrated join against
   `arm_tcp_pose.zarr` (what `rig_trace_decompose.py:41,60` does by hand).
10. **`grasp_label` is not wired to anything but its own CLI**: grep for `grasp_label` across
    `phantom/` and `tools/` returns one importer, `tools/label_grasps.py:23`. `eval/aggregate.py` and
    `eval/metrics.py` never see it, so the objective label that gates G2/G3 and produces the paper's
    success rate never enters the eval report, and the dual-label confusion matrix the completeness
    critic demanded (`grasp_label.confusion`, :323) can only be produced by hand.
11. **The pre-registration is stale.** `pipeline.md:389-395` is still the claims-of-record document
    (every eval module docstring cites "pipeline.md §8"): 5 systems × 5 tasks (fragile_grasp /
    slippery_place / insertion / wipe / regrasp — none exist) × 20 × ≥3 seeds ≈ 2,100 + 960 trials,
    occlusion variants, fragility instrumentation, `recovery_ratio` over `vision_only`.
    `configs/eval_campaign.example.yaml:4-11` encodes it verbatim. The rewrite is scheduled at
    D13/G4 — two days before the ablation session and after all three matrix arms are launched.

## What is genuinely ready

- The deploy levers are all flag-gated and, crucially, **always tagged** on/off into `meta.json`,
  so a run's arm can be reconstructed from the episode rather than the operator's memory
  (`run_deploy.py:519-532`) — for the `run_deploy` path.
- FT-A composes and guards itself; `--student`/`--mask-wrist` build and round-trip.
- `grasp_label` is careful, honest about its unvalidated positive class, and regression-tested.
- The P9 verdict guard reaches the eval path: `trial_runner._run_one:100-105` relabels the
  episode, so campaign episodes are trainable and judged. That one is fixed.

## Verdict

**NOT READY.** Every lever the paper needs to *run* exists; almost nothing the paper needs to
*measure* does. Besides rig time the blockers are: no interleaved-schedule generator and a
runner that is structurally arm-major with no placement or seed field; no statistics script
(no Wilson, no Fisher, no paired difference — scipy is not even installed); a headline metric
whose denominator the plan of record deleted; the student training on its own val split; and a
pre-registration document that describes a different experiment.

## Action items (specify, do not build)

| # | item | file | h | depends on |
|---|---|---|---|---|
| 1 | `run_episode(policy=, mode=)` per-episode override; hoist the runtime above the arm loop; reorder `run_campaign` to placement-major → arm-interleaved | `phantom/deploy/runtime.py:171`, `phantom/eval/trial_runner.py:67-79` | 3 | — |
| 2 | `tools/gen_ab_schedule.py`: pre-registered CSV (placement cell, arm order, per-episode seed, task) with a fixed RNG seed; `trial_runner` consumes it | new | 3 | 1 |
| 3 | Add `placement`, `seed`, `arm_tags`, `grasp_ok`, `z_close_mm` to `LEDGER_FIELDS`; plumb `seed` into `policy.rf._gen` via `run_deploy.episode_seed` (`run_deploy.py:319`) and tag it | `phantom/eval/trial_runner.py:31,90` | 2 | 1 |
| 4 | `phantom/eval/stats.py`: Wilson interval, Fisher exact (`math.comb`, no scipy), McNemar / paired per-placement difference + bootstrap on the paired delta; replace `cell_stats`'s percentile bootstrap | `phantom/eval/aggregate.py:32-56` | 3 | — |
| 5 | Rewrite the headline: drop `recovery_ratio`, report the three raw rates + paired deltas (teacher−student, student−no_distill) with CIs and Fisher p | `phantom/eval/aggregate.py:68-80` | 2 | 4 |
| 6 | Join `grasp_label.label_episode` into `metrics.trial_metrics`; emit `grasp_ok`, `z_close_mm`, `c_hold`, `lift_mm`, `stall` + the rule-vs-operator confusion matrix per arm | `phantom/eval/metrics.py:153`, `aggregate.py:84-96` | 2 | — |
| 7 | Define and implement the continuous primary: log `tcp_pose`/`z` per replan into `planner_trace.json`, add `commit_height_mm`, and either (a) record a per-episode object position at reset (one operator field / one taped cell id) so miss distance is defined, or (b) delete "miss distance" from the plan and the paper | `phantom/deploy/planner.py:486`, `docs/rig_session_v5.md:37` | 3 | — |
| 8 | `--split` on `distill_hid`/`finetune_hids`, defaulting to `train`, plus a `val` eval loop | `phantom/train/distill_hid.py:226` | 2 | — |
| 9 | Write the `no_distill` / `vision_only` launch lines into the playbook and update the comparative-systems table; state the shared dataset + shared deploy controller across arms (arm-parity pre-registration) | `docs/training_playbook.md:140-147,159` | 1 | 8 |
| 10 | Cap and de-duplicate `acc_lead_times` (cap at the replan period or the gate lookahead; one value per onset, not per sensor); fix the `event_f1` off-by-one and use all sensors — or cut RQ2 to a diagnostic per §4.2 | `phantom/eval/metrics.py:75-137` | 2 | — |
| 11 | Deploy-time tactile null (`--null tactile` mirroring `terminal_eval`'s `_null_obs_batch`) so D17's tactile ablation runs on the teacher checkpoint; or drop the arm | `phantom/scripts/run_deploy.py`, `phantom/deploy/planner.py:221` | 3 | — |
| 12 | Rewrite `pipeline.md:358-397` and `configs/eval_campaign.example.yaml` to the actual protocol (4 real tasks, 3 arms, 25 trials, no occlusion, no fragility instrumentation, single checkpoint per arm) — move it before D9, not D13 | `pipeline.md:389`, `configs/eval_campaign.example.yaml:4` | 2 | 5 |
| 13 | Regression tests for `trial_runner`/`aggregate`/`metrics` (mock-driver campaign smoke + a stats golden file) | `tests/` | 2 | 1,4 |

Total ≈ 30 h ≈ 2 focused dev-days, all off the rig, all before D15.
