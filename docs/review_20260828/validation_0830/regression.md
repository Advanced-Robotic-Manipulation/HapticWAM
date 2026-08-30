# LENS: REGRESSION AGAINST THE REVIEW — PHANTOM @ d9c40f2 (base 3539988)

All commands below were run on this Mac with `/Users/sannikov/GitHub/phantom/.venv/bin/python`,
mock drivers (`configs/hardware.nuc.mock.yaml`), the `--tiny` backbone and `configs/paths.local.yaml`.
Scratch under `.../scratchpad/validate/scratch/regression/`. Nothing under the repo was modified
(only gitignored `runs/` + `data/synthetic/` were written by the training smokes; `git status` clean).

## 0. Baseline

    .venv/bin/python -m pytest tests/ -q -p no:randomly
    -> 571 passed, 96 warnings in 329.73s

    git diff 3539988..HEAD -- '*.py' '*.sh' '*.yaml' | grep -iE '^\+.*(deferred|todo|not implemented|fixme)'
    -> 2 hits, both prose ("former DEFERRED note", "it used to say DEFERRED").
       grep -rniE 'deferred|TODO|FIXME' --include='*.py' phantom tools -> ONE line
       (phantom/deploy/planner.py:79, a comment recording that the DEFERRED note is gone).
    => No new deferral markers were introduced. Hygiene is good.

End-to-end exercises that ran clean:
  * `run_deploy --system teacher --task waffles --tiny --hardware configs/hardware.nuc.mock.yaml
     --episodes 1` (baseline) and again with `--parity-fixes --terminal-veto --k-seeds 4 --nfe 1
     --seed 4242` -> 5 replans, per-replan `terminal_veto` + `diag{nfe,guidance,k_seeds,
     head_dz_spread_mm,k_rejected,k_pick}` in planner_trace.json, meta tags
     `parity:on veto:pc0.50/pn0.90/r3 kseeds:4 seed:4242 zfloor:42mm`.
  * `tools/replay_rig.py --tiny --episodes <that episode>` -> per-replan head_dz +- seed std,
     trace_in_spread, OVERALL summary, JSON out.
  * `tools/label_grasps.py <episodes> --confusion` -> runs, prints the dual-label table.
  * `train_teacher --synthetic --tiny --acc-two-pass --contact-nll-beta 0.5 --contact-self-forcing
     --action-noise-per-strip --no-action-t-max-of-two --ema-decay 0.995 --cond-dropout 0` -> 4 steps,
     checkpoint records every knob; `run_deploy --ckpt <it>` rebuilds mc from the checkpoint and runs.
  * `train_teacher --student --mask-wrist` -> "STUDENT layout (vision_only arm): tactile inputs
     absent, wrist F/T MASKED", SequenceLayout(student=True, T=12).

## 1. Per-item verdicts

### P1 — offline metric conditions on demo-consistent inputs — **FIXED (with two caveats)**
`tools/replay_rig.py` (434 lines) + `tools/replay_deploy_path.py` rebuild the ObsSnapshot from a
recorded deploy episode's zarr + planner_trace and push it through `policy._batch_from_obs` /
`PhantomPolicy.replan`. Verified by replaying a mock deploy episode I recorded minutes earlier
(command above): 5/5 accepted replans rebuilt, K-seed spread and `trace_in_spread` reported.
`terminal_eval` hardened and verified:
    resolve_episodes(Path('data/synthetic/43cec1ae04'),'val')
    -> SystemExit "no manifest ... refusing to evaluate every episode ... pass --split all"
    --max-episodes default is None; medians/seed-std/pred_close_height_mm present; --null works.
Caveat A: the number that blessed v5_6 (docs/rig_session_v5.md:127, "124 val episodes") was produced
under the OLD silent-fallback, i.e. train+val; the hardened tool's own E9 run reports 78 val episodes
and endpoint 20.86 mm for the SAME v5_6/EMA — the doc still says 17.4 and v4 was never re-run.
Caveat B: neither replay_rig nor terminal_eval reports the model's ACC gate / p_evt at the replayed
state (replay_rig echoes the TRACE's `gate`, terminal_eval has no gate metric at all), so the
terminal veto's `--veto-p-close` cannot be calibrated offline. See findings 5 and 6.

### P2 — prev_chunk is the policy's own proposal — **FIXED behind `--parity-fixes`**
`SnapshotBuilder.prev_chunk_from_history` + `ObsSnapshot.prev_chunk` +
`PhantomPolicy._batch_from_obs` prefers it; `prev_cpk_step = round(latency/latent_dt)`; contact-state
dt from ring timestamps; `reactive` from consecutive fields_ds. Exercised live (`parity:on` tag,
episode ran). E3 support present: `replay_rig --prev-chunk {proposal,measured,zeros}`,
`--prev-cpk {chained,none}`. 20 unit tests in tests/test_deploy_levers.py cover the measured chain,
the executed gripper command, the first-replan zero case.

### P3 — nothing ties close-outcome to the lift; ACC gate decorative — **PARTIAL**
Deploy veto exists (`TerminalVeto`, `PlannerLoop._apply_veto`) and is the first path that lets the
gate modify a chunk; close-mask, phantom-grasp open+no-lift recovery, retry cap, all traced.
BUT (a) the "already low enough to close" escape is `z <= z_floor + 15 mm` where
`z_floor = tcp_z_min - 10 mm`, i.e. `tcp_z_min + 5 mm`, not the review's `tcp_z_min + 15 mm`;
measured band waffles 57 mm / Carton 81 / egg 65 / whiteboard 80, all BELOW the demo close band
(waffles demos close 46-88 mm) -> at low p_contact every demo-height close is masked (script
scratch/regression/veto_band.py). (b) `--veto-p-close 0.5` is uncalibrated and, per P1 caveat B,
cannot be calibrated with the shipped tools. (c) the λ:=0 ACC inference ablation (P3 step 2 /
§1.11 item 2) does not exist — `grep -rniE 'lambda.zero|acc_lambda|--acc-off'` returns nothing —
so the RQ2 attention-coupling claim is still unmeasured.

### P4 — executor phase lag / never-executed tail — **PARTIAL, deliberately deferred**
Latency levers shipped and are the recommended arm (`--nfe 1` = 172 ms replans per
docs/rig_session_v5.md; `--compile`/`--compile-mode` wired; `bench_inference --k-seeds` added so
K x NFE can be profiled on the deploy GPU). RTC committed-prefix inpainting / `t_exec0 = obs.t`
phase correction NOT implemented (`grep -rniE 'inpaint|skip.head|phase.correct'` -> nothing), and
E8's closed-loop surrogate does not exist: replay_rig replays recorded states open-loop only. This
matches the review's ordering (cut latency first) but E8 cannot be run as written.

### P5 — contact NLL owns the trunk gradient — **FIXED as flags, BLOCKED as a run**
`_nll_term(beta, detach_weight)` implements beta-NLL (`nll * var.detach()**beta`) and the
detached-weight variant; `--no-wrist-region-mse` zeroes the duplicate term; all recorded in the
checkpoint config; 6 unit tests assert the gradient identities. **But the FT-A launch line the
playbook and tools/provision_v5.sh print dies at startup** — see finding 1. Separately, the E1
verification probe the playbook names (`scratchpad/research/gradprobe.py`) is not in the repo
(`scratchpad/` does not exist here), so the review's gate on beta-NLL has no runnable artifact.

### P6 — few-NFE / persistent-noise offset; constant seed — **FIXED for run_deploy, NOT for run_eval**
`episode_seed()` + `policy.rf._gen = torch.Generator().manual_seed(ep_seed)` per episode, tag
`seed:<n>` verified in meta.json. `--nfe`, `--k-seeds K` with contact-consistent selection
(head_dz spread, k_rejected, k_pick in the trace), `--action-noise-per-strip` honoured in BOTH
`training_step` and `sample()`. The eval-campaign entry point was missed — finding 2.

### P7 — co-denoised GT contact is a shortcut; droppable-video claim false — **PARTIAL**
`--contact-self-forcing` implemented (packs the two-pass inner sample's cpk into the CONTACT x0
INPUT, GT stays the loss target; `_sf_pred_cpk` cleared on every `_acc_inputs_train` call so a stale
package cannot survive a step), requires `--acc-two-pass`, unit-tested at the tensor level, and
MEASURED offline (E9: commit ratio 0.94 -> 0.65 with the CONTACT frames zeroed). The group-causal
mask was not taken (fine — the review offered it as the alternative). The false claim is still in
the code the paper quotes: `phantom/model/sequence.py:193` still says video frames are "droppable at
inference without a train/test attention mismatch". Review §4.7 said fix or drop it.

### P8 — Robotiq OBJ is not a grasp signal — **FIXED as code, UNDER-VALIDATED**
`phantom/eval/grasp_label.py` implements the rule verbatim, `obj==2` is a stall FLAG only
(regression-tested), `tools/label_grasps.py` prints the dual label + confusion. Ran it on recorded
episodes. The recorded compute3 validation is 87/94 = 92.6% positive on demos (waffles 30/36 = 83%)
against the review's ">= 95%" bar, and the "grasp_ok AND stall on the 80 over-squeeze fails" leg is
not reported at all. Finding 9.

### P9 — unlabeled / contaminated rollouts would train — **FIXED**
Verified by execution (scratch/regression/p9_probe.py): a deploy episode is filed
`status='aborted' + tag 'unlabeled'` (seen live in the mock run), `is_trainable_episode` is False for
{unlabeled, contaminated, finalized+success=None+policy}; `manifest_split` drops tagged rows and
hard-fails on an unjudged rollout; `build_index` re-checks meta.json (the DAgger `--extra-data` path);
episodes are stamped with the BASE hardware hash (`config_hash 439840c8cd13e205` == `load_hardware()`
hash) with the envelope in `meta.deploy_overrides`; `tools/rederive_rollout_actions.py` added;
verdict prompt carries damage (`sd`/`fd`/`cd`); trial_runner writes the verdict back onto the episode.

### P10 — headline arm unbuildable / mis-built — **FIXED**
A: `--student` + `--mask-wrist` run (log above); `HHT._wrist_input` is the single choke point for
both OBS_PROPRIO and the ACC branch; `WRIST_MASKED_MODES=('vision_only','drop_tactile')` applied both
in `SnapshotBuilder.build` and via `mc.mask_wrist` in `run_deploy.build_policy`.
B: `distill_hid`, `dagger/relabel`, `finetune_hids` all build from
`PhantomModelConfig.from_dict(payload['configs']['model'])`; `assert_model_config_matches` refuses
drift and is backward compatible (`if k in saved`, so pre-flag checkpoints load). The D9 companion
item is NOT done: both programs still index every episode including the val split — finding 4.

### §1.11 (a) evaluation protocol — **MOSTLY NOT FIXED**
Fixed: the operator verdict is now written onto the episode, and damage has a field.
Not fixed: `seed` still never reaches the policy (`run_episode` has no seed argument;
`trial_runner._run_one` puts it in the ledger only) while `eval/aggregate.py` clusters its bootstrap
on it; `run_campaign` is still strictly system-major (interleaving structurally impossible);
`run_eval.py` builds `pol_args` without `parity_fixes/k_seeds/persistent_noise/guidance/nfe`, so an
eval campaign cannot run the same deploy controller the rig arms use (the review's arm-parity
kill-shot); `configs/eval_campaign.example.yaml:4` still names fragile_grasp/slippery_place/
insertion/wipe/regrasp. And the constant-seed bug is live on this path — finding 2.

### §1.11 (b) ACC injection numerically inert — **NOT FIXED** (no λ:=0 knob; see P3c).

### §1.11 (c) cond_dropout / guidance — **FIXED (mechanism)**
`5b3aeff`: the CFG null branch now adopts the conditional `prev_cpk` summary
(`rf.py` `_null_acc`: `acc_null.prev_cpk_summary_B_S = acc_inputs.prev_cpk_summary_B_S`), so a
guidance sweep is interpretable. `--guidance` exposed in run_deploy/terminal_eval/replay_rig. No
offline 1.0/1.5/2.0 A/B has been recorded yet.

### §1.11 (d) deploy safety — **PARTIAL**
Fixed and tested: the hitbox is evaluated on `self.clamp_target(target)` (safety.py:210), so a deep
descent is clamped at the z floor instead of ending the episode; joint start gate, speed cap,
max-replans 40 all present. NOT done: `configs/start_poses.yaml` still carries `q_n: 17..37` for the
joint stats AND for the `tcp_min/tcp_max` STOP-hitbox envelope (the file's own header says
"Regenerate both over the full dataset ... when it is on one box"), and `tools/gen_start_poses.py`
still has no `q_n != n` refusal (only `assert len(P) >= 20`, which egg's q_n=17 already violates for
the joint half). Finding 8.

## 2. What I would fix before the next rig hour / the next H100 hour
1. the `--init-weights` + FT-A crash (finding 1) — it blocks D4 entirely;
2. the veto band + p_close calibration (findings 3 and 6) — otherwise arm B can be 0/16 by
   construction and the G2 gate reads a controller artefact;
3. the eval-campaign seeding + lever parity (finding 2) if any paper number is produced by run_eval;
4. the distill/finetune val leak (finding 4) before the $70 student launch.
