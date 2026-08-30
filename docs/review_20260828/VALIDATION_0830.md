# PHANTOM end-to-end validation — HEAD `d9c40f2`, 2026-08-30

Seven lenses (Opus, each executing code on the Mac with `.venv/bin/python`, mock drivers,
`--tiny`) + four completed Codex passes. Three Codex passes (deploy-recipe, rollout-intake,
protocol-paper) and one Codex regression pass produced **no output** — quota exhausted
("try again at 1:48 PM") and a read-only sandbox that blocked `TMPDIR`, so `import torch`
itself failed. Those four are re-runnable and are listed as an action item, not as a verdict.

Twenty findings went to a two-voter adversarial confirm/refute round. **All twenty survived.**
Zero were refuted. Confidence per finding 0.72–0.96.

Everything below is reproduced-by-execution unless explicitly tagged `[unvoted]` (single lens,
no second voter) or `[codex]` (Codex-only, evidence from read-only probes).

---

## 1. READY / NOT READY by area

| # | Area | Verdict | Blocking items (must clear before the area is usable) |
|---|---|---|---|
| 1 | **Rig session recipe** (Session 4 / D5 A-B) | **NOT READY** | (a) `--terminal-veto` never arms on the recorded failure mode `planner.py:403`; (b) its `closed_at` latch never expires and reopens a real grasp `planner.py:379`; (c) the veto's `at_floor` escape sits 30–90 mm below the demo close band `planner.py:407`; (d) no loop pacing → arm B is a ~7 s episode at `--nfe 1` `planner.py:422`; (e) documented Arm A (`--seed 4242` + 1 episode/process) re-creates the constant-seed bug `docs/rig_session_v5.md:71`; (f) start-pose jitter is a separate unseeded RNG `run_deploy.py:553` |
| 2 | **FT-A retrain** (D4, ~6.5 H100-h, ~$16–40) | **NOT READY** | (a) the documented launch line dies at startup — P10B assert runs before the finetune-tolerant drift path `common.py:298`→`:269`, ahead of `train_teacher.py:325`; (b) `--event-band-weight 0` is not persisted and silently reverts on `--resume` `train_teacher.py:374`; (c) `[codex]` `h100x8` val runs on one shuffled, drop_last rank shard `common.py:584`; (d) `[codex]` `h100x8` without `torchrun` only warns and trains at effective batch 1 `compute.py:83` |
| 3 | **Eval tools** (replay/terminal_eval/grasp_label) | **NOT READY** | (a) the published E9 premise row is the GT-**pinned** condition read backwards `E9_premise_test.md:19`; (b) `replay_rig` implements none of the `--parity-fixes` construction `replay_rig.py:121,161`; (c) it seeds once per **run**, so results depend on `--episodes` order `replay_rig.py:399`, and no tool reads the new `seed:<n>` tag; (d) on a vetoed replan `trace["actions"]` is the veto's arithmetic, and `trace_in_spread` (gate G0) scores against it `planner.py:486`; (e) `grasp_label`'s 2.5 s hold tail is not guaranteed by `run_episode` `grasp_label.py:276` |
| 4 | **Safety** | **NOT READY unattended** (static envelope READY) | Verified sound: floor-is-a-clamp, hitbox STOP, `--max-tcp-speed`, joint/full-turn gate, `gripper_ctl`, per-episode seeding + tags. Blocking: (a) hitbox + z-floor fitted to `q_n` = 17–37 demos while `start_poses.yaml` says `n: 250` `configs/start_poses.yaml:24`; (b) every `STOP_EPISODE` leaves the gripper **commanded closed** on the gels `executor.py:131,252,334` `[unvoted]`; (c) the floor-is-clamp fix silently inverts if an operator sets `--hitbox-margin < --z-floor-margin` `run_deploy.py:454,472` `[unvoted]`; (d) no pacing (as §1) |
| 5 | **Rollout intake / D8 self-improvement** | **NOT READY** | (a) `EpisodeMeta.weight` does not exist and is silently dropped `schema.py:64`; (b) `distill_hid`/`finetune_hids` train on the held-out `val` split `distill_hid.py:226`, `finetune_hids.py:174`; (c) nothing enforces that `rederive_rollout_actions` ran `intake_recovery.py:152`; (d) the P8 auto-label scores a terminal-veto **retry success** as a failure — the exact number G2 is decided on `grasp_label.py:263`; (e) `dagger_driver` never forwards `--hardware` `[unvoted]`. P9 gates themselves **hold at every layer** |
| 6 | **Provisioning / rental** | **NOT READY** (hub + tarball **byte-verified**) | (a) it downloads only the v4 checkpoint and inits FT-A from it, against "FT-A: 3k steps from v5_6" `provision_v5.sh:205,271`; (b) `--run-name teacher_v5_batch0822` would overwrite the six shipped v5 checkpoints on the hub `[unvoted]`; (c) the pytest gate at `:245` is red — `test_replay_rig.py::test_rebuilt_snapshot_matches_the_snapshot_builder` is **flaky**, and `set -euo pipefail` swallows the gate's own error message; (d) checkpoint selection still says `terminal_eval` `:283` where the contract says replay; (e) no checkpoint egress/upload step anywhere in the repo; (f) no rig episodes on the box, so replay scoring is impossible there |
| 7 | **Paper protocol** | **NOT READY** | (a) `run_campaign` is structurally arm-major — interleaving is impossible `trial_runner.py:67`; (b) it never reseeds the sampler and its `seed` column is fictitious `trial_runner.py:74,90`; (c) **no statistics code exists** — no Wilson, no Fisher, no paired difference, scipy not installed; the shipped bootstrap CI is `[0.00, 0.00]` at 0/25 `aggregate.py:32`; (d) the headline metric is `nan` for the plan-of-record arms; (e) "miss distance" has no data source anywhere in the repo; (f) the student trains on its own val split; (g) `pipeline.md:389` still pre-registers a different experiment |

**Cross-cutting.** ~30 h of off-rig engineering stands between the rig hours and a paper
table. Nothing in the plan of record is blocked by hardware or by money; everything is
blocked by code that was written but never exercised through its own CLI.

---

## 2. Confirmed problems, ranked

Rank = (blocks a scheduled spend or rig hour) → (invalidates a number already quoted) →
(costs a gate) → (hygiene). `[2×]` = survived the two-voter round; `[unvoted]` = one lens;
`[codex]` = Codex only.

### P0 — blocks the next money or the next rig hour

**1. The FT-A launch line hard-fails at checkpoint load.** `[2×, conf 0.93–0.96]`
`phantom/train/common.py:298` → `assert_model_config_matches` raises at `:269`.
`load_phantom_checkpoint` is called from `train_teacher.py:316`, nine lines *before* the
finetune-tolerant `model_config_drift(..., tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)` at
`:325`. The ignore set at `common.py:264` is only `("student","mask_wrist",*TRAIN_ONLY_MODEL_FIELDS)`;
`cond_dropout_p`, `action_t_max_of_two`, `action_noise_per_strip` live in
`FINETUNE_MUTABLE_MODEL_FIELDS` (`config/model.py:54-56`), never in `TRAIN_ONLY`.
Both shipped checkpoints save `cond_dropout_p=0.1, action_t_max_of_two=True`, so **either**
init fails. `--cond-dropout 0` alone is enough. `docs/training_playbook.md:87-90` documents
the opposite. `tests/test_ft_a_fixes.py` calls the helper directly and never the CLI, which
is why 571 tests are green.
**Fix:** add `tolerate: frozenset = frozenset()` to `assert_model_config_matches` and
`load_phantom_checkpoint`; pass `FINETUNE_MUTABLE_MODEL_FIELDS` from the `--init-weights`
branch (`train_teacher.py:316`) and leave `--resume` (`:338`) strict. Add a CLI-level test
that invokes `main([... '--init-weights', ckpt, '--cond-dropout','0','--no-action-t-max-of-two', ...])`.
*Verified: with exactly that relaxation monkey-patched, the full bundle trains to completion.*

**2. `--terminal-veto` is a no-op on the failure mode it was written for.** `[2×, conf 0.85–0.86]`
`phantom/deploy/planner.py:403`: `closing = g_max > v.close_pos and (g_max - grip_now) > v.close_rise`,
where `grip_now` is the **current measured** aperture (`:470`). That makes 0.15 a per-replan
*rate* test. The rig's recorded terminal phase ramps 0.31→0.52 over several replans (per-replan
rise 0.02–0.06), so the mask never fires, `closed_at` is never armed and the phantom-grasp
recovery can never run. The class docstring claims it "reuses the TRAINING rule", which is
`train/common.py:358`'s **running minimum** (`pos - run_min > CLOSE_ABS_RISE`). Re-derived from
the repo's own `episodes_0828_commanded_vs_actual.txt`: the current rule fires on 5 of 23
episodes, the running-min rule on 9; on waffles (the A/B task) 2/6 vs 4/6.
**Fix:** carry `g_min` in `veto_state` (init at the first `grip_now`, update each replan) and
test `(g_max - state["g_min"]) > v.close_rise`. 3 lines.

**3. The veto's `closed_at` latch never expires — it reopens a successful grasp.** `[2×, conf 0.78–0.87]` `[codex corroborates]`
`planner.py:485` sets `veto_state["closed_at"]` on an accepted close; the only clear is inside
the recovery branch (`:379-381`). No comparison against `plan.t_created` or the replan index,
despite the docstring at `:308-310` ("the very next replan"). Reproduced: `p_none = [.2,.05,.05,.05,.95]`
→ `['close_allowed'×4, 'recovery_open']` **four replans after the close**, with the gripper
forced to 0.25 and every z delta zeroed. `--max-replans` defaults to 40, so a real grasp is
exposed to ~30 live triggers; a release, a transport frame where the gel loses the object, or
an egg held lightly all read `p_evt[none] > 0.9`. Three of those end the episode `veto_retry_cap`.
Codex adds two related state bugs: the latch is set from **plan acceptance**, not from an
executed close (so a close living only in the unexecuted chunk tail arms it), and after a
`close_masked` rewrite the next replan still consumes `prev_plan.cpk` from the **pre-veto**
chunk (`planner.py:471`, `policy.py:244`).
**Fix:** store the replan index with `closed_at`, fire recovery only while `n - closed_at_idx <= 1`,
clear unconditionally after that window; latch on the executor's entered gripper steps, not on
plan acceptance; clear/recompute `plan.cpk` whenever the veto mutates `plan.actions`.

**4. `PlannerLoop` has no pacing: `--max-replans 40` is a ~7 s episode at `--nfe 1`.** `[2×, conf 0.87–0.90]`
`planner.py:422-524` is a tight loop with no sleep; `control.model_tick_hz` is declared in all
three hardware yamls and **consumed nowhere**. Episode wall-time == `max_replans × latency`.
Measured with stubbed latency: 865 ms (`--nfe 5`) → 40 replans = 34.9 s; 172 ms (`--nfe 1`, the
team's own measured number) → 40 replans = **7.2 s**. Demos run 16–31 s, and the cap was raised
20→40 precisely as a wall-clock proxy at the NFE-5 cadence. The Session-4 arm B is `--nfe 1`.
**Fix:** add `--max-episode-s` (default 35) checked in `run()` beside `max_replans`, and log the
stop reason on the replan cap (today it breaks silently with `stopped_reason = None`).
Stopgap for the session: `--max-replans 200` whenever `--nfe < 5`.

**5. The veto's "already low enough to close" band sits below the entire demo close distribution.** `[unvoted, two lenses agree]`
`planner.py:320` hard-codes `z_margin = 0.015` with no CLI knob, and `:407` computes
`at_floor = z_now <= v.z_floor + v.z_margin` from the **already-lowered** floor
(`z_floor = tcp_z_min − 10 mm`), not the review's `tcp_z_min + 15 mm`. Measured bands vs the demo
close p95: Carton 81 mm vs ~129, egg 65 vs ~86, waffles 57 vs ~88, whiteboard 80 vs ~166 — and
the model's own predicted close height is 110.8 mm (E9). The escape hatch essentially never
fires, so every close in arm B hangs on `p_contact > 0.5`, a threshold that has never been
calibrated and that read 0.01 at the rig's observed closes.
**Fix:** `at_floor = z_now <= (stats.tcp_z_min if stats else v.z_floor) + v.z_margin` and raise
`z_margin` to the per-task demo close p95 (i.e. make the hatch `z <= Z_MAX[task]`). Expose it as
a flag. Combined with #2 this is the difference between an interpretable G2 and 0/16 on both arms.

**6. The documented Session-4 Arm A re-creates the constant-seed bug.** `[2×, conf 0.86–0.87]`
`docs/rig_session_v5.md:71`: `EXTRA="--seed 4242" ./GO_v5_waffles.sh 1` launches **one episode
per process**, and `episode_seed(base, i) = base + i` uses the *within-process* index
(`run_deploy.py:319-326`). Verified across two processes: both give 4242. With the GO script's
`--persistent-noise`, that is one identical noise tensor for every Arm-A episode, while Arm B
(unseeded) draws fresh — the exact asymmetry the same doc blames for v4-vs-v5 "never being a
model comparison".
**Fix:** run the whole arm block in one process (`./GO_v5_waffles.sh 10`), or drop `--seed` from
Arm A (the unseeded path already records `seed:<n>`), or mix a per-process nonce into
`episode_seed` when `--episodes == 1`. Doc + 3 lines.

**7. The STOP hitbox and the z floor are fitted to 17–37 demos while `start_poses.yaml` claims `n: 250`.** `[2×, conf 0.85–0.91]`
`configs/start_poses.yaml:24` and the file's own header: `tcp_min`/`tcp_max` (the STOP hitbox)
and `tcp_z_min` (the clamp) come from the compute3 `q_n` subset — Carton 20, egg 17, waffles 37,
whiteboard 21 — while `tcp_mean`/`tcp_std` come from 250. `load_start_stats`
(`phantom/deploy/start_pose.py:65-104`) never reads `q_n`; `tools/gen_start_poses.py:47` asserts
only `len(P) >= 20` and can silently desynchronize P/Z/lo/hi from Q. `REVIEW_SYNTHESIS §1.11`
made this blocking: "Land this before any rig episode is counted." Not landed.
An extremum statistic over ≤15% of the demos gates every episode: too tight laterally →
`hitbox_exit` → `safety_stop` → full RTDE control-script rebuild and a lost episode; too tight
in z → the clamp stops the descent above where demos actually grasp, biasing the A/B against
the arm that finally descends.
**Fix:** regenerate over the full 250/task set on compute3; add `assert len(P) == len(G) == len(Q)`
in `gen_start_poses.py` and a `q_n == n` refusal (or loud warning) in `load_start_stats`.

**8. Every `STOP_EPISODE` stops the arm and leaves the gripper commanded closed on the gels.** `[unvoted]`
`executor.py:252-255` → `_halt("safety_stop")` at `:131` sets `_grip_target = None` and the stop
event; `_grip_worker` (`:334`) exits without sending anything. The only later `gripper.move` is
the *next* episode's homing (`start_pose.py:208`). Reproduced: after a fingertip-protection stop
at tick 41 the gripper's last commanded target is 0.90. The `tactile_fz`/`tactile_depth` guard
exists **specifically** to protect the gels, and it leaves the fingers squeezing at the pad
ceiling through the label prompt and the "clear the arm's path" prompt — minutes, unattended.
Same after `hitbox_exit` and `veto_retry_cap`.
**Fix:** in `_halt`, before setting `_stop`, issue one `gripper.move(open_aperture, speed, force)`
on the reasons that mean "let go" (`tactile_*`, `wrench_limit`, `hitbox_exit`, `veto_retry_cap`);
add `GRIPPER_OPEN.sh` to the post-stop prompt text.

### P1 — invalidates a number that is already quoted, or a gate

**9. E9's headline P7 row is the GT-PINNED run, labelled and interpreted as its opposite.** `[2×, conf 0.92–0.93]`
`docs/review_20260828/E9_premise_test.md:19`. `tools/terminal_eval.py:306-312` zeroes
`events`/`cpk_*` for **every** mode except `contact_gt`, and `:285-286` additionally cond-pins the
CONTACT frames to GT only for `contact_gt`. So row 1 ("real inputs", commit 0.94) is *already*
the no-GT-package condition, and row 5 — labelled "CONTACT frames zeroed (no GT package)",
commit 0.65 — is the **pinned-GT** run. The repo's own test asserts exactly this
(`rec.zeroed("events") is (null != "contact_gt")`), and `git diff 756987e d9c40f2 -- tools/terminal_eval.py`
is empty, so the tool was byte-identical when the table was produced. Read correctly the
measurement says handing the model the true future contact package makes it commit **less** —
the opposite of the doc's Reading, of commit `d9c40f2`'s message, of the stated justification for
`--contact-self-forcing` in FT-A, and of the paper's P7 exposure-bias sentence.
Also: P7's own prescription was pinned-to-GT **vs pinned-to-zeros**; `terminal_eval` implements
pinned vs *unpinned*, which confounds pinning with GT content, and `contact_gt` pins **clean,
un-noised** frames — not a training condition either.
**Fix:** relabel row 5 as `--null contact_gt`, rewrite the Reading to the measured direction, add
a `contact_zero` mode that pins the CONTACT frames to a zeroed package, and re-run before any
number from that table is quoted or used to justify a flag.

**10. `replay_rig` implements none of the `--parity-fixes` construction, so it cannot replay Arm B.** `[2×, conf 0.72–0.90]` `[codex #4 corroborates]`
`tools/replay_rig.py:121,157-185`: `RigEpisode.snapshot()` never sets `snap.prev_chunk`, always
uses the nominal `dt_field = 1/field_ds_rate_hz` for `derive_timestep`, always derives `reactive`
from the **previous replan** (~1 s apart) rather than two consecutive `fields_ds` frames, and
`replay_episode` (`:336-338`) passes no `prev_cpk_step` (rf default 0 instead of
`round(latency/latent_dt)`). Those are exactly the four switches `SnapshotBuilder`/`replan` flip
under parity (`planner.py:246-248,266-270,280-284`; `policy.py:242-247`). `meta.tags` records
`parity:on/off` and `replay_rig` reads only `trace[0]["diag"]["nfe"/"guidance"]` — it does not
even warn. Measured on a real parity episode: `prev_chunk` MISMATCH on every replan, `reactive`
3× and 10× off. `replay_deploy_path.py:60` hard-codes `parity_fixes=False` too.
`reactive` is not inert — it feeds `AccGate.psi_react` → the per-block attention bias, so a wrong
value perturbs the sampled chunk.
**Fix:** add `--parity-fixes` to `replay_rig` (set `prev_chunk` from `measured_prev_chunk`, use
`ts[i]-ts[i-1]`, consecutive-frame `reactive`, pass `prev_cpk_step`), default it from the
episode's `parity:` tag, and hard-fail when flag and tag disagree.

**11. `replay_rig` seeds once per RUN — per-episode numbers depend on `--episodes` order.** `[2×, conf 0.85–0.93]`
`tools/replay_rig.py:399` seeds `policy.rf._gen` in `main()` before the loop; `replay_episode`
never reseeds (`policy.reset_episode()` only drops the held noise). Same episode, same `--seed 1000`:
alone → `trace_in_spread 0.50`; preceded by one other episode → `1.00`. That is exactly the
statistic GATE G0 thresholds at "≥4 of 6", so dropping an invalid episode or changing the glob
order can move the gate. Second half: no tool reads the `seed:<n>` tag that `ba61354` finally
made available (`run_deploy.py:671` writes it; a repo-wide grep finds no reader), and the
docstring still says "the recorded episodes carry `seed:none`".
**Fix:** reseed inside `replay_episode` (`manual_seed(args.seed + episode_index)`); add
`--seed-from-meta` that parses `seed:<n>`, forces `--seeds 1`, and reports the residual against
the recorded chunk.

**12. On a vetoed replan the trace holds the veto's arithmetic, and G0 scores against it.** `[unvoted, two lenses agree]`
`planner.py:371-415` mutates `plan.actions` in place (`a[:,6] = grip_now` on close_masked;
`open_aperture` + cumulative-z clamp on recovery_open) and `:486-495` writes
`"actions": plan.actions.tolist()` **after** that. The pre-veto proposal exists nowhere.
`replay_rig.py:343` reads `r["actions"]` both as the `prev_chunk="proposal"` feed and as the
reference for `trace_head_dz`/`trace_in_spread` — the G0 validity test — and never looks at
`r["terminal_veto"]`. Every Arm-B episode would be scored against a chunk the model never produced.
**Fix:** record `"actions_pre_veto"` in the trace row whenever the veto fires; have `replay_rig`
prefer it and warn when `terminal_veto.action != "none"`.

**13. `distill_hid` and `finetune_hids` train on the held-out val split.** `[2×, conf 0.94–0.95]`
`phantom/train/distill_hid.py:226` and `finetune_hids.py:174` build `C.WindowDataset(root, sampler)`
with no `episodes=`, so `windows.py:189` falls through to `list_episodes(root)` — every episode,
val included. Neither module defines `--split` or calls `manifest_split`; `train_teacher.py:379`
does it correctly. `terminal_eval` defaults to `--split val`. This is D9's own to-do
(`REVIEW_SYNTHESIS.md:517`), unfixed, and it also breaks RQ3 arm parity: `no_distill` trains via
`train_teacher --student` on the 991 train episodes while the HID student gets all 1115.
**Fix:** add `--split` (default `train`) to both, `eps = C.manifest_split(root, args.split)`,
pass `episodes=eps`; keep `--extra-data` merging after that. One line each plus the arg.
**Land this before the $70 / 28 h student run.**

**14. The eval-campaign path never reseeds and its `seed` column is fictitious.** `[2×, conf 0.80–0.89]`
`phantom/eval/trial_runner.py:74,90`: `run_campaign`/`_run_one` never touch `policy.rf._gen`
(pinned to `manual_seed(0)` at `rf.py:84`) and pass `seed` only into the ledger row;
`DeploymentRuntime.run_episode` has no seed parameter, and `reset_episode()` only clears the held
noise. Two fresh processes produce a bit-identical first chunk. Meanwhile `aggregate.py:43-47`
clusters its bootstrap on that never-applied seed, and the runner is resumable — a restart
replays the same noise prefix. `run_eval.py:55-59` also builds `pol_args` without
`persistent_noise/guidance/parity_fixes/k_seeds`, so a campaign cannot run the rig arms'
controller at all (the arm-parity kill-shot).
**Fix:** move seeding into `run_episode(seed=...)` (set `_gen`, `reset_episode_noise()`, append
`seed:<n>` to tags); have `_run_one` pass `seed*1000+trial`; thread the deploy-lever flags
through `run_eval`'s `pol_args`.

**15. `run_campaign` is structurally arm-major — the interleaved A/B the plan requires cannot run.** `[2×, conf 0.93–0.95]`
`phantom/eval/trial_runner.py:67-79` makes `for system in cfg["systems"]` outermost and opens a
`DeploymentRuntime` inside it; `__enter__`/`__exit__` are `rig.connect_all()` + `MasterClock.calibrate`
+ `SensorSession.start()` / `stop()` + `disconnect_all()`, so an arm switch is a full camera +
both tactile processes + RTDE teardown. `run_episode` takes no `policy`/`mode` override
(`policy_name` is metadata only), and the campaign YAML has **no placement axis at all**. Executed:
all teacher trials, then all student, then all no_distill. D5/D11/D15/D16 all specify "arms
interleaved per placement"; run as coded, any drift in table height, gel zeroing, lighting or
object wear over a 3-hour session is perfectly confounded with the arm.
**Fix:** add `policy=None, mode=None, seed=None` overrides to `run_episode` (~6 lines — they only
shadow `self.policy` for `PlannerLoop` and `self.mode` for `SnapshotBuilder`), hoist the
`with DeploymentRuntime(...)` above the system loop (`run_eval.py:52-59` already builds every
arm's policy in one process), and reorder to task → placement → shuffled arms → trial.

**16. No statistics code exists, and the shipped CI is degenerate at exactly the observed outcome.** `[2×, conf 0.88]`
`phantom/eval/aggregate.py:32-56` `cell_stats` is the only inferential code in the repo — a
percentile bootstrap. `grep -rniE 'wilson|fisher|binomtest|proportion_confint|mcnemar'` over every
`*.py`: zero hits. `import scipy`: ModuleNotFoundError. Executed: 0/25 → `95% CI [0.000, 0.000]`;
25/25 → `[1.000, 1.000]`; 13/25 → `[0.320, 0.720]`. Wilson would give `[0.000, 0.133]`. The last
three rig sessions produced 0/n. Also, with the plan's single checkpoint per arm there is one
seed group, so the "95% bootstrap CI over seeds" header and the docstring are both wrong — it is
a trial-level bootstrap. D15/D16 mandate "Wilson CIs + Fisher exact + the paired per-placement
differences", none of which any script can produce.
**Fix:** add `phantom/eval/stats.py` — closed-form Wilson, Fisher exact via `math.comb` (no new
dependency), paired per-placement difference with McNemar / a bootstrap on the paired delta —
and call it from `cell_stats`/`write_report` in place of the percentile bootstrap.

**17. `grasp_label` scores truncation as failure, and reads the FIRST close on a veto retry.** `[2×, conf 0.72–0.80]` + `[unvoted]`
`phantom/eval/grasp_label.py:275-281`: `t_release` falls back to the end of the recording, and
`runtime.py:241-246` stops the recorder immediately after the planner loop with no settle —
`executor.stop()` joins rather than playing the tail out. A grasp on one of the last replans, or
an episode ended by `veto_retry_cap` (which by construction fires ~1 replan after a close),
always fails `hold_s >= 2.0` and is reported with the **same reason string** as a genuine
no-hold failure; `--confusion` counts it as an ordinary rule-negative. Separately, `:263` takes
`close_index(pos)` = the *first* close, so an episode whose **second** close grasped and lifted
after a veto reopen is scored on the vetoed first close — constructed and reproduced:
`grasp_ok=False` on a successful grasp. G2 ("Arm B ≥3/16 tactile-confirmed grasps") is measured
on exactly the arm that runs the veto.
**Fix:** keep the recorder running `HOLD_START_S + MIN_HOLD_S` after the planner loop; add a
`hold_truncated` reason excluded from the confusion table as *unmeasurable*; evaluate every
close plateau, not only the first, and report the best.

**18. Provisioning downloads the wrong init checkpoint and would overwrite the shipped v5 run.** `[2×, conf 0.72–0.76]` `[codex #1 corroborates]`
`tools/provision_v5.sh:205` is the only checkpoint fetch (`teacher_v4_790eps/teacher_020000.pt`)
and `:271` inits the printed FT-A launch from it, while the plan of record is "FT-A: 3k steps from
v5_6" = `teacher_v5_batch0822/teacher_003000.pt` (present on the hub, 0.39 GB, byte-identical to
compute3's `DEMO.pt`). The same line reuses `--run-name teacher_v5_batch0822`, and `train_teacher`
does **not** refuse a populated run dir — uploading that output to the same hub folder overwrites
the six shipped v5 checkpoints. A v4-init FT-A is a defensible objective-only ablation, but the
script and the plan disagree with no comment saying which is intended (**Mikhail decision #3**).
**Fix:** fetch `teacher_v5_batch0822/teacher_003000.pt`, point `--init-weights` at it, switch to
`--run-name teacher_v5_ftA`, keep the v4 path as a commented control.

**19. The provisioning pytest gate is red at HEAD — and the test is flaky, not order-dependent.** `[verified here]`
`provision_v5.sh:245` runs `python -m pytest tests/ -q` and treats failure as fatal, after the
~100 GB pull. Measured today: `tests/test_replay_rig.py::test_rebuilt_snapshot_matches_the_snapshot_builder`
fails ~1 run in 4 **on its own file** (4 runs: pass, fail, pass, pass) and 2 of 8 in the 3-file
combination — so the provisioning lens's "deterministic collection-order repro" is wrong; it is a
wall-clock race. The assertion is `test_replay_rig.py:122` `np.allclose(snap.wrist_window, dsnap.wrist_window)`
— the rebuilt wrist F/T window picks different ring rows when the fixture episode is generated at
a slightly different phase (snapshot ts differ by ~21 µs). It passes 6/6 with `-p no:randomly`,
which only masks it. Secondary: `set -euo pipefail` + a failing pipeline exits before the
`[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PYTEST FAILED …"; }` clause, so the gate's own error
message is unreachable.
**Fix:** make the test deterministic (rebuild at the recorded snapshot timestamp, or widen the
window-selection tolerance the same ±1-row slack the `ur_state` check already allows); change the
gate to `if ! python -m pytest tests/ -q; then echo …; exit 1; fi`.

**20. Checkpoint selection in the runbook still says `terminal_eval`.** `[2×, conf 0.78–0.83]` `[codex]`
`provision_v5.sh:283` (and `:292,:296`, and `docs/recovery_demos_protocol.md:55`) print "SELECT the
best checkpoint with tools/terminal_eval.py", while `REVIEW_SYNTHESIS.md:482` says "Score FT-A on
**replay**, not `terminal_eval`" and P1 explains that terminal_eval teacher-forces GT video/ur_state/
prev_chunk and structurally cannot see the rig failure. v5_6 was blessed exactly this way and then
went 0/26.
**Fix:** replace the printed selection note with explicit `tools/replay_rig.py` commands on
held-out deploy episodes, raw **and** EMA per checkpoint. Keep terminal_eval as a per-task val
diagnostic only. (Blocked on #10/#11/#12 — and the rental has no rig episodes: **stage them**.)

### P2 — costs a downstream round, or a paper claim

**21. `EpisodeMeta.weight` does not exist; D8's weighting is silently a no-op.** `[2×, conf 0.87]`
`phantom/data/schema.py:64-95` has 15 fields, none named `weight`, and `from_dict` filters to
`__dataclass_fields__` with no warning; `windows.py:423` emits only `0.0 if is_failure_demo else 1.0`.
`windows_per_episode` is a default-8 argument `train_teacher` never passes, so there is no ×3
oversampling knob. `phantom/dagger/manifest.py:14-19` writes per-episode `weight` rows that no
training path reads. `REVIEW_SYNTHESIS.md:506-509` describes D8 as if the mechanism exists; ~40
rollouts against 1115 demos = 3.5% of windows at weight 1.0, i.e. a statistical no-op.
**Fix:** add `weight: float = 1.0` to `EpisodeMeta`; multiply it into `w["action_weight"]` at
`windows.py:423`; emit `round(windows_per_episode * weight)` items per episode in
`WindowSampler.build_index` (or add `--windows-per-episode`); have `intake_recovery.manifest()`
write the weight it computes; add a test asserting `weight: 3.0` in meta.json reaches the batch.

**22. Nothing enforces that `rederive_rollout_actions` ran before a rollout trains.** `[2×, conf 0.85]`
`tools/intake_recovery.py:152` (`manifest()`) and `phantom/data/windows.py:196` (`build_index`)
both gate only on `is_trainable_episode`; neither looks for `actions_plan.zarr` or the
`actions_rederived` tag (grep: those symbols exist only in the tool itself and one test).
Reproduced: a non-rederived rollout is indexed at `action_weight 1.0` and manifested silently.
The premise holds — `executor.py:232-241` records `plan.actions[k]`, the raw pre-clamp proposal,
stamped on the governor-warped `_play_time` clock — so a D6 checklist run out of order trains on
commands the z floor and hitbox actually refused, at a cadence the sampler reads as the 10 Hz grid,
and D8 oversamples exactly that pool in the commit band.
**Fix:** in `manifest()`, refuse any episode with `policy not in ('', 'teleop')` lacking
`actions_plan.zarr`/the tag, with the same REFUSING message; mirror it as a warning in `build_index`.

**23. Re-derivation writes the MEASURED aperture into the gripper channel.** `[unvoted]`
`rederive_actions()` sets `out[k-1,6] = gripper.zarr[:,0]` = `GripperState.position`. A teleop demo
records `grip_cmd = pilot.last_sent`, the **commanded** aperture, and the executor's own
`prev_chunk` history stores the commanded `a[6]`. Measured on the fixture: re-derived
`{0.05, 0.75}` vs demo `{0.0, 1.0}`. On the rig the difference *is* the grasp outcome — a
command of 1.0 that closes on a waffle pack reads back as the object width; a close on air reads
~1.0. So rollout actions both leak the outcome into the action target and disagree with demos on
the single channel the terminal commit is about. `tests/test_small_fixes_0829.py:320` asserts only
`act[:, :6]`. Codex flags the same class of bug in `replay_rig --prev-chunk measured`
(`replay_rig.py:188-205` reads `gripper.zarr`, not the command stream).
**Fix:** take the gripper channel from the commanded source (`actions_plan.zarr` / executor
history) and only fall back to the measured position when no command stream exists; extend the
test to channel 6.

**24. `--seed` does not make a trial reproducible: the homing jitter is a separate unseeded RNG.** `[2×, conf 0.85–0.87]`
`phantom/scripts/run_deploy.py:553` `rng = np.random.default_rng()` (no seed) → `move_to_start` →
`sample_start_pose`, which jitters the homing target by `clip(randn(6),-1,1) * tcp_std`. Measured
waffles σ: ±26.9/±29.0/±27.1 mm; realised per-episode spread ~19–21 mm per axis; the unpaired
A-minus-B z difference has std ~27 mm and a mean 3D separation of ~45 mm. That is the same order
as the 65–120 mm terminal error and far larger than the 3–6 mm endpoint deltas checkpoint
decisions turn on. Homing is on by default. `--seed`'s help says "reproducible noise draws for
paired trials". The realised pose is recoverable post hoc from `ur_state[0]`, so the harm is lost
power plus missing provenance, not unanalysable data.
**Fix:** `np.random.default_rng(ep_seed)`; append the realised `tcp_target`/`grip_target` to
`ep_tags` or `meta.deploy_overrides`; or pass `jitter_sigma=0` for A/B sessions.

**25. `--event-band-weight 0` is not persisted and silently reverts on `--resume`.** `[2×, conf 0.82–0.88]`
`train_teacher.py:374` stashes it as a bare attribute on `pm.rf`, read via
`getattr(self,"event_band_weight",None)` at `rf.py:449`. `save_phantom_checkpoint` writes only
hardware_shapes/hardware_hash/backbone/model/train/text_conditioning; the shipped tiny FT-A
checkpoint reports `loss.event = 0.5`. The `--resume` guard at `:336-346` states in its own comment
that "nothing may change — not even the objective knobs" and enforces that via `model_config_drift`
on `mc` — which this knob is not in. A spot-instance kill at step ~1800 plus a resume without the
flag brings back the term at weight 0.5 (~4× the action gradient per the script's own note) for the
remaining 1200 steps, and nothing in the artifact records it.
**Fix:** put `event_band_weight` in `TeacherTrainConfig` (lands in `configs.train`) or apply it as
`dataclasses.replace(mc.loss, event=...)`; re-apply from the payload on `--resume`.

**26. `[codex]` `h100x8` validation runs on one shuffled, dropped rank shard; the profile does not enforce `torchrun`.**
`common.py:584` `make_loader()` hardcodes `DistributedSampler(shuffle=True, drop_last=True)` and
`train_loop()` (`:870`) only evaluates when `is_main()`, so `val_*` misses windows and drifts with
shard composition — the in-run health signal for checkpoint selection on a rented H100.
Separately `compute.py:83` `check_world()` only warns: selecting `h100x8` without `torchrun` trains
at effective batch 1 instead of 8 and silently burns the rental.
**Fix:** build val loaders with `shuffle=False, drop_last=False` and no `DistributedSampler` (or
all-reduce eval sums/counts); make `check_world()` fatal for cluster profiles.

**27. `terminal_eval` cannot load a student checkpoint; `--null tactile` is not the student contrast.** `[unvoted]` + `[codex #3]`
`tools/terminal_eval.py:269` passes `student=False` into `build_model`, whose `builder.py:48`
asserts `mc.student == student` — so every D9/D12 student evaluation raises. `replay_rig.py:262`
got this right. And `--null tactile` zeroes `gel`/`fields`/`contact_state` but leaves `reactive`
**live**, while the student path forces `g_react = 0` (`rf.py:213`, `acc.py:95`) — so "tactile
nulled" still benefits from a tactile-derived channel the student never gets, overstating the
student/pivot claim. The three zeroed streams are also three different kinds of null: normalised
`fields` zero = the **dataset-mean field**, `contact_state` = true zeros, `gel` = a black image
(an uncontacted gel is bright and textured).
**Fix:** `student=mc.student`; zero `reactive` in tactile mode (or rename the mode); denormalise
the zeroed `fields` with the checkpoint's `norm_stats` and check whether the implied `mask_frac`
exceeds `tau_contact_area` before quoting E9's "16%" again.

**28. `replay_deploy_path` — the E0 discriminator — is CUDA/2B-only, untested, and models the pre-fix deploy.** `[2×, conf 0.72–0.78]`
`tools/replay_deploy_path.py:57-61` hardcodes `device="cuda", tiny=False, k_seeds=1,
parity_fixes=False` with no CLI flags; `grep -rln replay_deploy_path tests/` returns nothing. Its
`--deploy-rng` (`:76-80`, `manual_seed(0)` + a `fake_obs` warm-up) reproduces the pre-`ba61354`
deploy while its help says it reproduces "the rig's ACTUAL noise". E0 is GATE G0.
**Fix:** add `--device`/`--tiny`, plumb `--parity-fixes`/`--k-seeds`, add one CPU tiny smoke test
mirroring `tests/test_replay_rig.py`, and split `--deploy-rng` into `legacy` (documented as
pre-2026-08-29 only) and `from-meta`.

**29. K-seed selection scores contact from seed 0.** `[unvoted]` + `[codex #3]`
`phantom/inference/policy.py:256-258`: `p_evt0 = pred.acc.p_evt[0]` — row 0 of the K-expanded batch
— decides whether the "reject the timid mode" branch runs at all, while the selected row is `j`.
Codex reproduced a K=2 case where a bold high-contact seed 0 lets the timid low-contact seed 1 be
selected (`k_rejected 0, k_pick 1`), i.e. the lever keeps exactly the mode it exists to reject.
**Fix:** pass the full per-seed `p_evt[:,0]` into `_select_seed` and apply the rejection per candidate.

**30. `recovery_open` preserves lateral transport and can self-cancel into `hitbox_exit`.** `[codex]`
`planner.py:386` zeroes upward z but leaves x/y/rx/ry/rz; Codex reproduced
`veto_action recovery_open` → `x_after_rewrite 0.03` → `safety stop ['workspace_clamp','hitbox_exit']`.
Safe, but the retry path aborts the episode instead of re-descending.
**Fix:** on `recovery_open`, clamp x/y/rot deltas to zero so the arm holds pose while it reopens.

### P3 — hygiene, provenance, tooling

31. **`planner_trace` drops the per-seed head descents** — `planner.py:496` keeps only scalar diag
    entries, so `_select_seed`'s `head_dz_mm` list (the only record of what the K seeds proposed, and
    the quantity `replay_rig`'s spread must be validated against) is discarded. `[unvoted]`
32. **`planner_trace` carries no TCP pose** (`planner.py:486-497`), so even "z-at-close" needs a
    clock-calibrated join against `arm_tcp_pose.zarr`. `[unvoted]`
33. **"Miss distance" — the plan's primary continuous number — has no data source.** It occurs once
    in the repo, in prose (`docs/rig_session_v5.md:37`); nothing records an object pose
    (`grep object_pose|aruco|april` → zero hits). `[unvoted]`
34. **`grasp_label` is wired to nothing but its own CLI** — `eval/aggregate.py` and `metrics.py`
    never see it, so the objective label gating G2/G3 never enters the eval report. `[unvoted]`
35. **`acc_lead_times` is uncapped and per-sensor** (`metrics.py:75-137`): a gate that simply latched
    high at t=0.9 s reports **14.1 s** of "anticipation" on a system with a 0.9 s replan period and a
    0.5 s lookahead, twice (once per sensor). `event_f1` is off by one and uses only `sensors[0]`. `[unvoted]`
36. **`hitbox_margin > z_floor_margin` is an unasserted invariant.** `run_deploy.py:454` applies the
    hitbox, `:472` raises the floor; at `--hitbox-margin 5 --z-floor-margin 10` the 08-28 bug returns
    (`STOP ['workspace_clamp','hitbox_exit']`), and a generous `--z-floor-margin 40` puts the floor
    below the lowest demo. Both knobs are advertised in the rig doc. `[unvoted]`
37. **`zfloor:` tag lies under `--no-z-floor`** (`run_deploy.py:524` tags the raw workspace bound);
    `deploy_overrides` is the only honest record. `[unvoted]`
38. **A checkpoint with no ACC head fails open**: `p_evt = zeros(5)` → `p_none = 0` → `p_contact = 1`,
    so the close-mask and the K-seed rejection silently become no-ops that log `close_allowed`. `[unvoted]`
39. **`URArm.move_j` has no servo guard and ignores moveJ's return** (`drivers/real/ur.py:341-344`)
    while `move_l` has both and `MockArm` raises — and `--home-joints` issues the moveJ **first**, so
    a mock dry run is strictly stricter than the rig. Known since the 08-28 review; unfixed. `[unvoted]`
40. **`holdout_sessions` sorts session names reverse-alphabetically** with no notion of time, so the
    newest deploy-rollout session — the whole on-policy pool D8 exists to consume — is held out as
    `val`; and an episode placed as a real directory under `tasks/<task>/` reports session `"<task>"`,
    collapsing all of them into one pseudo-session. `[unvoted]`
41. **`dagger_driver` never forwards `--hardware`** (`dagger_driver.py:55-63`), so `distill_hid` falls
    back to `configs/hardware.yaml`, which differs from the rig in a shape field
    (`wrist_window_len` 125 vs 31) — the shape assert fires **after** the full relabel pass is paid
    for. `dagger_driver` also has no `--tiny`/`--synthetic`, so it cannot be smoke-run at all. `[unvoted]`
42. **The relabel output is orphaned**: `precompute_relabels` (named in `distill_hid.py:17`) does not
    exist, `HIDConfig.teacher_mode`/`relabel_dir` are never read, `dagger/manifest.py`'s
    `demo_weight`/`rollout_weight` have no reader. `--extra-data` also double-counts any rollout
    already `place`d under `tasks/`. `[unvoted]`
43. **`replay_rig --tiny` silently ignores `--ckpt`** while writing it into the output JSON; `--merge-lora`
    is ignored under `--tiny` and absent from the documented E0 recipe even though `run_deploy` always
    folds the LoRA; `--jpeg-quality` hard-imports `cv2`, a rig-only extra, and the exception is not
    caught. `chunk_metrics.close_step` is a bare `grip > 0.45` while the comment claims the
    `close_index` rule. `[unvoted]`
44. **`Z_MAX_MM` is a hardcoded dict** whose documented regeneration path
    (`rig_trace_decompose.py:30-33`) implements only `close_index`'s primary rule and not the
    max-relative fallback Carton needs (coverage 76% → 99%), so the regeneration measures Carton's
    z_close on a biased subset — the task the docstring itself calls thin (143 vs 144 mm).
    `gen_start_poses.py` emits no `z_close` at all, so the two per-task geometry tables are
    maintained with no cross-check. `[unvoted]` + `[codex #7]`
45. **`rig_trace_decompose.at()` is `searchsorted` with no `side="right"-1`**, so it samples the first
    **future** row (measured +3.8 ms), biasing `xyz@close` / window dz / `grip_actual` forward; and
    once rollouts are rederived it will read the measured `actions.zarr` and call it "commanded". `[codex]`
46. **`terminal_eval` has no runnable smoke** (`--synthetic` checkpoints save empty `norm_stats` →
    `KeyError: 'action'`; synthetic episodes contain no gripper close), and the E1 verification the
    playbook prescribes names a script that does not exist (`scratchpad/research/gradprobe.py`) and
    crashes on a training-mode build unless `build_model(..., inference=True)`. `[unvoted]`
47. **`sequence.py:193` still claims video frames are "droppable at inference without a train/test
    attention mismatch"** — the P7 claim the review said to fix or delete, still in the code the paper
    quotes. `[unvoted]`
48. **No λ:=0 ACC inference ablation exists** (`grep lambda.zero|acc_lambda|--acc-off` → nothing), so
    the RQ2 attention-coupling claim is unmeasured; and D17's "tactile on/off at deploy" has no
    implementation (`--system drop_tactile` sets `student=True`, a different layout, and
    `run_deploy` loads without `allow_missing`). `[unvoted]`
49. **The pre-registration is stale**: `pipeline.md:389-395` and `configs/eval_campaign.example.yaml:4-11`
    still describe 5 systems × 5 tasks (fragile_grasp / slippery_place / insertion / wipe / regrasp —
    none exist) × 20 × ≥3 seeds ≈ 3,000 trials, occlusion variants and `recovery_ratio` over a
    `vision_only` denominator the plan deleted. Every eval module docstring cites it. `[unvoted]`
50. **`v5_6`'s headline 17.4 mm was measured under the old silent train+val fallback.** The hardened
    `terminal_eval` reports **20.86 mm** for the same v5_6/EMA on 78 val episodes;
    `docs/rig_session_v5.md:127` still says 17.4, and v4 was never re-run. Nothing in the paper or the
    plan may quote 17.4 → 20.7 → 14.9 until both arms are re-scored with the hardened tool. `[unvoted]`

---

## 3. Action items — the next two weeks

Effort is engineering hours. "Blocks" = the activity that cannot start until it lands.

### 3.1 FIX-NOW — code, ≤ 1 day total, zero spend  (do these before anything is launched)

| # | Item | File:line | h | Blocks |
|---|---|---|---|---|
| F1 | `tolerate=` on `assert_model_config_matches`/`load_phantom_checkpoint`; pass `FINETUNE_MUTABLE_MODEL_FIELDS` from the `--init-weights` branch only; CLI-level regression test | `train/common.py:264,298`; `train_teacher.py:316` | 1.0 | **FT-A** |
| F2 | Veto close detector → running minimum `g_min` in `veto_state` | `deploy/planner.py:403` | 0.5 | **rig 1** |
| F3 | Veto `closed_at` → store the replan index, fire only while `n − idx <= 1`, clear unconditionally; latch on the executed close, not plan acceptance; clear `plan.cpk` on any rewrite | `deploy/planner.py:379,471,485` | 1.5 | **rig 1** |
| F4 | Veto `at_floor` band → measure from `stats.tcp_z_min`, raise `z_margin` to the per-task demo close p95, expose as a flag | `deploy/planner.py:320,407` | 0.75 | **rig 1** |
| F5 | `--max-episode-s` (default 35) checked beside `max_replans`; log the cap stop reason | `deploy/planner.py:422,523` | 0.75 | **rig 1** |
| F6 | Release the gripper in `_halt` on `tactile_*`/`wrench_limit`/`hitbox_exit`/`veto_retry_cap` | `deploy/executor.py:131` | 0.75 | **rig 1 unattended** |
| F7 | Assert `hitbox.z[0] <= workspace.z[0]` after both `model_copy`s; fix the `zfloor:` tag under `--no-z-floor`; assert the deployed checkpoint has an ACC head | `scripts/run_deploy.py:454,472,524` | 0.75 | **rig 1** |
| F8 | Seed the homing RNG from `ep_seed`; tag the realised start pose | `scripts/run_deploy.py:553,585` | 0.5 | **rig 1 pairing** |
| F9 | Record `actions_pre_veto` in the trace row; keep the per-seed `head_dz_mm` list; log `tcp_pose` per replan | `deploy/planner.py:486,496` | 0.75 | **G0, rig 1** |
| F10 | `--split` (default `train`) + `manifest_split` on `distill_hid` and `finetune_hids` | `train/distill_hid.py:226`; `finetune_hids.py:174` | 0.5 | **student $70** |
| F11 | Persist `event_band_weight` into `configs.train`; re-apply on `--resume` | `train_teacher.py:374` | 0.5 | **FT-A resume** |
| F12 | `terminal_eval` `student=mc.student`; zero `reactive` in `--null tactile` (or rename the mode) | `tools/terminal_eval.py:107,269` | 0.5 | D9/D12 |
| F13 | `intake_recovery.manifest()` refuses a rollout with no `actions_plan.zarr`/`actions_rederived`; warn in `build_index` | `tools/intake_recovery.py:152`; `data/windows.py:196` | 0.75 | D6/D8 |
| F14 | Rewrite the Arm-A line in the rig doc (one process per arm block, or drop `--seed`) | `docs/rig_session_v5.md:71` | 0.25 | **rig 1** |
| F15 | De-flake `test_rebuilt_snapshot_matches_the_snapshot_builder`; make the provisioning pytest gate report its own failure | `tests/test_replay_rig.py:122`; `tools/provision_v5.sh:245` | 1.0 | **provisioning** |
|  | **Total** | | **10.75** | |

Follow-on same-week fixes, ranked immediately after the above (not "fix-now" only because they
exceed one day): F16 `replay_rig` parity mode + per-episode reseed + `--seed-from-meta` +
prefer `actions_pre_veto` + honour `--ckpt` under `--tiny` (4 h, blocks G0 and every checkpoint
comparison); F17 regenerate `configs/start_poses.yaml` over the full 250/task on compute3 +
`q_n == n` refusal (2 h, needs compute3, blocks counted rig episodes); F18 the E9 relabel +
`contact_zero` mode + re-run (1.5 h, blocks the P7 paper sentence); F19 `EpisodeMeta.weight` +
window multiplier (1.5 h, blocks D8); F20 gripper channel of `rederive_rollout_actions` (1 h).

### 3.2 RIG SESSION 1 — what to run and what to log

**Pre-conditions:** F2–F9 + F14 landed and unit-tested; F17 (envelope regeneration) landed —
`REVIEW_SYNTHESIS §1.11` makes it blocking, "before any rig episode is counted"; `GRIPPER_OPEN.sh`
within reach; someone present (not unattended) until F6 has been exercised once on the real gripper.

**Arms** (waffles first — `q_n=37`, the best-covered envelope; then Carton):

```
# arm A — v5_6 baseline, NFE 5, no levers.  ONE process for the whole block:
EXTRA=""                                                    ./GO_v5_waffles.sh 10
# arm B — the lever bundle:
EXTRA="--nfe 1 --terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 35" \
                                                            ./GO_v5_waffles.sh 10
```

Do **not** pass `--seed` per single-episode process (P0 #6). If a seeded arm is wanted, run the
whole block in one process; the unseeded path already records `seed:<n>` per episode.
Interleave A/B per placement cell on the taped 3×3 grid; ≥16 episodes per arm to make G2's 3/16
readable; alternate which arm goes first per cell.

**Log per episode:** the placement cell id in the verdict prompt; operator verdict `s`/`f`/`c`
(+`d` for damage); `meta.tags` (all levers: `nfe`, `veto:`, `parity:`, `kseeds:`, `zfloor:`,
`hitbox:`, `vmax:`, `seed:`, `ckpt:`, `git:`); `planner_trace.json` with `actions_pre_veto`,
per-seed `head_dz_mm` and the per-replan `tcp_pose` (F9); the realised start pose (F8);
`tools/label_grasps.py --confusion` immediately after the session.

**Gates.** G0 (replay fidelity: trace inside the seed spread on ≥4 of 6 episodes) — **only
meaningful after F16**; do not compute it with the current `replay_rig`. G2 (Arm B ≥3/16
tactile-confirmed grasps) — read `grasp_ok` *and* the operator verdict, and treat any
`hold_truncated` episode as unmeasurable, not as a negative.

**Abort rules.** Two consecutive `safety_stop`s needing an RTDE control rebuild → stop and inspect
the envelope. Any protective stop or manual jog → re-run the joint gate before the next episode
(the 08-28 session lost 16 of 26 episodes to a wrist wrapped 360° / a flipped IK branch that the
TCP-only gate passed). Any `veto_retry_cap` → check whether the gripper is still commanded closed
(F6) before touching the rig.

**Expected cost:** ~3 h of rig time for 2 tasks × 2 arms × 16 episodes plus resets.

### 3.3 FT-A LAUNCH — exact sequence (~6.5 H100-h, ~$16–40)

Pre-conditions: F1, F11, F15 landed and pushed; `pack_repo.sh` re-run and the tarball re-uploaded;
Mikhail's decision on the init checkpoint (§3.5 #3). Sequence assumes **from v5_6**.

```bash
# --- Mac ---
cd ~/GitHub/phantom && .venv/bin/python -m pytest tests/ -q     # must be green, no --deselect
bash tools/pack_repo.sh /tmp/pack
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/tmp/pack/phantom_repo_latest.tar.gz",
  path_in_repo="dataset_v3_packed/phantom_repo_latest.tar.gz",
  repo_id="armteam/phantom-checkpoints", repo_type="model")
PY
scp /tmp/pack/provision_v5.<sha>.sh root@<vast>:~/      # the pinned script is NOT on the hub

# --- rental: H100 NVL 80GB, --disk 300 ---
export HF_TOKEN=hf_...                                   # rotate it first (§3.5 #8)
bash ~/provision_v5.<sha>.sh /workspace/phantom-v5       # ~15 min + ~100 GB
W=/workspace/phantom-v5

# 1. fetch the FT-A init checkpoint the script does not (P0 #18)
$W/.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil, os
p = hf_hub_download("armteam/phantom-checkpoints",
                    "teacher_v5_batch0822/teacher_003000.pt", repo_type="model")
d = "/workspace/phantom-v5/runs/teacher/teacher_v5_batch0822"; os.makedirs(d, exist_ok=True)
shutil.copy(p, d + "/teacher_003000.pt")
PY

# 2. smoke the REAL launch flags — 2 steps, the whole FT-A bundle (the shipped smoke omits it)
cd $W/phantom && $W/.venv/bin/python -m phantom.train.train_teacher \
  --data $W/data/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml \
  --run-name ftA_smoke --max-steps 2 \
  --init-weights $W/runs/teacher/teacher_v5_batch0822/teacher_003000.pt \
  --grasp-frac 0.3 --photo-aug 1.0 --acc-two-pass \
  --lr 2e-5 --lr-new-modules 6e-5 --warmup-steps 150 \
  --event-band-weight 0 --ema-decay 0.995 --cond-dropout 0 \
  --contact-nll-beta 0.5 --contact-self-forcing \
  --action-noise-per-strip --no-action-t-max-of-two \
  --batch-size 4 --grad-accum 2 --num-workers 8 --device cuda
# ^ MUST reach "step 2/2". Today, without F1, it raises RuntimeError model-config drift.
#   Note: --allow-config-drift deliberately dropped; if it is needed, find out why first.

# 3. launch — 3000 steps, ~5 h on H100 NVL, effective batch 8
unset HF_TOKEN
nohup $W/.venv/bin/python -m phantom.train.train_teacher \
  ... identical flags ... --run-name teacher_v5_ftA --max-steps 3000 \
  --ckpt-every 500 --eval-every 500 > train_ftA.log 2>&1 &
#   --run-name teacher_v5_ftA, NOT teacher_v5_batch0822 (that folder is the shipped v5 lineage)

# 4. EGRESS after every checkpoint — nothing in the repo does this; write it before you launch
$W/.venv/bin/python - <<'PY'
from huggingface_hub import HfApi; import sys
HfApi().upload_file(path_or_fileobj=sys.argv[1],
  path_in_repo="teacher_v5_ftA/" + sys.argv[1].rsplit("/",1)[-1],
  repo_id="armteam/phantom-checkpoints", repo_type="model")
PY

# 5. scoring: replay, not terminal_eval (REVIEW_SYNTHESIS:482) — needs rig episodes on the box.
#    scp the 08-28 rig session from compute3/NUC to $W/data/rig_0828/ BEFORE the run finishes,
#    then, per checkpoint, raw AND EMA:
$W/.venv/bin/python tools/replay_rig.py --ckpt <ckpt> --hardware configs/hardware.nuc.yaml \
    --episodes $W/data/rig_0828/ep_* --seed-from-meta --parity-fixes --merge-lora --out te.json
#    terminal_eval stays a per-task val diagnostic only.
```

If `h100x8` is ever selected, launch under `torchrun --nproc_per_node 8` and treat the in-run
`val_*` numbers as unreliable until Codex item #26 is fixed.

### 3.4 PAPER EXPERIMENTS — effort and dependency

| # | Item | Effort | Spend | Depends on |
|---|---|---|---|---|
| E1 | `phantom/eval/stats.py`: Wilson, Fisher (`math.comb`), McNemar / paired per-placement delta + bootstrap; replace `cell_stats`'s percentile bootstrap; fix the "over seeds" label | 3 h | — | — |
| E2 | `run_episode(policy=, mode=, seed=)` overrides; hoist the runtime above the arm loop; reorder `run_campaign` to placement-major → arm-interleaved; plumb the seed and tag it | 3 h | — | — |
| E3 | `tools/gen_ab_schedule.py`: pre-registered CSV (placement cell, arm order, per-episode seed, task) from a fixed RNG; `trial_runner` consumes it | 3 h | — | E2 |
| E4 | Ledger fields: `placement`, `seed`, `arm_tags`, `grasp_ok`, `z_close_mm`; thread the deploy levers through `run_eval`'s `pol_args` so a campaign runs the rig controller | 2 h | — | E2 |
| E5 | Join `grasp_label.label_episode` into `metrics.trial_metrics`; emit `grasp_ok`, `z_close_mm`, `c_hold`, `lift_mm`, `stall` + the rule-vs-operator confusion per arm | 2 h | — | F-block |
| E6 | Rewrite the headline: drop `recovery_ratio` (its `vision_only` denominator was deleted), report three raw rates + paired deltas with CIs and Fisher p | 2 h | — | E1 |
| E7 | Define the continuous primary: `commit_height_mm` from the new per-replan `tcp_pose`; then either record a per-episode placement cell id so **miss distance** is defined, or delete it from the plan and the paper | 3 h | — | F9 |
| E8 | Baselines: write the `no_distill` and `vision_only` launch lines into the playbook (today it has exactly one training command) and state the shared dataset + shared deploy controller as an arm-parity pre-registration | 1 h | — | F10 |
| E9 | `no_distill` insurance run | ~35 h wall | **$90** | F10, E8 |
| E10 | HID student run (D9) | ~28 h wall | **$70** | F10 (val leak) |
| E11 | Ablations: cap and de-duplicate `acc_lead_times`, fix the `event_f1` off-by-one and use all sensors — or demote RQ2 to a diagnostic | 2 h | — | — |
| E12 | Deploy-time tactile null (mirror `terminal_eval`'s null into `SnapshotBuilder`) so D17's tactile arm runs on the teacher checkpoint — or drop the arm; add a λ:=0 ACC knob for RQ2 | 3 h | — | — |
| E13 | Re-score v4 and v5_6 with the hardened `terminal_eval` on the real 78-episode val split, and stop quoting 17.4 mm | 1 h + GPU | — | — |
| E14 | Rewrite `pipeline.md:358-397` and `configs/eval_campaign.example.yaml` to the actual protocol (4 real tasks, 3 arms, 25 trials, no occlusion, no fragility instrumentation) — move it **before** D9, not D13 | 2 h | — | E6 |
| E15 | Regression tests for `trial_runner` / `aggregate` / `metrics` (mock-driver campaign smoke + a stats golden file) — today all three are untested | 2 h | — | E2, E1 |
|  | **Off-rig total** | **≈ 29 h** | **$160** | |

### 3.5 MIKHAIL-ONLY DECISIONS

1. **Arm A design.** One process for the whole arm block (keeps `--seed` meaningful, loses
   per-episode interleaving granularity) **vs** drop `--seed` entirely (fresh draw per episode,
   already recorded). This changes what "paired" means in the A/B. — *needed before rig 1.*
2. **Unattended rig.** Ilya/Petr run session 1 alone, or you present? Recommendation: **not
   unattended** until F6 (gripper release on stop) has been exercised once on the real gripper —
   today every safety stop leaves the fingers squeezing the gels at the pad ceiling.
3. **FT-A init: v5_6 (plan of record) or v4 (script as shipped).** v4-init is a cleaner
   objective-only ablation at equal steps; v5_6-init is what D4 gates on. They are different
   experiments and the artifacts are not comparable. — *needed before the rental.*
4. **The 17.4 mm number.** v5_6's headline was measured under the old silent train+val fallback;
   the hardened tool gives 20.86 mm on the real 78-episode val split and v4 was never re-run.
   Re-score both arms (E13, ~1 h + GPU) or strike the number from the docs — it cannot stay as is.
5. **E9 / P7.** Re-run the premise table with a `contact_zero` mode before any P7 sentence is
   written, or drop the exposure-bias claim from the paper. The current row argues the **opposite**
   of what it is quoted for, including for `--contact-self-forcing` in FT-A.
6. **Miss distance.** Instrument it (one taped placement-cell id per episode + a `commit_height_mm`
   from the trace) or delete it from the plan and the paper. Nothing records an object pose today.
7. **$160 of runs (no_distill $90 + student $70).** Both are gated on F10 (the val leak). Launch
   after F10, or accept that RQ3's headline is scored on training data.
8. **Rotate the HF token** before the next rental — it is exported into the provisioning
   environment, and the memory file has flagged this since 08-27.
9. **Scope.** ~29 h of off-rig work stands between the rig hours and a paper table. Accept the
   delay, or cut: drop occlusion, drop the 5-task pre-registration, drop RQ2 to a diagnostic.
10. **Re-run the four missing Codex passes** (deploy-recipe, rollout-intake, protocol-paper,
    regression-p1-p10) after the quota reset, with a sandbox mode that permits `TMPDIR` writes —
    the four that did complete produced 4 independent HIGH findings each, so the coverage gap is real.

---

## 4. Refuted findings

**The two-voter round refuted nothing: all 20 findings that went to a vote survived** (confidence
0.72–0.96), and the only downgrades were in scope, not in substance:

- *"the veto never fires"* — precise version: it fires on 2 of 6 recorded waffles episodes with the
  current rule vs 4 of 6 with the running-min rule; the recovery arm is still unreachable.
- *"the FT-A crash needs the whole bundle"* — refined: `--cond-dropout 0` alone is enough, and
  `action_noise_per_strip` is absent from both real saved configs, so only two fields actually drift.
- *"replay_rig's `prev_chunk` gap is unfixable today"* — `--prev-chunk measured` is a partial,
  default-off workaround for that one construction (and Codex shows it reads the wrong gripper source).
- *"a checkpoint A/B can flip on `--episodes` order"* — weaker than stated: a same-list A/B is
  noise-paired; the order dependence bites the per-episode G0 statistic.
- *"the trial-runner seed defect is new"* — it is the already-confirmed §1.11 item, narrowed to the
  one file that still has it.

Claims that the lenses themselves checked and **dismissed** (i.e. suspected problems that turned out
not to be problems), one line each:

- The z floor is a **clamp**, not a stop — `safety.py:211` evaluates the hitbox on `clamp_target`; a deep descent is clamped at 42 mm with `stop=None`. Fixed as claimed.
- The veto's interaction with `--parity-fixes` `prev_chunk` is **correct**: the executor's `_grip_hist` returns the executed 0.30 hold, not the policy's 0.80 proposal — which is what training feeds.
- The veto can **not** manufacture a `hitbox_exit` by lowering z (the clamp pins it above the hitbox's lower edge) — though Codex shows the *lateral* channel can (P2 #30).
- β-NLL is **not** broken: the 89%-contact LoRA gradient share at random init is a σ≈1 artifact; pinned to the measured v4/v5 regime, β=0.5 takes the contact/action ratio from ~35× to ~2.9×. P5 is genuinely fixed.
- `--event-band-weight 0` has **no truthiness bug** (`w.event if event_band_weight is None else float(...)`).
- The P9 gates **hold at every layer**: unlabeled/contaminated/unjudged rollouts are refused by `is_trainable_episode`, `manifest_split` and `build_index` independently; verified live.
- The base-hardware-hash stamp **works**: safety overrides change the hash but no shape-relevant field, so rollouts pass `train_teacher`'s CONFIG DRIFT check.
- `rederive_rollout_actions` is **numerically exact** on channels 0–5: integrating the re-derived Δ-xyz reproduces the measured TCP displacement to 0.0 mm; it is idempotent.
- The hub tarball **is** HEAD: `phantom_repo_latest.tar.gz` == `git archive HEAD` + `COMMIT`, and `pack_repo.sh` is byte-reproducible (same sha256).
- The intake **does** reproduce 1115 / 991 / 124 from the hub's own 325 metas via the repo's `holdout_sessions`; intake is idempotent.
- The per-episode seeding fix in `run_deploy` **is** live and device-portable: same seed → bit-identical draws, different seeds → max|diff| 1.80, unseeded → 4 unique seeds.
- `EXTRA` append order **is** last-wins, as the rig doc claims.
- The speed cap, joint/full-turn gate, `gripper_ctl`, meta-tag provenance and `deploy_overrides` all do exactly what they claim.
- P10A/P10B(B) are **fixed**: `--student --mask-wrist` builds and round-trips; every `build_model` call site derives `mc` from the checkpoint.
- No new deferral markers were introduced (`git diff 3539988..HEAD | grep -i 'deferred|todo|fixme'` → 2 prose hits); `571 passed` with `-p no:randomly`.
- The provisioning **pytest failure is not order-dependent** — measured here as a genuine flake (1 in 4 runs of the file alone), which corrects the provisioning lens's own characterization.

---

## Completeness critic

Ten gaps between what this validation exercised and what `REVIEW_SYNTHESIS §3` depends on.
Numbers in parentheses are the review's own completeness-critic bullets.

1. **No perception lens — extrinsics/lighting drift was never exercised (critic 1).** All seven
   lenses ran mock drivers and `--tiny` on the Mac; nothing compared a deploy cond frame against a
   demo cond frame. The review calls a re-aimed RealSense or the 08-18/20 lighting change a
   *sufficient standalone cause* of a terminal-phase height failure, and it would make every
   training-side result at G1 uninterpretable — so FT-A, the veto band and the whole D5 A/B rest on
   an unchecked premise. → Run the 1-minute check before rig 1: first cond frame of each 08-28
   deploy episode vs a same-task demo at the gated start pose (gripper pixel bbox + background
   SSIM); recalibrate or re-collect if it moved, and add it to the rig checklist.

2. **Gel baseline drift is nobody's lens, and the damage field still does not exist (critic 2).**
   No lens diffed per-session `fields_ds` depth / `mask_frac` baselines (demos through 08-22 vs the
   08-20/08-28 rollouts), yet `mask_frac` is the auto-label behind G2/G3 and every success number in
   the paper. And §3.2's operator verdict `+d for damage` has nowhere to land: `run_deploy.py:470-482`
   has no damage field, so D15's "damage % on egg" is a second metric with no data source (same
   class as #33 miss distance). → Add the `fields_ds` baseline diff to the offline day, a gel-zero +
   press-check to the rig checklist, and a `damage` field to `run_deploy` inside F7.

3. **The P8 success rule still has zero validated positive class (critic 3).** D2 called for
   validating the rule on all 1115 demos + the 39 rig rollouts; this validation found labeller bugs
   (#17, #34) but never re-ran the rule, and there are still no successful policy rollouts, so
   `c_hold ≥ 0.8` / 2 s hold / per-task `Z_MAX` have never seen a true positive the policy produced —
   while §3.2 keeps `grasp_ok` as a G2 input. → Re-run the 1115-demo + 39-rollout sweep on the fixed
   labeller before rig 1, and mark G2 provisional until ≥10 operator-confirmed policy grasps exist.

4. **Arm B is now a five-way bundle — the attribution problem got worse, not better (critic 5).**
   The critic asked D5 to be split into veto-only vs veto+parity+K-seed; §3.2's Arm B is
   `--nfe 1 --terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 35`, and no parity or K-seed
   ablation appears anywhere in §3.4. Whatever G2 says, the paper cannot name what made it grasp.
   → Split the 32 episodes into A (v5_6), B1 (veto only), B2 (veto+parity+K-seed) and pre-register
   which single intervention the abstract claims.

5. **K-seed × NFE was never profiled on the deploy machine (critic 6).** P0 #4's latency numbers
   (865 ms / 172 ms) are *stubbed on the Mac*; the review made NUC profiling a gate on the K-seed
   lever, and `--k-seeds 4` is now in the recommended arm without one — while `--max-episode-s 35`
   silently converts any latency surprise into truncated episodes that #17 then scores as failures.
   → Profile K × NFE on the rig NUC before the session; fall back to K=2–3 or score every other
   replan if the 0.9 s replan budget is missed, and set `--max-episode-s` from the measured number.

6. **E9 — the test of the paper's own premise — still feeds no gate, and the label space now
   collides (critic 8, and critic 7 untouched).** This validation shows the E9 table is read
   backwards (#9) *and* that `--null tactile` is not the student contrast (#27), yet the pivot
   threshold the review asked for (G1b: |Δ endpoint| < 3 mm, |Δ close_step| < 1 step ⇒ pivot to the
   analysis paper) appears nowhere; it is only Mikhail decision #5. Separately, the review's E7
   (guidance) and E8 (prefix inpainting) were never dispositioned — and §3.4 reuses the labels E7/E8/E9
   for *different* items, so "E9" now means both the premise test and the $90 `no_distill` run.
   → Add G1b with those numeric thresholds ahead of any run spend, explicitly cut or schedule the
   review's E7/E8, and renumber §3.4 (X1–X15) to stop the collision.

7. **The $160 of runs lost its success gate (critic 8/11).** The plan authorises the student only
   through G3 (teacher ≥40% on waffles centre over ≥10 interleaved episodes); §3.5 #7 re-gates both
   runs on F10 alone, and the strings G1, G3, G4 and G5 do not occur anywhere in this document —
   only G0 and G2 survived. The plan's kill-switches have been quietly dropped. → Restate E9/E10's
   pre-condition as "F10 landed **and** G3 passed", and carry G1/G1b/G3–G5 into §3 verbatim.

8. **Nothing in §3 has an owner, and two rig-blocking fixes sit below the fix-now line.** 10.75 h of
   F-work plus ~29 h of E-work are unassigned while §3.5 #2 asks who even runs the session; F17
   (regenerate `start_poses.yaml` over the full 250 — `§1.11`: "before any rig episode is counted")
   is a §3.2 pre-condition but lives in the follow-on list and needs compute3; F16 is likewise
   follow-on although §3.2 declares G0 meaningless without it. Also, "571 passed" holds only under
   `-p no:randomly`, i.e. the suite is green with the flake masked. → Put a name on every F/E item,
   promote F16/F17 into the FIX-NOW block with a hard "no episode counted until F17 lands" check,
   and require one green default-ordered `pytest` run before the tarball is re-uploaded.

9. **Rig hours and GPU dollars are still priced in successes, not attempts (critic 9/10).** §3.2
   books ~3 h for 64 episodes; at the plan's own 3.5 min that is 3.7 h before the last session's
   16/26 invalid rate, i.e. ~5.2 h against a 15–20 h total. §3.3 gives no per-instance ledger, no
   concurrency plan for D9's 35 h + 28 h pair, and no resume smoke — even though #25 proves a resume
   silently restores `event_band_weight 0.5`. → Re-price rig work in attempts at ≥1.4×, adopt §4.5
   (waffles only, 3 arms × 25) as the plan of record, and insert a kill-and-resume smoke between
   steps 2 and 3 of the FT-A sequence that asserts the objective knobs survive.

10. **No non-WAM baseline, and three of the D1 safety items are still missing (critic 12).** The
    review asked for one ACT/diffusion-policy baseline trained free on compute3 and reported at least
    on replay — the first question an ICRA reviewer asks of a 2B video-DiT teacher; it appears
    nowhere in this validation or in §3.4, so all three arms remain the same backbone. It also asked
    for force-abort + a protective-stop/E-stop watch + a *written* RTDE handover procedure for D7's
    auto-takeover; only the veto retry cap is confirmed, and "takeover", "handover" and "E-stop"
    occur zero times here. Without the handover procedure, D7 records the takeover transient as
    supervised demo action. → Add the compute3 ACT/DP baseline to §3.4, and land the protective-stop
    watch plus the written handover procedure before D7 collects a single auto-triggered recovery demo.
