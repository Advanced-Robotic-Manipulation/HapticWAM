# PHANTOM RE-VALIDATION — HEAD `820b4eb`, 2026-08-31

Lead-engineer synthesis of 7 Opus lenses (each executing code on the Mac with `.venv/bin/python`,
mock drivers, `--tiny`, plus read-only probes of compute3) + 5 completed Codex passes — the four
that were quota-blocked on 08-30 all ran this time. 32 findings went to a two-voter adversarial
round: **14 survived (11 unique after de-duplication), 18 were refuted or downgraded to
non-blocking.** Compare: 08-30 put 20 findings to a vote and refuted zero.

Baseline that everything below sits on: `pytest tests/ -q` → **661 passed** in default
(pytest-randomly) order, reproduced independently by three lenses.

---

## 1. SAFE-TO-SPEND

| Spend | Verdict | Conditions |
|---|---|---|
| **RIG SESSION 1** (Session 4 / D5 A-B, 4 rig hours, 2 operators) | **GO-WITH-CONDITIONS** | Seven, ~2 h of edits + one 5-min bench measurement. See C1-C7. |
| **FT-A RENTAL** (~6.5 H100-h, $16-40) | **GO-WITH-CONDITIONS** | Four, ~1.5 h, all before the instance is rented. See C8-C11. |
| **BASELINE + STUDENT RENTALS** (E9 `no_distill` $90 + E10 HID student $70 = **$160**, ~63 h wall) | **NO-GO** | Blocked on ~6 h of off-rig work *and* a completed FT-A. See N1-N6. |

### RIG SESSION 1 — GO-WITH-CONDITIONS

The recipe runs end to end. F5/F6/F7/F9/F17 are demonstrably live: `run_deploy --nfe 1
--terminal-veto --parity-fixes --k-seeds 4` exits 0, tags `parity:on veto:pc0.50/pn0.90/r3
kseeds:4 seed:<n> zfloor:32mm hitbox:60mm`, writes `tcp_pose` + `diag.head_dz_mm` + `k_pick` on
every trace row and `actions_pre_veto` on exactly the rewritten rows; the joint gate refuses
22/26 of the 08-28 starts (all 16 wrist-wrapped ones at 31-73σ); the executor shows zero
commanded-direction reversals at every replan period down to 60 ms and holds the speed cap to
4.7%. That is a genuinely different machine from the one that produced 0/26 on 08-28.

What is not ready is the **command line and the operator page**.

* **C1 (blocking, 1 line).** Arm B's `EXTRA` must gain `--max-replans 200`. Without it every
  arm-B episode ends at `replan_cap` after **7.2 s** against a 35 s arm A and 16-31 s demos —
  the A/B compares two different experiments and the terminal phase is never reached.
* **C2 (blocking for *unattended*, ~30 min).** `executor.py:163` issues the gripper release
  before invalidating the mailbox, and `executor.py:303` skips the release entirely on a UR
  protective stop. Fix both, or run the session **attended** with `./GRIPPER_OPEN.sh` at hand.
  This is Mikhail decision §3.5 #2 and it is still open.
* **C3 (blocking for arm B's interpretability, 4 lines).** Instrument or guard the terminal
  veto — see §2 #3. At minimum, print the per-episode `p_none` histogram in the preflight so
  the session produces the calibration data the veto's thresholds were never fitted on.
* **C4 (blocking for post-hoc analysis, ~15 min).** Persist the stop reason. `stopped_reason`
  exists only on `EpisodeResult` and a console line; the GO-script path writes nothing to
  `meta.json`, so after the session nothing on disk separates `replan_cap` /
  `episode_time_cap` / `safety_stop` / `veto_retry_cap` — which the abort rules, the
  unmeasurable-vs-failure call and the per-arm truncation rate all turn on.
* **C5 (doc, 20 min).** `docs/rig_session_v5.md:154-155` (retired 20.7→17.4 / 20.7→14.9) and
  `:119` (old z-floors) must be corrected before the operator reads them mid-session.
* **C6 (bench, 5 min, operator).** Jog to fingertip contact at the waffles station and read
  `tcp_pose[2]`. The waffles clamp floor is 31.5 mm and nothing in the repo records a measured
  table height. If you will not measure it, run waffles with `--z-floor 0.0465`.
* **C7 (accept, no work).** G0 will be unreadable on arm B (§2 #4) and `--seed-from-meta`
  will not reproduce the executed K-seed chunk (§2 #9). Both are post-session analysis
  defects, not session defects: the recorded numbers are correct, only the flags are wrong.
  Fix them while the session runs.

### FT-A RENTAL — GO-WITH-CONDITIONS

The code is ready and this is the strongest positive result of the round: on the **real shipped
v5_6 payload** (`teacher_v5_batch0822/teacher_003000.pt`, pulled from the hub), the documented
bundle gives `model_config_drift → HARD: {}`, `assert_model_config_matches(tolerate=FINETUNE_MUTABLE)`
PASSES, and the `--init-weights` norm-stats gate matches the hub's own `norm_stats.json` key for
key. F1 and F11 land. The drift matrix is exactly as wide as `FINETUNE_MUTABLE_MODEL_FIELDS` and
no wider. The launch will get past both startup gates. The **artifacts** are not ready.

* **C8 (blocking, 10 min).** Delete `--contact-self-forcing` from `tools/provision_v5.sh:275`
  (smoke) and `:295` (printed launch line), rewrite `:317`, and update
  `tests/test_ft_a_fixes.py:296` + `tests/test_fixnow_train_0830.py:623` to assert its
  **absence**. Three lenses and one Codex pass found this independently.
* **C9 (blocking, 5 min).** Re-run `tools/pack_repo.sh` **and upload the tarball**. The hub
  artifact is still `d9c40f2` — the commit 08-30 declared NOT READY. The COMMIT pin proves
  script↔tarball, never tarball↔HEAD. This aborts fail-safe ~30 s in if you forget, so it is
  a checklist item rather than a hazard, but it is currently wrong.
* **C10 (blocking, ~30 min).** Either de-flake `test_rebuilt_snapshot_matches_the_snapshot_builder`
  or accept a 1-in-6-to-1-in-20 re-provision. `provision_v5.sh:256` is fatal on any red test,
  after the ~100 GB pull, and the FT-A precondition forbids `--deselect`.
* **C11 (blocking if you plan to resume, ~45 min).** A `--resume` restores only
  `event_band_weight`. `grasp_frac`, `photo_aug`, `commit_band_weight` and `split` are not even
  in `configs.train`; `ema_decay` genuinely reverts. A spot-kill restart silently drops
  `--grasp-frac 0.3` — the entire point of FT-A — and stamps the reverted values into the
  final checkpoint. If you will not fix it, do not resume: restart from scratch, and change
  `--run-name` when you do (see §2 #8).

Also worth knowing before you rent: `--allow-config-drift` **is** needed (the shipped v5 run's
own log shows `991/991 episodes recorded under a DIFFERENT hardware config`, pre-existing, not
fix-induced), and the printed checkpoint-selection command is not executable as written —
`replay_rig` has no raw/EMA switch, so the prescribed RAW half of the comparison cannot be run.

### BASELINE + STUDENT RENTALS — NO-GO

Both arms *build and train* (`--student --split train` and `--student --mask-wrist --split train`
both exit 0 with the right log lines; F10 closed the val leak on all three programs). The money
is still not safe to spend, because nothing can score what it buys:

* **N1.** Neither launch line exists anywhere. `docs/training_playbook.md` has exactly one
  training command (FT-A). No step budget, no init checkpoint, no dataset pin, no arm-parity
  pre-registration. E8, 1 h, unwritten.
* **N2.** The headline metric these two runs exist to feed is `recovery_ratio =
  (student − vision_only)/(teacher − vision_only)`, whose `vision_only` denominator arm the plan
  deleted. Run on a plan-of-record session it evaluates to **`nan`**. E6, 0.5 h, unwritten.
* **N3.** There is no statistics code at all — `wilson|fisher|binomtest|proportion_confint|mcnemar`
  over `*.py` returns **0 hits**, `scipy` is not installed, `phantom/eval/stats.py` does not
  exist, and `aggregate.py` is byte-unchanged since `d9c40f2`. E1, 3 h.
* **N4.** `--resume` rewrites six of seven knobs (C11) and `upload_run_ckpts` skips by byte size
  only (§2 #8) — every teacher checkpoint is exactly 393,115,861 bytes. A preempted 35 h run
  that is relaunched under the same `--run-name` silently leaves attempt-1 weights on the hub
  under the name the selection step and the rig then trust. On a $90 run that is the whole run.
* **N5.** FT-A has not been run. These arms are supposed to be parity-matched against a lineage
  that does not exist yet.
* **N6.** Codex's protocol pass adds that the headline tables are not tactile-labeled at all
  (`aggregate` reads only the operator's y/n; `grasp_ok` never reaches it) and that the shipped
  CIs bootstrap over `seed`, which `trial_runner` never actually varies — pseudoreplication.

Unblock cost: N1+N2+N3 = 4.5 h, N4 = 1 h, plus FT-A. Then re-verdict.

---

## 2. CONFIRMED PROBLEMS, RANKED

Rank = (voids a scheduled spend or rig hour) → (corrupts a gate) → (invalidates a quoted
number) → (hygiene). File:line at `820b4eb`.

### MUST FIX BEFORE THE CORRESPONDING LAUNCH

**1. Arm B is a 7.2 s episode — `--max-replans` is never raised.** HIGH · rig session
`docs/rig_session_v5.md:74` · `phantom/scripts/run_deploy.py:198` · `phantom/deploy/planner.py:634`
`--max-replans` defaults to 40 and `PlannerLoop.run` checks the **count** cap before the
wall-clock budget, so `--max-episode-s 35` in the documented arm-B line is dead. Measured
through the real loop at the team's own latencies: arm B (`--nfe 1`, 172 ms) → 40 replans /
**7.2 s** / `stop='replan_cap'`; arm A (`--nfe 5`, 865 ms) → 40 replans / 34.8 s. Only
`--max-replans 200` reaches `episode_time_cap` at 35.0 s. `GO_ANY.sh` on compute3 (read live)
passes no `--max-replans`; `$EXTRA` is the sole injection point. This is VALIDATION_0830 P0 #4
verbatim: F5 shipped the code, nobody wrote the stopgap into the recipe.
**Fix:** `EXTRA="--nfe 1 --terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 35 --max-replans 200"`.
Better: in `run_deploy.main`, set `args.max_replans = None` when `--max-episode-s > 0` and the
user did not pass `--max-replans` explicitly.

**2. `provision_v5.sh` spends the paid FT-A run on the retracted `--contact-self-forcing`.**
HIGH · FT-A rental · `tools/provision_v5.sh:275,295,317`
`e06c33b` removed the flag from the recommended bundle (`docs/training_playbook.md:89`: *"do
not spend the primary FT-A run on it"*; `E9_premise_test.md:88-93`: *"E9 does not support it"*),
but that commit touched only the doc. The script the rental actually runs still carries it in
the 2-step smoke and in the copy-paste launch block, still prints the retracted P7 justification
at `:317`, and still asserts at `:320` that "the smoke already runs this exact bundle". It is
not inert — `rf.py:389` swaps the model's own predicted contact package into the network input.
And the divergence is **locked in by the suite**: `tests/test_ft_a_fixes.py:296` and
`tests/test_fixnow_train_0830.py:623` assert its presence and both pass. Verified harmless to
remove: on the real v5_6 payload, dropping it changes nothing at load (`HARD: {}` either way).
**Fix:** delete from `:275` and `:295`, rewrite `:317` to point at `E9_premise_test.md`, invert
both test assertions, and keep the flag as a commented ablation line the way the v4-init control
is kept.

**3. The terminal veto may be an anti-grasp on the rig's own gate statistics.** HIGH ·
rig session · `phantom/deploy/planner.py:415-490`
**This is the one finding the two voters genuinely split on, and it decides whether Arm B can
produce a grasp at all.** Measured premise (not disputed): across all 18 recorded 08-28
waffles/Carton traces, `p_none > 0.9` on **80.1% of 201 replans**, only 1 replan in 201 below
0.5, and at the first commanded close plus the next three replans `p_none` is 0.87-1.00.
Replaying that recorded `p_evt` through the **real** `_apply_veto` with the arm in the demo
close band ends **5 of 18 episodes in `veto_retry_cap`** (50 `close_allowed` → 27
`recovery_open`); four more reach retry 3/3 and escape only because the recorded trace runs out
at `--max-replans 20` — arm B gets 40, so 5/18 is a lower bound. The counter-argument (voter 2,
reproduced independently) is that those traces are the 0/26 session of air-closes, where high
`p_none` is a *correct* negative, and that during a real hold the supervised class is `hold`,
not `none` — so the replay assumes the miscalibration it is offered as evidence for. **Both are
right about the code; nobody has ACC data from a successful rig grasp, because there has never
been one.** Do not resolve this by argument.
**Fix (4 lines, removes the tail risk either way):** record `allowed_by` on the latch when a
close is permitted, and refuse to fire the recovery on a close that was allowed **only** by the
`at_floor` hatch — the hatch F4 opened is the only door on this data, since `p_contact ≤ 0.1`
everywhere. Plus print the per-episode `p_none` histogram in the preflight so Session 4
produces the calibration the thresholds were never fitted on.
**Related, same fix area, contested the same way** (`planner.py:401-413` vs `:486`): the
close-mask writes `a[:,6] = grip_now` and `_note_executed_close` reads that hold back as an
executed close, arming the phantom-grasp recovery on a close the veto itself prevented —
reproduced on a recorded open-loop ramp, *not* reproduced when the measured aperture tracks the
executed command. Codex independently found a third member of this family: the executed-close
latch is timestamp-based while the executor stamps every step entered in one tick with the same
`t0`, so a mid-burst planner read can skip a close (bounded at one replan of latency, not a
permanent miss). Treat `planner.py:396-413` + `:415-490` as one 30-minute review, not three.

**4. `replay_rig` declares every Arm-B replan uncomparable — GATE G0 is 0/N by construction.**
HIGH · rig session (post-hoc) · `tools/replay_rig.py:572` · `tools/replay_deploy_path.py:163`
`vetoed = bool(r.get("terminal_veto")) or pre_veto is not None` keys off the record's mere
existence, but `_apply_veto` writes a truthy dict on **every** replan while the veto is on —
including `action:"none"` and `"close_allowed"`, where `plan.actions` was never touched and no
`actions_pre_veto` row is written. Reproduced end to end: 8/8 and 5/5 rows come back
`trace_comparable: False` with the warning "compare against a VETO-REWRITTEN chunk (pre-F9
trace); their trace_in_spread is not a G0 signal" — on episodes where nothing was rewritten.
Arm A is unaffected, so the two arms are not even scored the same way. The repo's own test
misses it because it stores `terminal_veto` as a **string**, not the dict the planner writes.
Severity note: the *numbers* stay correct and `episode_summary` averages all rows regardless,
so this is a corrupted label and a false alarm, not a corrupted measurement — which is why one
voter refuted it. It still makes the gate unreadable on the arm it exists to validate.
**Fix:** `act = (r.get("terminal_veto") or {}).get("action"); vetoed = act in ("close_masked",
"recovery_open") or pre_veto is not None`. Same edit in `replay_deploy_path.py`. Fix the test
to inject the real dict shape.

**5. F6's gripper release races the worker that owns the socket, and never runs on a
protective stop.** MEDIUM (HIGH if the session is unattended) · rig session ·
`phantom/deploy/executor.py:163-167`, `:303`
`_halt` calls `_release_gripper()` **before** `self._grip_target = None` and `self._stop.set()`,
from the executor thread (`safety_stop`) and from the planner thread (`veto_retry_cap`).
`_grip_worker` calls itself "the single owner of gripper I/O" and re-checks `_stop` only
*before* `gripper.move`, so a target already latched sails past that check and lands **after**
the release. Measured on the real `ChunkExecutor` with a Robotiq-shaped single-socket gripper:
last command was a close in **1/30** trials at a 2 ms round trip, **2/40** at 2 ms and **8/40**
at a realistic 15 ms in an independent repro, **23/70** at 3 ms in a third. The log says
`RELEASED` either way. Separately, `safety.py:129-132` promotes a co-occurring `tactile_fz` /
`hitbox_exit` stop to `PROTECTIVE_STOP` (which outranks `STOP_EPISODE` in `_max`) while keeping
the letgo event in `verdict.events`, and `executor.py:303` calls `_halt("protective_stop")`
with no `events=` — so F6 never runs on exactly the coincidence a hard press into the table
produces. `tests/test_deploy_fixnow_0830.py:358-372` parametrizes six reasons and not that one.
Consequence: the fingers squeeze the DM-Tac gels through the blocking label prompt and the
"clear the arm's path" prompt — minutes, unattended.
**Fix:** reorder `_halt` to `self._stop.set(); self._grip_target = None; …; self._release_gripper()`,
put one shared lock around both `_grip_worker`'s `gripper.move` and `_release_gripper` (the
ordering swap alone leaves a window — the lock is the load-bearing half), pass
`events=verdict.events` on the `PROTECTIVE_STOP` branch, and add `protective_stop` to the F6
parametrization. Note `docs/rig_session_v5.md:135-137` already tells the operator to run
`./GRIPPER_OPEN.sh` if the gripper is stuck, which is the attended-session mitigation.

**6. The rig doc still quotes the numbers E13 was written to retire.** MEDIUM · rig session ·
`docs/rig_session_v5.md:154-155`, `:119`
`git blame` dates those lines to 159063d; E13 landed as a pure single-file **addition**
(`c422467`, 164 insertions), and `grep -rln E13_rescore docs/` returns nothing — the operator's
page has no path to the correction. Every number in that block is superseded: endpoint
20.7→17.4 becomes **20.82→18.23** (val124, 4 seeds, hardened tool); z-at-end −4.7→−2.4 becomes
+2.76→+4.16; commit ratio 1.72→1.43 becomes 1.28→1.10; close timing +0.7→−0.1 becomes
+0.19→−0.67; and the "new-batch holdout 20.7→14.9" has **no counterpart in E13 at all** (it
re-derives as 18.54→13.78 over the 46 `batch_20260822` episodes). The stale pair has already
propagated into `REVIEW_SYNTHESIS.md:20`. VALIDATION_0830 §3.5 #4 said "re-score **or** strike";
only the first half was done. `:119`'s z-floor table ("waffles 42, Carton 66, whiteboard 65,
egg 50") is also stale after F17 — `load_start_stats()` now gives floors 31.5 / 66.1 / 57.5 /
49.5 mm, and the live run prints `zfloor:32mm` against a doc that says 42.
**Fix:** two-line replacement plus a link to `E13_rescore.md`; regenerate `:119` from
`configs/start_poses.yaml`. 20 minutes.

**7. The provisioning pytest gate is still flaky.** HIGH · FT-A rental ·
`tests/test_replay_rig.py:137` · `phantom/deploy/planner.py:184,229`
F15 widened the admissible anchors to `{i_deploy, i_deploy-1}`, which narrowed the race but did
not close it. `SnapshotBuilder.build()` reads `rings["arm"].latest(self._n_arm)` to anchor the
wrist F/T grid, then a **second** `latest(1)` for `ur_state`; an instrumented probe measured the
row gap between the two reads at {0: 163, 1: 13, 2: 1} over 177 builds under load, and the
2-row case escapes both anchors. Observed failure rates disagree — 1/16 and 1/20 in two lenses,
0/60 in a third — so call it 1-in-6 to 1-in-20 per *file* run, ~0.5-5% per provisioning run.
`provision_v5.sh:256` is fatal on it, after the ~100 GB pull, and `--deselect` is forbidden.
**Fix (also fixes deploy):** in `SnapshotBuilder.build()`, take `ur_state` from the row already
fetched by the first `latest(self._n_arm)` call instead of issuing a second `latest(1)`. That
makes the snapshot internally self-consistent — wrist anchor == `ur_state` row, the way
training builds it — and removes the race from the runtime as well as the test.

**8. `--resume` silently rewrites the recipe; `upload_run_ckpts` skips by byte size only.**
HIGH · FT-A rental and both baseline rentals · `phantom/train/train_teacher.py:346-377,411-413`
· `tools/upload_run_ckpts.py:113`
F11 made `event_band_weight` a `TeacherTrainConfig` field and restores it on resume. Nothing
else. `grasp_frac`, `photo_aug`, `commit_band_weight` and `split` are read straight off `args`
and never reach `configs.train` at all; `ema_decay` is recorded but not re-applied. Measured:
resuming with only the model flags re-passed gave `ema_decay 0.999` (checkpoint: 0.995),
`photo_aug 0.0`, `commit_band_weight 1.0`, `grasp_frac 0.0` — and the new checkpoint records
the reverted values as if they had held for the whole run. (One sub-claim does **not** hold:
`opt.load_state_dict` + `sched.load_state_dict` do restore the LR, so only the *recorded* lr
lies.) Separately, `upload_run_ckpts.py:113` is `if have.get(p.name) == size: skip`, and every
PHANTOM teacher checkpoint is exactly 393,115,861 bytes — a same-name file on the hub is always
"already present" regardless of content, and the hub *does* expose the real sha256 via
`list_repo_tree(expand=True)`.
**Fix:** add the four missing fields to `TeacherTrainConfig`; in the `--resume` branch restore
every `configs.train` key the CLI did not explicitly override, or refuse the resume when any
differs — the same accept/refuse rule already written for `event_band_weight`. In
`remote_sizes`, pass `expand=True`, return `{name: e.lfs.sha256}`, and compare a local
`hashlib.sha256` (fall back to size when `lfs is None`). ~6 lines each.

**9. `--seed-from-meta` reproduces seed 0, not the chunk the K-seed lever executed.** MEDIUM ·
rig session (post-hoc) and FT-A checkpoint selection · `tools/replay_rig.py:472-482,549-563,656`
`run_deploy` records `kseeds:<K>` in `meta.tags` and `diag.k_pick` in every trace row.
`replay_rig` reads `seed:` and `parity:` and nothing else — the whole source has no occurrence
of `k_pick`, `kseeds` or `k_seeds`. Measured on the tiny model with a real recorded snapshot:
`--seeds 1` reproduces candidate 0 to 1.3e-07 while candidates 1-3 are 1.5-1.7 away; `--seeds 4`
reproduces **nothing** (2.16 off) because `build_x0` draws at B=4 and shifts the whole stream.
A direct probe of the selector found `k_pick != 0` in **310/400** replans with a `prev_plan`
present, so scoring against candidate 0 is the common case. And with `--seeds 1` — the value
the `--seed-from-meta` help prescribes for "an exact reproduction" — `trace_in_spread` becomes
a float-equality test over a one-element spread and reads 0.00 even for a perfect replay.
**Fix:** read `kseeds:<K>` from the tags and `diag.k_pick` from the row; under `--seed-from-meta`
force `--seeds K`, call `rf.sample` with `k_seeds=K` (not `_tile`), score the trace against row
`k_pick` while still reporting the K-spread, and hard-fail when `--seeds` disagrees with the
recorded `kseeds`.

**10. `E13_rescore.md`'s own headline compares a MEAN against a MEDIAN.** HIGH · paper ·
`docs/review_20260828/E13_rescore.md:154`
"Predicted close height is 115-117 mm on val124 while the demos close at 98 mm … ~17-19 mm above
where the demos do, and v5_6 buys only 2 mm of that." 115.30/117.44 are
`summary["pred_close_height_mm"]` (a **mean**); 98.109 is `median_gt_close_height_mm`. The GT
distribution is strongly multi-modal across tasks (per-task GT means: whiteboard 159.2, Carton
108.2, waffles 76.8, egg 72.7 → pooled mean 109.94 vs median 98.11). Like-for-like: mean−mean
7.50 → 5.37 mm (v5_6 buys 2.1), median−median 18.03 → 11.92 mm (buys **6.1**). The doc's
"17-19 mm" is exactly `mean_pred − median_gt`. Worse, the two aggregates are computed over
different row subsets; the paired per-window difference is only +4.79 mm mean for v4 and +6.00
for v5_6. So both quotable sentences fail: "roughly the same magnitude as the 3-6 cm closed-loop
miss" collapses at a 5.4-7.5 mm gap, and "the absolute height it closes at is essentially
unchanged" is contradicted by the doc's own median table. This is the document written
specifically to stop the project quoting a wrong number.
**Fix:** rewrite §3(2) with one statistic throughout, print the mean GT close height in the
mean table, and state which statistic the paper will quote.

### FIX IN THE NORMAL COURSE

**11. The RQ2 money plot is computed across two clocks.** HIGH severity, but no launch depends
on it · `phantom/eval/metrics.py:84,126`
`planner.py:162` sets `t_now = time.perf_counter()` and that becomes every `planner_trace.json`
row's `"t"` (HOST clock), while `recorder.py:125` writes every zarr `ts` through
`clock.host_to_master(t)` = host + `clock_calibration.offset` (on the rig, UR controller uptime
minus NUC perf_counter). `tools/rig_trace_decompose.py` applies `+off` and documents it;
`metrics.py` does not. Driven through the real functions on a synthetic episode with a true
0.500 s lead: offset 0 → leads `[0.5, 0.5]`; offset +1234 s → leads `[1234.5, 1234.5]`,
`event_f1 = nan`; offset −1234 s → leads `[]`. **The reported anticipation lead is literally the
clock offset.** Even at offset 0 every lead is double-counted, once per tactile sensor.
`git diff d9c40f2..HEAD -- phantom/eval/metrics.py` is empty — the fix campaign never touched
this file. VALIDATION #35 flagged the uncapped/per-sensor/off-by-one issues and not the clock.
**Fix (1.5 h):** read `off = EpisodeReader(ep).meta.clock_calibration.get("offset", 0.0)` and
use `r["t"] + off`; de-duplicate to one lead per onset across sensors; cap a lead at the ACC
lookahead; use all sensors in `event_f1` instead of `sensors[0]`.

**12. `--no-z-floor` alone is now a hard refusal on Carton and whiteboard, with wrong advice.**
MEDIUM · `phantom/scripts/run_deploy.py:421`
With no task floor the workspace bound is 30 mm while the hitbox floor is `tcp_z_min − 30` =
46.1 (Carton) / 37.5 (whiteboard), so `envelope_conflict` fires and `main` returns 2. The
printed remedy names `--z-floor-margin`, which is not in play under `--no-z-floor`; the only fix
is `--hitbox-margin ≥ 0.046`. The repo's own test exercises `--no-z-floor` with `hitbox=False`,
so the combination is uncovered. Fix the message and add the case.

**13. `is_letgo_reason` matches the whole `tactile_` prefix, so a 0.3 s sensor dropout drops the
object.** MEDIUM · `phantom/deploy/executor.py:38-43`
`tactile_<name>_stale` is a *freshness* event, not an over-force, and it opens the gripper to
0.232 — well open of a 0.5-0.6 grasp plateau. `hitbox_exit` is in the same set and the 08-28
excursions were at z = 290-673 mm, i.e. the object is dropped from transport height. On egg
that is D15's "damage %" metric writing itself. Fix: match `tactile_fz` / `tactile_depth`
explicitly, and gate the `hitbox_exit` release on a transport-height threshold.

**14. `veto_retry_cap` masks a `servo_stop` failure from `recover_control`.** LOW ·
`phantom/scripts/run_deploy.py:146` · `phantom/deploy/executor.py:144`
`_set_reason` is first-writer-wins, so after `request_stop("veto_retry_cap")` a later
`servo_stop failed` cannot set `servo_stop_failed`; `veto_retry_cap` is not in
`_CONTROL_DEAD_REASONS`, so the next episode skips `recover_control()` and `move_l` is refused
by the servo guard — the 2026-08-27 failure the code comment describes. Found independently by
Codex. Also the `retry_cap` trace row is written with `"diag": {}`, dropping that replan's
`head_dz_mm`.

**15. `chunk_metrics.close_step` is not `close_index`.** MEDIUM · `tools/replay_rig.py:114,362`
A bare `grip > 0.45` with a comment claiming it is "the same aperture rule as
terminal_eval / close_index" — but `close_index` has a max-relative fallback for wide grasps. On
a Carton-like chunk closing to 0.43: replay says `close_step = 16` ("never closed"),
`close_index` says 14. Carton is the second Session-4 task and 24% of its demos close below
0.45. VALIDATION #43, still unfixed.

**16. `replay_rig` does not mirror deploy's `_invalidate_cpk`.** MEDIUM ·
`tools/replay_rig.py:564`
F3 makes a veto rewrite set `plan.cpk = None`, so the rig's next replan runs with
`prev_cpk=None`; the replay chains `prev_cpk = pred.cpk.detach()` unconditionally. Instrumented
on a recorded veto episode: replans 3, 4, 5 had `prev_cpk=None` on the rig and a live package in
the replay. Those are exactly the terminal replans G0 and the trace decomposition are read off.

**17. `EpisodeMeta.weight` (F19) has a consumer and a CLI knob but no producer.** MEDIUM ·
`phantom/data/schema.py:99-102`
`grep` over the repo finds the dataclass default, the reader (`windows.py:439`) and the CLI help
— and no writer. `intake_recovery.manifest()` emits 13 keys and no `weight`;
`phantom/dagger/manifest.py` writes `weight` into a JSONL no trainer reads; the schema
docstring's formula `clip(1/(p_task+0.2), 1, 3)` has zero references to `p_task` anywhere. The
arithmetic is verified correct end to end (multiplicative with `commit_band_weight`, failure
demos pinned to 0, negatives clamped) — it simply never receives a non-default value.
VALIDATION #21's fix text asked for exactly this producer.

**18. `--parity-fixes` tag mismatch warns instead of refusing; the printed selection command
is not executable.** LOW · `tools/replay_rig.py:420,435,495-501` · `tools/provision_v5.sh:306-312`
F16 asked for a hard fail on flag/tag disagreement; it warns. And the printed FT-A
checkpoint-selection block prescribes "per checkpoint, RAW and EMA" via `replay_rig`, which has
no raw/EMA switch (`load_ema=True` is hardcoded), omits `--seed-from-meta`, and hardcodes
`--parity-fixes` on 08-28 episodes recorded `parity:off`. Also: the 08-28 rig session is **not
on the hub**, so "scp it before the run ends" is an unchecked manual precondition.

**19. Codex-only, deferred to the DAgger phase.** A non-rederived policy rollout passes the soft
`build_index` warn-gate into `dagger_driver`'s relabel and into `distill_hid --extra-data`
(`phantom/train/dagger_driver.py:50-57`, `phantom/dagger/relabel.py:92-99`,
`phantom/data/windows.py:209-217`); `behavior_match` and `finetune_hids` ignore `action_weight`
entirely. Unreachable today (the corpus is teleop, no DAgger round exists, `--extra-data`
appears in no launch script) and the hard gate at `tools/intake_recovery.py:164-167` holds — but
it must close before D8 self-improvement.

**20. Hygiene, one line each.** `label_grasps.py:115` divides by all episodes including
`inconclusive` abstentions (dilutes the printed rate; the confusion block already reports them
separately). `label_grasps` silently drops contaminated/aborted episodes with no excluded-count
line. `episode_qc.py:89` and `drivers/base.py:71` still describe `OBJ=2` as "held object",
contradicting the P8 fix. `docs/review_20260828/research/06_*.md:34` is still inverted.
`recovery_demos_protocol.md:55` still names `terminal_eval` as the selection tool.
`terminal_eval` reports `n` as row count, i.e. 8× the independent windows at `--seeds 8`.

---

## 3. EXPLICIT ANSWERS

### 3.1 Were any fix-induced bugs found?

**Yes — five, all in the deploy/replay veto-and-release subsystem, plus two protection
regressions.** The fix campaign's own artifacts:

| # | Introduced by | Defect | Where |
|---|---|---|---|
| 1 | F16 | `trace_comparable` false for every veto-on replan → G0 unreadable on Arm B | `replay_rig.py:572`, `replay_deploy_path.py:163` |
| 2 | F3 | deploy's new `_invalidate_cpk` is not mirrored in the replay → replay feeds a live `prev_cpk` where the rig fed `None` | `replay_rig.py:564` |
| 3 | F2+F3 | the close-mask's own hold command is read back as an executed close and arms the phantom-grasp recovery (contested; reproduced open-loop, not closed-loop) | `planner.py:401-413` vs `:486` |
| 4 | F2+F3+F4 | the veto went from inert to live and, on the only gate statistics that exist, may self-cancel every close (contested; 5/18 replayed episodes cap) | `planner.py:415-490` |
| 5 | F6 | the release is issued from a second thread before the mailbox is invalidated, and is skipped entirely on `PROTECTIVE_STOP` | `executor.py:163-167`, `:303` |
| 6 | F17 | the regenerated `start_poses.yaml` lowered the waffles clamp floor 42 → 31.5 mm (an honest statistic that shrinks protection as the dataset grows), and left `rig_session_v5.md:119` stale | `configs/start_poses.yaml:52` |
| 7 | F7 | `--no-z-floor` alone is now rc 2 on Carton/whiteboard, with a remedy message naming a flag that is not in play | `run_deploy.py:421` |

Two more are *incomplete* fixes rather than regressions: F5 added named stop reasons that are
never persisted to disk, and F11 restores one resume knob out of seven — which arguably makes a
resume **more** dangerous, because it now looks trustworthy.

The pattern is sharp and worth naming: **every fix-induced bug is in the veto/release path, and
every one of them is a fix that changed deploy without changing the tool that reads deploy.**
F2/F3/F4 made the veto live without updating `replay_rig`'s comparability rule, its `prev_cpk`
chaining, or the executed-close latch's understanding of who wrote the command. That is one
review, not seven fixes.

Notably **clean under adversarial probing**: F1, F8, F10, F12, F13, F18, F19, F20 — witnessed
live with no collateral. The objective mathematics is verified sound (β=1 reproduces the plain
MSE trunk gradient **bit-identically**; per-strip noise is strip-constant in `training_step`
*and* `sample`, exactly zero outside the strips; the P10B tolerate set is exactly the intended
one; `git diff d9c40f2..HEAD -- phantom/model/` is **empty**, and a 4-window collated batch is
sha256-identical between the two trees). Flags-off really is pre-fix.

### 3.2 Is the fix cycle converging?

**On code, decisively yes. On artifacts and documents, no — and that is now the dominant
failure mode.**

| | 2026-08-30 (`d9c40f2`) | 2026-08-31 (`820b4eb`) |
|---|---|---|
| Confirmed findings | **30** (8 P0, 12 P1, 10 P2) | **11 unique** (7 high, 4 medium, 0 crash-class) |
| Two-voter outcome | 20 voted, **0 refuted** | 32 voted, **18 refuted or downgraded** (56%) |
| Areas NOT READY | 7 / 7 | 7 / 7 |
| Off-rig work to a paper table | ~29-30 h | **~14.5 h** |
| Codex passes completed | 4 of 8 (quota) | **5 of 5** |
| Suite | 571 passed (`-p no:randomly`) | **661 passed**, default random order |
| Character of the top blocker | *the launch line crashes at checkpoint load* | *the launch line trains the wrong objective* |

Findings fell 63%. The refutation rate went from 0% to 56%, which is the strongest single signal:
the lenses now have to work much harder to find anything, and half of what they find dissolves
under a second reading. Severity descended a whole class — 08-30's P0s were *the code does not
work when you run it* (FT-A hard-fails at load, the veto never arms, every episode shares one
seed, no stop reasons, no checkpoint egress, the student trains on its own val split). Not one
of those survives. Today's highs are *the thing that runs is not the thing that was fixed*.

Three of the top four blockers are **one-line edits to a shell script and a markdown file**:
`--contact-self-forcing` still in `provision_v5.sh` after the playbook removed it,
`--max-replans` still absent from the arm-B `EXTRA` after F5 shipped the mechanism, and
`rig_session_v5.md` still quoting numbers `E13_rescore.md` retired. In every case the fix landed
on one branch and the artifact that consumes it lived on another. That is a **process gap, not
an engineering gap**, and it will recur on the next campaign unless it is mechanised.

**Recommended process fix (0.5 h, closes the whole class).** Add `tests/test_artifact_parity.py`:
(a) assert `provision_v5.sh`'s printed launch line is exactly the flag set
`docs/training_playbook.md` calls the recommended FT-A bundle, parsed from both; (b) assert no
retired number (`17.4`, `14.9`, the old z-floors) appears in `docs/rig_session_v5.md`; (c) assert
the arm-A/arm-B `EXTRA` lines in the rig doc parse through `run_deploy.build_parser()` into the
configuration the doc's own prose claims (`max_episode_s` actually binding). Every one of this
round's top three blockers turns red under that file.

---

## 4. PAPER-GAP COST LIST — priced

The fix campaign touched deploy, train, replay, `grasp_label`, `start_poses` and provisioning.
It touched **nothing** in `phantom/eval/{aggregate,metrics,trial_runner}.py` — `git diff --stat
d9c40f2..HEAD` lists none of them. Every §7 "Paper protocol" verdict from VALIDATION_0830 stands
verbatim, plus the clock bug no earlier lens caught. What landed and *did* shrink the gap: F9's
`tcp_pose` / `actions_pre_veto` / `head_dz_mm`, F8's `seed:` and `start:` tags, F10's `--split`,
F14's arm block, `grasp_label`'s `hold_truncated` and last-close rule — all exercised, all real.

| # | Item | h | Gates | Note |
|---|---|---|---|---|
| 1 | `docs/rig_session_v5.md:154-155` → E13 numbers, `:119` → real floors | 0.25 | rig 1 | also §2 #6 |
| 2 | `eval/metrics.py` clock offset + per-sensor de-dup + lookahead cap + all sensors in `event_f1` | 1.5 | RQ2 | §2 #11; without it the money plot **is** the clock offset |
| 3 | `phantom/eval/stats.py` — Wilson, Fisher (`math.comb`), McNemar, paired bootstrap, stdlib only; replace `cell_stats`'s "over seeds" bootstrap | 3.0 | every A/B number | 0 hits repo-wide today |
| 4 | Pre-registration rewrite: continuous primary, G2 demoted to tripwire | 0.5 | D5 read-out | 3/16 vs 0/16 gives Fisher p=0.226; B needs ≥5/16 for p<0.05 |
| 5 | `--cell` / `arm:` tags in `run_deploy` + verdict prompt | 1.0 | D11+ | D5 is centre-cell only, so not a rig-1 blocker |
| 6 | `commit_height_mm` from the trace, **same estimator both arms** | 2.0 | continuous primary | today arm A falls back to "z of the last replan" — not comparable to arm B |
| 7 | Stats input side: hardened join + tests, `grasp_ok` into the report, attempt denominator (attempted / valid / contaminated) | 2.0 | headline table | Codex: the tables are not tactile-labeled at all |
| 8 | Playbook baseline launch lines + arm-parity pre-registration; drop `recovery_ratio` | 1.5 | **$160 of runs** | N1/N2 |
| 9 | `pipeline.md §8` + `configs/eval_campaign.example.yaml` rewrite to the real protocol | 2.0 | paper §Experiments | currently pre-registers 5 nonexistent tasks |
| 10 | `trial_runner` / `run_eval`: fix (5 h) **or** quarantine as dead code (0.25 h) | 0.25-5 | — | `run_eval` cannot run arm B at all — `pol_args` omits every lever |
| 11 | Codex protocol delta: pool by `(task, system, occlusion)` not `system`; exclusion accounting; sensitivity/specificity/κ for the tactile rule; `terminal_eval` `n` inflated 8× by seeds | 1.5 | table honesty | new this round |
| 12 | RQ1/RQ2 ablation arms + non-WAM baseline (ACT / diffusion policy) | 0.25 (cut) / 6+ (keep) | RQ1/RQ2 | **Mikhail decision** |
| | **Minimum path to a defensible table** | **~15.5** | | plus $160 of runs, gated on #8 |
| | **Everything kept** | **~26** | | |

Down from ~29-30 h on 08-30. Items 3, 6 and 7 are the critical path: without them a perfect rig
session produces a CSV and no table.

---

## 5. REFUTED FINDINGS — one line each

Eighteen findings failed the two-voter round. Most were code-true and failed on **consequence**,
not on fact; those are marked *(true, not blocking)* and are worth doing as hygiene.

1. **"The terminal veto self-cancels — every allowed close is reopened"** — split 1-1; the cycle
   is real and reproduced, but the `p_none≈0.96` premise comes from the 0/26 session of genuine
   air-closes where high `p_none` is a correct negative, and during a real hold the supervised
   class is `hold`, not `none`. Carried forward as §2 #3, unresolved by argument.
2. **"The hub tarball is pre-fix and the COMMIT pin cannot detect it"** — mechanism true, but the
   gate sits *before* every expensive step and aborts fail-safe in ~30 s for ~2 MB; the
   catastrophic branch needs a stale pinned script that exists nowhere on this Mac. Kept as C9.
3. **"replay_rig marks every Arm-B replan uncomparable → G0 0/N"** *(merge-seams instance)* —
   same defect as the confirmed §2 #4; refuted here because the numbers stay correct, the
   summary averages all rows regardless, and the 08-28 episodes carry no `terminal_veto` key at
   all (`_apply_veto` first landed 08-29). Net: fix the flag, do not panic about the gate.
4. **"A UR protective stop leaves the gripper closed"** — code-true (confirmed by both voters);
   refuted as *blocking* because `rig_session_v5.md:135-137` already tells the operator to run
   `./GRIPPER_OPEN.sh` and `recover_control` blocks on an `input()` with a human at the pendant.
   *(true, not blocking — one-word fix, folded into §2 #5.)*
5. **"grasp_label scores post-release aperture drift as the grasp"** — the slice-relative
   fallback is real, but at realistic geometry (83 Hz gPO, 0.23 start aperture, 200 ms release)
   `_release_after` cuts on the ramp and `n_attempts` stays 1 for park peaks up to 0.40; the
   repro needs a synthetic single-sample step release at 20 Hz. *(one-line `ref_max` hardening.)*
6. **"The waffles clamp floor is 20.5 mm below the deepest demo"** — the 20.5 mm is measured
   against the 19-episode `val_eval` subset that F17 was commissioned to *discard*; against the
   250 episodes the code actually reads, 31.5 mm is 10 mm below the deepest demo — the identical
   margin Carton, egg and whiteboard all carry — and `apply_z_floor` takes `max(ws.z[0], floor)`
   so the N→∞ extrapolation resolves at 30 mm. The percentile hardening and the one-time table
   measurement remain good hygiene (C6).
7. **"No statistics code exists / the CI is degenerate at 0/16"** — every fact reproduces
   exactly; refuted as a *launch* blocker because the ledger plus `planner_trace.json` already
   persist everything the statistics need, so any Wilson/Fisher number is recoverable post hoc
   in an afternoon. Priced as §4 #3. *(true, not blocking.)*
8. **"The placement cell has no field"** — true, but D5 is **centre-cell only** by the plan's own
   descoping, so the pairing key is constant and pairing is interleave order plus the fielded
   `seed:` and `start:` tags. Becomes real at D11. Priced as §4 #5.
9. **"`_halt()` writes the gripper socket directly, racing the worker"** — the crash half is
   refuted (`RobotiqGripper._cmd` serializes every transaction under a mutex, 0/300
   `executor_crash`); the ordering half is confirmed and carried as §2 #5.
10. **"The executed-close latch misses closes on identical timestamps"** — mechanism real, but
    `_note_executed_close` tests running state against a persistent `in_close` flag, so a dropped
    step is re-detected on the next executed step: ≤1 replan of latency, never a permanent miss.
11. **"k-seed selection and the veto read different seeds' ACC beliefs"** — syntactically true
    (`p_evt[0]` vs `p_evt[j]`) but **numerically identical**: every ACC input is seed-invariant
    (`repeat_interleave`d conditioning, re-pinned each denoise step), measured
    `max|p_evt[j] − p_evt[0]| = 0.000e+00` bitwise across K=4 while the actions differ by 1.98.
    Cosmetic.
12. **"Non-rederived rollouts reach DAgger relabel and `--extra-data` training"** — true and
    deliberate: the hard gate is at intake, the `--extra-data` path is documented to "warn
    loudly", and a test pins warn-and-index. Unreachable today; carried as §2 #19 for D8.
13. **"P2 measured-history parity is opt-in"** — that is the pre-registered design;
    `REVIEW_SYNTHESIS.md:96` says verbatim "ship it behind `--parity-fixes` so the rig A/B can
    attribute it", and every episode is tagged `parity:on/off`. Defaulting it on would destroy
    the attribution.
14. **"P3 terminal-gate rules are off unless `--terminal-veto` is passed"** — same: arm A is the
    deliberate lever-free control, D17 exists to ablate it, and `run_deploy` hard-refuses the
    flag on a non-ACC checkpoint.
15. **"P4 chunk tail still never executes"** — true arithmetic, backwards conclusion: the fix's
    own rebase means *lower* latency executes a *shorter* prefix (L=0.172 s → steps 0-1), and the
    next chunk's head covers the dropped tail's wall-clock. Receding-horizon MPC working as
    designed.
16. **"Inconclusive tactile labels are counted as failures"** — true of one console line
    (`ok / len(ls)`), which already prints the abstention counts twice beside it; G2 is a
    numerator count, not a rate, and the tool has no per-arm grouping so the claimed A/B bias
    cannot arise through it. *(true, cosmetic — §2 #20.)*
17. **"Task-namespace mismatch auto-fails unknown tasks"** — `trial_runner` never calls
    `grasp_label` (campaign success is the operator's y/n), the veto is loud
    (`z_max_unknown_task:<task>`) and overridable via `--z-max`, and the five campaign task names
    exist only in a docstring and an example YAML.
18. **"The close-mask's own HOLD arms the phantom-grasp recovery"** — refuted at closed loop: the
    latch and the mask read the same measured aperture and the same running minimum, so when the
    measurement tracks the command the mask freezes it and nothing arms. Reproduced only against
    a scripted ramp that ignores its own mask. Carried as the "related" half of §2 #3, since the
    fix is the same 4 lines.

---

## 6. RECOMMENDED ORDER OF OPERATIONS

1. **~2 h of edits** (§2 #1, #2, #5, #6 + the stop-reason tag) — three of these are one-liners.
2. **`pack_repo.sh` + upload** (C9), then **rent the H100** and run FT-A. It is unblocked once
   #2 and #7 are done and it is the cheapest experiment on the board.
3. **Rig Session 1 in parallel**, attended, with the corrected arm-B line and the `p_none`
   histogram in the preflight.
4. **While both run:** `phantom/eval/stats.py`, `commit_height_mm`, the metrics clock fix, and
   `tests/test_artifact_parity.py`.
5. **Re-verdict the $160 of baseline/student rentals** after step 4 and a completed FT-A.

Standing items neither round has closed: **rotate the HF token** before the next rental
(flagged since 08-27, exported into the provisioning environment), and Mikhail decision §3.5 #2
(is Session 4 unattended?) — which condition C2 exists to answer.

---

## Completeness critic

Eight gaps between what this re-validation exercised and what the three scheduled spends depend
on. Parenthesised numbers are `VALIDATION_0830.md`'s own completeness-critic bullets; **six of its
ten are still open**, two of them verbatim untouched. Only critic 8's suite half (661 passed in
default random order) and critic 9's resume half (C11's "do not resume") were actually closed.

1. **Perception and gel-baseline drift were never exercised — second round running, zero lenses
   (0830 critic 1 and 2, verbatim untouched).** `extrinsic`, `lighting`, `RealSense`, `SSIM`,
   `cond frame`, `fields_ds` and `mask_frac` return **0 hits across all nine files in
   `validate2/`**. Every lens again ran mock drivers and `--tiny`; nothing compared a deploy cond
   frame or a gel baseline against a demo one. The review calls a re-aimed camera or the
   08-18/20 lighting change a *sufficient standalone cause* of the terminal-height failure, and
   `mask_frac` is the auto-label under every success number — so the FT-A rental, the veto
   argument in §2 #3 and the whole D5 A/B rest on an unchecked premise about the deploy
   distribution. The offline half is doable **today** without the rig: first cond frame of each
   08-28 deploy episode vs a same-task demo at the gated start pose, plus a per-session `fields_ds`
   depth/`mask_frac` diff (demos through 08-22 vs the 08-20/08-28 rollouts). **RIG-DAY CHECKLIST
   (hardware-only):** gel-zero + press-check and a camera re-aim/recalibrate decision before
   episode 1. Also still true: §2 #13 makes D15's "damage %" write itself, and `run_deploy` still
   has no `damage` field for the operator verdict to land in.

2. **K-seed × NFE latency was never profiled on the NUC, and this round's top blocking fix is
   derived from the Mac stub (0830 critic 5, still open — now load-bearing).** C1/§2 #1's
   `--max-replans 200` and the 35 s cap are computed from 172 ms / 865 ms, the same numbers 08-30
   flagged as *stubbed on the Mac*; `rig_recipe.md:102` states outright that **K=4 was never
   profiled at all**, and K=4 is in the recommended arm. If the real replan cost exceeds the 0.9 s
   budget, arm B truncates and C4's not-yet-persisted stop reason is the only thing that would
   reveal it — post hoc. **RIG-DAY CHECKLIST (hardware-only):** profile K×NFE on the deploy NUC
   before episode 1 and set both caps from the measurement; fall back to K=2-3 if 0.9 s is missed.
   Same unpriced-hours class (0830 critic 9): the session is still booked as "4 rig hours" in
   *episodes*, not attempts, and the joint gate now refuses **22/26** of the 08-28 starts — nothing
   budgets the operator's re-pose loop, and no lens measured how long a gate-passing start takes to
   set up.

3. **Arm B is now a six-lever bundle and the attribution split was never made (0830 critic 4,
   worse).** `--nfe 1 --terminal-veto --parity-fixes --k-seeds 4 --max-episode-s 35
   --max-replans 200`. §4 #4 usefully pre-registers a *read-out* threshold (B needs ≥5/16 for
   Fisher p<0.05) but nothing pre-registers what the abstract may attribute a win to, and §3.4-style
   parity/K-seed ablations still appear nowhere. §5 #13/#14 defend the levers as deliberate
   *A-vs-B* controls, which is correct and does not touch the *within-B* confound. B1 (veto only) /
   B2 (veto+parity+K-seed) do not exist in the plan of record; whatever G2 says, the paper cannot
   name the cause.

4. **The P8 success rule still has zero validated positive class, and the labeller got worse, not
   better (0830 critic 3, still open).** The 1115-demo + 39-rollout re-sweep on the fixed labeller
   was again not run; §2 #20 instead adds two *new* `label_grasps` defects (abstentions in the
   denominator, contaminated/aborted episodes silently dropped with no excluded count). §2 #3
   concedes the same hole from the other side — "nobody has ACC data from a successful rig grasp,
   because there has never been one" — which is precisely why the veto question is declared
   unresolvable by argument. So `grasp_ok` remains a G2 input that has never seen a true positive.
   **RIG-DAY CHECKLIST:** bank the first ≥10 operator-confirmed grasps as the labeller's *and* the
   veto's calibration set, and treat G2 as provisional until then.

5. **Spend verdict without a testable condition — the $160 NO-GO has no release test (0830 critic
   6 and 7, both still open).** The unblock rule reads "N1+N2+N3 = 4.5 h, N4 = 1 h, plus FT-A. Then
   re-verdict." **No pass criterion for FT-A is stated anywhere in the document** — no endpoint-mm
   threshold, no replay-score delta over v5_6, no checkpoint-selection rule that survives §2 #18's
   finding that the prescribed RAW-vs-EMA comparison is *not executable* (`replay_rig` hardcodes
   `load_ema=True`). Worse, the strings `G1`, `G1b`, `G3`, `G4`, `G5` occur **zero times** in this
   document: 08-30 asked for the plan's kill-switches to be carried in verbatim and for a numeric
   pivot threshold on the E9 premise test; instead the last two survivors are G0 — which §2 #4
   shows is unreadable on arm B by construction — and G2, which per bullet 4 has no validated
   positive class. The E-label collision 08-30 flagged also persists and has spread: §1 and §4 now
   use `E1`, `E6`, `E8`, `E9`, `E10` in a *third* numbering unrelated to both the review's E7/E8/E9
   and 08-30's §3.4.

6. **Two more conditions that are not tests.** (a) **C10** — "either de-flake
   `test_rebuilt_snapshot_matches_the_snapshot_builder` **or accept** a 1-in-6-to-1-in-20
   re-provision" — has no trigger and no budget: accepting means an unbounded number of ~100 GB
   re-pulls with no stated retry cap or abort rule, on a metered instance. (b) **C2** — "fix both,
   **or** run the session attended" — is Mikhail decision §3.5 #2, explicitly still open in §6, so
   at the moment of the GO-WITH-CONDITIONS verdict *it is undetermined which of the two verdicts
   applies*, and §2 #5's release race is HIGH or MEDIUM depending on an unanswered question. Also
   not a condition anywhere: the HF-token rotation is listed only as a "standing item neither round
   has closed" in §6, not as a checked precondition on the rental it protects.

7. **Retired numbers are still live, and this round retired another one without tracing where it
   went.** §2 #6 / C5 correctly flag `docs/rig_session_v5.md:154-155` (20.7→17.4, 20.7→14.9) and
   `:119` (42/66/65/50 mm floors), and note the stale pair *has already propagated* into
   `REVIEW_SYNTHESIS.md:20` — but the edit is **unapplied at `820b4eb`**, so at the time of reading
   every one of those numbers is still on the operator's page. Two further copies nobody was tasked
   with: the same 20.7→17.4 / 20.7→14.9 pair is quoted in the memory index's PHANTOM hot line, and
   `:119`'s floors disagree with what the live run prints (`zfloor:32mm`). Separately §2 #10 retires
   `E13_rescore.md:154`'s own "17-19 mm" headline as a mean-vs-median artifact (like-for-like:
   5.4-7.5 mm), which also retires the linkage "roughly the same magnitude as the 3-6 cm closed-loop
   miss" that the review narrative rests on — and **no lens grepped for where else 17-19 mm or that
   framing was copied**. The proposed `test_artifact_parity.py` (b) would cover only
   `rig_session_v5.md`; a repo-wide retired-number sweep is not on any list.

8. **D1's remaining safety items and the non-WAM baseline are still missing, and nothing has an
   owner (0830 critic 10 and 8, still open).** `handover`, `takeover` and `E-stop` return **0 hits
   across all of `validate2/`**: D7's auto-takeover recovery collection still has no written RTDE
   handover procedure, no protective-stop/E-stop watch and no force-abort — while §2 #5 shows
   `PROTECTIVE_STOP` is *exactly* the path that skips the gripper release, i.e. the coincidence a
   hard press into the table produces is both the safety hole and the un-instrumented one. Without
   the handover procedure, D7 records the takeover transient as supervised demo action. The
   ACT/diffusion-policy baseline was surfaced (§4 #12) but only as "Mikhail decision", so all three
   arms remain one backbone — the first question an ICRA reviewer asks of a 2B video-DiT teacher.
   And §6 orders the work without assigning a single name: `owner` appears once in the whole
   document, in prose about gripper I/O.
