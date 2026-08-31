# MERGE SEAMS lens — PHANTOM @ 820b4eb, re-validation 2026-08-31

Scope: bugs that live *between* the four fix branches + two merges that landed in
`d9c40f2..820b4eb`, not inside one of them. Everything below was produced by running
code on the Mac (`.venv/bin/python`, mock drivers, `--tiny`, cosmos importable).
Scratch: `.../scratchpad/validate2/scratch/merge-seams/`.

## Baseline

`.venv/bin/python -m pytest tests/ -q` → **661 passed, 0 failed** in 372 s, default
random ordering (no `-p no:randomly`). The F15 de-flake holds over a full run.
All tool `main()`s parse: gen_start_poses / label_grasps / terminal_eval / replay_rig /
rederive_rollout_actions / upload_run_ckpts → `--help` rc=0 (`intake_recovery` uses
subcommands, not `--help`; not a defect).

The full Arm-B bundle runs end to end through the real entry point:

    .venv/bin/python -m phantom.scripts.run_deploy --system teacher --task waffles --tiny \
      --hardware configs/hardware.nuc.mock.yaml --episodes 1 --nfe 1 --terminal-veto \
      --parity-fixes --k-seeds 4 --max-episode-s 30 --max-replans 8 --persistent-noise \
      --no-label-prompt --device cpu --out .../armb2      # rc=0

meta tags: `nfe1 g1.0 pnoise ckpt: git:820b4eb zfloor:32mm hitbox:30mm vmax:0.25
parity:on veto:pc0.50/pn0.90/r3 kseeds:4 seed:3476212045 unlabeled`; trace rows carry
`tcp_pose`, `terminal_veto`, and `diag.head_dz_mm` / `k_pick` (F9 landed).

---

## 1. `replay_rig` marks EVERY Arm-B replan uncomparable → G0 is 0/N by construction  (HIGH)

`tools/replay_rig.py:572` `vetoed = bool(r.get("terminal_veto")) or pre_veto is not None`.
`PlannerLoop` writes a **non-empty dict on every replan** whenever `--terminal-veto` is on —
`{"p_contact":…, "retries":0, "action":"none"|"close_allowed"|…}` (planner.py:415-493) — and
only writes `actions_pre_veto` when the action is `close_masked` / `recovery_open`
(planner.py:606). So `bool(dict)` is True for the 90+% of replans the veto did **not**
touch, `trace_comparable` is False everywhere, and the tool tells the operator the rows are
not a G0 signal.

Reproduced on the mock Arm-B episode above (veto action = `close_allowed` on all 8 replans,
no rewrite anywhere):

    ...: 8/8 replayed replans compare against a VETO-REWRITTEN chunk (pre-F9 trace);
         their trace_in_spread is not a G0 signal
    JSON: n_uncomparable_vetoed 8 / 8
    row0: trace_source='actions' trace_vetoed=True trace_comparable=False
    row2: trace_in_spread=True   trace_comparable=False   head_dz_std=902.8

The `trace_in_spread` numbers are fine (row 2 is genuinely in-spread); it is the
comparability flag and the warning that are wrong. VALIDATION_0830's own fix text said
"warn when `terminal_veto.action != 'none'`" — the implementation tested the dict's
truthiness instead. Fix: `act = (r.get("terminal_veto") or {}).get("action"); vetoed =
act in ("close_masked", "recovery_open") or pre_veto is not None`.

## 2. A UR protective stop leaves the fingers commanded closed on the gels  (HIGH)

`safety.py:129-132` sets `action = SafetyAction.PROTECTIVE_STOP` unconditionally when the
arm reports a protective stop, while every co-occurring `tactile_fz` / `wrench_limit` /
`hitbox_exit` event stays in `verdict.events`. `executor.py:302-304` then calls
`_halt("protective_stop")` **with no `events=`**, so `is_letgo_reason` sees only the generic
reason and F6's release never runs. The `events=` channel exists precisely because
"a tactile/wrench/hitbox stop must also open the fingers" behind a generic reason.

Reproduced (`test_pstop_gripper.py`, same executor, same chunk commanding 0.90, safety
verdict carrying BOTH a `tactile_fz` STOP_EPISODE event and a `protective_stop` event):

    verdict.action=stop   -> stopped_reason='safety_stop'      commands=[0.9, 0.25] last=0.25
    verdict.action=pstop  -> stopped_reason='protective_stop'  commands=[0.9]       last=0.9
    is_letgo_reason('protective_stop') = False

`tests/test_deploy_fixnow_0830.py:358-372` parametrizes six reasons and never
`protective_stop`. A hard press into the table is exactly the event that trips both the UR
protective stop and the gel force limit at once, and `docs/rig_session_v5.md:124` now tells
the operator "A let-go stop … now commands the fingers open itself".
Fix: pass `events=verdict.events` on the PROTECTIVE_STOP branch too (and add
`protective_stop` to the F6 parametrization).

## 3. `provision_v5.sh` still spends the FT-A rental on `--contact-self-forcing`  (HIGH)

`e06c33b` (branch `fixnow-startposes`, merged at 820b4eb) removed the flag from the
recommended bundle in `docs/training_playbook.md:89` — "Removed from the recommended FT-A
bundle 2026-08-30 … do not spend the primary FT-A run on it" — and
`docs/review_20260828/E9_premise_test.md` says "the E9 row cannot be quoted as the
justification for `--contact-self-forcing` on a paid H100 fine-tune".
`tools/provision_v5.sh` (branch `fixnow/train-0830`, merged at 4d2bcfb) still has it at
`:275` (the 2-step real-training smoke) and `:295` (the printed 3000-step launch line), with
the superseded P7 justification at `:317`. The script is the only thing that runs on the
rental, and its own text says "The 2-step smoke above already runs this exact bundle —
nothing to re-run by hand".

## 4. `replay_rig` chains `prev_cpk` through a replan the veto rewrote  (MEDIUM)

`planner.py:495 _invalidate_cpk` sets `plan.cpk = None` on `close_masked` / `recovery_open`
(F3), so the rig's next replan runs with `prev_cpk=None`. `tools/replay_rig.py:564`
unconditionally does `prev_cpk = pred.cpk.detach()` and never consults `r["terminal_veto"]`.
Exercised through the real `replay_episode()` with a stub sampler on a real recorded
episode whose replan 2 was marked `close_masked` (`replay_veto_cpk.py`):

    {'replan': 2, 'prev_cpk': 'cpk#2'}
    {'replan': 3, 'prev_cpk': 'cpk#3'}   <- deploy fed None here

## 5. `EpisodeMeta.weight` (F19) has a consumer and a CLI knob but no producer  (MEDIUM)

Matrix over 8 episode kinds through `WindowSampler.build_index` + `sample` +
`WindowDataset` (`matrix.py`, `matrix2.py`):

    kind                   trainable failure needs_rd indexed act_w
    success_demo           True      False   False    2       1.00
    fail_demo_task         True      True    False    2       0.00
    fail_demo_verdict      True      True    False    2       0.00
    weighted_rollout       True      False   False    2       3.00
    unlabeled_rollout      False     False   True     0       (refused)
    contaminated           False     False   False    0       (refused)
    veto_retry_rollout     True      False   True     2       1.00  (+ "NO re-derived actions" warning)
    weighted_faildemo      True      True    False    2       0.00
    commit_band_weight=2.0 multiplies only in-band windows; weight 3.0 * band 2.0 = 6.0.

Every gate behaves. But `grep` over the whole repo finds **no writer**: the only
occurrences of the field are the dataclass default (`schema.py:102`), the reader
(`windows.py:439`), and the CLI help (`train_teacher.py:142`). `intake_recovery.manifest()`
emits 13 keys and no `weight`; `phantom/dagger/manifest.py` writes `weight` into a JSONL no
trainer reads. The schema docstring at `schema.py:99` states the formula
`clip(1/(p_task+0.2), 1, 3)` — `grep -rn p_task` over the repo: zero hits outside that
comment. VALIDATION_0830 #21's fix text asked for exactly this producer
("have `intake_recovery.manifest()` write the weight it computes") and for
`round(windows_per_episode * weight)` items in `build_index`; neither landed.

## 6. `docs/rig_session_v5.md` still quotes the numbers E13 replaced  (MEDIUM)

`:154-155` — "endpoint error v4 20.7 mm -> v5_6 17.4 mm … new-batch holdout 20.7 -> 14.9 mm".
`E13_rescore.md §3` (landed on the other branch, c422467): "The correct statement is
20.8 mm -> 18.2 mm", and VALIDATION #50: "Nothing in the paper or the plan may quote
17.4 → 20.7 → 14.9". This is the doc the session is run from.
Also `:119` — the z-floor table "waffles 42, Carton 66, whiteboard 65, egg 50 mm" is stale
after F17 regenerated `start_poses.yaml`: `load_start_stats()` now gives
tcp_z_min 41.5 / 76.1 / 67.5 / 59.5 mm, i.e. floors 31.5 / 66.1 / 57.5 / 49.5 mm (the mock
run printed `zfloor:32mm` for waffles against the doc's 42).

## 7. `--seed-from-meta` reproduces seed 0, not the chunk the rig executed  (MEDIUM)

`run_deploy` records `seed:<n>` **and** `kseeds:<K>`, and the trace records `diag.k_pick` —
which seed the K-seed lever actually selected. `replay_rig` reads `seed:` and `parity:` and
nothing else. Measured with the tiny model on a real recorded snapshot (`kseed_repro.py`,
same generator seed for all three):

    shapes  rig(k_seeds=4) (4,16,7)   replay --seeds 1 (1,16,7)   replay --seeds 4 (4,16,7)
    max|rig[0]-rep1[0]| = 1.34e-07        <- --seeds 1 reproduces seed 0 exactly
    max|rig   -rep4|    = 2.159           <- --seeds 4 does NOT reproduce the K-batch
    |rig[1]-rep1[0]|_max = 1.581  |rig[2]-…| = 1.528  |rig[3]-…| = 1.698

(`--seeds 4` diverges because `build_x0` itself draws from `self._gen` at B=4 instead of
B=1, shifting the whole stream.) So when `k_pick != 0` — the entire point of `--k-seeds 4`
in Arm B — there is no invocation of `replay_rig` that reproduces the executed chunk, and
the `--help` prescribes the K that reproduces a *rejected* candidate. Second half: with
`--seeds 1`, `summarize()`'s `trace_in_spread = hz.min() <= trace <= hz.max()` collapses to
an exact-equality test on a one-element spread; the measured run reported
`head_dz_std 0.0, trace_in_spread 0.00` on all 8 replans.

## 8. In Arm B the veto and the K-seed rejection read different rows of `p_evt`  (MEDIUM)

`policy.py:256-258` feeds `_select_seed` `1 - p_evt[0]` (row 0 of the K-expanded batch) while
the returned `Plan.p_evt` is row `j` (the selected seed), and `planner._apply_veto` reads
`plan.p_evt[0]`, i.e. seed *j*'s. Both levers are keyed to the **same** number,
`--veto-p-close` (`run_deploy.py:123` sets `close_p`; `build_veto` sets `p_close`), so on one
replan the rejection can conclude "we are in contact, keep every candidate" from seed 0 while
the veto concludes "not in contact, mask the close" from seed j. This is the merge-time face
of the still-open VALIDATION #29; it only becomes self-inconsistent now that Arm B runs
`--terminal-veto` and `--k-seeds 4` together.

---

## Seams checked and found SOUND (negative results)

- **`ThinStartStatsError` / `allow_thin_q` × every caller.** The shipped
  `configs/start_poses.yaml` carries `q_n: 250` against `n: 250` for all four tasks;
  `load_start_stats()` returns cleanly with no warning (`Carton 76.1 / egg 59.5 /
  waffles 41.5 / whiteboard 67.5 mm`), `run_deploy` (`:540`, `:647`) and
  `tools/episode_qc.py:73` both import and run. Nothing crashes on the shipped file.
- **Veto band × F17.** `veto_z_margin = Z_MAX_MM[task]/1000 - tcp_z_min`, so the band TOP is
  `Z_MAX_MM` regardless of the regenerated `tcp_z_min` — F17 lowering waffles by 10.5 mm
  does not move the hatch. `envelope_conflict` still holds at the new floors.
- **Stop-reason table.** `replan_cap` / `episode_time_cap` (new, planner-side) and
  `veto_retry_cap` reach `EpisodeResult.stopped_reason` and are correctly *absent* from
  `_CONTROL_DEAD_REASONS` (no RTDE rebuild needed) and from `_SESSION_FATAL_REASONS`.
  Nothing downstream (`run_deploy` 792-820, `dagger/rollout.py:29`, `trial_runner.py:112`)
  branches on the string, so the new non-`None` reasons do not change any path.
  `veto_retry_cap` *is* in `LETGO_STOP_REASONS`, verified releasing in the live mock run
  (`stop (hitbox_exit): gripper RELEASED to 0.23`).
- **F1 / F11 scoping.** `tolerate_model_fields` defaults to `frozenset()`; only
  `train_teacher.py:325` (`--init-weights`) passes `FINETUNE_MUTABLE_MODEL_FIELDS`.
  `--resume` (`:348`), deploy, distill, replay and terminal_eval all stay strict.
  `--allow-config-drift` only affects the *data* hardware-hash check (`:444`), not the model
  drift check, so provisioning's use of it does not defeat F1. `cfg = dataclasses.replace(...)`
  at `:368` happens before the `pm.rf.event_band_weight` assignment at `:405`.
- **Veto × parity `prev_chunk`.** Under `--parity-fixes` the intent channel is rebuilt from
  the MEASURED arm ring plus `executor.gripper_cmd_at` (the commands actually sent), so a
  chunk the veto rewrote is represented by what the arm actually did — the semantics are
  right. `SnapshotBuilder.prev_chunk_from_history` is the single implementation shared by
  deploy and `replay_rig.measured_prev_chunk`.
- **`_halt` release ordering.** `_release_gripper()` runs before `_stop.set()`, so in
  principle the gripper worker's `GRIP_FLUSH_CYCLES` deadband flush could re-send the close
  inside that window. **Could not reproduce** over 6 attempts with a 15 ms-blocking mock
  gripper and a pending sub-deadband residual (`test_halt_race.py`: `[0.9, 0.25]` every
  time) — the flush needs `stable_for == 5` exactly and `tgt != last_sent`. Reporting it as
  a negative result, not a finding.
- **Episode-kind matrix (P9 gates × weight × commit band).** All eight kinds behave as
  intended; see §5's table. `unlabeled` and `contaminated` are refused at both
  `build_index` and `intake_recovery.manifest`; a non-re-derived rollout is indexed with a
  loud warning and refused by the manifest (F13 as specified).
