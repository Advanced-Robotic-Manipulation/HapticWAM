# LENS: what still stands between a good rig session and a paper table
HEAD 820b4eb, 2026-08-31. Nothing under ~/GitHub/phantom was modified.
Scratch: scratchpad/validate2/scratch/gap/

## Bottom line
The fix campaign touched **deploy, train, replay, grasp_label, start_poses, provisioning**.
It touched **nothing** in `phantom/eval/{aggregate,metrics,trial_runner}.py` (`git diff --stat
d9c40f2..HEAD` lists none of them). Every §7 "Paper protocol" verdict from VALIDATION_0830
therefore still stands verbatim, plus one new clock bug that no earlier lens caught.

Total remaining off-rig engineering to a paper table: **~14.5 h** (down from ~29 h — F9's
`tcp_pose`/`actions_pre_veto`/`head_dz_mm`, F8's `start:` tag, F10's `--split`, F14's arm block
and grasp_label's `hold_truncated`/last-close all landed and are exercised below).

## 1. Statistics — still zero code
`grep -rniE 'wilson|fisher|binomtest|proportion_confint|mcnemar' --include='*.py'` → **0 hits**.
`import scipy` → ModuleNotFoundError. `phantom/eval/stats.py` does not exist.
`phantom/eval/aggregate.py` is byte-unchanged since d9c40f2.

Executed `scratch/gap/ab_aggregate.py` — a synthetic plan-of-record session (waffles, 16
episodes/arm, arm B 3/16, arm A 0/16) through the real `aggregate()` + `write_report()`:

    ('waffles','teacher','False'): n=16 success=0.000 CI=[0.000, 0.000]   <- degenerate
    ('waffles','student','False'): n=16 success=0.188 CI=[0.000, 0.375]
    headline ('waffles','False'): {'recovery_ratio': nan, 'retention': nan}

The report.md that comes out has a nan headline row and a [0.00,0.00] CI on the arm the last
three rig sessions actually produced. Reference values the missing module would give
(`scratch/gap/stats_ref.py`, stdlib only, no scipy needed):

    arm A 0/16  Wilson 95% [0.000, 0.194]
    arm B 3/16  Wilson 95% [0.066, 0.430]
    Fisher exact two-sided (3,13 vs 0,16): p = 0.2258
    paired McNemar over placement cells (B-wins 3, A-wins 0): p = 0.2500
    power: B needs >= 5/16 vs 0/16 to clear p<0.05 (4/16 -> 0.1012, 5/16 -> 0.0434)

So **G2 as pre-registered ("Arm B >= 3/16") cannot separate the arms** — it is a tripwire, not a
result. REVIEW_SYNTHESIS:668 says exactly this and asks for a continuous primary; neither
rig_session_v5.md nor VALIDATION_0830 §3.2 carried that change.
COST: stats.py 3 h; pre-registration rewrite 0.5 h.

## 2. Interleaving — trial_runner is unchanged; the manual flow IS the protocol
Executed `scratch/gap/campaign_order.py` (real `run_campaign`, stub runtime, 2 arms x 4 trials):

    RUN ORDER: armA armA armA armA armB armB armB armB
    ENTER/EXIT: connect(armA) -> disconnect(armA) -> connect(armB) -> disconnect(armB)
    run_episode kwargs: {'task','tags','max_replans','policy_name'}   # no seed/policy/mode/max_episode_s
    ledger header: timestamp,campaign,task,system,seed,trial,occlusion,episode_path,success,
                   damage,stopped_reason,n_replans,notes                # no placement, no grasp_ok, no z_close

`run_eval.py:55-59` still builds `pol_args` without persistent_noise/guidance/parity_fixes/
k_seeds/terminal_veto, and `build_policy` reads them all with `getattr(..., default)` — so a
campaign silently runs NFE-default, veto-off, parity-off: it cannot run arm B at all.
=> The Session-4 manual flow (`GO_v5_waffles.sh` -> `run_deploy.main`) is the de-facto protocol,
and `phantom/eval/trial_runner.py` + `run_eval.py` are dead code for rig 1. They also have zero
tests (`ls tests/ | grep -iE 'trial|aggregate|metric|stats|campaign'` -> nothing).
COST: fix E2/E4 5 h, or quarantine 0.25 h.

## 3. The per-episode join — `scratch/gap/join_rows.py` (46 lines) is the spec
Mock Session-4 session built with the real EpisodeWriter + real tag set + F9-shaped
planner_trace (`scratch/gap/mk_session.py`, 8 episodes, A/B interleaved over cells c0-c3).
`tools/label_grasps.py --confusion --json-out` runs clean on it (2 OK / 5 not-ok / 100%
agreement). The join produces the paper row:

  episode  arm cell verdict grasp_ok hold_trunc n_close commit_mm commit_src  z_close_mm seed  start damage ckpt nfe

What exists and is easy:
  * verdict     meta.success + 'contaminated' tag + meta.damage           OK
  * grasp_ok    grasp_label.label_episode (+ hold_truncated, n_close_attempts)  OK
  * z_close_mm  grasp_label (LAST close attempt — the veto-retry fix)     OK
  * seed        `seed:<n>` tag (F8/ba61354)                               OK
  * start pose  `start:x,y,zmm/g<ap>` tag (F8)                            OK
  * ckpt/git/levers  `ckpt: git: parity: veto: kseeds: nfeN zfloor: hitbox: vmax:` tags  OK

What is missing or awkward — and these are the deliverable's findings:
  * **arm** has no field. Reconstructed heuristically: `B if veto!=off or parity==on or
    kseeds!=1`. A "parity-only" or "kseeds-only" ablation arm is silently mislabelled B, and
    an NFE-only arm is mislabelled A.
  * **cell** has no field. The rig doc says "note the cell id in the verdict prompt";
    `run_deploy.parse_verdict` keeps only the first token as the verdict and dumps the rest into
    `meta.notes` as free text. My regex recovered 7 of 8; episode 004 (operator typed just `f`)
    has **no cell at all** and is unusable for any paired statistic. This is the pairing key for
    McNemar / paired-delta — the entire reason the session is interleaved.
  * **commit height** has no recorded event. With `--terminal-veto` ON I can read the
    `close_allowed` replan's `tcp_pose`; with the veto OFF (arm A) `terminal_veto` is `None` in
    every row and the join falls back to "z of the last replan" — a different estimator. Arm A
    and arm B commit heights are then not comparable, which is fatal for the continuous primary.
    Fixable from what is already logged: scan `actions_pre_veto or actions` for `[:,6]` crossing
    the training `close_index` rule, for both arms.
  * `tools/label_grasps.py` **silently drops contaminated/aborted episodes**: 8 on disk ->
    "== 7 episodes"; `--include-unfinalized` -> 8. No excluded-count line either way, so the
    attempt denominator the paper needs (attempted / valid / contaminated) is not printed.
COST: cell+arm tags in run_deploy 1 h; commit_height 2 h; eval/stats.py input side (this
script, hardened + tested) 2 h.

## 4. Baselines — they RUN, they are undocumented
Executed `scratch/gap/baselines.py` (real `train_teacher.main`, --tiny --synthetic --cpu):

    no_distill : --student --split train                 -> rc=0, ckpt written, student=True  mask_wrist=False
    vision_only: --student --mask-wrist --split train    -> rc=0, ckpt written, student=True  mask_wrist=True
    log: "STUDENT layout (no_distill arm)" / "STUDENT layout (vision_only arm): ... wrist F/T MASKED"

Both parse and train. F10 landed (`--split` default `train` on train_teacher, distill_hid and
finetune_hids; all three call `C.manifest_split`) so the val leak is gone from all three arms.
Still lacking: neither line appears anywhere in `docs/training_playbook.md` (grep: only FT-A);
no step budget / cost / init checkpoint / dataset pre-registration; and `aggregate.write_report`
still headlines `recovery_ratio = (student - vision_only)/(teacher - vision_only)`, which the
plan deleted — hence the `nan` in §1.
COST: playbook lines + arm-parity pre-registration 1 h; drop recovery_ratio 0.5 h.

## 5. Claims inventory
| claim in the docs | evidence path today |
|---|---|
| v4 -> v5_6 terminal endpoint 20.8 -> 18.2 mm (val124) | **REAL** — E13_rescore.md, `tools/terminal_eval.py` on compute3 |
| ... but `docs/rig_session_v5.md:154-155` still prints `20.7 -> 17.4` and `20.7 -> 14.9` | **STALE / RETRACTED** (VALIDATION #50, decision #4) |
| P7 exposure bias / `--contact-self-forcing` | **RESOLVED AND WITHDRAWN** — E9 re-run with `contact_zero`; playbook e06c33b drops the flag from the bundle. Correct. |
| tactile premise (null tactile costs 16% commit) | **REAL** — E9 table, `--null tactile` now zeroes `reactive` too |
| grasp success (G2/G3) | **REAL rule** (`phantom/eval/grasp_label.py`, `tools/label_grasps.py`) but wired to nothing: `grep grasp_ok phantom/eval/{aggregate,metrics,trial_runner}.py` -> 0 hits |
| success % with CIs / arm comparison | **MISSING** — §1 |
| "miss distance" (rig_session_v5.md:37, REVIEW_SYNTHESIS:525) | **MISSING** — grep object_pose/aruco/april -> 0 hits; no placement field |
| `commit_height_mm` (E7's continuous primary) | **MISSING** — grep -> 0 hits repo-wide |
| z-at-close | **REAL** — grasp_label.z_close_mm and rig_trace_decompose `deploy` |
| ACC lead time + event F1 (RQ2 money plot) | **BROKEN** — clock-frame bug, §6 |
| RQ2 alpha:=0 (ACC vs CASA) | **MISSING** — `alpha` is a learned scalar (`acc.py:99`), no override knob |
| RQ1 tactile-as-image / VAE-only variant | **MISSING** — grep vae_only/tactile_as_image -> 0 hits |
| non-WAM baseline (ACT / diffusion policy) | **MISSING** — grep -> 0 hits |
| pipeline.md §8 pre-registration | **STALE** — 5 nonexistent tasks, 5 systems, >=3 seeds, ~2,100+960 trials, occlusion, recovery_ratio headline; `configs/eval_campaign.example.yaml` identical; every eval docstring cites it |

## 6. NEW BUG the fix campaign did not touch: the RQ2 numbers are computed across two clocks
`phantom/deploy/planner.py:162` `t_now = time.perf_counter()` -> `snap.t` -> every
`planner_trace.json` row's `"t"` (HOST clock). `phantom/recording/recorder.py:125`
`self.clock.host_to_master(t)` -> every zarr `ts` (MASTER clock = host + `clock_calibration.offset`;
on the rig `offset = median(UR controller uptime - NUC perf_counter)`, `drivers/real/ur.py:301`
`r.getTimestamp()`). `tools/rig_trace_decompose.py` applies `+off` explicitly and says so in its
docstring. `phantom/eval/metrics.py` does not: `acc_lead_times` (l.82) and `event_f1` (l.125)
compare `r["t"]` to stream `ts` raw.

Executed `scratch/gap/clock_join.py` — one synthetic episode, gate crossing 0.5 s before a
single contact onset, run through the real `M.acc_lead_times` / `M.event_f1`:

    offset=     0.0s  acc_lead_times=[0.5, 0.5]      event_f1=0.000   (truth: ONE onset, lead 0.50 s)
    offset=  1234.0s  acc_lead_times=[1234.5, 1234.5] event_f1=nan    (truth: ONE onset, lead 0.50 s)

The reported "anticipation lead" IS the clock offset. Even at offset 0 the lead is reported
twice (once per tactile sensor) — the duplication VALIDATION #35 flagged, still live.
COST: 1.5 h (add the offset, de-duplicate, cap at the lookahead, use all sensors in event_f1).

## 7. Costed remainder
| item | h |
|---|---|
| rig_session_v5.md:154-155 -> E13 numbers | 0.25 |
| eval/metrics clock offset + de-dup + cap | 1.5 |
| phantom/eval/stats.py (Wilson / Fisher / McNemar, stdlib) + call it from cell_stats | 3.0 |
| pre-registration: continuous primary + G2 as tripwire | 0.5 |
| `--cell` / `arm:` tags in run_deploy + prompt | 1.0 |
| commit_height_mm from the trace, same estimator both arms | 2.0 |
| eval/stats.py input side (join_rows hardened + tests) + grasp_label into the report + attempt denominator | 2.0 |
| playbook baseline lines + arm-parity pre-registration; drop recovery_ratio | 1.5 |
| pipeline.md §8 + eval_campaign.example.yaml rewrite | 2.0 |
| trial_runner: fix (5 h) or quarantine (0.25 h) | 0.25-5 |
| RQ1/RQ2 ablation arms + non-WAM baseline: decide to cut, or budget | 0.25 (cut) / 6+ (keep) |
| **total to a defensible table** | **~14.5** |

VERDICT: NOT READY — the rig session can now be RUN and LOGGED correctly, but nothing in the
repo turns its episodes into a table: no statistics, no pairing key, no comparable continuous
primary, and the one RQ2 number that does have code is computed across two clocks.
