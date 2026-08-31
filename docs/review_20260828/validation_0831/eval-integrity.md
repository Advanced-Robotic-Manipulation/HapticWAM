# Lens: EVALUATION-TOOL INTEGRITY — re-validation at HEAD `820b4eb` (2026-08-31)

Everything below was produced by running the tools through their own entry points on this
tree (`.venv/bin/python`, mock drivers, `--tiny`), plus read-only re-derivations from the
E9/E13 JSONs still on compute3. Scratch: `scratch/eval-integrity/`.

## What I exercised

| # | Check | Method | Result |
|---|---|---|---|
| 1 | replay_rig determinism | `replay_rig.main()` twice, tiny init pinned, sha256 of the JSON | **PASS** — bit-identical |
| 1b | order / subset independence | same episode alone / preceded / followed by another | **PASS** — per-replan `head_dz` identical; seeds from CRC32(dirname), recorded per episode |
| 2 | snapshot parity vs `SnapshotBuilder` | real mock deploy runs (`parity_fixes=True` and legacy), every `ObsSnapshot` field diffed | **PASS** apart from documented ring-read races + the JPEG recording |
| 2b | `--parity-fixes` == deploy's construction | `prev_chunk` 6/6 exact, `contact_state` 6/6 exact, measured-dt + consecutive-frame reactive | **PASS on the snapshot**; **FAILS on `prev_cpk`** (finding 2) |
| 3 | `--null` semantics | `terminal_eval.main()` per mode, batch captured at `rf.sample` | **PASS** — every mode zeroes exactly what its help claims, incl. `reactive`; `null_semantics` in every JSON |
| 3b | terminal_eval determinism / subsets | twice + `--max-episodes 2` | **PASS** — bit-identical; capped rows identical to the full run |
| 4 | grasp_label truth table | 11 synthetic episodes through `label_episode` | mostly correct; two edge cases (findings 7, 8) |
| 5 | E9 / E13 numbers vs the compute3 JSONs | re-derived on compute3 | **E9 exact**; **E13 tables exact**, but the E13 narrative mixes statistics (finding 4) and the rig doc still carries the superseded numbers (finding 1) |
| 6 | `trace_comparable` / `trace_vetoed` in G0 | real veto-armed deploy episode replayed | **BROKEN** (findings 3, 5) |

## Findings, in the order I would fix them

### 1 (HIGH) `docs/rig_session_v5.md:154-155` still quotes the numbers E13 replaced

The campaign wrote `E13_rescore.md` ("The correct statement is `20.8 mm -> 18.2 mm`… not the
3.3 mm / 16% that `20.7 -> 17.4` claimed") and edited `rig_session_v5.md` in `b364747` — but
left the **Offline numbers behind v5_6** block untouched. `git diff d9c40f2..HEAD --
docs/rig_session_v5.md | grep -E '17\.4|20\.7|14\.9'` is empty.

Every number in that block is superseded (hardened `terminal_eval`, 4 seeds, val124):

| rig doc line 154-155 | hardened E13 (same split) |
|---|---|
| endpoint v4 20.7 → v5_6 17.4 | 20.82 → 18.23 |
| z-at-end −4.7 → −2.4 | +2.76 → +4.16 |
| commit ratio 1.72 → 1.43 | 1.28 → 1.10 |
| close timing +0.7 → −0.1 | +0.19 → −0.67 |
| new-batch holdout 20.7 → 14.9 | **18.54 → 13.78** (re-derived here) |

The holdout cell is not in E13 either; I re-derived it as `val124 \ val78` = the 46
`batch_20260822` episodes, 184 windows (`/tmp/holdout46.py` on compute3).

### 2 (HIGH) the replay does not mirror deploy's `_invalidate_cpk`

F3 added `PlannerLoop._invalidate_cpk` — a veto rewrite sets `plan.cpk = None`, so
`policy.replan` feeds `prev_cpk=None` on the NEXT replan. `replay_rig.replay_episode`
chains `prev_cpk = pred.cpk.detach()` unconditionally (`:564`). Instrumented
`PhantomRectifiedFlow.sample` on a recorded veto episode: replans 3, 4, 5 (every replan
after a `close_masked`) had `prev_cpk=None` on the rig and a live package in the replay.
Those are exactly the terminal replans E0/G0 and the trace decomposition are read off.

### 3 (HIGH) `trace_vetoed` / `trace_comparable` are wrong for EVERY Arm-B replan

`planner.py:594` writes `"terminal_veto": veto_rec` for every replan once `--terminal-veto`
is on; when the veto does nothing `veto_rec` is `{"p_contact":…, "retries":0, "action":"none"}`
— a **truthy dict**. `replay_rig.py:572` `vetoed = bool(r.get("terminal_veto")) …` therefore
flags every replan, and `:584` marks any replan without `actions_pre_veto` uncomparable.
Reproduced on a real recorded episode: replans 0 and 1 (`action: "none"`, the model's own
chunk in `actions`) came back `trace_vetoed=True, trace_comparable=False` with the log line
"compare against a VETO-REWRITTEN chunk (pre-F9 trace)". `replay_deploy_path.py:162-163` has
the identical expression. The repo test passes only because it injects
`trace[i]["terminal_veto"] = "close_masked"` — a **string**, not the dict the planner writes.

### 4 (HIGH) `E13_rescore.md:154` compares a MEAN against a MEDIAN

"Predicted close height is 115-117 mm on val124 while the demos close at 98 mm … ~17-19 mm
above where the demos do, and v5_6 buys only 2 mm of that." 115.30/117.44 are
`pred_close_height_mm` (mean); 98.109 is `median_gt_close_height_mm`. The GT distribution is
strongly right-skewed (mean 109.94, median 98.11 — whiteboard closes at ~181 mm, waffles at
~103), so mixing them inflates the gap ~3×:

| like-for-like | v4 | v5_6 | v5_6 buys |
|---|---|---|---|
| mean pred − mean GT | 7.50 | 5.37 | 2.1 mm |
| median pred − median GT | 18.03 | 11.92 | **6.1 mm** |
| doc (mean pred − median GT) | 19.33 | 17.19 | 2.1 mm |

The conclusion "the absolute height it closes at is essentially unchanged" holds under
mean-mean but not under median-median, where the fine-tune closes a third of the gap.

### 5 (MEDIUM) the G0 statistic averages the rows it just called uncomparable

`episode_summary` (`:615`) means `trace_in_spread` over **all** rows; `SUMMARY_KEYS` has no
notion of `trace_comparable`. On my run: reported 0.667, honest (comparable rows only) 0.500;
`abs_head_dz_err` 916.8 vs 775.8. On a pre-F9 trace (the 08-28 rig episodes that exist today)
every vetoed replan silently counts toward fidelity.

### 6 (MEDIUM) `chunk_metrics.close_step` is not `close_index`

`CLOSE_THR = 0.45` with the comment "same aperture rule as terminal_eval / close_index"
(`:114`), used as a bare `grip > 0.45` (`:362`). `close_index` has a max-relative fallback for
wide grasps. On a Carton-like chunk closing to 0.43: replay `close_step = 16` ("never closed"),
`close_index` = 14. Carton is the second session-4 task and 24% of its demos close below 0.45.
`close_step` / `trace_close_step` are printed and in the JSON. (VALIDATION #43, unfixed.)

### 7 (LOW) `grasp_label` abstains on a truncated close-on-AIR

`inconclusive` fires when the only reasons are `hold_truncated`, `c_hold`, `lift` (`:403-406`).
`c_hold` is a MEAN over the measured window, so 0.00 is a measured zero, not a truncation
artifact. Truth-table case 7 (truncated, legal height, **no contact at all**, lift 0.0) →
`inconclusive=True`, counted `trunc_none`. Every `veto_retry_cap` episode lands here by
construction (the cap fires ~1 replan after the close, so the recording is always truncated),
so the veto's own phantom-grasp detections leave the confusion table as "unmeasurable".

### 8 (LOW) F6's gripper release is indistinguishable from a real let-go

F6 opens the gripper in `_halt` on `tactile_*` / `wrench_limit` / `hitbox_exit` /
`veto_retry_cap`. That reopen is a `_release_after` release, so `released=True`,
`hold_truncated=False`, and a genuine grasp cut short by a fingertip-protection stop is scored
`no_*` with reasons `['hold','lift']` — byte-identical to a real let-go (truth-table cases 8
and 9). Nothing writes `EpisodeResult.stopped_reason` into `meta`, so it is not recoverable
post hoc either.

### 9 (LOW) `--parity-fixes` tag mismatch is a warning, not a refusal

F16 asked to "default it from the episode's `parity:` tag, and hard-fail when flag and tag
disagree". Replaying a `parity:on` episode without the flag exits 0 and writes the JSON; the
mismatch is one WARNING line among the per-replan log. The JSON does record both
(`parity_fixes:false`, `recorded_parity:true`), so it is recoverable.

## Things I checked and did NOT find a problem with

- **De-flaked snapshot test (F15).** 5 consecutive default-ordered runs of
  `tests/test_replay_rig.py`: 15 passed each time. The two-anchor acceptance is honest:
  `SnapshotBuilder.build()` really does read `rings["arm"]` twice (`latest(self._n_arm)` for
  the F/T anchor, then `latest(1)` for `ur_state`), so deploy's own snapshot can be internally
  skewed by one 125 Hz sample. My own rebuild-at-`t` comparison shows the residual: 5/6 exact
  on `wrist_window` and `ur_state`, worst |Δ| 0.15 (mock F/T units) / 0.03 mm. It is bullet 2
  of the module's own "Known fidelity gaps" list.
- **CRC32 per-episode seeding.** Documented in the module docstring and in `--seed`'s help,
  recorded per episode as `seed` + `seed_source`, `zlib.crc32` is platform-stable, and
  `episode_seed(7,'ep_a') - episode_seed(0,'ep_a') == 7`. Order/subset independence verified
  through `main()`.
- **`--seed-from-meta`.** `run_deploy.episode_seed` returns `os.urandom(4)` when `--seed` is
  omitted and tags it, so unseeded rig episodes are replayable; `seed:none` is refused.
- **All eight `--null` modes.** Asserted on the real batch tensors: `none` touches nothing but
  the GT package; `tactile` zeroes gel/fields/contact_state/**reactive**; `wrist` only wrist;
  `obs`/`all` the classifier-free null + black video; `contact_zero`/`contact_gt` pin the
  CONTACT frames; `events`/`cpk_*` zeroed everywhere but `contact_gt`. `null_semantics` in
  every JSON matches.
- **E9.** Every cell of the published table (incl. per-task) reproduces from
  `/tmp/e9_{none,tactile,contact_zero,contact_gt}.json`, and the "bit-identical to 2026-08-29"
  claim for `tactile`/`contact_gt` is exact (max |Δ endpoint| = 0 over 312 windows).
  `wrist`/`prev_cpk` carry no `null_semantics`, consistent with the doc saying they were not
  re-run.
- **E13 tables.** All 4 runs re-derive exactly from their rows (means, medians, per-task,
  across-seed std, n=496/312, `skipped:0`, `split:val`, `seeds:4`, `ema:true`).
- **grasp_label** on the veto retry (last close scored), truncation vs real release, `z_close`
  never abstained, OBJ never entering `grasp_ok`.

## Verdict

**NOT READY.** The tools are deterministic, order-independent and honest about their `--null`
semantics, and E9/E13's JSONs back every published cell — but the replay's Arm-B path has two
fix-induced parity/bookkeeping defects that corrupt GATE G0 (`prev_cpk` not invalidated;
`trace_comparable` false for every Arm-B replan), and the rig doc still hands the operator the
offline numbers E13 was written to retire.
