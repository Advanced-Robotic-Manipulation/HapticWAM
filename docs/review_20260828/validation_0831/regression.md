# LENS: per-fix regression witnesses + collateral damage — HEAD 820b4eb (2026-08-31)

Everything below was produced on this Mac with `.venv/bin/python`, mock drivers and
`--tiny`, through the real entry points (`run_deploy.main`, `train_teacher.main`,
`tools/replay_rig.py`, `tools/terminal_eval.py` helpers, pytest). No file under
`/Users/sannikov/GitHub/phantom` was modified; scratch under
`.../scratchpad/validate2/scratch/regression/`.

## 0. Baseline

    .venv/bin/python -m pytest tests/ -q
    661 passed, 129 warnings in 367.06s          # DEFAULT (pytest-randomly) order, no -p no:randomly

    for i in 1..8: pytest tests/test_replay_rig.py -q   ->  15 passed  x8
    (F15: the flake was ~1 run in 4 on this file before)

## 1. F1-F20 table

| F# | witnessed? | collateral-suspect | verdict |
|----|------------|--------------------|---------|
| F1 tolerate= on init-weights | YES — ran the *playbook* bundle (no `--contact-self-forcing`) through `train_teacher.main`: `WARNING --init-weights: training-only model flags differ ... {'cond_dropout_p': (0.1, 0.0), 'action_t_max_of_two': (True, False), 'contact_nll_beta': (None, 0.5), 'action_noise_per_strip': (False, True)}` then `step 2/2 ... total=10.8409`, `RC 0`, saved `cond_dropout_p 0.0 / action_t_max_of_two False` | `--resume` still strict (test); `tolerate` widens the ignore set only for the `--init-weights` branch. BUT the *shipped* launch line (`tools/provision_v5.sh:275,295`) still carries `--contact-self-forcing` → finding F-3 | clean in code, stale in the runbook |
| F2 running-min close rule | YES — on the repo's recorded ramp (0.31→0.52, cmd = meas+0.04) the mask now fires from replan 3 (`close_masked`), the old rate rule never did | the running minimum is never reset, so once the aperture has risen >0.15 above the episode min the mask stays armed for the rest of the episode | see F3 |
| F3 executed-close latch + expiry + cpk clear | PARTLY — expiry works (`recovery` only at n-idx<=1; tests + my run); `_invalidate_cpk` fires on both rewrites | **BROKEN**: `close_masked` writes `a[:,6] = grip_now`; `_note_executed_close` reads that back off `entered_grip_after` and arms `closed_idx`. Reproduced: `['none','none','none','close_masked','close_masked','recovery_open','close_masked','recovery_open','close_masked','recovery_open','close_masked','retry_cap']` → `veto_retry_cap` | **finding F-1 (high)** |
| F4 demo-close band | YES — computed through `build_veto` on the real `configs/start_poses.yaml`: band top = `Z_MAX_MM` exactly (Carton 144, egg 101, waffles 103, whiteboard 181 mm) vs the old 57–81 mm | on waffles/egg the band top is BELOW the model's own predicted close height (110.8 mm, E9) → every arm-B close hangs on the uncalibrated `p_contact`. That is what P0 #5 asked for; noted, not a defect | clean |
| F5 `--max-episode-s` + named cap | YES — end-to-end mock episode: `replan cap reached (8 replans, 0.4 s)` and `stop=replan_cap` in the ledger (was silent `None`); `--max-episode-s 35` default reaches `PlannerLoop.run` | none seen | clean |
| F6 gripper release on let-go stops | YES — real end-to-end mock run: `WARNING phantom.deploy.executor: stop (hitbox_exit): gripper RELEASED to 0.23 — if it is still closed, run ./GRIPPER_OPEN.sh` | `_halt` issues the release BEFORE `self._grip_target = None` / `self._stop.set()`. Probed: during the release move `_grip_target = 0.9`, `_stop.is_set() = False` | **finding F-6 (medium)** |
| F7 envelope assert / honest zfloor / ACC assert | YES — `--hitbox-margin 0.005 --z-floor-margin 0.010` → `UNSAFE ENVELOPE ...` rc 2; `zfloor:none` under `--no-z-floor`; `assert_acc_head` before `build_model` | `--no-z-floor` alone now returns **rc 2** on Carton (hitbox floor 46 mm vs workspace 30 mm) and whiteboard (38 vs 30); the printed remedy names `--z-floor-margin`, which is not in play | **finding F-5 (medium)** |
| F8 seeded homing + start tag | YES — `pytest -k "homing_jitter or realised_start"` 2 passed (both drive `run_deploy.main`) | `ep_tags` is rebuilt from `cond_tags` inside the loop, so no start-tag accumulation | clean |
| F9 trace provenance | YES — real mock episode with `--k-seeds 4`: `diag keys: ['guidance','head_dz_mm','head_dz_spread_mm','k_pick','k_rejected','k_seeds','nfe']`, `head_dz_mm: [2882.01,-550.9,668.43,-1481.51]`, `tcp_pose` on every row incl. the retry-cap row | the new `diag` filter accepts numeric lists; `event_logits` (ndarray) still excluded, JSON serialization OK | clean |
| F10 `--split` on the HID programs | YES (tests, `test_distill_hid_trains_on_the_train_split` / `test_finetune_hids_...`) | default flipped to `train`; `--split all` reproduces the old behaviour | clean |
| F11 persist event_band_weight | YES — the FT-A checkpoint I produced carries `configs.train.event_band_weight = 0.0` while `mc.loss.event` stays 0.5; resume restore/refusal covered by tests | `apply_overrides` treats `0.0` correctly (`is not None`) | clean |
| F12 terminal_eval student/reactive | YES — `student=mc.student` at :381 and the sampler at :395; `null_semantics('tactile').batch_streams_zeroed == ['contact_state','fields','gel','reactive']` | `inference=True` added to `build_model`; E9's own re-run reproduces 2026-08-29 to the last digit, so numerically inert | clean |
| F13 rederive gate | YES (tests; `intake_recovery.manifest` REFUSES, `build_index` warns) | `needs_rederive` only fires for `policy not in ('','teleop')`, so teleop demos and the 325 recovery demos are untouched | clean |
| F14 rig-doc arms | YES — one process per arm block, `--seed` warning, abort rules, `--max-episode-s 35` in Arm B | the same doc still quotes the retired offline numbers 30 lines later | **finding F-7 (medium)** |
| F15 de-flake + pytest gate | YES — 8/8 green on `test_replay_rig.py`, 661 passed in default order; `if ! python -m pytest ...` gate with a log tail | none | clean |
| F16 replay parity / per-episode seed / seed-from-meta / actions_pre_veto / `--ckpt` under `--tiny` | PARTLY — `--seed-from-meta` reads the recorded tag (`seed used 1599432872 meta`); order-independence proven in-process (same episode, list order reversed → bit-identical summary); `--parity-fixes` runs and warns on a `parity:off` episode | **`trace_comparable` is False on EVERY replan of a veto-on episode** (`bool(r.get("terminal_veto"))` is truthy for `action:"none"`/`"close_allowed"` too): 8/8 rows flagged, plus a false "pre-F9 trace" warning | **finding F-4 (high)** |
| F17 start_poses regenerated + q_n gate | YES — shipped file has `q_n: 250 == n: 250` on all four tasks; `load_start_stats` raises `ThinStartStatsError` otherwise (tests) | `tools/episode_qc.py` also calls `load_start_stats()` and would now refuse a thin file — correct, but it is an offline tool | clean |
| F18 E9 relabel + contact_zero | YES — `null_semantics` prints the two switches for all 8 modes; `contact_zero` = zeroed package + pinned frames; every run prints/records them | none | clean |
| F19 EpisodeMeta.weight + commit band | YES (tests; `--commit-band-weight` rides through the real CLI) | `action_weight = 0 if failure else max(0, weight)`; failure demos still forced to 0 | clean |
| F20 rederived gripper channel | YES (tests) | falls back to the measured aperture only with a loud warning | clean |

## 2. Findings (detail is in the structured output)

F-1 (high) planner.py:410 — the veto reopens a grasp it never allowed.
F-2 (high) grasp_label.py:232 — `close_attempts` re-runs `close_index` on a slice, so the
          Carton max-relative fallback fires on post-release aperture drift.
F-3 (high) provision_v5.sh:275,295 — `--contact-self-forcing` still in the paid FT-A line.
F-4 (high) replay_rig.py:572 — every veto-on replan declared uncomparable → G0 unreadable.
F-5 (med)  run_deploy.py:570 — `--no-z-floor` now rc 2 on Carton/whiteboard.
F-6 (med)  executor.py:163 — release issued before the mailbox is invalidated.
F-7 (med)  rig_session_v5.md:154 — retired 17.4 / 20.7 / 14.9 still quoted.
F-8 (low)  provision_v5.sh:311 — the printed checkpoint-selection replay omits `--merge-lora`
          / `--seed-from-meta` and passes `--parity-fixes` on parity:off 08-28 episodes.
F-9 (low)  recovery_demos_protocol.md:55 — still names terminal_eval as the selection tool.

## 3. Things I checked and did NOT find a problem in

- `check_world` fatal only for `nproc_per_node > 1`; the committed default target is
  `rtx5090` (nproc 1), so no single-GPU run is affected.
- `make_loader(shard=False)` also flips `drop_last`; only the val loader uses it.
- `_invalidate_cpk` → `prev_cpk=None` is the same path the first replan already takes.
- `ep_tags` is rebuilt from `cond_tags` every episode (no start-tag accumulation).
- The `diag` list filter keeps the trace JSON-serializable with `--k-seeds 4`.
- `replay_rig --tiny` is nondeterministic process-to-process because the tiny backbone is
  random-init and nothing seeds the global RNG — pre-existing, not a regression; it does mean
  `--tiny` numbers must never be compared across processes.
