# Lens: EVALUATION TOOLS VALIDITY — phantom @ d9c40f2

Scope: `tools/replay_rig.py`, `tools/replay_deploy_path.py`, `tools/terminal_eval.py`,
`tools/rig_trace_decompose.py`, `phantom/eval/grasp_label.py`, `tools/label_grasps.py`.
Question: **can checkpoint decisions be made on these numbers?**

Everything below was executed on this Mac with `/Users/sannikov/GitHub/phantom/.venv/bin/python`
against mock-driver deploy episodes generated through the real `DeploymentRuntime` path
(the same construction `tests/test_replay_rig.py` uses), plus a tiny random checkpoint
(`build_model(tiny=True, load_base=False)` + `save_phantom_checkpoint`).
Scratch: `.../scratchpad/validate/scratch/evaltools/`.

## 0. Baseline — the suites pass

    .venv/bin/python -m pytest tests/test_replay_rig.py -q            -> 4 passed
    .venv/bin/python -m pytest tests/test_terminal_eval_null.py \
        tests/test_grasp_label.py tests/test_terminal_eval_close.py -q -> 41 passed

So this is not a "the fixes do not run" report. Every problem below is a **validity**
problem: a number that is computed successfully and means something other than what the
tool (or the doc that quotes it) says.

## 1. E9 — the paper-premise table has one row inverted

`docs/review_20260828/E9_premise_test.md` row 5 is labelled
"CONTACT frames zeroed (no GT package)" with commit 0.94 -> 0.65, and the Reading says
"the co-denoised GT contact package carries MORE of the terminal commit than the live
tactile input ... this is P7 (exposure bias) measured, and **the direct argument for
`--contact-self-forcing` in FT-A**". The commit message agrees:
`docs: E9 premise test — ... zeroed GT contact package costs 31%`.

But the header of that same file says the five rows are `--null {none,tactile,wrist,prev_cpk,contact_gt}`,
and in `tools/terminal_eval.py` there is **no mode that zeroes the CONTACT frames as a
distinct condition**:

    terminal_eval.py:306-312   if not pins_contact(args.null):
                                   for k in list(batch):
                                       if k == "events" or k.startswith("cpk_"):
                                           batch[k] = torch.zeros_like(batch[k])

i.e. `--null none` (row 1, "real inputs", commit 0.94) **already** hands the model a zeroed
GT package with the CONTACT frames co-denoised from noise — the deploy condition.
`--null contact_gt` is the opposite: it keeps the GT package and cond-PINS the CONTACT
frames (`contact_pinned_layout`, `terminal_eval.py:133-142`).

The repo's own test asserts exactly this:

    .venv/bin/python -m pytest \
      "tests/test_terminal_eval_null.py::test_every_null_mode_runs_and_nulls_what_it_names" \
      -q -k "none or contact_gt"        -> 2 passed
    # test body: assert rec.zeroed("events") is (null != "contact_gt")
    #            pinned = [bool(m[CONTACT].all()) for m in rec.cond_masks]
    #            assert all(pinned) is (null == "contact_gt")

and `git diff --stat 756987e d9c40f2 -- tools/terminal_eval.py` is empty, so the tool was
byte-identical when the E9 table was produced.

**Therefore row 5 is the GT-PINNED run.** Read correctly, the measurement says: *handing the
model the true future contact package makes it commit LESS* (0.94 -> 0.65) and close ~1.8
steps EARLIER — which is not "the head reads the answer off GT contact tokens". The premise
sentence, the row label and the commit message all state the direction backwards, and that
row is the stated empirical justification for one of the two headline FT-A flags on a paid
H100 fine-tune and for the P7 exposure-bias claim in the paper.

(Note also that P7's own prescription in REVIEW_SYNTHESIS is "CONTACT frames cond-pinned to
GT **vs zeros**". terminal_eval implements pinned-GT vs *unpinned*. The comparison is still
informative, but it is not the one P7 asked for, and no `--null contact_zero_pinned` exists.)

## 2. `replay_rig` cannot replay a `--parity-fixes` episode

`SnapshotBuilder` has three parity switches (planner.py:83-96): measured `prev_chunk`,
measured contact-state `dt`, consecutive-frame `reactive`; and `PhantomPolicy.replan`
additionally passes `prev_cpk_step = round(L/latent_dt)` under parity (policy.py:246-247).
`replay_rig.RigEpisode.snapshot()` implements **none** of them: it never sets
`snap.prev_chunk`, always uses the nominal `dt_field = 1/field_ds_rate_hz`
(replay_rig.py:161), always derives `reactive` from the previous *replan*
(replay_rig.py:183-184), and calls `rf.sample(...)` without `prev_cpk_step`
(replay_rig.py:335-338).

I generated two mock deploy episodes, one with `parity_fixes=False` and one with
`parity_fixes=True`, kept every `ObsSnapshot` the runtime handed the policy, and extended
the test's parity check from `(ur_state, wrist_window)` to **every** ObsSnapshot field
(`scratch/evaltools/parity.py`):

    == parity: 6 trace rows, 6 deploy snapshots
     replan 3 : rgb=DIFF 56  wrist_window=OK  ur_state=OK  gel=OK  fields=OK
                contact_state=DIFF 19.3  reactive=DIFF 0.001381  prev_chunk=MISMATCH(replay=None)
     replan 4 : ... reactive=DIFF 0.0178   prev_chunk=MISMATCH(replay=None)
     replan 5 : ... reactive=DIFF 0.01626  prev_chunk=MISMATCH(replay=None)
    deploy reactive: [0.0, 0.001688, 0.001731, 0.001686, 0.00776,  0.001773]
    replay reactive: [0.0, 0.001701, 0.001694, 0.003067, 0.025559, 0.018034]   # 3x and 10x off

On the non-parity episode the same check is clean apart from the documented +-1-row slack
(`ur_state=DIFF 0.012` once, `reactive` within 2e-3), so the tool *is* faithful to the
legacy deploy — it is the new arm it cannot follow. `meta.tags` records `parity:on/off`
(run_deploy.py:529) and `replay_rig` reads only `trace[0]["diag"]["nfe"/"guidance"]`
(replay_rig.py:300-304), so it does not even warn.

Session 4's Arm B is `--parity-fixes` (docs/rig_session_v5.md), and D4's instruction is
"Score FT-A on **replay**, not terminal_eval". Replaying those episodes silently conditions
on the legacy intent channel — the very channel E3/P2 is trying to measure.

## 3. `replay_rig` results depend on the order of `--episodes`, and cannot reproduce a seeded rig episode

`main()` seeds once, before the episode loop (`replay_rig.py:399`):

    policy.rf._gen = torch.Generator().manual_seed(args.seed)

`replay_episode` then calls `policy.reset_episode()` (which clears `_episode_noise`) but never
reseeds, so every episode after the first starts from wherever the previous episode's replans
left the generator. Same episode, same `--seed 1000`, same tiny checkpoint:

    replay_rig.py --tiny --seed 1000 --episodes B
      head_dz per replan: [-2691.68, 1183.05, -2435.75, 256.05, -1796.75, -1473.04]
    replay_rig.py --tiny --seed 1000 --episodes A B
      head_dz per replan (episode B): [-597.32, -64.04, -620.40, -281.22, -727.68, -362.58]

So E0's "trace inside the K-seed spread on >=4 of 6 episodes" (GATE G0) and any FT-A-vs-v5_6
replay comparison change if the episode list or its order changes — which it will, because
invalid episodes are dropped per-checkpoint.

Second half of the same defect: since `ba61354`, `run_deploy` seeds the sampler **per episode**
and records it (`episode_seed`, run_deploy.py:319-324, 665-671; tag `seed:<n>`, line 671).
`replay_rig`'s docstring still says "the recorded episodes carry `seed:none`, so the trace can
only be checked against the spread" (replay_rig.py:16-18) — stale for every episode recorded
from now on. Neither tool reads `seed:<n>` out of `meta.tags`, so the strongest available
validity check (exact bit-reproduction of the recorded chunk) is unavailable exactly when it
finally became possible.

## 4. On a `--terminal-veto` episode the trace chunk is a scripted chunk, and `trace_in_spread` compares against it

`PlannerLoop._apply_veto` rewrites `plan.actions` **in place** — `a[:,6] = grip_now`
(close_masked, planner.py:415), `a[:,6] = v.open_aperture` plus a cumulative-z clamp
(recovery_open, planner.py:386-394) — and `run()` writes `"actions": plan.actions.tolist()`
*after* that call (planner.py:486-495). `replay_rig` builds `trace_m = chunk_metrics(r["actions"])`
(replay_rig.py:343-345) and reports `trace_head_dz`, `trace_grip_max`, `head_dz_err` and
`trace_in_spread` against it, and never looks at `r["terminal_veto"]`.

On a vetoed replan those "trace" values are the veto's arithmetic, not a model sample, so
`trace_in_spread` is guaranteed-uninformative there — and `trace_in_spread` is the G0 gate.
(`--prev-chunk proposal` conditioning on the post-veto chunk is correct, since deploy's
`prev_plan` is the same mutated object; only the comparison is wrong.)

## 5. `replay_deploy_path` — the E0 discriminator has no smoke path and models the pre-fix deploy

    .venv/bin/python tools/replay_deploy_path.py --ckpt <tiny.pt> --hardware <hw> --episodes <ep> --nfe 1
    -> AssertionError: Torch not compiled with CUDA enabled        # dargs.device = "cuda", line 57

Forcing `device="cpu"` gets one step further and then:

    FileNotFoundError: .../cosmos-predict2.5-2b/robot/action-cond/..._ema_bf16.pt   # tiny=False, line 57

There is no `--device`, no `--tiny`, and no test anywhere (`grep -rln replay_deploy_path tests/`
returns nothing) — so the tool that produced the constant-seed finding cannot be exercised at all
off a GPU box with the full 2B weights. `dargs` also pins `parity_fixes=False, k_seeds=1`
(line 59-61) and builds no `TerminalVeto`, so it cannot be aimed at a Session-4 Arm-B episode.

`--deploy-rng` (line 46-50, "reproduce the rig's ACTUAL noise") hardcodes
`manual_seed(0)` + one warm-up replan on `fake_obs` (lines 76-80). That is precisely the
**pre-2026-08-29** deploy. After `ba61354`, `run_deploy` overwrites `_gen` with
`manual_seed(episode_seed(args.seed, i))` *after* the warm-up, so the flag reproduces nothing
for a new episode, and its help text does not say so. What is needed is a second mode:
read `seed:<n>` from `meta.tags`, `manual_seed(n)`, `reset_episode_noise()`, **no** warm-up.

## 6. `grasp_label`'s hold rule needs recording that `run_deploy` does not guarantee

`grasp_ok` requires `hold_s = t_release - (t_close + 0.5) >= 2.0 s` **and** `lift >= 50 mm`
(grasp_label.py:275-316), where `t_release` falls back to the end of the recording
(`t_end = min(g_ts[-1], t_ts[-1])`, line 276-280). `DeploymentRuntime.run_episode` ends the
recording immediately after `planner.run()` returns — `planner.stop(); executor.stop();
recorder.stop()` (runtime.py:241-246) — with no post-close hold. A grasp that happens on one
of the last replans (or that ends the episode via `veto_retry_cap`) therefore cannot pass:

    # mock deploy episode whose policy commands gripper 0.9 from replan 4 of 6
    .venv/bin/python tools/label_grasps.py <ep> --hw <hw> --confusion
    ep_teacher_whiteboard_1788097650_000  whiteboard  ..  -  t_cl 2.0  z_cl 240  hold 0.0  c_hold 0.00  lift 0
      reasons: z_close 240mm > 181mm; hold 0.0s < 2.0s; c_hold 0.00 < 0.80; lift 0mm < 50mm

`hold 0.0s < 2.0s` here is *truncation*, not failure, and nothing in the output or in
`GraspLabel.reasons` distinguishes the two; the `--confusion` table that G2/G3 read counts it
as `no_none`. Rig sessions run `--max-replans 40` (~36 s) against 16-31 s demos, so the tail
usually exists — but it is not guaranteed, and G2 is a 16-episode decision.

## 7. `terminal_eval` hard-fails on any student checkpoint

    terminal_eval.py:269   pm = build_model(hw, paths, student=False, tiny=args.tiny, mc=mc, ...)
    builder.py:48          assert mc.student == student

    .venv/bin/python tools/terminal_eval.py --ckpt <tiny_student.pt> --data ... --tiny --split all
    -> AssertionError: mc.student must agree with the student flag

`replay_rig.build_policy` got this right (`student=mc.student`, replay_rig.py:262); terminal_eval
was missed by the same pass. D9/D12 evaluate the HID student offline. (It also omits
`inference=True`, which is only an activation-checkpointing difference — harmless.)

## 8. `--null tactile` is three different kinds of null, and it is not the student ablation

`WindowSampler.sample` normalises `fields` (`windows.py:385`) but leaves `contact_state`
(`:398`) and `gel` (`:386`) raw. `null_batch(..., "tactile")` zeroes all three
(terminal_eval.py:107-109), so the model is handed:

* `fields`  -> the **dataset-mean** field (normalised zero), not an uncontacted gel;
* `contact_state` -> true zeros (no wrench, no area, no `mask_frac`);
* `gel` -> a **black** image, which is off-distribution in the other direction (an
  uncontacted gel is a bright textured frame).

The docstring calls this "exactly the streams the student layout has no frames for"
(terminal_eval.py:22-24, and the test name `test_tactile_null_drops_exactly_the_student_deleted_streams`),
but the student layout *removes the frames*, which — by the review's own P7 argument about
`drop_video` shifting `pos_emb_t` — is not the same intervention. The E9 headline ("removing
the live tactile input costs 16% of terminal commitment") and gate G1b are read off this.
One-minute check before quoting it again: denormalise the zeroed `fields` tensor with the
checkpoint's `norm_stats` and run `derived.derive_timestep` on it; if the resulting `mask_frac`
is above `tau_contact_area`, "tactile nulled" is literally "average contact".

## 9. `--jpeg-quality` hard-imports cv2, which is a rig-only extra

    replay_rig.py:134   import cv2
    .venv/bin/python tools/replay_rig.py --tiny ... --jpeg-quality 30
    -> ModuleNotFoundError: No module named 'cv2'

`opencv-python` is only in the `hw` extra of `pyproject.toml` (line 47, "Real hardware drivers
(recording rig / deployment box only)"), and the repo's own codec
(`phantom/data/jpeg_codec.py:25-42`) falls back to Pillow precisely because cv2 may be absent.
The exception is also not one of `(KeyError, IndexError, FileNotFoundError)` caught at
replay_rig.py:316, so it kills the whole run rather than skipping a replan. The image-fragility
probe is therefore unrunnable on any box that is not a rig.

Related, same file: `--merge-lora` is only honoured in the non-tiny branch
(replay_rig.py:267) and is silently ignored under `--tiny`; and the E0 recipe in the module
docstring (lines 22-27) omits `--merge-lora` even though `run_deploy.build_policy`
**always** folds the LoRA (`run_deploy.py:97-99`), so E0 as documented runs a numerically
different network from the rig.

Also: `chunk_metrics`'s `close_step` is a bare `grip > 0.45` (replay_rig.py:70, 217) while the
comment claims "same aperture rule as terminal_eval / close_index"; `close_index`
(common.py:342-363) additionally requires a 0.15 rise over the running minimum. Every replan
after the gripper has closed therefore reports `close_step = 0`, and the episode-summary mean
of `close_step` (replay_rig.py:359-368) is dominated by those.

## 10. `Z_MAX_MM` is a hand-written literal and the documented way to re-derive it uses a different close rule

`grasp_label.Z_MAX_MM` (waffles 103 / Carton 144 / egg 101 / whiteboard 181 mm) is a hardcoded
dict whose docstring says to re-derive it with `tools/rig_trace_decompose.py demos <task_dir>`
after any table or TCP-offset change. But `rig_trace_decompose.close_idx` (lines 30-33)
implements only `close_index`'s **primary** rule (`>0.45` after a `>0.15` rise) and not the
max-relative fallback that `close_index` needs for Carton, which "never crosses 0.45"
(common.py:346-353: coverage 76% -> 99% with the fallback). So the regeneration procedure
measures z_close on a biased 76% subset of Carton — the task whose headroom the docstring itself
calls thin ("143 vs 144 mm").

`tools/gen_start_poses.py`, which regenerates the *other* per-task envelope table
(`configs/start_poses.yaml`: `tcp_z_min`, `tcp_min/tcp_max`, `q_mean/q_std`), emits no
`z_close` statistic at all (`grep z_close tools/gen_start_poses.py` -> nothing), so the two
per-task geometry tables that gate the rig (safety floor) and the labels (grasp_ok) are
maintained independently with no cross-check.

## Verdict

**NOT READY.** The tools run and are well tested at the unit level, but three of the numbers
they will be asked to produce in the next two weeks do not mean what the docs say: the
published E9 premise row is the GT-pinned condition read as its opposite; `replay_rig` cannot
follow the `--parity-fixes` / `--terminal-veto` arm it is meant to score and is not
reproducible across `--episodes` orderings or against the new per-episode `seed:<n>`; and
`grasp_label`'s hold criterion depends on recording tail that `run_deploy` does not guarantee.
Fixes 1-4 are each under ~30 lines; until they land, replay numbers can rank checkpoints only
within one fixed episode list, and no E9/P7 sentence should go into the paper.
