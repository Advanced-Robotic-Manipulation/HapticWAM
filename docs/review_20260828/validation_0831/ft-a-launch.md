# Lens: FT-A LAUNCH END-TO-END — PHANTOM @ 820b4eb, 2026-08-31

Everything below was produced by executing code on the Mac (`.venv/bin/python`, cosmos
importable, `--tiny/--synthetic`, mock/no drivers) plus read-only HF hub queries. Scratch:
`/private/tmp/.../validate2/scratch/ft-a-launch/`. Nothing under `~/GitHub/phantom` was modified
(`git status --short` empty before and after; `pack_repo.sh` wrote only into scratch).

## 0. Verdict

**NOT READY.** The *code* is ready — the F1/F11 drift-and-resume fixes are real and I reproduced
them on the actual shipped v5_6 checkpoint. What is not ready is the **artifact and the runbook**:
the hub repo tarball is still `d9c40f2` (pre-fix), the printed launch line still spends the paid
run on `--contact-self-forcing` (removed from the recommended bundle on 2026-08-30, and pinned in
place by two tests), and the provisioning pytest gate is still a live coin-flip.

## 1. `tools/provision_v5.sh` — static trace

`bash -n tools/provision_v5.sh` → clean.

Section by section:

| lines | what | verdict |
|---|---|---|
| 16-38 | `set -euo pipefail`, zstd/git/nvidia-smi preflight, disk warn, venv, `huggingface_hub>=1.20,<2` | OK |
| 50-58 | repo tarball + `COMMIT` pin | **see §2 — the pin cannot see staleness** |
| 60-72 | cosmos repo pinned to `a2c298b`, deps, `torch.cuda.is_available()` early abort | OK |
| 74-126 | cosmos weights, dataset tarballs w/ per-task `.complete` resume | OK |
| 128-202 | recovery intake, 1115-episode assert, per-stream zarr open, manifest bijection | OK |
| **204-215** | **FT-A init = `teacher_v5_batch0822/teacher_003000.pt`** (v5_6), v4 kept as a commented control | **FIXED** (was P0 #18) |
| 217-250 | `paths.local.yaml`, GPU verify, text-cache preflight | OK |
| **252-261** | pytest gate, now `if ! python -m pytest ...` | error reporting FIXED, **the test is still flaky — §4** |
| 263-283 | 2-step REAL smoke with the FULL bundle | runs the bundle, but the bundle is wrong — §3 |
| 285-328 | printed launch line + notes | `--run-name teacher_v5_ftA` FIXED; `--contact-self-forcing` NOT — §3 |

**Every flag in the printed launch line exists in `train_teacher`'s parser** (checked
programmatically — rendered the echo block with `W/FTA_INIT/BS/GA/NW` bound, regex-extracted 25
long options, matched against `add_argument` strings scraped from `train_teacher`):

```
flags in launch line: ['--acc-two-pass','--action-noise-per-strip','--allow-config-drift',
 '--batch-size','--ckpt-every','--cond-dropout','--contact-nll-beta','--contact-self-forcing',
 '--data','--device','--ema-decay','--eval-every','--event-band-weight','--grad-accum',
 '--grasp-frac','--hardware','--init-weights','--lr','--lr-new-modules','--max-steps',
 '--no-action-t-max-of-two','--num-workers','--photo-aug','--run-name','--warmup-steps']
KNOWN: 41
MISSING FROM PARSER: []
```

## 2. The hub tarball is still the pre-fix commit

```
$ python - <<'PY'  # read the COMMIT out of the tarball the rental will unpack
from huggingface_hub import hf_hub_download; import tarfile
p = hf_hub_download("armteam/phantom-checkpoints","dataset_v3_packed/phantom_repo_latest.tar.gz",repo_type="model")
t = tarfile.open(p); print("HUB TARBALL COMMIT =", t.extractfile("COMMIT").read().decode().strip())
print("has tools/upload_run_ckpts.py:", "tools/upload_run_ckpts.py" in t.getnames())
PY
HUB TARBALL COMMIT = d9c40f2
has tools/upload_run_ckpts.py: False
```

`d9c40f2` is the commit the 2026-08-30 validation declared NOT READY. `provision_v5.sh:55-57`
compares the tarball's `COMMIT` against `REPO_COMMIT`, which `pack_repo.sh:22` bakes into the
*script*. So the gate proves **script ↔ tarball agreement**, never **tarball ↔ HEAD**. Both
branches are bad:

* operator scp's the script they already have (`provision_v5.d9c40f2.sh` — the only one that was
  ever produced, per `docs/review_20260828/validation_0830/provisioning.md:175`): prints
  `repo commit d9c40f2 verified` and provisions the unfixed tree. `git show d9c40f2:tools/provision_v5.sh`
  inits FT-A from **v4**, prints `--run-name teacher_v5_batch0822` (the folder holding the six
  SHIPPED v5 checkpoints), and its smoke omits the objective knobs — so the smoke PASSES and the
  paid 3000-step launch then dies at load on the F1 drift crash.
* operator re-runs `pack_repo.sh` (I did, into scratch: `packed 820b4eb`, pin
  `REPO_COMMIT=${REPO_COMMIT:-820b4eb}`, egress helper present) but forgets to re-upload: the
  COMMIT gate fires loudly — after the ~15 min / ~100 GB pull.

Nothing in the repo uploads the tarball; the upload is a hand-typed `HfApi().upload_file` snippet
that lives only in `VALIDATION_0830.md:571-577`.

## 3. `--contact-self-forcing` is still in the paid launch, and two tests pin it there

`e06c33b` removed it from the playbook: `docs/training_playbook.md:89` now reads *"Removed from
the recommended FT-A bundle 2026-08-30 … do not spend the primary FT-A run on it"*, and
`E9_premise_test.md:88-93` says E9 "provides no exposure-bias gap and no offline evidence that
self-forcing helps".

The thing that actually runs still has it, twice:

```
tools/provision_v5.sh:275   --contact-nll-beta 0.5 --contact-self-forcing \        # the smoke
tools/provision_v5.sh:295 echo "    --contact-nll-beta 0.5 --contact-self-forcing \\"   # the launch
tools/provision_v5.sh:317 ... "--contact-self-forcing denoises ACTION alongside the model's OWN ..."
```

and rendering the launch block confirms the operator copy-pastes it:

```
$ bash printlaunch.sh
  cd /workspace/phantom-v5/phantom && nohup .../python -m phantom.train.train_teacher \
    ...
    --contact-nll-beta 0.5 --contact-self-forcing \
```

Worse, the divergence is **locked in by the test suite**:

```
tests/test_ft_a_fixes.py:296        for flag in ("--contact-nll-beta 0.5", "--contact-self-forcing", ...
tests/test_fixnow_train_0830.py:623 for flag in ("--contact-nll-beta 0.5", "--contact-self-forcing", ...
```

Deleting the flag from the script turns both tests red — so this is not a one-line edit anybody
will do by accident; it has to be done deliberately.

I verified the corrected bundle *runs*: on the REAL v5_6 payload,
`model_config_drift(saved, mc, tolerate=FINETUNE_MUTABLE)` with `contact_self_forcing=False`
gives `HARD: {}` and the same `soft` set as with it, so removing the flag costs nothing at load.

## 4. The provisioning pytest gate

Error reporting: **fixed**. I reproduced the gate in isolation against a deliberately failing test
file — it prints the tail, prints `PYTEST FAILED — full log: …`, and exits 1 under
`set -euo pipefail`.

Flakiness: **not fixed, only narrowed.**

```
$ .venv/bin/python -m pytest tests/ -q            → 661 passed in 364.96s   (RC=0)
$ for i in 1..6: pytest tests/test_replay_rig.py -q
run 1..5 RC=0 (15 passed)   run 6 RC=1  1 failed, 14 passed
$ for i in 1..16: pytest tests/test_replay_rig.py::test_rebuilt_snapshot_matches_the_snapshot_builder -q
FAILURES: 1 / 16
E  AssertionError: replan 1: deploy's wrist window matches neither the rebuild at its own
   ur_state row 220 nor the one at 219
```

Root cause (unchanged by the fix): `SnapshotBuilder.build()` reads `rings["arm"].latest(self._n_arm)`
to anchor the wrist F/T grid at `ts_a[-1]`, then a **second** `rings["arm"].latest(1)` for
`ur_state`. Any samples landing between the two reads leave `ur_state` newer than its own wrist
anchor. The fix widened the admissible anchors to `{i_deploy, i_deploy-1}`; the race can advance
by ≥2 rows, so it still escapes. `provision_v5.sh:256` is fatal on it.

Two fixes, both small: (a) test-side, widen the anchor search to the rows spanned by one
`_n_arm` read; (b) better — build `ur_state` from the row already fetched in the first `latest()`
call, which makes the snapshot internally self-consistent (wrist anchor == `ur_state` row, the
way training builds it) and removes the race from deploy as well as from the test.

## 5. The FT-A bundle, end to end, through the real CLI

Driver: `drive.py` monkeypatches `load_paths().runs_root` into scratch and calls
`train_teacher.main(argv)`.

1. **base checkpoint** (v5_6's shape of problem):
   `--tiny --synthetic --acc-two-pass --max-steps 1 --ckpt-every 1 --run-name v5like_base`
   → `cond_dropout_p 0.1 | action_t_max_of_two True | action_noise_per_strip False |
   contact_nll_beta None | rope time_true | acc.sa two_pass | train.event_band_weight None`.

2. **FT-A, documented bundle, WITHOUT `--contact-self-forcing`**, 2 steps, `--ckpt-every 1`:
   ```
   INFO train_teacher: FT-A objective knobs active (non-default): {'contact_nll_beta': 0.5, 'action_noise_per_strip': True}
   WARNING train_teacher: --init-weights: training-only model flags differ from the checkpoint
     (checkpoint -> this run): {'cond_dropout_p': (0.1, 0.0), 'action_t_max_of_two': (True, False),
      'contact_nll_beta': (None, 0.5), 'action_noise_per_strip': (False, True)}
   INFO train_teacher: packed event-band MSE weight overridden: 0 (config 0.5)
   INFO ... step 1/2 ... step 2/2 ... checkpoint saved teacher_000002.pt
   EXIT 0
   ```
   product config: `cdp 0.0 atmot False anps True beta 0.5 csf False loss.event 0.5`,
   `train.event_band_weight 0.0`, `ema_decay 0.995`, `lr 2e-05`, `step 2`. **F1 and F11 land.**

   *(Aside: with `--grasp-frac 0.3` the synthetic fixture trips the coverage guard —
   `only 0/4 episodes weightable ({'band_outside_range': 4})`. That is the guard doing its job on
   synthetic gripper traces, not a defect; the rental smoke exercises it on real episodes.)*

3. **resume** (no `--event-band-weight`):
   `--resume: packed event-band MSE weight 0 restored from the checkpoint` →
   `overridden: 0 (config 0.5)` → `step 3/3`, EXIT 0. Resuming *with* the same
   `--event-band-weight 0` (what an operator copy-pastes) also works: `step 3/4 … 4/4`, EXIT 0.

4. **in-run validation** — built a scratch data root (`tasks/approach_contact/ep_* ` symlinks +
   `manifests/all.jsonl`, 3 train / 1 val) because the synthetic root has no manifest:
   `val: 8 windows from 1 episodes` → `EVAL step 1 val_action_v_mse=2.0437 …`,
   `EVAL step 2 …`. `--eval-every 500` will produce the health signal the runbook relies on.

5. **the product loads everywhere**:
   * `run_deploy.build_policy(--system teacher --ckpt <ftA> --tiny --terminal-veto)` →
     `model config from checkpoint: rope=time_true cond_dropout=0.00 acc=two_pass`,
     `EMA weights applied: 156 tensors (40 LoRA)`, `POLICY BUILT: PhantomPolicy`,
     `mc.action_noise_per_strip = True`. The `--terminal-veto` ACC-head assert passes.
   * `tools/terminal_eval.py --tiny --ckpt <ftA>` → `load_phantom_checkpoint` succeeds; it then
     dies at `ns["mean"]["action"]` because a synthetic-trained checkpoint carries identity norm
     stats. Fixture limitation, not a tool defect (the real root has `norm_stats.json`).
   * `tools/replay_rig.py --tiny --ckpt <ftA> --episodes data/episodes/deploy/20260827/ep_*`
     → 2 episodes × 3 replans, per-episode seeding (`seed 420460 (base + episode name)`),
     OVERALL block, JSON written. Same with `--parity-fixes`. `--seed-from-meta` correctly
     refuses episodes tagged `seed:none` with an actionable message.

## 6. Drift matrix (through `train_teacher.main`, not the helper)

`matrix.py` calls `main()` once per (field × mode) against the pre-flag base checkpoint:

```
field                          --init-weights          | --resume
cond_dropout_p                 OK(rc=0)                | REFUSED RuntimeError: model-config drift {'cond_dropout_p': ...}
action_t_max_of_two            OK(rc=0)                | REFUSED RuntimeError: ... {'action_t_max_of_two': ...}
action_noise_per_strip         OK(rc=0)                | REFUSED RuntimeError: ... {'action_noise_per_strip': ...}
contact_nll_beta               OK(rc=0)                | REFUSED SystemExit: --resume model-config drift ...
contact_nll_detach_w           OK(rc=0)                | REFUSED SystemExit: ...
wrist_region_mse               OK(rc=0)                | REFUSED SystemExit: ...
contact_self_forcing           OK(rc=0)                | REFUSED SystemExit: ...
mask_wrist                     REFUSED SystemExit      | REFUSED SystemExit
rope_time_mode                 REFUSED RuntimeError    | REFUSED RuntimeError
acc.self_anticipation          REFUSED RuntimeError    | REFUSED RuntimeError
student                        REFUSED RuntimeError    | REFUSED RuntimeError
```

The `tolerate=` relaxation is exactly as wide as `FINETUNE_MUTABLE_MODEL_FIELDS` and no wider —
`rope_time_mode`, `acc`, `mask_wrist` and `student` are all still fatal on `--init-weights`.
`--resume` is strict on every field, plus the `event_band_weight` train-config check
(`--resume event-band weight drift: checkpoint 0.0, this run 0.5` reproduced).

**On the REAL shipped v5_6 payload** (`teacher_v5_batch0822/teacher_003000.pt`, 0.39 GB,
downloaded from the hub):

```
saved: cond_dropout_p 0.1 | action_t_max_of_two True | action_noise_per_strip <absent>
       contact_nll_beta <absent> | rope_time_mode time_true | acc.self_anticipation two_pass
       step 3000 | has ema True | hardware_shapes.wrist_window_len 31

documented bundle (NO --contact-self-forcing) vs v5_6:
  HARD (fatal): {}
  soft (warn) : {'cond_dropout_p': (0.1, 0.0), 'action_t_max_of_two': (True, False),
                 'contact_nll_beta': (None, 0.5), 'action_noise_per_strip': (False, True)}
  assert_model_config_matches(tolerate=FINETUNE_MUTABLE): PASS
  assert_model_config_matches(strict): FAIL (expected)  ← the pre-F1 crash, still there for other callers

--init-weights norm_stats gate vs the hub's dataset_v3_packed/norm_stats.json:
  ckpt keys == data keys == ['action','area','cpk_d_disp','cpk_d_fz','fields','ur_state','wrench','wrist_ft']
  MISMATCHES: none -> gate passes
```

So the FT-A launch will get past both startup gates on the real artifacts. That is the single
most important positive result of this lens.

### per-strip noise does not leak onto a pre-flag checkpoint

`run_deploy.build_policy` rebuilds `mc` from `payload["configs"]["model"]`
(`run_deploy.py:52-62`), so:

```
base(pre-flag)     mc.action_noise_per_strip=False
ftA(per-strip)     mc.action_noise_per_strip=True
```

`replay_rig.build_policy:401-403` and `terminal_eval:374-376` do the same. A v5_6 deploy/replay is
unaffected by the flag; only the FT-A artifact samples strip noise, which is what
`rf.py:570` requires ("sampling MUST draw the same kind of ACTION noise the model was trained to
denoise"). No regression.

## 7. `tools/upload_run_ckpts.py`

Dry-run against the real hub folder, with three local files planted to cover all three branches:

```
$ python tools/upload_run_ckpts.py <scratch>/teacher_v5_batch0822 --dry-run
INFO: skip teacher_v5_batch0822/teacher_000500.pt (already on the hub, 0.39 GB)
INFO: would upload .../teacher_003500.pt -> .../teacher_003500.pt (0.00 GB)
INFO: would upload .../train_v5.log -> .../train_v5.log (0.00 GB)
INFO: DRY RUN: 2 uploaded, 1 already present -> armteam/phantom-checkpoints/teacher_v5_batch0822/
```

Collection and skip logic are right, and nothing was uploaded. **But the skip is byte-SIZE only**
(`upload_run_ckpts.py:113`), and the `teacher_000500.pt` it skipped was 393,115,861 bytes of
**zeros** — content it has never seen. Every PHANTOM teacher checkpoint is exactly 393,115,861
bytes, so a relaunched FT-A run under the same `--run-name` silently keeps the stale hub file. The
hub does expose the real digest:

```
teacher_v5_batch0822/teacher_003000.pt 393115861 lfs: BlobLfsInfo(sha256='7edcb833…', …)
```

so `list_repo_tree(..., expand=True)` + a local sha256 is a ~6-line fix.

Minor: with `--dry-run` and no token, `remote_sizes` swallows the 401 in its `except` and reports
every file as new. Harmless for the documented loop (which always has `HF_TOKEN`), misleading for
a bare sanity check.

## 8. Checkpoint selection as printed is not executable

`provision_v5.sh:306-312` (and `VALIDATION_0830.md` §3.3 step 5) instruct: *"per checkpoint, RAW
and EMA: python tools/replay_rig.py --ckpt <ckpt> …"*. `replay_rig` has **no** raw/EMA switch —
`--help` lists none, and `load_ema=True` is hardcoded at `replay_rig.py:420` and `:435`
("deploy loads EMA … the replay must too"). `terminal_eval` has `--no-ema`; the tool the runbook
moved selection *to* does not. The RAW half of the prescribed comparison cannot be run.

Two related loose ends in the same block, worth folding into the same fix:
* the printed replay command omits `--seed-from-meta`, which F16 added precisely so a replay
  reproduces the rig's recorded draw;
* it hardcodes `--parity-fixes` for the 08-28 episodes, which were recorded `parity:off`;
  `replay_rig.py:495-501` only **warns** on the mismatch (VALIDATION F16 asked for a hard fail).
* the 08-28 rig session is **not on the hub** (`rig_sessions/` holds only `rig_20260818.tar.gz`,
  `rig_20260820.tar.gz`, `rig_20260820_final.tar.gz`), so "scp it before the run ends" stays a
  manual precondition with no check anywhere.

## 9. `--allow-config-drift`

`VALIDATION_0830.md` §3.3 dropped it from the documented sequence with the note *"if it is needed,
find out why first"*. It is needed, and now I can say why: the shipped v5 run's own log
(`teacher_v5_batch0822/train_v5.log`, fetched from the hub) says

```
WARNING train_teacher: CONFIG DRIFT (allowed): 991/991 episodes were recorded under a DIFFERENT
  hardware config than 'configs/hardware.nuc.yaml'.
```

— 1115 per-episode warnings, every episode. `configs/hardware.nuc.yaml` is **unchanged** by the fix
campaign (`git diff d9c40f2..HEAD -- configs/hardware.nuc.yaml` empty; last touched 54b7943,
2026-08-16), and so is `phantom/config/hardware.py`, so this is pre-existing, not fix-induced. But
an operator following §3.3 verbatim gets `SystemExit: CONFIG DRIFT` after the 100 GB pull, and
nothing in either document records the 991/991 answer.

## 10. What I could not exercise

* the real 2-step GPU smoke (needs the box + the 100 GB dataset). Everything upstream of it —
  flag parsing, the drift gate on the real v5_6 payload, the norm-stats gate against the hub's
  own `norm_stats.json` — is verified here, so the smoke's remaining risk is VRAM and
  `--grasp-frac` coverage on real episodes.
* `h100x8` / torchrun (`configs/compute.yaml` target is `rtx5090`, so the single-process launch
  line passes `check_world(1)`; the v5 log confirms `compute target: rtx5090`).
