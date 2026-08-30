# Lens: PROVISIONING + HUB — validation of phantom @ d9c40f2

Everything below was executed, not read. Scratch under
`.../scratchpad/validate/scratch/provisioning/`.

## 1. What is provably fine

**The hub repo tarball is exactly HEAD.**
```
$ tar -xzf dataset_v3_packed/phantom_repo_latest.tar.gz -C unpack && cat unpack/COMMIT
d9c40f2
$ diff <(git archive --format=tar HEAD | tar -tf - | sort) <(tar -tzf ...tar.gz | sort)
2a3
> COMMIT                       # the only difference
$ bash tools/pack_repo.sh $S/pack && shasum -a 256 $S/pack/*.tar.gz dataset_v3_packed/*.tar.gz
87a74ab4ec1164642df0e8cb392c7298753ba3bc30fa771b91fef4bb8a7f8f04  (both)
```
`pack_repo.sh` is byte-reproducible against the blob on the hub. The pinned script it
emits pins `REPO_COMMIT=d9c40f2` and `bash -n` clean.

**All the new levers ride in the tarball.** grep inside the *unpacked hub tarball*:
`--nfe --terminal-veto --parity-fixes --k-seeds --seed --max-tcp-speed --home-joints
--z-floor --hitbox-margin` all present in `phantom/scripts/run_deploy.py`;
`--contact-nll-beta --contact-self-forcing --action-noise-per-strip
--no-action-t-max-of-two --ema-decay --cond-dropout --acc-two-pass` all present in
`phantom/train/train_teacher.py`.

**Hub layout verified with `list_repo_tree` on the exact paths** (never
`list_repo_files` on the episodes repo):
- `dataset_v3_packed/`: 8 task tarballs (72.0 GB), `batch_20260822.tar.zst` 30.65 GB,
  `manifests.tar`, `norm_stats.json`, `text_embeddings.pt` 0.82 GB,
  `phantom_repo_latest.tar.gz` 1.26 MB. **No `provision_v5.<sha>.sh`.**
- `teacher_v4_790eps/teacher_020000.pt` 0.39 GB + `train_v4.log`.
- `teacher_v5_batch0822/`: teacher_000500…003000.pt (6), 14 `te_*.json`, `train_v5.log`,
  `probe_summary.txt`, `ckpt_watch.log`.
- `rig_sessions/`: only `rig_20260818` and `rig_20260820(_final)` — **no 08-28**.

**The batch_20260822 intake reproduces the documented split.** I pulled all 325
`meta.json` from `armteam/phantom-episodes/archive/20260822_*` (34 sessions) and ran
the repo's own `holdout_sessions()`:
```
tags: {'full':325,'batch_20260822':325,'deliberate_failure':45,'undergrasp':45}
tasks: waffles 70 / waffles_fail 15 / Carton 70 / Carton_fail 15 / egg 70 / egg_fail 15 / whiteboard 70
added 325 val 46 train 279  =>  total manifest: train 991 val 124
```
So 1115 / 991 / 124 is what the rental will get, and no episode is refused by the new
P9 `is_trainable_episode` gate (all finalized, no `contaminated`/`unlabeled`).

**Intake is idempotent.** Ran normalize/place/manifest twice on a synthetic 1115-episode
tree; second pass appends 0 rows, keeps `intake_holdout.json`, assert block still passes.

**Stream names in the validation loop are correct**: `hw.tactile.sensors` = `[left, right]`,
`tactile_stream()` yields exactly the 10 tactile names the script hardcodes; the store uses
only stock `numcodecs.Blosc`/`VLenBytes`, so bare `zarr.open` (no phantom import) is safe.

**The FT-A objective bundle itself trains.** Tiny/synthetic 2-step run with
`--contact-nll-beta 0.5 --contact-self-forcing --action-noise-per-strip
--no-action-t-max-of-two --ema-decay 0.995 --cond-dropout 0 --acc-two-pass` completes and
checkpoints.

## 2. The blocker — the FT-A launch line cannot start

`train_teacher.py:313-331` has two drift checks. Line 316 calls
`C.load_phantom_checkpoint(...)`, which internally calls
`assert_model_config_matches` (`common.py:236-274`) and raises on any config difference
outside `("student","mask_wrist",*TRAIN_ONLY_MODEL_FIELDS)`. Only *afterwards*, at
line 326, does the fine-tune-tolerant check
`model_config_drift(saved_mc, mc, tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)` run.

```
DEAD (mutable but hard-failed earlier): ['action_noise_per_strip','action_t_max_of_two','cond_dropout_p']
```
Those are three of the six FT-A flags. Against the **real** shipped checkpoints:
```
v5_6  (teacher_v5_batch0822/teacher_003000.pt): action_t_max_of_two True, cond_dropout_p 0.1
v4    (teacher_v4_790eps/teacher_020000.pt)  : action_t_max_of_two True, cond_dropout_p 0.1
assert_model_config_matches(payload, mc_ftA)
  -> RuntimeError: model-config drift vs checkpoint:
     {'cond_dropout_p': {'checkpoint':0.1,'model':0.0},
      'action_t_max_of_two': {'checkpoint':True,'model':False}}
model_config_drift(..., tolerate=FINETUNE_MUTABLE_MODEL_FIELDS)
  -> hard: {}   soft: [action_noise_per_strip, action_t_max_of_two, cond_dropout_p,
                        contact_nll_beta, contact_self_forcing]
```
Reproduced end-to-end on the tiny model too (checkpoint trained without the bundle,
`--init-weights` + bundle → same RuntimeError, exit non-zero).

`tests/test_ft_a_fixes.py:218-243` unit-tests `model_config_drift` in isolation and never
goes through `--init-weights`, which is why 570 passing tests miss it.

**The provision smoke does not catch it either**: `provision_v5.sh:251-258` runs
`--init-weights` *without* the FT-A flags, so it prints `READY` and then a launch line
(lines 268-278) that dies in the first second. Line 290 is only a comment asking the
operator to re-run the smoke with the bundle by hand.

## 3. The pytest gate is red at HEAD

`provision_v5.sh:245` runs `python -m pytest tests/ -q` and treats failure as fatal.
Two full runs on this repo, same result:
```
1 failed, 570 passed, 96 warnings in 337.34s
FAILED tests/test_replay_rig.py::test_rebuilt_snapshot_matches_the_snapshot_builder
```
It passes alone (4.2 s) and fails in collection order. Minimal deterministic repro:
```
$ pytest tests/test_deploy_levers.py tests/test_deploy_parity_fixes.py tests/test_replay_rig.py -q
1 failed, 64 passed
$ pytest tests/test_deploy_levers.py tests/test_replay_rig.py -q          -> 40 passed
$ pytest tests/test_deploy_parity_fixes.py tests/test_replay_rig.py -q    -> 29 passed
```
Assertion is `test_replay_rig.py:122` `np.allclose(snap.wrist_window, dsnap.wrist_window,
atol=1e-5)` — the rebuilt window is completely different rows, snapshot ts differ by 21 µs.
Not CPU load (3× under 8 busy cores: all pass). So it is state left by the two deploy
suites, and the provisioning gate hard-fails after the whole ~100 GB pull.

Secondary: the gate's own error message is unreachable. `set -euo pipefail` + a failing
pipeline exits before the `[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PYTEST FAILED …" }`
clause. Verified with a 4-line repro: prints only `before`, exit 1, no message.

## 4. What FT-A needs that provisioning does not do

1. **Init checkpoint.** `REVIEW_SYNTHESIS.md:478` — "FT-A: 3k steps **from v5_6**". The
   script downloads only `teacher_v4_790eps/teacher_020000.pt` (line 205) and its launch
   line inits from it. `teacher_v5_batch0822/teacher_003000.pt` is never fetched.
2. **Run name.** The launch line reuses `--run-name teacher_v5_batch0822` — the shipped v5
   run and hub folder. `train_teacher` does **not** refuse a populated run dir (verified:
   a second run with the same name wrote `teacher_000001.pt` next to `teacher_000002.pt`).
   Uploading FT-A output to the same hub folder would overwrite the six shipped v5
   checkpoints, including the `teacher_003000.pt` that compute3's `DEMO.pt`/`v5_6` is
   byte-identical to.
3. **Checkpoint egress.** `grep upload_file|upload_folder|HfApi(` over `tools/ phantom/`
   returns only the NUC episode uploaders and provision's own `HfApi()`. Nothing uploads
   a training run. v5's hub folder carries a `ckpt_watch.log` from a watcher that is not
   in the repo. A destroyed rental (as happened to the v5 vast box) loses the run.
4. **Replay scoring.** D4 says "score FT-A on **replay**, not `terminal_eval`", but the
   rental has no rig episodes: provisioning downloads none, and the hub's `rig_sessions/`
   stops at 20260820. The 08-28 session that every P1/E2/H1 conclusion rests on exists
   only on compute3/the NUC. `terminal_eval` *does* work on the provisioned box
   (`resolve_episodes` reads `<data>/../manifests/all.jsonl`, so
   `--data $W/data/phantom-episodes/tasks --split val` gives the 124), but the READY block
   prints no ready-to-paste command for it.

## 5. The manifest gate is self-referential

`provision_v5.sh:191` computes the expectation from the file the intake just wrote:
`exp = {"train": 712 + hold["train"], "val": 78 + hold["val"]}`, `hold` = `intake_holdout.json`.
I ran the full assert block (lines 163-201, verbatim) on a synthetic tree built from the
hub's real 790-row `all.jsonl` + a synthetic 325-episode batch whose sessions hold out
differently:
```
episodes under tasks/ now: 1115
manifest: {'train': 997, 'val': 118} | expect {'train': 997, 'val': 118}
manifest: bijection onto disk, unique, disjoint splits OK
EXIT=0
```
997/118 passes every assert. Nothing pins 991/124. And the split is not archived: the hub's
`manifests.tar` is still the **790-row v4** manifest (`{'train':712,'val':78}`), so the
1115-row `all.jsonl` and `intake_holdout.json` that v5_6's 17.4 mm was measured against
exist nowhere but a destroyed rental. It happens to be reproducible (§1), but a future
`--val-min-eps` change or a re-tar would silently move the val set with no assert firing.
(`manifests.tar` is also missing `whiteboard.jsonl`/`whiteboard_fail.jsonl`; harmless —
per-task jsonl are written, never read.)

## 6. Exact rental sequence that would actually work today

```bash
# --- Mac, repo clean ---
cd ~/GitHub/phantom && bash tools/pack_repo.sh /tmp/pack
python - <<'PY'   # only if HEAD moved
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/tmp/pack/phantom_repo_latest.tar.gz",
  path_in_repo="dataset_v3_packed/phantom_repo_latest.tar.gz",
  repo_id="armteam/phantom-checkpoints", repo_type="model")
PY
scp /tmp/pack/provision_v5.d9c40f2.sh root@<vast>:~/     # the pinned script is NOT on the hub

# --- rental: H100 NVL 80GB, --disk 300 ---
export HF_TOKEN=hf_...            # read on armteam/* + NVIDIA license accepted
bash ~/provision_v5.d9c40f2.sh /workspace/phantom-v5      # ~15 min + ~100 GB
#   ^ TODAY this aborts at the pytest gate (§3). Until that is fixed:
#     pytest tests/ -q --deselect tests/test_replay_rig.py::test_rebuilt_snapshot_matches_the_snapshot_builder

# --- everything below is manual; the script does none of it ---
W=/workspace/phantom-v5
$W/.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil, os
p = hf_hub_download("armteam/phantom-checkpoints",
                    "teacher_v5_batch0822/teacher_003000.pt", repo_type="model")
d = "/workspace/phantom-v5/runs/teacher/teacher_v5_batch0822"; os.makedirs(d, exist_ok=True)
shutil.copy(p, d + "/teacher_003000.pt")
PY
cd $W/phantom
# smoke the REAL launch flags (2 steps) before paying for 3000:
$W/.venv/bin/python -m phantom.train.train_teacher \
  --data $W/data/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml \
  --allow-config-drift --run-name ftA_smoke --max-steps 2 \
  --init-weights $W/runs/teacher/teacher_v5_batch0822/teacher_003000.pt \
  --grasp-frac 0.3 --photo-aug 1.0 --acc-two-pass \
  --lr 2e-5 --lr-new-modules 6e-5 --warmup-steps 150 \
  --event-band-weight 0 --ema-decay 0.995 --cond-dropout 0 \
  --contact-nll-beta 0.5 --contact-self-forcing \
  --action-noise-per-strip --no-action-t-max-of-two \
  --batch-size 4 --grad-accum 2 --num-workers 8 --device cuda
#   ^ TODAY: RuntimeError model-config drift (cond_dropout_p, action_t_max_of_two) — §2
unset HF_TOKEN
nohup $W/.venv/bin/python -m phantom.train.train_teacher \
  ... same flags ... --run-name teacher_v5_ftA --max-steps 3000 \
  --ckpt-every 500 --eval-every 500 > train_ftA.log 2>&1 &

# scoring on the box (replay needs rig episodes that are not here):
$W/.venv/bin/python tools/terminal_eval.py --ckpt $W/runs/teacher/teacher_v5_ftA/teacher_00XXXX.pt \
  --data $W/data/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml \
  --seeds 4 --split val --out /tmp/te_ftA_XXXX.json
# egress — write this yourself, nothing in the repo does it:
#   HfApi().upload_file(..., path_in_repo="teacher_v5_ftA/teacher_00XXXX.pt", ...)
```

## Verdict
**NOT READY.** Provisioning itself is sound and hub-verified, but the run it exists to
launch cannot start (§2), the last gate is red at HEAD (§3), and the FT-A init checkpoint,
run name, checkpoint egress and replay data are all missing from the script (§4).
