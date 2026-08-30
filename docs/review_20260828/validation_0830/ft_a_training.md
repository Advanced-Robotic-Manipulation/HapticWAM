# Lens: FT-A TRAINING PATH — validation of phantom @ d9c40f2

Scratch: `/private/tmp/claude-501/-Users-sannikov/a5af36c2-8450-43d1-9a34-fa0cdb820706/scratchpad/validate/scratch/ft_a/`
Repo untouched (`git status --porcelain` empty before and after; only gitignored `runs/`, `data/synthetic/` written).

## 0. Verdict

**NOT READY.** The FT-A bundle *as documented* — the launch line
`tools/provision_v5.sh` prints and `docs/training_playbook.md:69-74` reproduces — **cannot start**:
`--init-weights <v5_6> --no-action-t-max-of-two --cond-dropout 0 --action-noise-per-strip`
raises `RuntimeError: model-config drift vs checkpoint` inside `load_phantom_checkpoint`
before the first step. The P10B hardening (`common.py:236-275`) fires *before* the
finetune-tolerant drift check in `train_teacher.py:325`, so the code path the playbook
promises ("`--init-weights` warns (rather than fails)") is unreachable. Everything else
I exercised works once that one guard is relaxed.

## 1. What I ran

### 1.1 baseline tiny checkpoint (v5 flags)

```
.venv/bin/python -m phantom.train.train_teacher --tiny --synthetic --acc-two-pass \
  --max-steps 2 --ckpt-every 1 --device cpu --run-name ftbase_scratch
```
→ OK. `step 2/2 action_v_mse=2.6746 contact_nll=5.3016 total=12.7040`, two checkpoints written.

### 1.2 the documented FT-A bundle → HARD FAIL

```
.venv/bin/python -m phantom.train.train_teacher --tiny --synthetic --acc-two-pass \
  --contact-nll-beta 0.5 --contact-self-forcing --action-noise-per-strip \
  --no-action-t-max-of-two --ema-decay 0.995 --cond-dropout 0 \
  --init-weights runs/teacher/ftbase_scratch/teacher_000002.pt \
  --ckpt-every 1 --max-steps 2 --device cpu --run-name fta_scratch
```
```
File "phantom/train/common.py", line 298, in load_phantom_checkpoint
    assert_model_config_matches(payload, model)
RuntimeError: model-config drift vs checkpoint: {'cond_dropout_p': {'checkpoint': 0.1,
'model': 0.0}, 'action_t_max_of_two': {'checkpoint': True, 'model': False},
'action_noise_per_strip': {'checkpoint': False, 'model': True}}
```

Reproduced against a **v5_6-shaped** checkpoint (new flag keys stripped exactly as
`tests/test_v5_train_fixes.py::test_drift_check_tolerates_training_only_flags` strips them,
so `action_noise_per_strip`/`contact_nll_beta` are *absent* from the saved config the way
they are in the real v5_6/v4 payloads):

```
v5-like saved keys of interest: {'cond_dropout_p': 0.1, 'action_t_max_of_two': True,
 'action_noise_per_strip': '<absent>', 'contact_nll_beta': '<absent>'}
RuntimeError: model-config drift vs checkpoint: {'cond_dropout_p': {'checkpoint': 0.1,
'model': 0.0}, 'action_t_max_of_two': {'checkpoint': True, 'model': False}} — ...
```

and `--cond-dropout 0` **alone** (one documented lever, no bundle) is enough:

```
RuntimeError: model-config drift vs checkpoint: {'cond_dropout_p': {'checkpoint': 0.1, 'model': 0.0}}
```

Why the existing tests miss it: `test_drift_check_tolerates_training_only_flags` calls
`train_teacher.model_config_drift(...)` **directly**; nothing exercises the CLI's
`--init-weights` path. And `provision_v5.sh:251-257` runs its 2-step real smoke with the
**v5 flags only** — the comment at :290 even says "re-run it with the bundle appended
before launching", which nobody does, so the failure first appears at paid launch.

Minimal fix (verified): teach `load_phantom_checkpoint` / `assert_model_config_matches` a
`tolerate` set and pass `FINETUNE_MUTABLE_MODEL_FIELDS` from `train_teacher.py:316`
(`--init-weights` only; `--resume` keeps the strict set). I reproduced the whole run with
exactly that relaxation monkey-patched (`scratch/ft_a/run_fta.py`) and it works:

```
WARNING train_teacher: --init-weights: training-only model flags differ from the checkpoint
 (checkpoint -> this run): {'cond_dropout_p': (0.1, 0.0), 'action_t_max_of_two': (True, False),
 'contact_nll_beta': (None, 0.5), 'contact_self_forcing': (False, True),
 'action_noise_per_strip': (False, True)}
INFO phantom.train.common: EMA weights applied: 156 tensors (40 LoRA)
INFO phantom.train.common: step 2/2 action_v_mse=2.8066 contact_nll=6.7031 total=14.1832
```

### 1.3 checkpoint round-trip / consumers (all OK, with the guard relaxed)

* `configs.model` round-trips: `PhantomModelConfig.from_dict(cm).to_dict() == cm` → True,
  and carries `contact_nll_beta 0.5, contact_self_forcing True, action_noise_per_strip True,
  action_t_max_of_two False, cond_dropout_p 0.0, rope_time_mode time_true`.
* `configs.train` round-trips through `TeacherTrainConfig(**ct)`, `ema_decay 0.995`.
* EMA really ran: 156 shadow tensors, all map onto raw weights, rel-L2 raw↔EMA
  median 6.5e-3 / max 9.95e-1 (i.e. not a copy). `EMA.__init__` is called *after*
  `--init-weights` loads (common.py:802), so the shadow starts at the loaded EMA weights —
  no cold-start lag.
* `run_deploy.build_policy` loads it: `per_strip=True t_max2=False cond_dropout 0.0 beta 0.5
  selfforce True` — mc comes from the payload, and the P10B assertion passes.
* `tools/terminal_eval.py --tiny --ckpt <FT-A>` gets past build + `load_phantom_checkpoint`
  (assertion accepts). It then stops at `no windows with a gripper close found` — the
  synthetic episodes have no detectable close, so this tool has **no runnable smoke path**
  (see §3). It also `KeyError: 'action'`s on any `--synthetic`-trained checkpoint because
  that path saves `norm_stats {'mean': [], 'std': []}`; I worked around it by splicing real
  `dump_norm_stats.py` output into a scratch copy of the checkpoint.
* `tools/replay_rig.py --tiny` **ignores `--ckpt` entirely** (`build_policy`, replay_rig.py:254)
  while still writing `"ckpt": args.ckpt` into the output JSON (replay_rig.py:400) — a
  provenance trap for the D4 "score FT-A on replay" step.
* All 25 flags the provision launch line prints exist in `train_teacher`'s parser, and the
  real parser accepts the whole line:
  `{'contact_nll_beta': 0.5, 'contact_self_forcing': True, 'action_noise_per_strip': True,
    'action_t_max_of_two': False, 'ema_decay': 0.995, 'cond_dropout': 0.0,
    'acc_two_pass': True, 'event_band_weight': 0.0, 'init_ema': True}`.
* `pytest tests/test_ft_a_fixes.py` → 17 passed.

### 1.4 loss bookkeeping and gradients

All terms finite under the bundle; magnitudes sane (`action_v_mse ~2.5`, `contact_nll ~5.8`,
`total ~13.2` on the random-init tiny model).

Per-term **LoRA** squared-gradient share, one batch, tiny teacher (`scratch/ft_a/gradprobe.py`):

| term | v5 recipe | FT-A bundle |
|---|---|---|
| contact_nll | 82.5 % | 89.4 % |
| action_v_mse | 4.3 % | 5.7 % |

That looks like the P5 fix failing — it is not. At **random init** the sigma head emits
`log σ ≈ 0`, so β-NLL's `σ^{2β}` weight is ≈ 1 and the flag is a no-op. Pinning the sigma
head's output to the **measured v4/v5 regime** (`log σ = −2.35`, `1/σ² ≈ 110`) shows the fix
does what P5 claims (`scratch/ft_a/beta_probe.py`):

```
beta= None  |g_contact|=2.8439e+01  |g_action|=8.2097e-01  ratio=  34.64
beta=  0.5  |g_contact|=4.6399e+00  |g_action|=1.5822e+00  ratio=   2.93
beta=  1.0  |g_contact|=5.3407e-01  |g_action|=1.7922e+00  ratio=   0.30
```

So the documented β=0.5 takes the contact/action LoRA grad-norm ratio from ~35× to ~3×
(β=1 over-corrects to 0.3×). **The bundle's β=0.5 is well-chosen and P5 is genuinely fixed.**
Two consequences worth knowing before the run:
1. the *logged* `contact_nll` is multiplied by ~σ ≈ 0.095, so FT-A loss curves are **not**
   comparable to the v4/v5 curves (v5 logged −2.9…−3.7);
2. the σ head's own gradient is scaled by the same detached factor, so σ calibration
   (the speed governor's input) learns ~10× slower under β=0.5.
Neither is a blocker; both should be stated in the playbook.

**The verification step the playbook prescribes does not exist and does not run.**
`docs/training_playbook.md:92-95` says to confirm with `scratchpad/research/gradprobe.py` —
there is no `scratchpad/` in the repo. And a per-term probe (several `autograd.grad` calls
on one graph) **crashes on the training build**, because cosmos wraps every block in a
selective-activation-checkpoint wrapper unless `build_model(..., inference=True)`:

```
SAC=True   contact_nll=6.124671 action=2.137881 |g_contact|=2.369045e+00
   second grad call FAILED: RuntimeError: shape '[2, 14, 16]' is invalid for input of size 35840
SAC=False  contact_nll=6.124671 action=2.137881 |g_contact|=2.369045e+00
   second grad call OK |g_action|=4.850938e-01
```

Losses and the first gradient are **bit-identical** with SAC on/off, so training itself is
unaffected (one `.backward()` per micro-step) — but E1 as written on D2 dies on the H100
unless the probe builds with `inference=True`.

## 2. Other things I confirmed working

* `--contact-self-forcing` correctly clears `_sf_pred_cpk` on **every** `_acc_inputs_train`
  call (rf.py:196) so a stale package can never leak into a later step, and the
  self-forced package is `detach()`ed out of the two-pass no-grad sample.
* `total_loss` handles `--event-band-weight 0` correctly (`w.event if event_band_weight is
  None else float(...)` — no truthiness bug).
* `action_noise_per_strip` is honoured in `training_step`, `sample()` and
  `build_x_t_at` — including the `reuse_noise` cache and the `k_seeds` batch expansion.
* All `build_model(...)` call sites now pass an `mc` derived from the checkpoint
  (relabel, distill_hid, finetune_hids, run_deploy, terminal_eval, replay_rig) — P10B(B) is fixed.
* `--init-weights` norm-stats guard compares every key, mean **and** std.

## 3. Residual issues (ranked)

1. **[HIGH]** `--init-weights` + any FT-A model flag hard-fails (§1.2). Launch blocker.
2. **[MEDIUM]** `--event-band-weight 0` — in both the shipped v5 and the FT-A launch lines —
   is set as a bare attribute on `pm.rf` (`train_teacher.py:374`) and is written to **no**
   part of the checkpoint (`configs.train` has no such key; `configs.model.loss.event`
   stays 0.5). A `--resume` after a spot-instance kill silently reverts the objective to
   `w.event=0.5`, i.e. reinstates the 4×-the-action-gradient term the flag exists to remove,
   and the recipe is not recoverable from the artifact.
3. **[MEDIUM]** The documented P5 verification (E1, D2, gate for spending the FT-A run) is
   unrunnable: the script it names does not exist and the multi-term probe crashes on a
   training-mode build.
4. **[MEDIUM]** `provision_v5.sh`'s real-training smoke does not use the bundle it prints,
   so the guard that fires at launch is never exercised by the provisioning step whose
   entire purpose is exercising launch guards.
5. **[LOW]** `replay_rig.py --tiny` silently ignores `--ckpt` yet records it in the output.
6. **[LOW]** `terminal_eval.py` has no working smoke: `--synthetic`-trained checkpoints have
   empty `norm_stats` (`KeyError: 'action'`), and synthetic episodes contain no gripper
   close (`no windows with a gripper close found`).
