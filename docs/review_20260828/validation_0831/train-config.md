# Lens: TRAINING-CONFIG INTEGRITY on the tiny model — HEAD 820b4eb, 2026-08-31

Everything below was produced by running code. Scratch: `.../validate2/scratch/train-config/`.
Nothing under `/Users/sannikov/GitHub/phantom` was modified; both the pre-fix (`d9c40f2`)
and post-fix (`820b4eb`) trees were `git archive`d into scratch with their own
`paths.local.yaml` so no run wrote into the repo.

## 0. Baseline
`.venv/bin/python -m pytest tests/ -q` (default random order) -> **661 passed in 349 s**.
`tests/test_fixnow_train_0830.py -q -p no:randomly` -> 29 passed in 19 s.
The `test_replay_rig` flake (F15) did not reproduce in this run.

## 1. beta-NLL: the gradient property holds EXACTLY  (`e1_beta_grad.py`)
Real tiny layout, real `contact_hetero_nll`, `log sigma` pinned to the logged v4/v5 regime
(-2.35, 1/sigma^2 = 109.95). Gradient taken w.r.t. `x0_pred` and projected on the plain
per-sigma-group MSE gradient:

    beta=None  loss=  -3.60473  g/g_plain= 109.9472  max|g - r*g_plain|=3.7e-09
    beta=0.0   loss=  -3.60473  g/g_plain= 109.9472
    beta=0.25  loss=  -1.11321  g/g_plain=  33.9538
    beta=0.5   loss=  -0.34378  g/g_plain=  10.4856   (= sqrt(1/sigma^2))
    beta=1.0   loss=  -0.03279  g/g_plain=   1.0000   max|g - r*g_plain|=0.0  exact_equal=True
    detach_weight            g/g_plain=   1.0000   max=0.0

`beta=1.0` gives the trunk **bit-identically** the plain-MSE gradient. `beta=0` == flag off.
P5's core claim is sound.

## 2. Per-term LoRA gradient norms — the P5 point  (`e6_gradratio.py`)
Same weights, same batch, same generator seed; only the objective flags change. One fresh
forward per term (SAC cannot be backwarded twice). 40 LoRA tensors / 53 248 params.

**Sigma head at random init (sigma ~ 1) — the "89% at random init is an artifact" regime:**

    no-flag        |g_c|=8.67e-01 |g_a|=2.51e-01 ratio= 3.461  contact sq-share=87.8%
    beta=0.5       |g_c|=9.28e-01                ratio= 3.701  89.2%
    beta=1.0       |g_c|=9.98e-01                ratio= 3.982  90.5%

**Sigma head pinned to the logged v4/v5 regime (log sigma = -2.35):**

    no-flag                       |g_c|=8.796e+00 |g_a|=2.507e-01 ratio= 35.091  sq-share=99.87%
    --contact-nll-beta 0.5        |g_c|=8.401e-01                 ratio=  3.352  sq-share=87.11%
    --contact-nll-beta 1.0        |g_c|=8.024e-02                 ratio=  0.320  sq-share= 5.81%
    --contact-nll-detach-weight   |g_c|=2.562e+00                 ratio= 10.222  sq-share=98.43%
    FULL BUNDLE (b=.5+per-strip)  |g_c|=8.840e-01 |g_a|=6.505e-01 ratio=  1.359  sq-share=62.64%

Reduction vs no-flag: **beta=0.5 -> 10.47x, beta=1.0 -> 109.62x, bundle -> 25.82x,
detach_weight -> only 3.43x.** The detach variant is documented as the interchangeable
alternative ("Pick one") but leaves ~10x contact dominance, because the sigma head's own
`d.detach()/var + log_var` term still back-propagates into the SHARED contact trunk at
1/sigma^2 scale. (My pinning attenuates the sigma path by 50x, so the real gap is worse.)

## 3. Flags off == pre-fix
`git diff d9c40f2..HEAD -- phantom/model/` is EMPTY, so `rf.training_step`, `sample()` and
`ace/losses.py` are byte-identical to the pre-fix tree. The only default-path changes are in
`windows.py` (`action_weight` now `meta.weight`, default 1.0) and `common.py`
(`commit_band_weight` default 1.0 -> branch skipped; `drop_last=shard` -> unchanged for the
train loader). Verified by dumping a 4-window collated batch from BOTH trees against one
shared synthetic episode root and sha256-ing every tensor: `diff batch_OLD.json batch_NEW.json`
differs only in the printed `windows.py` path. **Batches bit-identical.**

Caveat (pre-existing): `build_model` runs BEFORE `train_loop`'s `torch.manual_seed`, so
model init is unseeded — two identical `--tiny --synthetic --max-steps 3` runs of the SAME
tree gave `step 1 total=10.5035` and `total=10.0984`. Bit-identity therefore cannot be
demonstrated at the CLI level for a from-scratch run (and the from-scratch baselines
`no_distill` / `vision_only` are not reproducible from the recorded `seed`). FT-A is
unaffected: `--init-weights` overwrites all 156 trainable tensors.

## 4. `--action-noise-per-strip`  (`e3_strip_noise.py`, `e9_combos.py`)
Hooked `rf.net`'s forward-pre-hook, recovered eps from `x_t` in `training_step`, and captured
the first sampling step's `x`. apf=4, 7 strips, lat_w=10, lat_h=8:

    per_strip=False  training_step: max within-strip spread 5.711e+00, pad absmax 3.660, std 0.9905
                     sample()     : max within-strip spread 5.856e+00, pad absmax 3.859, std 0.9895
    per_strip=True   training_step: max within-strip spread 0.000e+00, pad absmax 0.000, std 0.5234
                     sample()     : max within-strip spread 0.000e+00, pad absmax 0.000, std 0.5174

Constant within every (frame, action) strip, exactly zero outside the strips, in BOTH paths.
std 0.517 ~ sqrt(apf/lat_c)=0.5 confirms the strips tile lat_w completely.

Train/eval combinations, exercised with two real tiny checkpoints (one saved with the flag,
one without), through `load_phantom_checkpoint`:

    ckpt per_strip=True   build False tolerate=none   -> REFUSED (model-config drift)
                          build False tolerate=FT_MUT -> LOADS
                          build True  (either)        -> LOADS
    ckpt per_strip=False  build True  tolerate=none   -> REFUSED
                          build True  tolerate=FT_MUT -> LOADS

`run_deploy.build_policy:59`, `replay_rig:403`, `terminal_eval:376` all do
`PhantomModelConfig.from_dict(payload["configs"]["model"])`, so the eval flag is ALWAYS the
checkpoint's — the illegal combination is unreachable through a CLI, and is refused when
constructed by hand. `tolerate=FINETUNE_MUTABLE_MODEL_FIELDS` is passed only by
`train_teacher --init-weights`. Correct.

## 5. `EpisodeMeta.weight` x `--commit-band-weight`  (`e4_weights.py`, `e4b_band.py`)
Seven synthetic episodes, meta.json hand-set to weight {absent, 0, 0.5, 1, 3, -2} plus a
`grasp_slip_fail` at weight 3. `EpisodeMeta.from_dict` round-trip: absent -> 1.0, else the
literal value. Effective per-window `action_weight`, with the close time pinned so the band
intersects the admissible anchor range (band = [tc-1.5, tc-0.2]):

    cbw   meta.weight   before   lo-edge  inside  hi-edge   after
    1.0   absent/1.0      oob      1       1       1        1
    1.0   0.0             oob      0       0       0        0
    1.0   0.5             oob      0.5     0.5     0.5      0.5
    1.0   3.0             oob      3       3       3        3
    1.0   -2.0            oob      0       0       0        0
    1.0   3.0 + _fail     oob      0       0       0        0
    2.0   1.0             oob      2       2       2        1
    2.0   0.5             oob      1       1       1        0.5
    2.0   3.0             oob      6       6       6        3
    2.0   3.0 + _fail     oob      0       0       0        0
    3.0   3.0             oob      9       9       9        3

Multiplicative, band inclusive at both edges, failure demos pinned to 0, negatives clamped
to 0. Arithmetic is correct. **But nothing writes `weight` into meta.json** (finding 4).

## 6. EMA + checkpoint payload  (`e5_bundle_cli.py`)
Full recommended bundle (playbook, i.e. WITHOUT `--contact-self-forcing`) from a base tiny
checkpoint, through `train_teacher.main`, 3 steps: all terms finite, totals 7.4746 / 10.9426 /
11.1248. `C.EMA` spied: `{'decay': 0.995, 'updates': 3}` (base run: 0.999). Payload:

    configs.model : contact_nll_beta 0.5, action_noise_per_strip true, action_t_max_of_two false,
                    cond_dropout_p 0.0, contact_self_forcing false, wrist_region_mse true
    configs.train : ema_decay 0.995, event_band_weight 0.0, lr 2e-05, lr_new_modules 6e-05,
                    warmup_steps 150
    ABSENT        : commit_band_weight, grasp_frac, photo_aug, split
    ema           : 156 keys, all matched to raw, max|ema-raw| 1.25e-06

## 7. P10B tolerate set  (`e7_p10b.py`)
Swept all 19 `PhantomModelConfig` fields, strict vs `tolerate=FINETUNE_MUTABLE_MODEL_FIELDS`:
**zero mismatches against the intended sets.** `student`/`mask_wrist`/TRAIN_ONLY always
ignored; `cond_dropout_p`, `action_t_max_of_two`, `action_noise_per_strip` raise strictly and
are tolerated only with the FT-A set; every layout/behaviour field (rope, nfe, hht_dim, acc,
lora, loss, feature_align, contact_obs_frames, use_action_adaln_intent,
drop_video_at_inference) raises in both. `model_config_drift` puts the same three in `soft`
for `--init-weights` and in `hard` for `--resume`.

## 8. `--resume` restores exactly ONE knob  (`e8_resume.py`, `e10_resume_knobs.py`)
Checkpoint `configs.train`: ema_decay 0.995, lr 2e-05, lr_new_modules 6e-05, warmup_steps 150,
event_band_weight 0.0. Resumed with only the *model* flags re-passed (what a spot-instance
restart script would do):

    RESUMED RUN ACTUALLY USED: {'ema_decay': 0.999, 'lr': 0.0001, 'lr_new_modules': 0.0003,
      'warmup_steps': 1, 'event_band_weight': 0.0,
      'datasets': [{'grasp_frac': 0.0, 'photo_aug': 0.0, 'commit_band_weight': 1.0}]}
    post-resume checkpoint records: ema_decay=0.999 lr=0.0001 warmup_steps=1 event_band_weight=0.0

Only `event_band_weight` (F11) came back. Everything else silently reverted AND the new
checkpoint now records the reverted values as if they had held for the whole run.

## 9. The provisioning script contradicts the corrected playbook
`e06c33b` ("playbook: drop --contact-self-forcing from the recommended ft-a bundle") is NEWER
than `fad0a92` (the provisioning rewrite). `tools/provision_v5.sh:275` (2-step smoke) and
`:295` (the printed launch line the operator copies for the 3000-step paid run) still carry
`--contact-self-forcing`, and `tests/test_fixnow_train_0830.py:623` +
`tests/test_ft_a_fixes.py:296` ASSERT that it is there.

## 10. `docs/rig_session_v5.md:153-155` still quotes 20.7 -> 17.4 mm
Section header "Offline numbers behind v5_6 (terminal_eval, 124 val episodes x 2 seeds, EMA)".
`E13_rescore.md` (same fix batch) measured 20.82 -> 18.23 mm on val124 with 4 seeds and the
hardened tool and says the 17.4 number "cannot stay as is". No pointer, no correction.

## Verdict
NOT READY. The objective mathematics is sound and every new flag round-trips, but the paid
launch line still enables a retracted flag, a resume silently rewrites the recipe, and one
already-published number was not corrected where it is read.
