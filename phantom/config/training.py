"""Training-program configurations (plain dataclasses; each train script builds
one from CLI args + defaults). Hardware/backbone/model configs are loaded
separately and snapshotted into every checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass(frozen=True)
class CommonTrainConfig:
    run_name: str = "run"
    seed: int = 0
    lr: float = 1e-4
    lr_new_modules: float = 3e-4     # HHT/ACC/heads may use a higher LR than LoRA
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 20_000
    batch_size: int = 1              # per GPU
    grad_accum: int = 8
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 50
    ckpt_every: int = 1000
    eval_every: int = 1000
    num_workers: int = 4
    activation_checkpointing: bool = True
    # fp32 master copies in the optimizer: pure-bf16 stepping froze every
    # param whose update rounds below the bf16 ulp (v3: acc beta_raw and the
    # tactile-encoder norm scales never left init — v4 audit 2026-08-14)
    fp32_master: bool = True
    synthetic: bool = False          # swap dataset for SyntheticEpisodeGenerator
    tiny: bool = False               # tiny backbone preset (CPU smoke)
    device: str = "cuda"
    out_dir: str = ""                # default: runs_root/<program>/<run_name>

    def to_dict(self) -> dict:
        return asdict(self)

    # v6 data fix: subtract each pad's per-episode wrench zero offset (median
    # of the first N rows) from contact_state and the cpk_wrench target.
    # 0 = raw (v4/v5 checkpoints). Persisted in EVERY program's checkpoint
    # (students inherit their teacher's value) so deploy / replay / the next
    # program read it from the payload, never from a flag.
    wrench_baseline_rows: int = 0

@dataclass(frozen=True)
class TactilePretrainConfig(CommonTrainConfig):
    """(1) Contact-play self-supervised pretraining of TactileFieldEncoder."""
    run_name: str = "tactile_pretrain"
    mask_ratio: float = 0.5
    mask_patch: int = 16             # patch size of the masking grid (field pixels)
    w_masked_recon: float = 1.0
    w_cross_channel: float = 1.0     # predict force channels from displacement channels
    max_steps: int = 30_000
    batch_size: int = 32
    grad_accum: int = 1
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class TeacherTrainConfig(CommonTrainConfig):
    """(2) Teacher PHANTOM fine-tune: LoRA + new modules, joint RF objective."""
    run_name: str = "teacher"
    tactile_pretrain_ckpt: str = ""  # output of program (1); empty = train from scratch
    freeze_tactile_steps: int = 0    # optional warmup with the tactile conv frozen
    max_steps: int = 50_000
    # packed-event-band MSE weight override (None = use mc.loss.event). It
    # lives HERE so it lands in the checkpoint's `configs.train` and can be
    # re-applied on --resume: `--event-band-weight 0` used to be a bare
    # attribute on pm.rf, so a spot-instance kill + resume silently brought
    # the term back at 0.5 (validation 2026-08-30 F11).
    event_band_weight: float | None = None
    # ... and the rest of the recipe, for exactly the same reason: these four
    # were read straight off `args` and reached NO part of the checkpoint, so
    # a resume reverted them to the defaults below AND recorded the reverted
    # values as if they had governed the whole run (2026-08-31 #8).
    split: str = "train"                 # manifest subset the run trains on
    grasp_frac: float = 0.0              # windows anchored in the pre-close band
    photo_aug: float = 0.0               # scene-camera photometric jitter strength
    commit_band_weight: float = 1.0      # ACTION-loss multiplier in the commit band


@dataclass(frozen=True)
class HIDConfig(CommonTrainConfig):
    """(3) Haptic-Imagination Distillation into the sensor-free student."""
    run_name: str = "hid"
    teacher_ckpt: str = ""                       # required
    teacher_mode: Literal["cached", "online"] = "cached"
    relabel_dir: str = ""                        # cached teacher outputs (precompute pass)
    w_traj: float = 1.0                          # trajectory distillation on contact package
    w_event: float = 1.0                         # event CE vs teacher
    w_behavior: float = 1.0                      # action-frame velocity matching
    w_ground: float = 0.5                        # GT RF grounding (actions/video/wrist)
    saliency_kappa: float = 4.0                  # s_tau = 1 + kappa * (1 - p_evt[none])
    confidence_temp: float = 1.0                 # c_tau = exp(-sigma_T / temp)
    # round-2 knobs (09-05, PR #6): both default to the round-1 behaviour
    teacher_nfe: int = 0            # teacher imagination NFE: 0 = legacy nfe//2, -1 = the teacher's own nfe, N = N
    w_sigma: float = 0.0            # student sigma head supervised vs GT (contact_hetero_nll + sigma_reg); 0 = untrained (round 1)
    feature_align: bool = False                  # ablation row
    w_feature_align: float = 0.1
    dagger_round: int = 0                        # 0 = offline; 1,2 = DAgger rounds
    max_steps: int = 30_000
    # traj_distill target space (code defect, 2026-09-18):
    #   "roundtrip"    — pack(unpack(teacher CONTACT frames)), what every
    #                    shipped checkpoint was distilled against;
    #   "raw_latents"  — the teacher's raw CONTACT latents, which carry no
    #                    re-packed CoP bump on no-contact steps.
    traj_target: Literal["roundtrip", "raw_latents"] = "roundtrip"
    # action grounding on on-policy DAgger rollouts (code defect,
    # 2026-09-18): "failure_demo" = the shipped rule (a `success is False`
    # verdict zeroes the action weight, so a judged rollout grounds nothing);
    # "judged_rollouts" = a verdict-judged POLICY rollout keeps its
    # per-episode weight, deliberate failure demos still at 0. Named for the
    # EPISODES it re-admits: what gets grounded is the rollout's own
    # re-derived (measured delta-EE) actions, never a teacher relabel.
    rollout_action_weight: Literal["failure_demo", "judged_rollouts"] = "failure_demo"
    # sha256[:12] of the teacher checkpoint this student was distilled from.
    # Recorded so a --resume can accept the SAME teacher at a different path
    # (rental workspaces move) and refuse a different one.
    teacher_ckpt_sha12: str = ""
    # ... and the data recipe, for the same reason the teacher carries it
    # (2026-08-31 #8): these four were read straight off `args`
    # and reached NO part of the student checkpoint, so a --resume that
    # forgot them silently reverted the window recipe to train/0.0/0.0/1.0
    # (code defect, 2026-09-18 (c)).
    split: str = "train"                 # manifest subset the run trains on
    grasp_frac: float = 0.0              # windows anchored in the pre-close band
    photo_aug: float = 0.0               # scene-camera photometric jitter strength
    commit_band_weight: float = 1.0      # ACTION-loss multiplier in the commit band


@dataclass(frozen=True)
class HIDSConfig(CommonTrainConfig):
    """(4) HID-S: force-safety fine-tune (AWR + KL leash) of the post-HID student."""
    run_name: str = "hids"
    student_ckpt: str = ""                       # required: post-HID student
    tau_quantile: float = 0.93                   # auto-calibrated per task from successes
    lambda_F: float = 1.0                        # penalty scale
    beta_awr: float = 1.0                        # exp(A / beta) weighting temperature
    kl_coef: float = 1.0                         # velocity-MSE leash to frozen reference
    max_weight: float = 20.0                     # AWR weight clip
    lr: float = 1e-5
    max_steps: int = 5_000
