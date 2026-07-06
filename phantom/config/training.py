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
    synthetic: bool = False          # swap dataset for SyntheticEpisodeGenerator
    tiny: bool = False               # tiny backbone preset (CPU smoke)
    device: str = "cuda"
    out_dir: str = ""                # default: runs_root/<program>/<run_name>

    def to_dict(self) -> dict:
        return asdict(self)


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
    feature_align: bool = False                  # ablation row
    w_feature_align: float = 0.1
    dagger_round: int = 0                        # 0 = offline; 1,2 = DAgger rounds
    max_steps: int = 30_000


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
