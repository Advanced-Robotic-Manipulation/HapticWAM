"""PHANTOM model configuration: sequence-layout counts, loss weights, LoRA,
ACC/ACE knobs. Hardware-derived sizes come from HardwareConfig; backbone sizes
from BackboneConfig — this config only holds PHANTOM-specific choices."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

# Contact-event ontology — the ONE vocabulary shared by derived labels, ACC and ACE.
EVENTS: tuple[str, ...] = ("none", "onset", "hold", "slip", "release")
N_EVENTS: int = len(EVENTS)
EVENT_IDX: dict[str, int] = {e: i for i, e in enumerate(EVENTS)}


@dataclass(frozen=True)
class AccConfig:
    d_acc: int = 256               # per-branch projection width
    d_e: int = 256                 # fused embedding width
    hidden: int = 512              # phi MLP hidden
    react_hidden: int = 32         # psi MLP hidden for the reactive (CASA) term
    lambda_bias_init: float = 0.0  # per-block bias scale init (0 => no-op at step 0)
    beta_init: float = 1.0         # event up-weighting init (softplus-parameterized)
    self_anticipation: Literal["gt_noised", "two_pass"] = "gt_noised"
    gt_noise_scale: float = 0.1    # noise added to GT package in gt_noised mode
    alpha_entropy_weight: float = 0.0  # optional prior on the fusion coefficient


@dataclass(frozen=True)
class LossWeights:
    action: float = 1.0            # λ_a
    video: float = 0.1             # λ_v (coarse guidance, not a fidelity target)
    contact: float = 1.0           # λ_c (events + deltas)
    wrist: float = 0.5             # λ_w
    event: float = 0.5             # λ_evt (EventReadout CE)
    gate_bce: float = 0.2          # ACC auxiliary
    sigma_reg: float = 0.01        # mild penalty keeping log-sigma bounded


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: int = 16
    target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    dropout: float = 0.0


@dataclass(frozen=True)
class PhantomModelConfig:
    student: bool = False
    drop_video_at_inference: bool = False
    rope_time_mode: Literal["aligned", "append"] = "aligned"
    use_action_adaln_intent: bool = True   # feed prev chunk through pretrained AdaLN path
    hht_dim: int = 256                     # shared embedding width of the small encoders
    contact_obs_frames: int = 1
    nfe: int = 5                           # denoising steps per replan
    loss: LossWeights = field(default_factory=LossWeights)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    acc: AccConfig = field(default_factory=AccConfig)
    feature_align: bool = False            # Efficient-WAM-style hidden alignment (HID ablation)

    def to_dict(self) -> dict:
        return asdict(self)
