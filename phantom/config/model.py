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


# PhantomModelConfig fields that change ONLY the training objective — no
# module shape, no sampling behaviour, nothing the deployed forward reads.
# `assert_model_config_matches` (P10B) ignores these: a checkpoint fine-tuned
# with a different contact-loss shape is still the same model at deploy.
TRAIN_ONLY_MODEL_FIELDS: frozenset[str] = frozenset({
    "contact_nll_beta", "contact_nll_detach_weight",
    "wrist_region_mse", "contact_self_forcing",
})
# ...plus the fields a FINE-TUNE may legitimately flip relative to the
# checkpoint it initializes from. `action_noise_per_strip` also changes
# SAMPLING, so it stays in the deploy-side P10B check — but adopting it is
# the whole point of the FT-A bundle, so --init-weights only warns.
# `action_t_max_of_two` and `cond_dropout_p` are read exclusively by
# `training_step` (verified 2026-08-29).
FINETUNE_MUTABLE_MODEL_FIELDS: frozenset[str] = TRAIN_ONLY_MODEL_FIELDS | frozenset({
    "action_noise_per_strip", "action_t_max_of_two", "cond_dropout_p",
})


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
    # aligned:   v3 behavior — integer positions 1+k*(t_gen//t_len) clamped to
    #            t_gen; with H=16@10Hz on the 4Hz backbone this collapses to
    #            [1,2,3,3]: the last two ACTION frames are exactly exchangeable
    #            and 0.1-1.6s of actions is stamped as 1-3s (v4 audit).
    # time_true: fractional positions at the physical time each ACTION frame
    #            describes, in latent-frame units ([0.4,0.8,1.2,1.6] for the
    #            current rig config) — RoPE consumes float positions natively.
    rope_time_mode: Literal["aligned", "append", "time_true"] = "aligned"
    # p of dropping ALL observation conditioning for a training sample
    # (classifier-free style) — enables obs-guidance at sampling and forces
    # the action head to actually read observations when they are present
    cond_dropout_p: float = 0.0
    # per-group timesteps: ACTION frames get max(t, t') of two draws — biases
    # their supervision toward the high-noise band that few-NFE sampling
    # actually visits first (informative band ~t in [0.87,1] under tiling)
    action_t_max_of_two: bool = False
    # --- FT-A objective knobs (review 2026-08-28, P5/P6/P7). All default to
    # the shipped v4/v5 behaviour: a checkpoint trained before they existed
    # trains and evaluates bit-identically with every one of them off.
    #
    # P5: the contact heteroscedastic NLL sends 2r/sigma^2 into the shared
    # trunk; with the logged log sigma ~ -2.35 that is a ~110x weight on the
    # contact residual and the action objective trains at ~1/10-1/50 rate.
    # contact_nll_beta = beta-NLL (Seitzer 2022): multiply the NLL term by
    # (sigma^2)^beta detached, beta in [0, 1] (1 = the trunk sees exactly the
    # plain-MSE gradient). None = off.
    contact_nll_beta: float | None = None
    # P5 alternative: the sigma head trains on the DETACHED residual
    # (d.detach()/var + log var) and the trunk on plain MSE (+ d), so sigma
    # stays calibrated for the governor while the trunk is no longer scaled
    # by 1/sigma^2.
    contact_nll_detach_weight: bool = False
    # P5: lambda_w * wrist_region_mse double-supervises packed channel 15,
    # which is ALSO the NLL's `wrist` sigma group. False drops the duplicate.
    wrist_region_mse: bool = True
    # P7: pack the ACC two-pass inner sample's OWN predicted contact package
    # into the CONTACT x0 the ACTION frames are denoised alongside (GT stays
    # the loss target) — self-forcing, zero extra forward passes. Requires
    # acc.self_anticipation == "two_pass".
    contact_self_forcing: bool = False
    # P6: draw ACTION noise PER STRIP — eps_action = ActionPacker.pack(
    # randn(B, H, A)) — so every latent cell of one action value shares one
    # noise draw instead of 160-192 i.i.d. cells whose strip mean is a small
    # fixed offset. Honoured by training_step AND sample() (deploy/train must
    # match); the checkpoint records it and run_deploy rebuilds from it.
    action_noise_per_strip: bool = False
    # Zero the wrist F/T window before it reaches the WristTCN (HHT.obs_frames
    # and HHT.wrist_feature). The INPUT ABLATION the comparative systems need:
    # without it `vision_only` / `no_distill` / `drop_tactile` are input-
    # identical to `student` (both fuse [WristTCN(wrist) || URStateMLP] into
    # OBS_PROPRIO), so the recovery_ratio denominator in eval/aggregate.py has
    # no producible arm and a reviewer attributes any student gain to the
    # surviving wrist F/T signal (P10A, review 2026-08-28). The window is
    # still RECORDED in every mode — only the model stops reading it.
    mask_wrist: bool = False
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

    @classmethod
    def from_dict(cls, d: dict) -> "PhantomModelConfig":
        """Rebuild from a checkpoint's configs['model'] snapshot. Unknown keys
        are ignored (forward compat); missing keys keep the current defaults —
        the caller decides whether that is acceptable. Deployment MUST build
        the model with the SAVED config: e.g. a time_true-rope checkpoint
        rebuilt with default rope silently shifts every action frame's
        temporal phase (v4 audit: inference used to construct defaults)."""
        d = dict(d)
        kw: dict = {}
        for f in ("student", "drop_video_at_inference", "rope_time_mode",
                  "use_action_adaln_intent", "hht_dim", "contact_obs_frames",
                  "nfe", "feature_align", "cond_dropout_p",
                  "action_t_max_of_two", "mask_wrist",
                  "contact_nll_beta", "contact_nll_detach_weight",
                  "wrist_region_mse", "contact_self_forcing",
                  "action_noise_per_strip"):
            if f in d:
                kw[f] = d[f]
        if isinstance(d.get("loss"), dict):
            kw["loss"] = LossWeights(**{k: v for k, v in d["loss"].items()
                                        if k in LossWeights.__dataclass_fields__})
        if isinstance(d.get("lora"), dict):
            kw["lora"] = LoRAConfig(**{k: v for k, v in d["lora"].items()
                                       if k in LoRAConfig.__dataclass_fields__})
        if isinstance(d.get("acc"), dict):
            kw["acc"] = AccConfig(**{k: v for k, v in d["acc"].items()
                                     if k in AccConfig.__dataclass_fields__})
        return cls(**kw)
