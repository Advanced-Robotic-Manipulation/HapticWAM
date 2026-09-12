"""ACE grouped training objective (pipeline.md §4):

    L = lambda_a L_act + lambda_v L_vid + lambda_c (L_evt + L_delta)
        + lambda_w L_F/T + L_ACC

realized on the joint rectified-flow denoiser as per-frame-group losses:
  VIDEO_GEN  — velocity MSE (small weight; guidance, not fidelity)
  CONTACT    — heteroscedastic NLL on the x0 prediction (sigma from SigmaHead)
               + event CE from EventReadout
  ACTION     — velocity MSE (what actually executes)
  wrist term — the wrist channel region of the CONTACT frames (lambda_w)
  ACC aux    — gate BCE (contact-within-Δ) + event CE at t+1
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phantom.config.model import LossWeights
from phantom.model.ace.heads import SIGMA_GROUPS
from phantom.model.acc import AccOutput
from phantom.model.sequence import FrameGroup, SequenceLayout


def group_velocity_mse(v_pred: torch.Tensor, v_target: torch.Tensor,
                       layout: SequenceLayout, group: FrameGroup,
                       weights: torch.Tensor | None = None,
                       channels: slice | None = None) -> torch.Tensor:
    """Velocity MSE over one frame group.

    `weights` (B,) optionally scales each sample's contribution — used to zero
    the ACTION term on deliberate-failure demos while their contact/event
    supervision is kept. A batch of only zero-weight samples yields no
    gradient rather than a NaN.

    `channels` restricts the loss to a channel range. For the ACTION group
    this must be the packer's live channels (slice(0, actions_per_frame)):
    ActionPacker writes only apf of the lat_c channels, so averaging all of
    them spent 75% of the "action loss" on denoising constant zero padding —
    diluting the action gradient 4x and pinning the reported metric near its
    no-conditioning floor (v3 audit, 2026-08-14)."""
    sl = layout.frame_slice(group)
    pred, tgt = v_pred[:, :, sl].float(), v_target[:, :, sl].float()
    if channels is not None:
        pred, tgt = pred[:, channels], tgt[:, channels]
    if weights is None:
        return F.mse_loss(pred, tgt)
    per_sample = ((pred - tgt) ** 2).flatten(1).mean(1)          # (B,)
    w = weights.to(per_sample.device, per_sample.dtype).reshape(-1)
    return (per_sample * w).sum() / w.sum().clamp_min(1e-6)


def _nll_term(d_B_Tc: torch.Tensor, log_var: torch.Tensor,
              beta: float | None, detach_weight: bool) -> torch.Tensor:
    """One sigma group's contact term.

    Default (beta=None, detach_weight=False) is exactly the historical
    `d / sigma^2 + log sigma^2` — bit-identical, no extra ops.

    P5 knobs (review 2026-08-28), both aimed at the same thing: the trunk's
    gradient through `x0_pred` is `2r/sigma^2`, and the logged regime is
    `log sigma ~ -2.35` (`1/sigma^2 ~ 110`), so the contact residual owns the
    LoRA while the action objective trains at ~1/10-1/50 rate.

      beta (beta-NLL, Seitzer et al. 2022): scale the whole term by
      `(sigma^2)^beta` DETACHED. beta=1 makes the trunk's gradient exactly the
      plain-MSE gradient (the 1/sigma^2 and the detached sigma^2 cancel) while
      the sigma head still sees a proper NLL; beta=0 is the unmodified NLL.

      detach_weight: the sigma head trains on the DETACHED residual
      (`d.detach()/var + log var` — sigma stays calibrated for the speed
      governor and the HID confidence weights) and the trunk trains on plain
      `d`. The two paths are added, so the returned VALUE is no longer
      comparable across the flag; the gradient is what changes on purpose.
    """
    var = log_var.exp()
    nll = (d_B_Tc.detach() if detach_weight else d_B_Tc) / var + log_var
    if beta is not None:
        nll = nll * var.detach().pow(beta)
    return nll + d_B_Tc if detach_weight else nll


def contact_hetero_nll(x0_pred: torch.Tensor, x0_target: torch.Tensor,
                       log_sigma_B_Tc_K: torch.Tensor,
                       layout: SequenceLayout,
                       group_channels: dict[str, list[int]] | None = None,
                       beta: float | None = None,
                       detach_weight: bool = False) -> torch.Tensor:
    """Heteroscedastic NLL d/sigma^2 + log sigma^2 on the CONTACT frames' x0,
    PER SIGMA GROUP: each SigmaHead channel is supervised against the residual
    of its own packed channels (sigma_group_channels), so the per-group sigma
    the speed governor and HID confidence weights consume is individually
    calibrated. Falls back to the scalar-aggregate version when no channel map
    is given (legacy).

    `beta` / `detach_weight` rebalance the trunk gradient — see `_nll_term`.
    Both default to off = the shipped v4/v5 objective."""
    sl = layout.frame_slice(FrameGroup.CONTACT)
    d = (x0_pred[:, :, sl].float() - x0_target[:, :, sl].float()) ** 2  # (B,C,Tc,H,W)
    if group_channels is None:
        d_B_Tc = d.mean(dim=(1, 3, 4))                   # (B, Tc)
        log_var = 2.0 * log_sigma_B_Tc_K.mean(-1)        # (B, Tc)
        return _nll_term(d_B_Tc, log_var, beta, detach_weight).mean()
    terms = []
    for k, name in enumerate(SIGMA_GROUPS):
        chans = group_channels.get(name)
        if not chans:
            continue
        d_B_Tc = d[:, chans].mean(dim=(1, 3, 4))         # (B, Tc)
        log_var = 2.0 * log_sigma_B_Tc_K[..., k]         # (B, Tc)
        terms.append(_nll_term(d_B_Tc, log_var, beta, detach_weight))
    return torch.stack(terms, dim=-1).mean()


def wrist_region_mse(x0_pred: torch.Tensor, x0_target: torch.Tensor,
                     layout: SequenceLayout, wrist_channel: int) -> torch.Tensor:
    """lambda_w term: the wrist-F/T channel of the CONTACT frames.

    NOTE (P5): channel 15 is ALSO the NLL's `wrist` sigma group, so this term
    is a second supervision of the same cells at weight lambda_w. Kept for
    checkpoint parity; `PhantomModelConfig.wrist_region_mse=False`
    (--no-wrist-region-mse) drops it."""
    sl = layout.frame_slice(FrameGroup.CONTACT)
    return F.mse_loss(x0_pred[:, wrist_channel, sl].float(),
                      x0_target[:, wrist_channel, sl].float())


def event_ce(event_logits_B_Tc_E: torch.Tensor, events_B_Tc: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(event_logits_B_Tc_E.flatten(0, 1),
                           events_B_Tc.flatten(0, 1))


def acc_losses(acc: AccOutput, gate_label_B: torch.Tensor,
               event_next_B: torch.Tensor, alpha_entropy_weight: float = 0.0) -> dict:
    out = {
        # .float() BEFORE the clamp: in bf16 `1 - 1e-6` rounds to 1.0 and the
        # clamp is a no-op — a saturated gate (logit ~6.9 suffices in bf16)
        # on a negative label logged BCE=100 and carried a dead gradient.
        # Unreachable pre-54b7943 (labels were 100% positive); real now.
        "acc_gate_bce": F.binary_cross_entropy(acc.g.float().clamp(1e-6, 1 - 1e-6),
                                               gate_label_B.float()),
        "acc_event_ce": F.cross_entropy(acc.event_logits.float(), event_next_B),
    }
    if alpha_entropy_weight > 0:
        a = acc.alpha.clamp(1e-6, 1 - 1e-6)
        out["acc_alpha_entropy"] = alpha_entropy_weight * (
            a * a.log() + (1 - a) * (1 - a).log()).mean()
    return out


def event_band_mse(x0_pred: torch.Tensor, x0_target: torch.Tensor, layout,
                   event_channel: int) -> torch.Tensor:
    """Reconstruction MSE on the packed contact-EVENT channel.

    That channel belongs to no SigmaHead group, so the grouped contact NLL
    (introduced at 8ebe429; the legacy aggregate loss covered all 16
    channels) never supervised it — yet ContactPacker.unpack() derives
    `cpk.event` from it at sampling and flatten_summary() feeds those
    probabilities to ACC (two-pass training AND deploy). Plain MSE, weighted
    like the EventReadout CE, keeps the head shapes and checkpoint format
    unchanged (Codex review 2026-08-26)."""
    sl = layout.frame_slice(FrameGroup.CONTACT)
    d = (x0_pred[:, event_channel, sl].float() - x0_target[:, event_channel, sl].float()) ** 2
    return d.mean()


def total_loss(parts: dict[str, torch.Tensor], w: LossWeights,
               event_band_weight: float | None = None) -> torch.Tensor:
    # event_band_weight: train-time override for the packed-event-band MSE
    # (defaults to w.event). A checkpoint trained before that term existed
    # starts at contact_event_mse ~0.9 vs action_v_mse ~0.1 (H100 probe
    # 2026-08-27), so a low-LR fine-tune of it should run this at 0 (or tiny)
    # to keep the gradient budget on the action objective; from-scratch runs
    # keep the default.
    w_band = w.event if event_band_weight is None else float(event_band_weight)
    total = (w.action * parts["action_v_mse"]
             + w_band * parts.get("contact_event_mse", torch.zeros(())).to(
                 parts["action_v_mse"].device)
             + w.contact * parts["contact_nll"]
             + w.event * parts["event_ce"]
             + w.wrist * parts["wrist_mse"]
             + w.gate_bce * parts.get("acc_gate_bce", torch.zeros(())).to(
                 parts["action_v_mse"].device)
             + w.event * parts.get("acc_event_ce", torch.zeros(())).to(
                 parts["action_v_mse"].device)
             + w.sigma_reg * parts.get("sigma_reg", torch.zeros(())).to(
                 parts["action_v_mse"].device))
    # lambda_v = 0 (the video-objective ablation, --loss-video 0) SKIPS the
    # term rather than scaling it by zero: the VIDEO_GEN frames stay in the
    # layout (same token count, same attention, same RoPE) but no gradient
    # flows from them, so the arm differs from the control in the objective
    # alone. `video_v_mse` is still reported in `parts` as a diagnostic.
    if "video_v_mse" in parts and float(w.video) != 0.0:
        total = total + w.video * parts["video_v_mse"]
    if "acc_alpha_entropy" in parts:
        total = total + parts["acc_alpha_entropy"]
    return total
