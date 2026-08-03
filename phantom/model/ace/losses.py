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
                       weights: torch.Tensor | None = None) -> torch.Tensor:
    """Velocity MSE over one frame group.

    `weights` (B,) optionally scales each sample's contribution — used to zero
    the ACTION term on deliberate-failure demos while their contact/event
    supervision is kept. A batch of only zero-weight samples yields no
    gradient rather than a NaN."""
    sl = layout.frame_slice(group)
    pred, tgt = v_pred[:, :, sl].float(), v_target[:, :, sl].float()
    if weights is None:
        return F.mse_loss(pred, tgt)
    per_sample = ((pred - tgt) ** 2).flatten(1).mean(1)          # (B,)
    w = weights.to(per_sample.device, per_sample.dtype).reshape(-1)
    return (per_sample * w).sum() / w.sum().clamp_min(1e-6)


def contact_hetero_nll(x0_pred: torch.Tensor, x0_target: torch.Tensor,
                       log_sigma_B_Tc_K: torch.Tensor,
                       layout: SequenceLayout,
                       group_channels: dict[str, list[int]] | None = None) -> torch.Tensor:
    """Heteroscedastic NLL d/sigma^2 + log sigma^2 on the CONTACT frames' x0,
    PER SIGMA GROUP: each SigmaHead channel is supervised against the residual
    of its own packed channels (sigma_group_channels), so the per-group sigma
    the speed governor and HID confidence weights consume is individually
    calibrated. Falls back to the scalar-aggregate version when no channel map
    is given (legacy)."""
    sl = layout.frame_slice(FrameGroup.CONTACT)
    d = (x0_pred[:, :, sl].float() - x0_target[:, :, sl].float()) ** 2  # (B,C,Tc,H,W)
    if group_channels is None:
        d_B_Tc = d.mean(dim=(1, 3, 4))                   # (B, Tc)
        log_var = 2.0 * log_sigma_B_Tc_K.mean(-1)        # (B, Tc)
        return (d_B_Tc / log_var.exp() + log_var).mean()
    terms = []
    for k, name in enumerate(SIGMA_GROUPS):
        chans = group_channels.get(name)
        if not chans:
            continue
        d_B_Tc = d[:, chans].mean(dim=(1, 3, 4))         # (B, Tc)
        log_var = 2.0 * log_sigma_B_Tc_K[..., k]         # (B, Tc)
        terms.append(d_B_Tc / log_var.exp() + log_var)
    return torch.stack(terms, dim=-1).mean()


def wrist_region_mse(x0_pred: torch.Tensor, x0_target: torch.Tensor,
                     layout: SequenceLayout, wrist_channel: int) -> torch.Tensor:
    """lambda_w term: the wrist-F/T channel of the CONTACT frames."""
    sl = layout.frame_slice(FrameGroup.CONTACT)
    return F.mse_loss(x0_pred[:, wrist_channel, sl].float(),
                      x0_target[:, wrist_channel, sl].float())


def event_ce(event_logits_B_Tc_E: torch.Tensor, events_B_Tc: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(event_logits_B_Tc_E.flatten(0, 1),
                           events_B_Tc.flatten(0, 1))


def acc_losses(acc: AccOutput, gate_label_B: torch.Tensor,
               event_next_B: torch.Tensor, alpha_entropy_weight: float = 0.0) -> dict:
    out = {
        "acc_gate_bce": F.binary_cross_entropy(acc.g.clamp(1e-6, 1 - 1e-6).float(),
                                               gate_label_B.float()),
        "acc_event_ce": F.cross_entropy(acc.event_logits.float(), event_next_B),
    }
    if alpha_entropy_weight > 0:
        a = acc.alpha.clamp(1e-6, 1 - 1e-6)
        out["acc_alpha_entropy"] = alpha_entropy_weight * (
            a * a.log() + (1 - a) * (1 - a).log()).mean()
    return out


def total_loss(parts: dict[str, torch.Tensor], w: LossWeights) -> torch.Tensor:
    total = (w.action * parts["action_v_mse"]
             + w.contact * parts["contact_nll"]
             + w.event * parts["event_ce"]
             + w.wrist * parts["wrist_mse"]
             + w.gate_bce * parts.get("acc_gate_bce", torch.zeros(())).to(
                 parts["action_v_mse"].device)
             + w.event * parts.get("acc_event_ce", torch.zeros(())).to(
                 parts["action_v_mse"].device)
             + w.sigma_reg * parts.get("sigma_reg", torch.zeros(())).to(
                 parts["action_v_mse"].device))
    if "video_v_mse" in parts:
        total = total + w.video * parts["video_v_mse"]
    if "acc_alpha_entropy" in parts:
        total = total + parts["acc_alpha_entropy"]
    return total
