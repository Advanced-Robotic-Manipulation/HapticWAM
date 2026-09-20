"""ACC — Anticipatory Contact Coupling (pipeline.md §3).

    e_t = phi([ w_{t-k:t} || z_vis_t || a_intent_t || c_hat_{t+1|t-1} ])
    g_ant = sigma(W_g e),  p_evt = softmax(W_e e),  alpha = sigma(W_alpha e)
    g_react = sigma(psi(||x_t - x_{t-1}||))          (CASA term, teacher only)
    g = alpha * g_ant + (1 - alpha) * g_react        (alpha == 0 recovers CASA;
                                                      student: alpha -> 1, no react)

Injection into attention happens in PhantomDiT via attention_bias (key-only
bias `lambda_block * beta(p_evt) * g` on the haptic token group). Training
targets are auto-derived (§1): event CE at t+1, gate BCE vs contact-within-Δ.

Self-anticipation input during TRAINING: cfg.self_anticipation
  "gt_noised" (default) — the GT contact package at t plus config-scale noise
      (documented approximation; avoids a second forward);
  "two_pass" — caller runs a no-grad 1-NFE forward first and feeds the real
      prev-replan prediction (used for the final teacher runs and the
      lead-time evaluation). At DEPLOYMENT the policy always feeds the true
      previous replan's package.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.config.hardware import HardwareConfig
from phantom.config.model import AccConfig, N_EVENTS


@dataclass
class AccInputs:
    wrist_feat_B_D: torch.Tensor            # WristTCN(w_{t-k:t}) — from HHT
    intent_B_H_A: torch.Tensor              # previously committed action chunk
    prev_cpk_summary_B_S: torch.Tensor      # ContactPacker.flatten_summary(c_hat_prev)
    react_score_B: torch.Tensor | None      # CASA statistic (None in the student)


class AccOutput(dict):
    """keys: g, g_ant, g_react, alpha, p_evt, event_logits"""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


class AccGate(nn.Module):
    def __init__(self, cfg: AccConfig, hw: HardwareConfig, *, wrist_dim: int,
                 z_vis_dim: int, cpk_summary_dim: int, student: bool):
        super().__init__()
        self.cfg = cfg
        self.student = student
        d = cfg.d_acc
        A = hw.control.action_dim
        H = hw.control.chunk_horizon
        self.wrist_proj = nn.Linear(wrist_dim, d)
        self.vis_proj = nn.Linear(z_vis_dim, d)
        self.intent_mlp = nn.Sequential(nn.Linear(H * A, d), nn.SiLU(), nn.Linear(d, d))
        self.cpk_mlp = nn.Sequential(nn.Linear(cpk_summary_dim, d), nn.SiLU(),
                                     nn.Linear(d, d))
        self.phi = nn.Sequential(nn.Linear(4 * d, cfg.hidden), nn.SiLU(),
                                 nn.Linear(cfg.hidden, cfg.d_e), nn.SiLU())
        self.W_g = nn.Linear(cfg.d_e, 1)
        self.W_e = nn.Linear(cfg.d_e, N_EVENTS)
        self.W_alpha = nn.Linear(cfg.d_e, 1)
        # reactive (CASA) path: scalar statistic -> gate
        self.psi_react = nn.Sequential(nn.Linear(1, cfg.react_hidden), nn.SiLU(),
                                       nn.Linear(cfg.react_hidden, 1))
        # beta: learnable per-event up-weighting (softplus-positive)
        self.beta_raw = nn.Parameter(torch.full((N_EVENTS,), cfg.beta_init))

    def forward(self, inp: AccInputs, z_vis_B_D: torch.Tensor) -> AccOutput:
        B = z_vis_B_D.shape[0]
        # inputs arrive from mixed-precision producers (the trunk's embedding
        # path emits fp32 z_vis under bf16 training) — normalize to our dtype
        dt = self.vis_proj.weight.dtype
        e = self.phi(torch.cat([
            self.wrist_proj(inp.wrist_feat_B_D.to(dt)),
            self.vis_proj(z_vis_B_D.to(dt)),
            self.intent_mlp(inp.intent_B_H_A.reshape(B, -1).to(dt)),
            self.cpk_mlp(inp.prev_cpk_summary_B_S.to(dt)),
        ], dim=-1))
        # probabilities are formed in fp32: a bf16 sigmoid already rounds to
        # exactly 1.0 at logit ~6.9, so casting AFTER it (losses.py clamps
        # `.float()`) cannot restore precision and the BCE gradient on a
        # saturated false-positive gate is dead (2026-08-26)
        event_logits = self.W_e(e).float()
        p_evt = F.softmax(event_logits, dim=-1)
        g_ant = torch.sigmoid(self.W_g(e).float()).squeeze(-1)

        if self.student or inp.react_score_B is None:
            alpha = torch.ones(B, device=e.device, dtype=torch.float32)
            g_react = torch.zeros_like(g_ant)
        else:
            alpha = torch.sigmoid(self.W_alpha(e).float()).squeeze(-1)
            g_react = torch.sigmoid(
                self.psi_react(inp.react_score_B.reshape(B, 1).to(e.dtype)).float()).squeeze(-1)
        g = alpha * g_ant + (1.0 - alpha) * g_react

        return AccOutput(g=g, g_ant=g_ant, g_react=g_react, alpha=alpha,
                         p_evt=p_evt, event_logits=event_logits)

    def beta_of_events(self, p_evt: torch.Tensor) -> torch.Tensor:
        """beta(p_evt): expected event up-weighting, (B,) — up-weights
        onset/slip/release through the learned per-event beta."""
        beta = F.softplus(self.beta_raw)
        return (p_evt * beta).sum(-1)
