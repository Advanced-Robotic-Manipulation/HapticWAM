"""ACE readout heads on the CONTACT-frame hidden states (pipeline.md §4):
EventReadout (primary event logits) and SigmaHead (heteroscedastic per-group
log-variance -> HID confidence weights + the runtime speed governor)."""

from __future__ import annotations

import torch
import torch.nn as nn

from phantom.config.model import N_EVENTS

# per-output-group sigma channels (order is fixed; consumers index by name)
SIGMA_GROUPS: tuple[str, ...] = ("d_disp", "d_fz", "mask", "cop", "slip",
                                 "wrench", "wrist")


class EventReadout(nn.Module):
    """(B, Tc, S, D) contact hidden -> (B, Tc, N_EVENTS) logits."""

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model),
                                 nn.Linear(d_model, hidden), nn.SiLU(),
                                 nn.Linear(hidden, N_EVENTS))

    def forward(self, h_B_Tc_S_D: torch.Tensor) -> torch.Tensor:
        return self.net(h_B_Tc_S_D.float().mean(dim=2))


class SigmaHead(nn.Module):
    """(B, Tc, S, D) -> (B, Tc, K) log sigma per output group. Clamped to keep
    the heteroscedastic NLL well-behaved."""

    LOG_SIGMA_RANGE = (-5.0, 3.0)

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model),
                                 nn.Linear(d_model, hidden), nn.SiLU(),
                                 nn.Linear(hidden, len(SIGMA_GROUPS)))

    def forward(self, h_B_Tc_S_D: torch.Tensor) -> torch.Tensor:
        log_sigma = self.net(h_B_Tc_S_D.float().mean(dim=2))
        return log_sigma.clamp(*self.LOG_SIGMA_RANGE)

    @staticmethod
    def governor_sigma(log_sigma_B_Tc_K: torch.Tensor) -> torch.Tensor:
        """Scalar uncertainty per future step for the speed governor:
        mean sigma over the safety-relevant groups (d_fz, slip, mask)."""
        idx = [SIGMA_GROUPS.index(g) for g in ("d_fz", "slip", "mask")]
        return log_sigma_B_Tc_K[..., idx].exp().mean(-1)
