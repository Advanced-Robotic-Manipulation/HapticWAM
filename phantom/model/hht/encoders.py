"""Small trained HHT encoders (pipeline.md §6c): contact-state MLP (~1M),
wrist-F/T temporal conv (~2M, also feeds ACC), UR-state MLP (~1M).
All input dims derive from the hardware config."""

from __future__ import annotations

import torch
import torch.nn as nn

from phantom.config.hardware import HardwareConfig


class ContactStateMLP(nn.Module):
    """Per-finger contact-state vector -> embedding.
    Input (B, F, contact_state_dim) = [wrench(6) | area | CoP(2) | slip | mask_frac]."""

    def __init__(self, hw: HardwareConfig, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hw.contact_state_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, cs_B_F_D: torch.Tensor) -> torch.Tensor:
        return self.net(cs_B_F_D)                      # (B, F, dim)


class WristTCN(nn.Module):
    """Dilated causal temporal conv over the wrist F/T window
    (B, L, 6) with L = hw.wrist_ft.window_len -> (B, dim)."""

    def __init__(self, hw: HardwareConfig, dim: int = 256, channels: int = 64,
                 n_blocks: int = 4):
        super().__init__()
        self.in_dim = hw.wrist_ft.dim
        layers = []
        c_in = self.in_dim
        for i in range(n_blocks):
            d = 2 ** i
            layers.append(_CausalConvBlock(c_in, channels, dilation=d))
            c_in = channels
        self.tcn = nn.ModuleList(layers)
        self.head = nn.Linear(channels, dim)

    def forward(self, w_B_L_6: torch.Tensor) -> torch.Tensor:
        h = w_B_L_6.transpose(1, 2)                    # (B, 6, L)
        for blk in self.tcn:
            h = blk(h)
        return self.head(h[..., -1])                   # last timestep -> (B, dim)


class _CausalConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, k: int = 3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.act = nn.SiLU()
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = self.act(self.conv2(nn.functional.pad(y, (self.pad, 0))))
        return y + self.res(x)


class URStateMLP(nn.Module):
    """(B, ur_state_dim) -> (B, dim)."""

    def __init__(self, hw: HardwareConfig, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hw.ur_state_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, q_B_D: torch.Tensor) -> torch.Tensor:
        return self.net(q_B_D)
