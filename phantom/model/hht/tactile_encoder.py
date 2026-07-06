"""TactileFieldEncoder (~5M): the trained conv encoder over the 8-channel
tactile field stack [D_t || F_t] (pipeline.md §6c) — one tensor, one encoder.

Input (per finger): (C8, H, W) with (H, W) = hw.tactile.field — everything
sized from the config. Downsamples by stride-2 stages until the spatial map is
<= ~12x12, FiLM-conditioned on the contact-state embedding.

Also defines the two SSL pretrain heads (masked reconstruction +
cross-channel force-from-displacement prediction) used by
train/pretrain_tactile.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.config.hardware import HardwareConfig
from phantom.data.derived import channel_slices


def _n_stages(h: int, w: int, target: int = 12) -> int:
    n = 0
    while max(h, w) > target and n < 6:
        h, w = (h + 1) // 2, (w + 1) // 2
        n += 1
    return n


class TactileFieldEncoder(nn.Module):
    def __init__(self, hw: HardwareConfig, out_dim: int = 256, film_dim: int | None = None):
        super().__init__()
        self.hw = hw
        C8 = hw.tactile.field_ch
        H, W = hw.tactile.field.hw
        stages = _n_stages(H, W)
        chans = [C8] + [min(256, 32 * 2 ** i) for i in range(stages)]
        blocks = []
        for i in range(stages):
            blocks.append(nn.Sequential(
                nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1),
                nn.GroupNorm(min(8, chans[i + 1]), chans[i + 1]),
                nn.SiLU(),
                nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1),
                nn.GroupNorm(min(8, chans[i + 1]), chans[i + 1]),
                nn.SiLU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_ch = chans[-1]
        self.out_dim = out_dim
        self.head = nn.Linear(self.out_ch, out_dim)
        self.film = nn.Linear(film_dim, 2 * self.out_ch) if film_dim else None
        self._stages = stages

    # ------------------------------------------------------------------
    def forward_spatial(self, x_BF_C_H_W: torch.Tensor,
                        film_BF_D: torch.Tensor | None = None) -> torch.Tensor:
        """(B*F, C8, H, W) -> (B*F, out_ch, h', w') pre-pool feature map."""
        h = x_BF_C_H_W
        for blk in self.blocks:
            h = blk(h)
        if self.film is not None and film_BF_D is not None:
            scale, shift = self.film(film_BF_D).chunk(2, dim=-1)
            h = h * (1 + scale[..., None, None]) + shift[..., None, None]
        return h

    def forward(self, x_BF_C_H_W: torch.Tensor,
                film_BF_D: torch.Tensor | None = None) -> torch.Tensor:
        """(B*F, C8, H, W) -> (B*F, out_dim) pooled feature."""
        h = self.forward_spatial(x_BF_C_H_W, film_BF_D)
        return self.head(h.mean(dim=(-2, -1)))


# ---------------------------------------------------------------------------
# SSL pretrain heads (program 1)
# ---------------------------------------------------------------------------

class TactilePretrainModel(nn.Module):
    """Encoder + light deconv decoders for the two contact-play objectives:
      (a) masked-patch reconstruction of all 8 channels;
      (b) cross-channel: predict the 5 force channels (shear + dist force)
          from displacement-only input (force channels zeroed) — free native
          supervision (pipeline.md §7).
    """

    def __init__(self, hw: HardwareConfig, out_dim: int = 256):
        super().__init__()
        self.hw = hw
        self.encoder = TactileFieldEncoder(hw, out_dim=out_dim)
        C8 = hw.tactile.field_ch
        ch = channel_slices(hw.tactile)
        self.force_slices = (ch["shear"], ch["dist_force"])
        n_force = (ch["shear"].stop - ch["shear"].start
                   + ch["dist_force"].stop - ch["dist_force"].start)
        self.decoder_recon = self._make_decoder(self.encoder.out_ch, C8)
        self.decoder_cross = self._make_decoder(self.encoder.out_ch, n_force)

    def _make_decoder(self, in_ch: int, out_ch: int) -> nn.Module:
        ups = []
        ch = in_ch
        for _ in range(self.encoder._stages):
            nxt = max(16, ch // 2)
            ups += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, nxt, 3, padding=1), nn.SiLU()]
            ch = nxt
        ups.append(nn.Conv2d(ch, out_ch, 3, padding=1))
        return nn.Sequential(*ups)

    # ------------------------------------------------------------------
    def mask_patches(self, x: torch.Tensor, ratio: float, patch: int,
                     gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero out a random `ratio` of (patch x patch) tiles. Returns
        (masked input, boolean mask map (B,1,H,W) of the HIDDEN region)."""
        B, C, H, W = x.shape
        gh, gw = (H + patch - 1) // patch, (W + patch - 1) // patch
        keep = torch.rand(B, 1, gh, gw, device=x.device, generator=gen) > ratio
        mask = keep.repeat_interleave(patch, 2).repeat_interleave(patch, 3)[..., :H, :W]
        return x * mask, ~mask

    def _decode_to(self, dec: nn.Module, feat: torch.Tensor, hw_out: tuple[int, int]) -> torch.Tensor:
        y = dec(feat)
        if y.shape[-2:] != hw_out:
            y = F.interpolate(y, size=hw_out, mode="bilinear", align_corners=False)
        return y

    def training_losses(self, x_B_C_H_W: torch.Tensor, mask_ratio: float,
                        mask_patch: int) -> dict[str, torch.Tensor]:
        B, C, H, W = x_B_C_H_W.shape
        # (a) masked reconstruction
        x_masked, hidden = self.mask_patches(x_B_C_H_W, mask_ratio, mask_patch)
        feat = self.encoder.forward_spatial(x_masked)
        recon = self._decode_to(self.decoder_recon, feat, (H, W))
        w = hidden.float()
        loss_recon = ((recon - x_B_C_H_W) ** 2 * w).sum() / (w.sum() * C + 1e-6)
        # (b) cross-channel prediction: zero the force channels at input
        x_disp_only = x_B_C_H_W.clone()
        target_force = []
        for sl in self.force_slices:
            target_force.append(x_B_C_H_W[:, sl])
            x_disp_only[:, sl] = 0.0
        target_force = torch.cat(target_force, dim=1)
        feat2 = self.encoder.forward_spatial(x_disp_only)
        pred_force = self._decode_to(self.decoder_cross, feat2, (H, W))
        loss_cross = F.mse_loss(pred_force, target_force)
        return {"masked_recon": loss_recon, "cross_channel": loss_cross}
