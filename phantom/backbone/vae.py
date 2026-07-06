"""Frozen Wan2.1 video-VAE wrapper (Cosmos tokenizer). Encodes pixel windows
to 16-ch latents; used for scene RGB and gel images (HHT frozen paths).

Real weights path comes from paths.yaml (tokenizer.pth). For CPU smoke tests a
FakeVAE with the same interface (strided average pooling to the right latent
geometry) stands in — the model code is agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from phantom.config.backbone import BackboneConfig
from phantom.config.paths import PathsConfig

log = logging.getLogger(__name__)


class VAEBase:
    """Interface: encode (B,3,T,H,W) in [-1,1] -> (B,16,T',H/8,W/8); decode inverse."""
    lat_ch: int
    spatial_comp: int
    temporal_comp: int

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def latent_t(self, pixel_t: int) -> int:
        return 1 + (pixel_t - 1) // self.temporal_comp


class WanVAE(VAEBase):
    """Wraps cosmos's Wan2pt1VAEInterface (frozen, no grad)."""

    def __init__(self, bb: BackboneConfig, paths: PathsConfig,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from phantom.backbone.loader import setup_cosmos
        setup_cosmos(paths)
        from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

        self.lat_ch = bb.lat_ch
        self.spatial_comp = bb.spatial_comp
        self.temporal_comp = bb.temporal_comp
        self.device = device
        self.dtype = dtype
        self._vae = Wan2pt1VAEInterface(vae_pth=str(paths.cosmos_tokenizer))
        log.info("Wan2.1 VAE loaded from %s", paths.cosmos_tokenizer)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._vae.encode(pixels.to(self.device, self.dtype))

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self._vae.decode(latents.to(self.device, self.dtype))


class FakeVAE(VAEBase):
    """Deterministic, weight-free stand-in with the exact latent geometry.
    Encode: space-to-channel average pooling + fixed random projection to 16ch.
    Only for smoke tests — carries NO pretrained-latent semantics."""

    def __init__(self, bb: BackboneConfig, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.lat_ch = bb.lat_ch
        self.spatial_comp = bb.spatial_comp
        self.temporal_comp = bb.temporal_comp
        self.device = device
        self.dtype = dtype
        g = torch.Generator().manual_seed(0)
        self._proj = torch.randn(bb.lat_ch, 3, generator=g) / 3.0

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = pixels.shape
        Tl = self.latent_t(T)
        # temporal: first frame + strided mean of the rest
        idx_groups = [pixels[:, :, :1]]
        rest = pixels[:, :, 1:]
        if rest.shape[2]:
            rest = rest.reshape(B, C, Tl - 1, self.temporal_comp, H, W).mean(3)
            idx_groups.append(rest)
        x = torch.cat(idx_groups, dim=2)                       # (B, 3, Tl, H, W)
        x = F.avg_pool3d(x, (1, self.spatial_comp, self.spatial_comp))
        proj = self._proj.to(x.device, x.dtype)
        lat = torch.einsum("lc,bcthw->blthw", proj, x)
        return lat.to(self.dtype)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        B, L, Tl, h, w = latents.shape
        proj = self._proj.to(latents.device, latents.dtype)
        x = torch.einsum("lc,blthw->bcthw", proj, latents)     # crude pseudo-inverse-ish
        x = F.interpolate(x.flatten(0, 1).unsqueeze(1),
                          scale_factor=(1, self.spatial_comp, self.spatial_comp),
                          mode="nearest").squeeze(1).reshape(B, 3, Tl, h * self.spatial_comp,
                                                             w * self.spatial_comp)
        T = 1 + (Tl - 1) * self.temporal_comp
        return F.interpolate(x, size=(T, x.shape[-2], x.shape[-1]), mode="nearest")


def make_vae(bb: BackboneConfig, paths: PathsConfig | None, *, fake: bool = False,
             device: str = "cpu", dtype: torch.dtype = torch.float32) -> VAEBase:
    if fake or paths is None:
        return FakeVAE(bb, device=device, dtype=dtype)
    return WanVAE(bb, paths, device=device, dtype=dtype)
