"""HHT: Heterogeneous Haptic Tokenizer (pipeline.md §6c).

Assembles the observation latent frames of the extended sequence:
  OBS_GEL     — 2-finger gel-image VAE latents fused by a 1x1 conv (teacher)
  OBS_MECH    — tactile-field conv encoder spatial map + contact-state FiLM (teacher)
  OBS_PROPRIO — [WristTCN || URStateMLP] -> projected spatial map (both stages)

Observation frames are FRAME_REPLACE conditioning (pinned, never denoised), so
they may be produced by trainable encoders without giving the denoiser a
moving RF target.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.backbone.vae import VAEBase
from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import HardwareConfig
from phantom.config.model import PhantomModelConfig
from phantom.model.hht.encoders import ContactStateMLP, URStateMLP, WristTCN
from phantom.model.hht.tactile_encoder import TactileFieldEncoder
from phantom.model.sequence import FrameGroup, SequenceLayout


class HHT(nn.Module):
    def __init__(self, hw: HardwareConfig, bb: BackboneConfig, mc: PhantomModelConfig,
                 vae: VAEBase, *, student: bool):
        super().__init__()
        self.hw = hw
        self.bb = bb
        self.student = student
        self.mask_wrist = bool(mc.mask_wrist)   # input ablation, see below
        self.vae = vae                      # frozen; not an nn submodule on purpose
        d = mc.hht_dim
        lat = bb.lat_ch
        self.lat_hw = (bb.lat_h, bb.lat_w)

        # -- proprio path (both teacher and student)
        self.phantom_wrist_tcn = WristTCN(hw, dim=d)
        self.phantom_ur_mlp = URStateMLP(hw, dim=d)
        pr_h, pr_w = max(1, bb.lat_h // 4), max(1, bb.lat_w // 4)
        self._pr_hw = (pr_h, pr_w)
        self.phantom_proprio_proj = nn.Linear(2 * d, lat * pr_h * pr_w)

        if not student:
            # -- gel path: per-finger VAE latents fused 1x1 (2F*lat -> lat)
            self.phantom_gel_fuse = nn.Conv2d(hw.n_fingers * lat, lat, kernel_size=1)
            # -- mech path: field conv encoder + contact-state FiLM + conv head
            self.phantom_contact_mlp = ContactStateMLP(hw, dim=d)
            self.phantom_tactile_enc = TactileFieldEncoder(hw, out_dim=d, film_dim=d)
            self.phantom_mech_head = nn.Conv2d(
                hw.n_fingers * self.phantom_tactile_enc.out_ch, lat, kernel_size=1)

    # ------------------------------------------------------------------
    def load_pretrained_tactile(self, ckpt_path: Path) -> None:
        """Load contact-play SSL weights (train/pretrain_tactile.py output).

        The SSL model has no FiLM layer (film_dim=None), while this encoder
        does — load non-strict and assert the ONLY missing keys are film.*
        (which stay at their zero-init no-op, see TactileFieldEncoder)."""
        assert not self.student, "student HHT has no tactile encoder"
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        missing, unexpected = self.phantom_tactile_enc.load_state_dict(
            payload["encoder"], strict=False)
        bad_missing = [k for k in missing if not k.startswith("film.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"SSL tactile checkpoint mismatch: missing={bad_missing} "
                f"unexpected={list(unexpected)} ({ckpt_path})")

    # ------------------------------------------------------------------
    def _wrist_input(self, batch: dict) -> torch.Tensor:
        """batch['wrist'] (B,L,6), zeroed when mask_wrist is set.

        The ONE place the wrist window enters the model — both OBS_PROPRIO
        (via obs_frames) and the ACC leading-signal branch (wrist_feature) go
        through here, so masking is total: the model's output becomes
        invariant to the recorded F/T window (the WristTCN still contributes
        its bias/constant path, which is what an ablated input looks like).
        This is what makes vision_only / no_distill / drop_tactile genuinely
        different inputs from the tactile-free student (P10A).
        """
        w = batch["wrist"]
        return torch.zeros_like(w) if self.mask_wrist else w

    def obs_frames(self, batch: dict) -> dict[FrameGroup, torch.Tensor]:
        """batch keys (from WindowSampler / policy ObsSnapshot):
        gel (B,F,3,res_h,res_w), fields (B,F,Hf,Wf,C8),
        contact_state (B,F,Dcs), wrist (B,L,6), ur_state (B,Dur).
        Returns {group: (B, lat_ch, 1, lat_h, lat_w)}."""
        out: dict[FrameGroup, torch.Tensor] = {}
        wrist = self.phantom_wrist_tcn(self._wrist_input(batch))
        ur = self.phantom_ur_mlp(batch["ur_state"])
        pr = self.phantom_proprio_proj(torch.cat([wrist, ur], dim=-1))
        B = pr.shape[0]
        pr = pr.reshape(B, -1, *self._pr_hw)
        pr = F.interpolate(pr, size=self.lat_hw, mode="bilinear", align_corners=False)
        out[FrameGroup.OBS_PROPRIO] = pr.unsqueeze(2)

        if self.student:
            return out

        # gel: all fingers' single frames through the frozen VAE in ONE call
        # (fingers folded into the batch dim — one launch instead of F)
        gel = batch["gel"]                                     # (B, F, 3, H, W)
        Fn = gel.shape[1]
        lat = self.vae.encode(gel.flatten(0, 1).unsqueeze(2))[:, :, 0]  # (B*F, lat, h, w)
        gel_lat = lat.reshape(B, Fn * lat.shape[1],
                              *lat.shape[-2:]).to(pr.dtype)    # (B, F*lat, h, w)
        out[FrameGroup.OBS_GEL] = self.phantom_gel_fuse(gel_lat).unsqueeze(2)

        # mech: conv encoder spatial maps (FiLM on contact state), fused 1x1
        fields = batch["fields"]                               # (B, F, Hf, Wf, C8)
        cs_emb = self.phantom_contact_mlp(batch["contact_state"])   # (B, F, d)
        x = fields.permute(0, 1, 4, 2, 3).flatten(0, 1)        # (B*F, C8, Hf, Wf)
        feat = self.phantom_tactile_enc.forward_spatial(
            x, film_BF_D=cs_emb.flatten(0, 1))                 # (B*F, C, h', w')
        feat = feat.reshape(B, -1, *feat.shape[-2:])           # (B, F*C, h', w')
        feat = F.interpolate(feat, size=self.lat_hw, mode="bilinear", align_corners=False)
        out[FrameGroup.OBS_MECH] = self.phantom_mech_head(feat).unsqueeze(2)
        return out

    def wrist_feature(self, batch: dict) -> torch.Tensor:
        """(B, d) — the WristTCN feature reused by ACC (leading signal)."""
        return self.phantom_wrist_tcn(self._wrist_input(batch))
