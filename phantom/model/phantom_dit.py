"""PhantomDiT: Cosmos-Predict2.5-2B action-chunk-conditioned DiT extended with
the PHANTOM sequence (LFA action frames + ACE contact frames + HHT observation
frames), frame-type embeddings, per-frame RoPE positions, and the ACC
attention bias (pipeline.md §3/§4/§6).

Subclasses ActionChunkConditionedMinimalV1LVGDiT (the class the released
robot/action-cond checkpoint was trained as — verified by its
action_embedder_B_D.fc1 in-dim of action_dim x 4). The forward is a
re-implementation of the parent's (the parent forward is monolithic) calling
the same submodules, so the pretrained computation on video frames is
preserved exactly; every new parameter is named phantom_* and initialized so
step 0 is a no-op on the pretrained paths:
  - frame-type embeddings init to zeros,
  - ACC per-block bias scales (lambdas) init to 0,
  - structural mask only FORBIDS attention (new frames), never changes
    video-frame -> video-frame attention.

This module may only be imported after phantom.backbone.loader.setup_cosmos().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange

from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import HardwareConfig
from phantom.config.model import PhantomModelConfig
from phantom.model.acc import AccGate, AccInputs, AccOutput
from phantom.model.attention_bias import (BiasHolder, acc_key_bias,
                                          install_bias_hooks,
                                          structural_bias_tokens)
from phantom.model.sequence import FrameGroup, SequenceLayout

from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (  # noqa: E501
    ActionChunkConditionedMinimalV1LVGDiT,
)


@dataclass
class PhantomNetOutput:
    velocity_B_C_T_H_W: torch.Tensor
    acc: AccOutput | None
    contact_hidden_B_Tc_S_D: torch.Tensor | None   # final-block hidden of CONTACT tokens


class PhantomDiT(ActionChunkConditionedMinimalV1LVGDiT):
    def __init__(self, *, layout: SequenceLayout, mc: PhantomModelConfig,
                 bb: BackboneConfig, hw: HardwareConfig,
                 cpk_summary_dim: int | None = None, **cosmos_kwargs):
        super().__init__(**cosmos_kwargs)
        # non-module attrs (no params): default layout + configs
        self.layout = layout
        self.mc = mc
        self.bb_cfg = bb

        D = self.model_channels
        self.phantom_frame_type_emb = nn.Embedding(len(FrameGroup), D)
        self.phantom_frame_type_emb_3d = nn.Embedding(len(FrameGroup), 3 * D)
        nn.init.zeros_(self.phantom_frame_type_emb.weight)
        nn.init.zeros_(self.phantom_frame_type_emb_3d.weight)
        self.phantom_bias_lambdas = nn.Parameter(
            torch.full((self.num_blocks,), mc.acc.lambda_bias_init))

        if cpk_summary_dim is None:
            from phantom.model.ace.packing import ContactPacker
            cpk_summary_dim = ContactPacker(hw, layout).summary_dim()
        self.phantom_acc = AccGate(mc.acc, hw, wrist_dim=mc.hht_dim,
                                   z_vis_dim=D, cpk_summary_dim=cpk_summary_dim,
                                   student=layout.student)

        self.phantom_bias_holder = BiasHolder()
        install_bias_hooks(self.blocks, self.phantom_bias_holder)
        self._rope_cache: dict = {}
        self._structural_cache: dict = {}

    # ------------------------------------------------------------------
    # custom RoPE: per-frame temporal positions (rope_time_mode)
    # ------------------------------------------------------------------
    def _phantom_rope(self, layout: SequenceLayout, fps: torch.Tensor | None,
                      device: torch.device) -> torch.Tensor:
        key = (layout.slots, layout.drop_video, str(device),
               None if fps is None else float(fps.flatten()[0]))
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        pe = self.pos_embedder
        Hp = layout.lat_h // self.patch_spatial
        Wp = layout.lat_w // self.patch_spatial
        t_pos = torch.from_numpy(
            layout.rope_frame_positions(self.mc.rope_time_mode)).float().to(device)

        h_theta = 10000.0 * pe.h_ntk_factor
        w_theta = 10000.0 * pe.w_ntk_factor
        t_theta = 10000.0 * pe.t_ntk_factor
        h_freqs = 1.0 / (h_theta ** pe.dim_spatial_range.float().to(device))
        w_freqs = 1.0 / (w_theta ** pe.dim_spatial_range.float().to(device))
        t_freqs = 1.0 / (t_theta ** pe.dim_temporal_range.float().to(device))

        if pe.enable_fps_modulation and fps is not None:
            t_pos = t_pos / float(fps.flatten()[0]) * pe.base_fps
        seq_h = torch.arange(Hp, dtype=torch.float32, device=device)
        seq_w = torch.arange(Wp, dtype=torch.float32, device=device)
        half_t = torch.outer(t_pos, t_freqs)             # (T, dt/2)
        half_h = torch.outer(seq_h, h_freqs)
        half_w = torch.outer(seq_w, w_freqs)

        em = torch.cat([
            half_t[:, None, None, :].expand(-1, Hp, Wp, -1),
            half_h[None, :, None, :].expand(len(t_pos), -1, Wp, -1),
            half_w[None, None, :, :].expand(len(t_pos), Hp, -1, -1),
        ] * 2, dim=-1)                                   # (T, Hp, Wp, head_dim)
        rope = rearrange(em, "t h w d -> (t h w) 1 1 d").float()
        self._rope_cache[key] = rope
        return rope

    def _structural(self, layout: SequenceLayout, device: torch.device) -> torch.Tensor:
        key = (layout.slots, layout.drop_video, layout.video_attend, str(device))
        cached = self._structural_cache.get(key)
        if cached is None:
            cached = structural_bias_tokens(layout, device=device)
            self._structural_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    def _action_adaln_embeddings(self, action: torch.Tensor | None,
                                 layout: SequenceLayout, B: int,
                                 device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Pretrained per-latent-frame action-AdaLN path fed with the INTENT
        chunk (previously committed actions). Video frame k in 1..t_video-1
        receives the embedding of action group k-1 (the pretrained semantics);
        all non-video frames receive zeros. -> (B, T_total, D), (B, T_total, 3D)."""
        D = self.model_channels
        T = layout.t_total
        emb = torch.zeros(B, T, D, device=device, dtype=dtype)
        emb3 = torch.zeros(B, T, 3 * D, device=device, dtype=dtype)
        if action is None or not self.mc.use_action_adaln_intent \
                or not layout.has(FrameGroup.VIDEO_GEN):
            return emb, emb3
        apf = self._num_action_per_latent_frame
        n_groups = layout.t_video_gen
        need = n_groups * apf
        assert action.shape[1] >= need, (
            f"intent chunk has {action.shape[1]} actions; the pretrained AdaLN path "
            f"needs {need} (= {n_groups} latent frames x {apf})")
        a = action[:, :need].reshape(B, n_groups, apf * action.shape[-1])
        a_emb = self.action_embedder_B_D(a)              # (B, n_groups, D)
        a_emb3 = self.action_embedder_B_3D(a)
        vid = layout.frame_slice(FrameGroup.VIDEO_GEN)
        emb[:, vid] = a_emb.to(dtype)
        emb3[:, vid] = a_emb3.to(dtype)
        return emb, emb3

    # ------------------------------------------------------------------
    def forward(  # type: ignore[override]
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor | None = None,              # intent chunk (B, H, A)
        acc_inputs: AccInputs | None = None,
        layout: SequenceLayout | None = None,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_contact_hidden: bool = True,
        crossattn_projected: bool = False,   # crossattn_emb already through crossattn_proj
        **kwargs,
    ) -> PhantomNetOutput:
        layout = layout or self.layout
        B = x_B_C_T_H_W.shape[0]
        # normalize the input latent to the net's parameter dtype — callers on
        # mixed-precision paths (sample() latents, VAE fp32 outputs) may hand
        # us fp32 under bf16 training
        net_dtype = self.x_embedder.proj[1].weight.dtype \
            if hasattr(self.x_embedder, "proj") else next(self.parameters()).dtype
        x_B_C_T_H_W = x_B_C_T_H_W.to(net_dtype)
        crossattn_emb = crossattn_emb.to(net_dtype)
        device, in_dtype = x_B_C_T_H_W.device, x_B_C_T_H_W.dtype
        assert x_B_C_T_H_W.shape[2] == layout.t_total, (
            f"input T={x_B_C_T_H_W.shape[2]} != layout T={layout.t_total}")

        # 1) condition-mask channel (parent VIDEO branch)
        x_B_C_T_H_W = torch.cat(
            [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)],
            dim=1)

        # 2) rectified-flow timestep scaling (parent)
        timesteps_B_T = timesteps_B_T * self.timestep_scale
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        if timesteps_B_T.shape[1] == 1:
            timesteps_B_T = timesteps_B_T.expand(-1, layout.t_total)

        # 3) patch embedding (+ padding-mask channel inside)
        if padding_mask is None:
            padding_mask = torch.zeros(B, 1, x_B_C_T_H_W.shape[-2],
                                       x_B_C_T_H_W.shape[-1], device=device,
                                       dtype=in_dtype)
        x_B_T_H_W_D, rope_standard, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask)
        # every named layout mode goes through the phantom rope (which reads
        # layout.rope_frame_positions for the mode); ONLY "append" means "use
        # the backbone's standard sequential rope". A `== "aligned"` test here
        # silently dropped the v4 "time_true" mode into the untested append
        # geometry while the checkpoint recorded time_true (readiness audit
        # 2026-08-14 — the fix was provably inert).
        rope_emb = (rope_standard if self.mc.rope_time_mode == "append"
                    else self._phantom_rope(layout, fps, device))

        # 4) cross-attention context
        if self.use_crossattn_projection and not crossattn_projected:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        context_input = crossattn_emb

        # 5) t embeddings + action AdaLN (intent) + frame-type embeddings
        a_emb, a_emb3 = self._action_adaln_embeddings(action, layout, B, device,
                                                      torch.float32)
        group_ids = torch.from_numpy(layout.group_of_frame()).to(device)
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = (t_embedding_B_T_D + a_emb
                                 + self.phantom_frame_type_emb(group_ids).unsqueeze(0))
            adaln_lora_B_T_3D = (adaln_lora_B_T_3D + a_emb3
                                 + self.phantom_frame_type_emb_3d(group_ids).unsqueeze(0))
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # 6) ACC gate from leading signals + pooled visual tokens
        # NOTE: the holder is cleared at the START of the next forward, not in
        # a finally block — activation checkpointing re-runs the blocks during
        # backward and must see the SAME bias tensors it saw in forward.
        acc_out: AccOutput | None = None
        holder = self.phantom_bias_holder
        holder.clear()
        holder.structural = self._structural(layout, device)
        if acc_inputs is not None:
            vis_groups = [FrameGroup.VIDEO_COND]
            if not layout.student:
                vis_groups += [g for g in (FrameGroup.OBS_GEL, FrameGroup.OBS_MECH)
                               if layout.has(g)]
            pooled = [x_B_T_H_W_D[:, layout.frame_slice(g)].mean(dim=(1, 2, 3))
                      for g in vis_groups]
            z_vis = torch.stack(pooled, 0).mean(0).float()
            acc_out = self.phantom_acc(acc_inputs, z_vis)
            beta = self.phantom_acc.beta_of_events(acc_out.p_evt)
            holder.acc_key = acc_key_bias(layout, acc_out.g, beta)
            holder.lambdas = list(self.phantom_bias_lambdas)

        # 7) transformer blocks with the bias hook armed
        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D, t_embedding_B_T_D, context_input,
                rope_emb_L_1_1_D=rope_emb,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb)

        contact_hidden = None
        if return_contact_hidden and layout.has(FrameGroup.CONTACT):
            sl = layout.frame_slice(FrameGroup.CONTACT)
            h = x_B_T_H_W_D[:, sl]                        # (B, Tc, Hp, Wp, D)
            contact_hidden = h.flatten(2, 3)              # (B, Tc, Hp*Wp, D)

        # 8) output head
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D,
                                       adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        velocity = self.unpatchify(x_B_T_H_W_O)
        return PhantomNetOutput(velocity_B_C_T_H_W=velocity, acc=acc_out,
                                contact_hidden_B_Tc_S_D=contact_hidden)
