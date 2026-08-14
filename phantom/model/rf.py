"""PhantomRectifiedFlow: the joint rectified-flow model over the extended
sequence — owns the PhantomDiT, HHT, fixed packers, readout heads, frozen VAE
and text provider; implements FRAME_REPLACE conditioning, the grouped training
objective, and the few-NFE Euler sampler (pipeline.md §4/§6e).

RF conventions (transcribed from the cosmos rectified-flow stack):
    x_t = (1 - t) * x0 + t * eps,   velocity target v = eps - x0,
    t ~ logit-normal, time-shifted by t <- shift*t / (1 + (shift-1)*t),
    net receives t*1000 (timestep_scale=0.001 inside the net).
FRAME_REPLACE: conditioning frames (layout.cond_mask) are pinned to their
clean values with per-frame timestep 0 and excluded from every loss
(denoise_replace_gt_frames semantics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from phantom.backbone.text_embedding import TextEmbeddingProvider
from phantom.backbone.vae import VAEBase
from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import HardwareConfig
from phantom.config.model import PhantomModelConfig
from phantom.model.ace import losses as L
from phantom.model.ace.heads import EventReadout, SigmaHead
from phantom.model.ace.packing import (_CH_WRIST, ActionPacker, ContactPackage,
                                       ContactPacker, sigma_group_channels)
from phantom.model.acc import AccInputs, AccOutput
from phantom.model.sequence import FrameGroup, SequenceLayout


@dataclass
class PhantomPrediction:
    actions_B_H_A: torch.Tensor            # normalized; denormalize with NormStats
    cpk: ContactPackage
    event_logits_B_Tc_E: torch.Tensor
    log_sigma_B_Tc_K: torch.Tensor
    governor_sigma_B_Tc: torch.Tensor
    acc: AccOutput | None
    x_final_B_C_T_H_W: torch.Tensor


def package_from_batch(batch: dict) -> ContactPackage:
    """GT ContactPackage from a WindowSampler batch (cpk_* keys)."""
    return ContactPackage(
        event=batch["events"], d_disp=batch["cpk_d_disp"], d_fz=batch["cpk_d_fz"],
        mask=batch["cpk_mask"], cop=batch["cpk_cop"], slip=batch["cpk_slip"],
        wrench=batch["cpk_wrench"], wrist=batch["cpk_wrist"])


class PhantomRectifiedFlow(nn.Module):
    def __init__(self, net, hht, hw: HardwareConfig, bb: BackboneConfig,
                 mc: PhantomModelConfig, layout: SequenceLayout, vae: VAEBase,
                 text: TextEmbeddingProvider):
        super().__init__()
        self.net = net
        self.hht = hht
        self.hw = hw
        self.bb = bb
        self.mc = mc
        self.layout = layout
        self.vae = vae            # frozen, deliberately not registered
        self.text = text
        self.c_pack = ContactPacker(hw, layout)
        self.a_pack = ActionPacker(hw, layout)
        self.phantom_event_head = EventReadout(net.model_channels)
        self.phantom_sigma_head = SigmaHead(net.model_channels)
        g = torch.Generator().manual_seed(0)
        self._gen = g

    @property
    def device(self):
        return next(self.net.parameters()).device

    @property
    def dtype(self):
        return next(self.net.parameters()).dtype

    # ------------------------------------------------------------------
    def _time_shift(self, t: torch.Tensor) -> torch.Tensor:
        s = self.bb.rf_shift
        return s * t / (1.0 + (s - 1.0) * t)

    def _sample_t(self, B: int, device) -> torch.Tensor:
        u = torch.randn(B, generator=self._gen) * self.bb.rf_logit_normal_std \
            + self.bb.rf_logit_normal_mean
        return self._time_shift(torch.sigmoid(u)).to(device)

    # ------------------------------------------------------------------
    def build_x0(self, batch: dict, layout: SequenceLayout | None = None,
                 *, encode_gen: bool = True
                 ) -> tuple[torch.Tensor, torch.Tensor, AccInputs]:
        """Assemble the clean extended sequence + cond mask + ACC inputs.

        encode_gen=False (sampling): VIDEO_GEN x0 content is dead at sampling
        (those frames start from pure noise and are never cond-pinned), so
        only the single conditioning frame goes through the VAE — the policy
        tiles the current camera frame across the window, and encoding the
        other 12 identical pixel frames is pure latency."""
        layout = layout or self.layout
        dev, dt = self.device, self.dtype
        video = batch["video"].to(dev, dt)                        # (B, Tpix, 3, H, W)
        B = video.shape[0]
        pix = video if encode_gen else video[:, :1]
        vid_lat = self.vae.encode(pix.permute(0, 2, 1, 3, 4)).to(dt)  # (B,16,Tv,h,w)

        x0 = torch.zeros(B, layout.lat_c, layout.t_total, layout.lat_h, layout.lat_w,
                         device=dev, dtype=dt)
        x0[:, :, layout.frame_slice(FrameGroup.VIDEO_COND)] = vid_lat[:, :, :1]
        if layout.has(FrameGroup.VIDEO_GEN) and encode_gen:
            x0[:, :, layout.frame_slice(FrameGroup.VIDEO_GEN)] = vid_lat[:, :, 1:]

        obs_batch = {k: v.to(dev, dt) if torch.is_tensor(v) else v
                     for k, v in batch.items()
                     if k in ("gel", "fields", "contact_state", "wrist", "ur_state")}
        obs = self.hht.obs_frames(obs_batch)
        for g, frame in obs.items():
            if layout.has(g):
                x0[:, :, layout.frame_slice(g)] = frame.to(dt)

        gt_cpk = package_from_batch(batch).to(dev)
        x0[:, :, layout.frame_slice(FrameGroup.CONTACT)] = \
            self.c_pack.pack(gt_cpk).to(dt)
        x0[:, :, layout.frame_slice(FrameGroup.ACTION)] = \
            self.a_pack.pack(batch["action_chunk"].to(dev)).to(dt)

        cond = torch.from_numpy(layout.cond_mask_T()).to(dev)
        cond_mask = cond.reshape(1, 1, -1, 1, 1).expand(
            B, 1, -1, layout.lat_h, layout.lat_w).to(dt)

        acc_inputs = self._acc_inputs_train(batch, gt_cpk, obs_batch)
        return x0, cond_mask, acc_inputs

    def _acc_inputs_train(self, batch: dict, gt_cpk: ContactPackage,
                          obs_batch: dict) -> AccInputs:
        """Training-time self-anticipation per mc.acc.self_anticipation:

        'gt_noised' — GT package + noise stands in for the previous replan's
        prediction (fast proxy; the gate sees leaked GT — never report gate
        lead-time from a gt_noised-trained model);
        'two_pass'  — a no-grad short sample() produces the model's OWN
        predicted package, exactly like deployment's prev_cpk (the inner pass
        falls back to the noised-GT summary to terminate the recursion — the
        one-level approximation of an infinite replan history)."""
        dev, dt = self.device, self.dtype
        mode = self.mc.acc.self_anticipation
        if mode == "two_pass" and not getattr(self, "_in_anticipation_pass", False):
            self._in_anticipation_pass = True
            try:
                with torch.no_grad():
                    pred = self.sample(batch, nfe=max(2, self.mc.nfe // 2))
                summary = self.c_pack.flatten_summary(pred.cpk.detach()).to(dt)
            finally:
                self._in_anticipation_pass = False
        else:
            summary = self.c_pack.flatten_summary(gt_cpk).to(dt)
            noise = torch.randn(summary.shape, generator=self._gen) \
                .to(dev, dt) * self.mc.acc.gt_noise_scale
            summary = summary + noise
        react = batch.get("reactive")
        return AccInputs(
            wrist_feat_B_D=self.hht.wrist_feature(obs_batch).to(dt),
            intent_B_H_A=batch["prev_chunk"].to(dev, dt),
            prev_cpk_summary_B_S=summary,
            react_score_B=None if (self.layout.student or react is None)
            else react.to(dev, dt),
        )

    # ------------------------------------------------------------------
    def prepare_denoise(self, batch: dict, t_B: torch.Tensor,
                        eps_by_group: dict[FrameGroup, torch.Tensor] | None = None):
        """Build (x0, x_t, cond_mask, acc_inputs, t_B_T) at externally chosen t
        and (optionally) externally chosen per-group noise — lets HID noise the
        teacher's and student's SHARED groups identically even though their
        layouts differ. Returns also the eps actually used (per group)."""
        layout = self.layout
        x0, cond_mask, acc_inputs = self.build_x0(batch)
        B = x0.shape[0]
        dev, dt = x0.device, x0.dtype
        cond_T = torch.from_numpy(layout.cond_mask_T()).to(dev)
        t_B = t_B.to(dev, dt)
        t_B_T = t_B.reshape(B, 1).expand(B, layout.t_total).clone()
        t_B_T[:, cond_T] = 0.0
        eps = torch.randn(x0.shape, generator=self._gen).to(dev, dt)
        used: dict[FrameGroup, torch.Tensor] = {}
        for slot in layout.slots:
            g = slot.group
            if eps_by_group is not None and g in eps_by_group:
                eps[:, :, slot.t_slice] = eps_by_group[g].to(dev, dt)
            used[g] = eps[:, :, slot.t_slice]
        t_full = t_B.reshape(B, 1, 1, 1, 1)
        x_t = (1 - t_full) * x0 + t_full * eps
        x_t = torch.where(cond_mask.bool(), x0, x_t)
        return x0, x_t, cond_mask, acc_inputs, t_B_T, used

    # ------------------------------------------------------------------
    @staticmethod
    def _null_obs_batch(batch: dict) -> dict:
        """Copy of the batch with every OBSERVATION input nulled (classifier-
        free dropout). Targets (action_chunk, cpk_*, events, gate_label) and
        intent (prev_chunk) are untouched — only what the model perceives."""
        out = dict(batch)
        for k in ("video", "gel", "fields", "contact_state", "wrist",
                  "ur_state", "reactive"):
            if k in out and torch.is_tensor(out[k]):
                out[k] = torch.zeros_like(out[k])
        return out

    def training_step(self, batch: dict) -> dict[str, torch.Tensor]:
        layout = self.layout
        x0, cond_mask, acc_inputs = self.build_x0(batch)
        B = x0.shape[0]
        dev, dt = x0.device, x0.dtype

        t = self._sample_t(B, dev).to(dt)                          # (B,)
        cond_T = torch.from_numpy(layout.cond_mask_T()).to(dev)
        t_B_T = t.reshape(B, 1).expand(B, layout.t_total).clone()
        if self.mc.action_t_max_of_two and layout.has(FrameGroup.ACTION):
            # per-group timestep for ACTION: max of two draws biases its
            # supervision toward the high-noise band few-NFE sampling visits
            # first (velocity targets are t-independent, so only the noising
            # level and the timestep embedding change)
            t_act = torch.maximum(t, self._sample_t(B, dev).to(dt))
            t_B_T[:, layout.frame_slice(FrameGroup.ACTION)] = t_act.reshape(B, 1)
        t_B_T[:, cond_T] = 0.0                                     # clean cond frames

        text = batch.get("text")
        if self.mc.cond_dropout_p > 0 and float(torch.rand(
                (), generator=self._gen)) < self.mc.cond_dropout_p:
            # classifier-free conditioning dropout: this sample trains the
            # UNCONDITIONAL action/contact distribution — zero every
            # observation input jointly (prev_chunk stays: it is intent, not
            # observation). Enables obs-guidance at sampling.
            x0, cond_mask, acc_inputs = self.build_x0(
                self._null_obs_batch(batch), layout)
            text = [""] * B

        eps = torch.randn(x0.shape, generator=self._gen).to(dev, dt)
        t_frame = t_B_T.reshape(B, 1, layout.t_total, 1, 1).to(dt)
        x_t = (1 - t_frame) * x0 + t_frame * eps
        x_t = torch.where(cond_mask.bool(), x0, x_t)               # FRAME_REPLACE

        out = self.net(
            x_B_C_T_H_W=x_t, timesteps_B_T=t_B_T * 1000.0,
            crossattn_emb=self.text.get(B, text),
            condition_video_input_mask_B_C_T_H_W=cond_mask,
            action=batch["prev_chunk"].to(dev, dt),
            acc_inputs=acc_inputs, layout=layout,
            fps=torch.full((B,), self.bb.fps, device=dev))

        v_pred = out.velocity_B_C_T_H_W
        v_target = eps - x0
        x0_pred = x_t - t_frame * v_pred

        event_logits = self.phantom_event_head(out.contact_hidden_B_Tc_S_D)
        log_sigma = self.phantom_sigma_head(out.contact_hidden_B_Tc_S_D)

        # deliberate-failure demos supervise contact/event/gate but must not
        # train action imitation (windows.py sets action_weight=0 for them)
        act_w = batch.get("action_weight")
        if act_w is not None:
            act_w = torch.as_tensor(act_w, device=dev).reshape(-1)
        parts: dict[str, torch.Tensor] = {
            # channels: only the packer's live cells — averaging the zero
            # padding diluted the action gradient 4x and floored the metric
            "action_v_mse": L.group_velocity_mse(
                v_pred, v_target, layout, FrameGroup.ACTION, act_w,
                channels=slice(0, layout.actions_per_frame)),
            "contact_nll": L.contact_hetero_nll(
                x0_pred, x0, log_sigma, layout,
                group_channels=sigma_group_channels(self.hw.n_fingers)),
            "event_ce": L.event_ce(event_logits, batch["events"].to(dev)),
            "wrist_mse": L.wrist_region_mse(x0_pred, x0, layout, _CH_WRIST),
            "sigma_reg": (log_sigma ** 2).mean(),
        }
        if layout.has(FrameGroup.VIDEO_GEN):
            parts["video_v_mse"] = L.group_velocity_mse(v_pred, v_target, layout,
                                                        FrameGroup.VIDEO_GEN)
        if out.acc is not None:
            parts.update(L.acc_losses(out.acc, batch["gate_label"].to(dev),
                                      batch["events"][:, 0].to(dev),
                                      self.mc.acc.alpha_entropy_weight))
        parts["total"] = L.total_loss(parts, self.mc.loss)
        return parts

    # ------------------------------------------------------------------
    def velocity_at(self, batch: dict, x_t: torch.Tensor, t_B_T: torch.Tensor,
                    cond_mask: torch.Tensor, acc_inputs: AccInputs):
        """Single net call at externally-chosen (x_t, t) — used by HID behavior
        matching and the HID-S KL leash (shared-(x_t,t) velocity comparison)."""
        B = x_t.shape[0]
        return self.net(
            x_B_C_T_H_W=x_t, timesteps_B_T=t_B_T * 1000.0,
            crossattn_emb=self.text.get(B, batch.get("text")),
            condition_video_input_mask_B_C_T_H_W=cond_mask,
            action=batch["prev_chunk"].to(self.device, self.dtype),
            acc_inputs=acc_inputs, layout=self.layout,
            fps=torch.full((B,), self.bb.fps, device=self.device))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, batch: dict, *, nfe: int | None = None,
               prev_cpk: ContactPackage | None = None,
               drop_video: bool | None = None,
               guidance_scale: float = 1.0) -> PhantomPrediction:
        """Few-NFE Euler sampling of the joint sequence (the per-replan denoise).

        guidance_scale > 1 applies observation-guidance (classifier-free):
        v = v_null + s * (v_obs - v_null), doubling the per-step cost. Only
        meaningful for checkpoints trained with cond_dropout_p > 0."""
        nfe = nfe or self.mc.nfe
        drop_video = self.mc.drop_video_at_inference if drop_video is None else drop_video
        layout = (SequenceLayout.build(self.bb, self.mc, self.hw,
                                       student=self.layout.student, drop_video=True)
                  if drop_video else self.layout)
        x0, cond_mask, acc_inputs = self.build_x0(batch, layout, encode_gen=False)
        x0_null = acc_null = ctx_null = None
        if guidance_scale != 1.0:
            x0_null, _, acc_null = self.build_x0(
                self._null_obs_batch(batch), layout, encode_gen=False)
        if prev_cpk is not None:  # deployment: true previous-replan package
            acc_inputs.prev_cpk_summary_B_S = \
                self.c_pack.flatten_summary(prev_cpk.to(self.device)).to(self.dtype)
        B = x0.shape[0]
        dev, dt = x0.device, x0.dtype
        cond_T = torch.from_numpy(layout.cond_mask_T()).to(dev)

        x = torch.randn(x0.shape, generator=self._gen).to(dev, dt)
        x = torch.where(cond_mask.bool(), x0, x)

        # step-invariant conditioning, computed ONCE per replan (not per NFE
        # step): the projected text context (the 100352->1024 projection reads
        # a ~100M-param matrix — running it per step is pure waste), the
        # intent chunk on-device, and the fps tensor
        ctx = self.text.get(B, batch.get("text")).to(dt)
        ctx_projected = False
        if getattr(self.net, "use_crossattn_projection", False):
            ctx = self.net.crossattn_proj(ctx)
            ctx_projected = True
        prev_chunk = batch["prev_chunk"].to(dev, dt)
        fps = torch.full((B,), self.bb.fps, device=dev)
        if guidance_scale != 1.0:
            ctx_null = self.text.get(B, [""] * B).to(dt)
            if ctx_projected:
                ctx_null = self.net.crossattn_proj(ctx_null)

        ts = self._time_shift(torch.linspace(1.0, 0.0, nfe + 1)).to(dev, dt)
        acc_out: AccOutput | None = None
        contact_hidden = None
        for i in range(nfe):
            t_cur, t_next = ts[i], ts[i + 1]
            t_B_T = t_cur.expand(B, layout.t_total).clone()
            t_B_T[:, cond_T] = 0.0
            out = self.net(
                x_B_C_T_H_W=x, timesteps_B_T=t_B_T * 1000.0,
                crossattn_emb=ctx, crossattn_projected=ctx_projected,
                condition_video_input_mask_B_C_T_H_W=cond_mask,
                action=prev_chunk,
                acc_inputs=acc_inputs, layout=layout,
                fps=fps)
            v = out.velocity_B_C_T_H_W
            if guidance_scale != 1.0:
                out_u = self.net(
                    x_B_C_T_H_W=torch.where(cond_mask.bool(), x0_null, x),
                    timesteps_B_T=t_B_T * 1000.0,
                    crossattn_emb=ctx_null, crossattn_projected=ctx_projected,
                    condition_video_input_mask_B_C_T_H_W=cond_mask,
                    action=prev_chunk,
                    acc_inputs=acc_null, layout=layout,
                    fps=fps)
                v = out_u.velocity_B_C_T_H_W + guidance_scale * (
                    v - out_u.velocity_B_C_T_H_W)
            x = x + (t_next - t_cur) * v
            x = torch.where(cond_mask.bool(), x0, x)               # keep cond pinned
            acc_out = out.acc
            contact_hidden = out.contact_hidden_B_Tc_S_D

        event_logits = self.phantom_event_head(contact_hidden)
        log_sigma = self.phantom_sigma_head(contact_hidden)
        cpk = self.c_pack.unpack(x[:, :, layout.frame_slice(FrameGroup.CONTACT)].float())
        actions = self.a_pack.unpack(x[:, :, layout.frame_slice(FrameGroup.ACTION)].float())
        return PhantomPrediction(
            actions_B_H_A=actions, cpk=cpk, event_logits_B_Tc_E=event_logits,
            log_sigma_B_Tc_K=log_sigma,
            governor_sigma_B_Tc=SigmaHead.governor_sigma(log_sigma),
            acc=acc_out, x_final_B_C_T_H_W=x)
