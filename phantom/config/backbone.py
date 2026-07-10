"""Backbone configuration: the frozen Cosmos-Predict2.5-2B robot/action-cond
net kwargs, transcribed from the repo's LazyConfig experiment chain
(action/configs/action_conditioned/net.py + experiment/exp_2B_action_conditioned_
rectify_flow.py), plus latent-geometry constants and a `tiny()` preset for
CPU smoke tests.

`verify_against_state_dict` guards against transcription drift (risk R7).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class BackboneConfig:
    # --- DiT kwargs (transcribed; do not change for the real checkpoint) ---
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    in_channels: int = 16          # latent channels (net adds +1 cond mask, +1 padding mask)
    out_channels: int = 16
    adaln_lora_dim: int = 256
    use_adaln_lora: bool = True
    crossattn_proj_in_channels: int = 100352
    crossattn_emb_channels: int = 1024
    use_crossattn_projection: bool = True
    pos_emb_cls: str = "rope3d"
    pos_emb_learnable: bool = True
    rope_h_extrapolation_ratio: float = 3.0
    rope_w_extrapolation_ratio: float = 3.0
    rope_t_extrapolation_ratio: float = 1.0
    use_wan_fp32_strategy: bool = True
    timestep_scale: float = 0.001
    atten_backend: str = "torch"   # forced: SDPA (weight-free; required by the ACC bias hook)
    # action conditioning (ActionChunkConditionedMinimalV1LVGDiT)
    action_dim: int = 7
    num_action_per_chunk: int = 12  # 12 actions over 13 pixel frames -> 4 per latent frame

    # --- video geometry (robot/action-cond post-train operating point) ---
    frames_pix: int = 13           # pixel frames per window
    res_h: int = 256
    res_w: int = 320
    fps: float = 4.0
    # position-table BOUNDS baked into the released checkpoint (NOT the
    # operating resolution): the DiT sizes its pos tables from
    # max_img_* // patch_spatial and saves them in the state dict
    # (pos_embedder.seq = arange(max(len_h, len_w, len_t)) = 128 for the
    # robot/action-cond .pt). Must match the checkpoint to load; only needs
    # to be >= the ACTUAL patched-latent extent (res/16 = 16x20 here).
    # First real-weights load on the 5090 (2026-07-10) caught this: we
    # passed res_w=320 -> table 160 vs checkpoint 128.
    max_img_h: int = 256
    max_img_w: int = 256

    # --- VAE latent constants (Wan2.1) ---
    lat_ch: int = 16
    spatial_comp: int = 8
    temporal_comp: int = 4

    # --- rectified-flow training constants (transcribed) ---
    rf_shift: float = 5.0
    rf_logit_normal_mean: float = 0.0
    rf_logit_normal_std: float = 1.0

    # --- text conditioning ---
    text_emb_seq_len: int = 512
    text_emb_dim: int = 100352

    # ---------- derived ----------
    @property
    def t_video(self) -> int:
        """Latent video frames: 1 + (frames_pix - 1) / temporal_comp."""
        assert (self.frames_pix - 1) % self.temporal_comp == 0
        return 1 + (self.frames_pix - 1) // self.temporal_comp

    @property
    def lat_h(self) -> int:
        assert self.res_h % self.spatial_comp == 0
        return self.res_h // self.spatial_comp

    @property
    def lat_w(self) -> int:
        assert self.res_w % self.spatial_comp == 0
        return self.res_w // self.spatial_comp

    @property
    def tokens_per_frame(self) -> int:
        assert self.lat_h % self.patch_spatial == 0 and self.lat_w % self.patch_spatial == 0
        return (self.lat_h // self.patch_spatial) * (self.lat_w // self.patch_spatial)

    @property
    def actions_per_latent_frame(self) -> int:
        assert self.num_action_per_chunk % (self.t_video - 1) == 0
        return self.num_action_per_chunk // (self.t_video - 1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny(cls) -> "BackboneConfig":
        """CPU-testable preset: same structure, small dims. NOT checkpoint-compatible."""
        return cls(
            model_channels=64,
            num_blocks=2,
            num_heads=2,
            adaln_lora_dim=16,
            crossattn_proj_in_channels=128,
            crossattn_emb_channels=64,
            frames_pix=13,
            res_h=64,
            res_w=80,
            text_emb_seq_len=8,
            text_emb_dim=128,
        )

    # ---------- checkpoint guard ----------
    def verify_against_state_dict(self, sd: dict) -> None:
        """Assert the transcribed kwargs agree with a raw checkpoint state dict
        (keys prefixed 'net.'). Raises AssertionError with the offending key."""
        def shape(key: str) -> tuple:
            return tuple(sd[key].shape)

        patch_in = (self.in_channels + 2) * self.patch_spatial ** 2 * self.patch_temporal
        assert shape("net.x_embedder.proj.1.weight") == (self.model_channels, patch_in), \
            f"x_embedder mismatch: {shape('net.x_embedder.proj.1.weight')} vs ({self.model_channels}, {patch_in})"
        act_in = self.action_dim * self.actions_per_latent_frame
        assert shape("net.action_embedder_B_D.fc1.weight")[1] == act_in, \
            f"action embedder in-dim {shape('net.action_embedder_B_D.fc1.weight')[1]} != {act_in}"
        assert shape("net.crossattn_proj.0.weight") == (
            self.crossattn_emb_channels, self.crossattn_proj_in_channels), "crossattn_proj mismatch"
        n_blocks = len({k.split(".")[2] for k in sd if k.startswith("net.blocks.")})
        assert n_blocks == self.num_blocks, f"num_blocks {n_blocks} != {self.num_blocks}"
        head_dim = self.model_channels // self.num_heads
        assert shape("net.blocks.0.self_attn.q_norm.weight") == (head_dim,), \
            f"head_dim mismatch: {shape('net.blocks.0.self_attn.q_norm.weight')} vs ({head_dim},)"
