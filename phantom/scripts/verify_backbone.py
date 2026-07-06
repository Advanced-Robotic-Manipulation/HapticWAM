"""Golden-parity verification of the backbone integration (risk R6/R7).

Two checks:
 1. SUBCLASS PARITY (any machine with the weights): a PhantomDiT loaded with
    the released checkpoint, run through the PARENT class's forward (pure
    video path, zero-init phantom modules unused), must produce EXACTLY the
    same output as a freshly constructed repo-native
    ActionChunkConditionedMinimalV1LVGDiT loaded with the same weights.
    Guards subclassing + key remapping + transcribed kwargs.
 2. SHIM PARITY (Linux with real transformer_engine installed): rerun with
    the real TE — if outputs match the shimmed run's saved reference, the
    RMSNorm/RoPE shims are numerically faithful. Until this has passed on the
    training box, treat shimmed-machine full-checkpoint numbers as smoke-only.

    python -m phantom.scripts.verify_backbone [--save-ref out.pt] [--check-ref out.pt]
"""

from __future__ import annotations

import argparse
import logging

import torch

from phantom.backbone import compat
from phantom.backbone import loader as bl
from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import load_hardware
from phantom.config.model import PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.model.sequence import SequenceLayout

log = logging.getLogger("verify_backbone")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-ref", default="", help="save outputs for cross-machine shim parity")
    ap.add_argument("--check-ref", default="", help="compare against saved reference")
    args = ap.parse_args(argv)

    hw = load_hardware()
    paths = load_paths()
    paths.validate(require_cosmos=True)
    bb = BackboneConfig()
    mc = PhantomModelConfig()
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    log.info("TE shimmed: %s", compat.is_shimmed() or "not yet (installed on import)")

    layout = SequenceLayout.build(bb, mc, hw, student=False)
    net = bl.build_phantom_net(bb, mc, hw, layout, paths, device=args.device, dtype=dtype)
    bl.load_base_weights(net, paths.cosmos_checkpoint, verify=bb)
    net.eval()

    from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (  # noqa: E501
        ActionChunkConditionedMinimalV1LVGDiT,
    )
    ref = ActionChunkConditionedMinimalV1LVGDiT(
        max_img_h=bb.res_h, max_img_w=bb.res_w, max_frames=bb.t_video,
        in_channels=bb.in_channels, out_channels=bb.out_channels,
        patch_spatial=bb.patch_spatial, patch_temporal=bb.patch_temporal,
        model_channels=bb.model_channels, num_blocks=bb.num_blocks,
        num_heads=bb.num_heads, atten_backend="torch",
        crossattn_emb_channels=bb.crossattn_emb_channels,
        use_crossattn_projection=bb.use_crossattn_projection,
        crossattn_proj_in_channels=bb.crossattn_proj_in_channels,
        pos_emb_cls=bb.pos_emb_cls, pos_emb_learnable=bb.pos_emb_learnable,
        use_adaln_lora=bb.use_adaln_lora, adaln_lora_dim=bb.adaln_lora_dim,
        rope_h_extrapolation_ratio=bb.rope_h_extrapolation_ratio,
        rope_w_extrapolation_ratio=bb.rope_w_extrapolation_ratio,
        rope_t_extrapolation_ratio=bb.rope_t_extrapolation_ratio,
        use_wan_fp32_strategy=bb.use_wan_fp32_strategy,
        timestep_scale=bb.timestep_scale, action_dim=bb.action_dim,
        temporal_compression_ratio=bb.actions_per_latent_frame,
    ).to(args.device, dtype)
    bl.load_base_weights(ref, paths.cosmos_checkpoint)
    ref.eval()

    # fixed inputs: the standard 4-latent-frame video path
    g = torch.Generator().manual_seed(0)
    B, T = 1, bb.t_video
    x = torch.randn(B, bb.lat_ch, T, bb.lat_h, bb.lat_w, generator=g).to(args.device, dtype)
    tt = torch.full((B, T), 500.0).to(args.device, dtype)
    text = torch.randn(B, bb.text_emb_seq_len, bb.text_emb_dim, generator=g) \
        .to(args.device, dtype)
    cond = torch.zeros(B, 1, T, bb.lat_h, bb.lat_w).to(args.device, dtype)
    cond[:, :, :1] = 1.0
    act = torch.randn(B, bb.num_action_per_chunk, bb.action_dim, generator=g) \
        .to(args.device, dtype)
    pad = torch.zeros(B, 1, bb.lat_h, bb.lat_w).to(args.device, dtype)
    fps = torch.full((B,), bb.fps, device=args.device)

    kw = dict(timesteps_B_T=tt, crossattn_emb=text,
              condition_video_input_mask_B_C_T_H_W=cond, action=act,
              fps=fps, padding_mask=pad)
    with torch.no_grad():
        out_ref = ref(x_B_C_T_H_W=x, **kw)
        # parent-class forward on the PhantomDiT instance (bypasses extensions)
        out_ours = ActionChunkConditionedMinimalV1LVGDiT.forward(
            net, x_B_C_T_H_W=x, **kw)

    diff = (out_ref.float() - out_ours.float()).abs().max().item()
    log.info("subclass parity: max |diff| = %.3e", diff)
    assert diff < 1e-4, f"subclass parity FAILED (max diff {diff})"
    print(f"SUBCLASS PARITY OK (max diff {diff:.2e}); TE shim active: {compat.is_shimmed()}")

    if args.save_ref:
        torch.save({"out": out_ref.float().cpu(), "shimmed": compat.is_shimmed()},
                   args.save_ref)
        print(f"reference saved to {args.save_ref} — rerun with --check-ref on the "
              f"other machine (real TE) to verify shim parity")
    if args.check_ref:
        ref_payload = torch.load(args.check_ref, map_location="cpu", weights_only=True)
        shim_diff = (ref_payload["out"] - out_ref.float().cpu()).abs().max().item()
        print(f"SHIM PARITY: max |diff| vs reference = {shim_diff:.3e} "
              f"({'OK' if shim_diff < 5e-2 else 'FAILED — do not trust shimmed numbers'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
