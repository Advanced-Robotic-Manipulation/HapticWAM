"""[Linux/5090] Deployment entry point: the §6e receding-horizon loop on the
real rig (or fully mocked for a dry run on any machine).

    python -m phantom.scripts.run_deploy --system teacher --ckpt <teacher.pt> \
        --task fragile_grasp [--episodes 3] [--max-replans 20]
    python -m phantom.scripts.run_deploy --system student --ckpt <student.pt> ...
    # dry run without hardware or checkpoint:
    python -m phantom.scripts.run_deploy --system teacher --tiny --task smoke
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data.schema import NormStats
from phantom.deploy.planner import SYSTEM_MODES
from phantom.deploy.runtime import DeploymentRuntime
from phantom.inference.policy import PhantomPolicy
from phantom.train import common as C
from phantom.train.builder import build_model

log = logging.getLogger("run_deploy")


def build_policy(args, hw, paths) -> PhantomPolicy:
    student = args.system != "teacher"
    dtype = torch.bfloat16 if (args.device == "cuda" and not args.tiny) else torch.float32
    # build with the checkpoint's OWN model config (rope time mode, acc
    # feedback mode, ...) — building code defaults silently changes the
    # deployed geometry vs what was trained (v4 audit 2026-08-14)
    mc = payload = None
    if args.ckpt:
        payload = torch.load(str(Path(args.ckpt)), map_location="cpu",
                             weights_only=False)
        saved_mc = payload.get("configs", {}).get("model")
        if isinstance(saved_mc, dict):
            from phantom.config.model import PhantomModelConfig
            mc = PhantomModelConfig.from_dict(saved_mc)
            log.info("model config from checkpoint: rope=%s cond_dropout=%.2f "
                     "acc=%s", mc.rope_time_mode, mc.cond_dropout_p,
                     mc.acc.self_anticipation)
    pm = build_model(hw, paths, student=student, tiny=args.tiny, mc=mc,
                     load_base=not args.tiny, device=args.device, dtype=dtype,
                     inference=True)
    norm = NormStats.identity()
    if args.ckpt:
        payload = C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw,
                                            load_ema=args.ema, payload=payload)
        if payload.get("norm_stats"):
            ns = payload["norm_stats"]
            norm = NormStats(
                mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
                std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})
    from phantom.backbone import loader as bl
    if args.ckpt and not args.tiny:
        # exact: fold LoRA deltas into the base weights (removes the adapter
        # matmuls from every hot linear)
        bl.merge_lora(pm.rf.net)
    if getattr(args, "fp8", False):
        # bench-only escape hatch: torchao 0.18 rowwise FP8 currently produces
        # NaN actions on this stack (measured 2026-08-13) — never expose it as
        # a deploy flag until the parity check in bench_inference passes
        bl.quantize_fp8(pm.rf.net)
    if getattr(args, "flex", False):
        bl.enable_flex_attention(pm.rf.net)
    if getattr(args, "compile", False):
        bl.compile_blocks(pm.rf.net, mode=getattr(args, "compile_mode", "default")
                          or "default")
    return PhantomPolicy(pm, norm, nfe=args.nfe, drop_video=args.drop_video,
                         task_text=(args.text or args.task),
                         persistent_noise=getattr(args, "persistent_noise", False))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEM_MODES, required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--task", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-replans", type=int, default=20)
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--drop-video", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the DiT blocks (adds ~1-2 min warmup)")
    ap.add_argument("--compile-mode", default="default",
                    help="torch.compile mode (e.g. reduce-overhead for cudagraphs)")
    ap.add_argument("--persistent-noise", action="store_true",
                    help="hold the sampling noise fixed within an episode — "
                         "fresh noise per replan re-rolls the plan direction "
                         "(rig: consec-replan cosine 0.16-0.35)")
    ap.add_argument("--flex", action="store_true",
                    help="FlexAttention self-attn (block-sparse structural mask; "
                         "action parity vs SDPA verified to ~6e-4)")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    out_root = Path(args.out) if args.out else \
        paths.episodes_root() / "deploy" / time.strftime("%Y%m%d")

    policy = build_policy(args, hw, paths)
    if getattr(args, "compile", False):
        # compile warms up on the first forward — do it OUTSIDE the episode
        # (it ate 1-2 min inside the first episode on the rig)
        try:
            from phantom.scripts.bench_inference import fake_obs
            log.info("compile warmup replan...")
            policy.replan(fake_obs(hw, teacher=(args.system == "teacher")),
                          None, np.zeros(6))
            policy.reset_episode()
        except Exception:
            log.exception("compile warmup failed (continuing)")
    with DeploymentRuntime(hw, policy, mode=args.system, out_root=out_root) as rt:
        for i in range(args.episodes):
            if hw.mode.drivers == "real":
                input(f"episode {i + 1}/{args.episodes}: reset scene, Enter to start...")
                # RealSense auto-exposure needs seconds after the stream opens
                # to settle; the first plans otherwise condition on dark frames
                # (collection never hit this — its camera runs the whole
                # session). Field-verified 2026-08-11: unsettled AE made the
                # policy flail confidently and slam the table.
                log.info("camera AE settle...")
                time.sleep(5.0)
            res = rt.run_episode(task=args.task, text=args.text,
                                 max_replans=args.max_replans,
                                 policy_name=f"{args.system}")
            log.info("episode %d: %s (replans=%d stop=%s safety_events=%d)",
                     i, res.episode_path, res.n_replans, res.stopped_reason,
                     res.safety_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
