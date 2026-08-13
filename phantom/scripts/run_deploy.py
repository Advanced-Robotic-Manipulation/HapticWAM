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
    pm = build_model(hw, paths, student=student, tiny=args.tiny,
                     load_base=not args.tiny, device=args.device, dtype=dtype,
                     inference=True)
    norm = NormStats.identity()
    if args.ckpt:
        payload = C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw,
                                            load_ema=args.ema)
        if payload.get("norm_stats"):
            ns = payload["norm_stats"]
            norm = NormStats(
                mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
                std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})
    if getattr(args, "compile", False):
        from phantom.backbone.loader import compile_blocks
        compile_blocks(pm.rf.net)
    return PhantomPolicy(pm, norm, nfe=args.nfe, drop_video=args.drop_video,
                         task_text=(args.text or args.task))


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
