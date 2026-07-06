"""[Linux/5090] Collect one DAgger round of student rollouts (rig wears the
sensors; full streams recorded), then hand off to phantom.train.dagger_driver.

    python -m phantom.scripts.run_dagger_round --round 1 --ckpt <student_r0.pt> \
        --tasks fragile_grasp slippery_place --rollouts-per-task 50
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.dagger.rollout import collect_rollouts
from phantom.scripts.run_deploy import build_policy

log = logging.getLogger("run_dagger_round")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--rollouts-per-task", type=int, default=50)
    ap.add_argument("--max-replans", type=int, default=20)
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--drop-video", action="store_true")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    args.system = "student"

    hw = load_hardware(args.hardware)
    paths = load_paths()
    policy = build_policy(args, hw, paths)
    out_root = paths.episodes_root() / "dagger" / f"round_{args.round}" \
        / time.strftime("%Y%m%d")

    for task in args.tasks:
        collect_rollouts(hw, policy, out_root, task=task,
                         n_rollouts=args.rollouts_per_task,
                         dagger_round=args.round, max_replans=args.max_replans)
    print(f"\nRollouts recorded under {out_root}. Next:")
    print(f"  python -m phantom.train.dagger_driver --round {args.round} "
          f"--teacher-ckpt <teacher.pt> --demos <demos_root> --rollouts {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
