"""Resident policy server: load a checkpoint once, keep it warm, serve
replans to any number of consecutive `run_deploy --policy-server` launches.

Terminal A (once):
    .venv/bin/python -m phantom.scripts.policy_server \
        --ckpt runs/teacher_v5_batch0822/v5_6.pt --hardware configs/hardware.nuc.yaml

Terminal B (as many times as you like, starts in seconds):
    run_deploy ... --policy-server auto

`--probe` prints the running server's checkpoint and exits 0 (used by
PICK.sh to auto-detect), exits 1 if no server is listening.
"""

from __future__ import annotations

import argparse
import logging
import sys

from phantom.inference.remote import (DEFAULT_PORT, PHANTOM_AUTHKEY,
                                      PolicyServer, RemotePolicy)

log = logging.getLogger("phantom.policy_server")


def probe(port: int) -> int:
    try:
        pol = RemotePolicy(("127.0.0.1", port))
        print(pol.info["ckpt"])
        pol.close()
        return 0
    except Exception:
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint to load and hold")
    ap.add_argument("--hardware", default="configs/hardware.nuc.yaml")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--system", default="teacher")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="print the running server's ckpt and exit")
    # accepted for build_policy compatibility; per-launch values come from the
    # client's configure call
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--drop-video", action="store_true")
    ap.add_argument("--terminal-veto", action="store_true", default=True,
                    help="assert the ACC head exists at load (veto launches "
                         "must not discover its absence mid-session)")
    args = ap.parse_args()

    if args.probe:
        return probe(args.port)
    if not args.ckpt:
        ap.error("--ckpt is required (unless --probe)")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    from phantom.config.hardware import load_hardware
    from phantom.config.paths import load_paths
    from phantom.scripts.run_deploy import build_policy

    hw = load_hardware(args.hardware)
    paths = load_paths()
    # build_policy reads a handful of attrs argparse above does not define
    args.text = ""
    args.task = ""
    args.persistent_noise = True
    args.parity_fixes = False
    args.k_seeds = 1
    args.veto_p_close = 0.5
    log.info("loading %s ...", args.ckpt)
    policy = build_policy(args, hw, paths)
    srv = PolicyServer(policy, ckpt=args.ckpt)
    if not args.no_warmup:
        srv.warmup(hw, teacher=(args.system == "teacher"))
    try:
        srv.serve_forever(port=args.port)
    except KeyboardInterrupt:
        log.info("server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
