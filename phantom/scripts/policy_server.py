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

from phantom.inference.policy import NULL_IMAGINATION_MODES
from phantom.inference.remote import (DEFAULT_PORT, PHANTOM_AUTHKEY,
                                      PolicyServer, RemotePolicy)

log = logging.getLogger("phantom.policy_server")


def probe(port: int) -> int:
    """Print `<ckpt> <idle|busy> <sha8|->` and exit 0 when a server answers
    (busy servers answer too — issue #8); exit 1 when nothing listens;
    exit 2 when something listens but does not complete the handshake
    (AMBIGUOUS — launchers must not fall back to a local load on this)."""
    try:
        info = RemotePolicy.probe(("127.0.0.1", port))
    except RemotePolicy.Unreachable as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception:
        return 1
    print(f"{info.get('ckpt')} {'busy' if info.get('busy') else 'idle'} "
          f"{info.get('ckpt_sha') or '-'}")
    return 0


def digest(path: str) -> str | None:
    """sha256 (first 12 hex) of the checkpoint FILE the server loaded — the
    identity a launch records instead of a mutable basename like BEST.pt
    (issue #9)."""
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return None


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
    ap.add_argument("--action-time-origin", choices=("inference_ready", "observation"),
                    default="inference_ready")
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--drop-video", action="store_true")
    # inference-latency levers (measured with phantom.scripts.bench_inference; parity
    # against the plain path is required before a warm rig server runs with them)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the DiT blocks (compile happens in the warm-up replan)")
    ap.add_argument("--compile-mode", default="default",
                    help="torch.compile mode: default | reduce-overhead (cudagraphs) | max-autotune")
    ap.add_argument("--flex", action="store_true",
                    help="FlexAttention self-attention with a block mask (needs the flex kernels)")
    # CAUSAL PROBE (not a latency lever): corrupt the imagined contact package
    # for every replan this server serves. Owned here, not per launch —
    # serve_bg.sh passes the MODELS.tsv extras column to THIS process, and
    # run_deploy adopts (and tags) whatever `info` reports.
    ap.add_argument("--null-imagination", choices=NULL_IMAGINATION_MODES,
                    default="none",
                    help="none | prev_cpk (zero ContactPackage as the ACC "
                         "intent channel on every replan) | contact_zero "
                         "(CONTACT frames cond-pinned to the zero package). "
                         "Same mechanics as tools/terminal_eval.py --null")
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
    sha = digest(args.ckpt)
    log.info("checkpoint digest sha256[:12]=%s (%s)", sha, args.ckpt)
    levers = {"compile": bool(args.compile), "compile_mode": args.compile_mode if args.compile else None,
              "flex": bool(args.flex), "fp8": False}
    log.info("inference levers: %s", levers)
    # startup banner: this one changes the PLANS, not the latency — every
    # replan this server serves is a probe arm until it is restarted
    null_imag = str(getattr(policy, "null_imagination", "none") or "none")
    if null_imag == "none":
        log.info("imagination null: none (intact model)")
    else:
        log.warning("IMAGINATION NULL: %s — every replan this server serves has "
                    "its imagined contact package corrupted (%s). Attached "
                    "launches adopt and tag null:%s.", null_imag,
                    "zero ContactPackage on the ACC intent channel"
                    if null_imag == "prev_cpk" else
                    "CONTACT frames cond-pinned to the zero package", null_imag)
    srv = PolicyServer(policy, ckpt=args.ckpt, ckpt_sha=sha, levers=levers)
    if not args.no_warmup:
        srv.warmup(hw, teacher=(args.system == "teacher"))
    try:
        srv.serve_forever(port=args.port)
    except KeyboardInterrupt:
        log.info("server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
