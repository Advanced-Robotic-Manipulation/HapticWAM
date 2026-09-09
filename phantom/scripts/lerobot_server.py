"""Resident policy server for an external LeRobot baseline (target: pi05).

Same wire, same ownership rules and the same `--probe` contract as
`phantom.scripts.policy_server` — it IS `PolicyServer`, only the policy object
differs — so `run_deploy --policy-server`, `PICK.sh`'s probe and `SERVE.sh`'s
one-port-per-slot layout all work unchanged.

Terminal A (once):
    .venv/bin/python -m phantom.scripts.lerobot_server \
        --ckpt <pi05 checkpoint dir> --port 7790 \
        --hardware configs/hardware.nuc.yaml --task "pick up the egg"

Terminal B (per episode batch):
    run_deploy ... --policy-server 127.0.0.1:7790 --system student

lerobot is imported ONLY inside `load_lerobot_policy`, so this module is
importable (and testable) on a machine without it.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from phantom.inference.remote import PolicyServer
from phantom.scripts.policy_server import digest, probe

log = logging.getLogger("phantom.lerobot_server")

DEFAULT_PORT = 7790
# checkpoint files that define a LeRobot artifact's identity
CKPT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "config.json",
              "train_config.json")


def dir_digest(path: str) -> str | None:
    """sha256[:12] of a checkpoint DIRECTORY — the identity `run_deploy`
    records instead of a mutable path (issue #9, extended to LeRobot's
    multi-file checkpoints). Hashes each weight/config file's relative path,
    size and content, in sorted order, so a swapped weight file and a renamed
    directory are both visible."""
    p = Path(path)
    if p.is_file():
        return digest(str(p))
    if not p.is_dir():
        return None
    files = sorted({f for g in CKPT_GLOBS for f in p.rglob(g) if f.is_file()},
                   key=lambda f: str(f.relative_to(p)))
    if not files:
        log.warning("no checkpoint files under %s — ckpt_sha will be null", p)
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(p)).encode())
        h.update(str(f.stat().st_size).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()[:12]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="serve a LeRobot policy over the phantom policy protocol")
    ap.add_argument("--ckpt", help="LeRobot checkpoint directory (or HF repo id)")
    ap.add_argument("--policy-type", default="pi05",
                    help="LeRobot policy type name (default pi05)")
    ap.add_argument("--hardware", default="configs/hardware.nuc.yaml")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default="",
                    help="episode instruction sent as the `task` batch key; "
                         "run_deploy's --text/--task overrides it at attach")
    ap.add_argument("--action-space", default="delta", choices=("delta", "absolute"),
                    help="delta: the chunk IS [dx..drz, grip] per 10 Hz step "
                         "(default). absolute: the six pose channels are base-"
                         "frame poses, differenced with derived.pose_delta")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--image-key", default="observation.images.scene")
    ap.add_argument("--state-key", default="observation.state")
    ap.add_argument("--pad-mode", default="hold", choices=("hold", "repeat"),
                    help="short chunk padding: hold (zero pose-delta + last "
                         "aperture, default) or repeat the last row")
    ap.add_argument("--no-task-key", action="store_true",
                    help="omit the `task` key for a policy that is not "
                         "language-conditioned")
    ap.add_argument("--no-processors", action="store_true",
                    help="do NOT load the checkpoint's LeRobot pre/post "
                         "processor pipelines. They carry the dataset "
                         "normalisation stats, pi05's state->prompt "
                         "discretisation and the tokeniser, so skipping them "
                         "feeds the model garbage — for a policy that ships "
                         "no pipeline only")
    ap.add_argument("--skip-contract-check", action="store_true",
                    help="do not verify the checkpoint's camera key and action "
                         "dimension against the deploy contract")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="print the running server's ckpt and exit "
                         "(same contract as policy_server --probe)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.probe:
        return probe(args.port)
    if not args.ckpt:
        ap.error("--ckpt is required (unless --probe)")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    from phantom.config.hardware import load_hardware
    from phantom.inference.lerobot_policy import (LeRobotPolicy,
                                                  check_checkpoint_contract,
                                                  load_lerobot_policy)

    hw = load_hardware(args.hardware)
    inner, pre, post = load_lerobot_policy(
        args.ckpt, policy_type=args.policy_type, device=args.device,
        with_processors=not args.no_processors)
    if not args.skip_contract_check:
        check_checkpoint_contract(inner, image_key=args.image_key,
                                  action_dim=hw.control.action_dim,
                                  chunk_horizon=hw.control.chunk_horizon)
    policy = LeRobotPolicy(
        inner, hw, task_text=args.task, action_space=args.action_space,
        image_size=args.image_size, image_key=args.image_key,
        state_key=args.state_key, pad_mode=args.pad_mode, device=args.device,
        use_task_key=not args.no_task_key,
        preprocessor=pre, postprocessor=post)
    sha = dir_digest(args.ckpt)
    log.info("checkpoint digest sha256[:12]=%s (%s)", sha, args.ckpt)
    srv = PolicyServer(policy, ckpt=str(args.ckpt), ckpt_sha=sha)
    if not args.no_warmup:
        # `teacher=False`: this policy reads no tactile stream, so the
        # synthetic snapshot must be the student-shaped one.
        srv.warmup(hw, teacher=False)
    log.warning("LeRobot baseline: --terminal-veto / --parity-fixes / "
                "--k-seeds>1 / --drop-video are REFUSED (see "
                "docs/pi05_baseline.md); sigma is 0 so the speed governor "
                "runs at full scale for every chunk.")
    try:
        srv.serve_forever(port=args.port)
    except KeyboardInterrupt:
        log.info("server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
