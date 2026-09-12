#!/usr/bin/env python3
"""Dedicated inference-only PHANTOM server for a simulator campaign.

Invoke with the live PHANTOM Python environment and an explicit repository,
checkpoint, dedicated port, and output directory. No hardware driver/runtime
or run_deploy module is imported. Existing rig servers are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--expected-sha256")
    p.add_argument("--hardware", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--system", choices=("auto", "teacher", "student"), default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--action-time-origin", choices=("inference_ready", "observation"),
                   default="inference_ready")
    p.add_argument("--nfe", type=int, default=5)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--k-seeds", type=int, default=1)
    p.add_argument("--task-text", default="waffles")
    p.add_argument(
        "--parity-fixes", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--persistent-noise", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--drop-video", action="store_true")
    p.add_argument(
        "--raw",
        action="store_true",
        help="Explicit last-step weight ablation; default EMA",
    )
    p.add_argument("--no-warmup", action="store_true")
    return p


def build_policy(args, metadata):
    """Match run_deploy's model construction without importing its runtime."""
    import numpy as np
    import torch

    from phantom.backbone.loader import merge_lora
    from phantom.config.hardware import load_hardware
    from phantom.config.model import PhantomModelConfig
    from phantom.config.paths import load_paths
    from phantom.data.schema import NormStats
    from phantom.inference.policy import PhantomPolicy
    from phantom.train.builder import build_model
    from phantom.train.common import load_phantom_checkpoint

    checkpoint = args.ckpt.resolve()
    digest = sha256_file(checkpoint)
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError("Checkpoint hash differs from campaign's pinned artifact")
    before = checkpoint.stat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while loading")
    saved = payload["configs"]["model"]
    mc = PhantomModelConfig.from_dict(saved)
    system = "student" if mc.student else "teacher"
    if args.system != "auto" and args.system != system:
        raise ValueError(
            f"Requested {args.system} but saved checkpoint architecture is {system}"
        )
    if not args.raw and not payload.get("ema"):
        raise ValueError("EMA required but absent; raw weights require explicit --raw")
    ns = payload.get("norm_stats")
    if not ns:
        raise ValueError("Checkpoint normalizers are required for campaign inference")
    for key, value in ns["std"].items():
        if not np.isfinite(value).all() or (np.asarray(value) <= 0).any():
            raise ValueError(f"Invalid normalizer standard deviations: {key}")
    hw_path = args.hardware or args.repo / "configs/hardware.nuc.yaml"
    hw, paths = load_hardware(hw_path), load_paths()
    paths.validate(require_cosmos=True)
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    metadata.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "checkpoint_size_bytes": before.st_size,
            "system": system,
            "saved_model_config": saved,
            "hardware_path": str(hw_path.resolve()),
            "hardware_sha256": sha256_file(hw_path),
            "hardware_config_hash": hw.config_hash(),
            "weights": "raw" if args.raw else "EMA",
            "dtype": str(dtype),
            "normalizers_sha256": hashlib.sha256(
                json.dumps(ns, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "backbone_checkpoint": str(paths.cosmos_checkpoint),
            "backbone_sha256": sha256_file(paths.cosmos_checkpoint),
            "text_cache": paths.cosmos_text_embedding_cache,
            "text_cache_sha256": sha256_file(paths.cosmos_text_embedding_cache)
            if paths.cosmos_text_embedding_cache
            else None,
            "wrench_baseline_rows": int(
                payload.get("configs", {})
                .get("train", {})
                .get("wrench_baseline_rows", 0)
            ),
            "inference_source_sha256": {
                name: sha256_file(args.repo / name)
                for name in (
                    "phantom/inference/policy.py",
                    "phantom/inference/remote.py",
                    "phantom/inference/action_timing.py",
                    "phantom/train/common.py",
                    "phantom/train/builder.py",
                    "phantom/config/model.py",
                    "phantom/model/rf.py",
                )
            },
        }
    )
    write_json(args.out / "server.json", metadata)
    pm = build_model(
        hw,
        paths,
        student=mc.student,
        mc=mc,
        tiny=False,
        load_base=True,
        device=args.device,
        dtype=dtype,
        inference=True,
    )
    load_phantom_checkpoint(
        checkpoint, pm.rf, hw=hw, load_ema=not args.raw, payload=payload
    )
    norm = NormStats(
        mean={k: np.asarray(v, np.float32) for k, v in ns["mean"].items()},
        std={k: np.asarray(v, np.float32) for k, v in ns["std"].items()},
    )
    merge_lora(pm.rf.net)
    policy = PhantomPolicy(
        pm,
        norm,
        nfe=args.nfe,
        guidance=args.guidance,
        k_seeds=args.k_seeds,
        parity_fixes=args.parity_fixes,
        persistent_noise=args.persistent_noise,
        drop_video=args.drop_video,
        task_text=args.task_text,
        close_p=0.5,
        action_time_origin=getattr(args, "action_time_origin", "inference_ready"),
    )
    policy.wrench_baseline_rows = metadata["wrench_baseline_rows"]
    cache = getattr(pm.rf.text, "_cache", {})
    if args.task_text and args.task_text not in cache:
        raise ValueError(
            f"Task text {args.task_text!r} is absent from loaded embedding cache"
        )
    metadata["loaded_task_cache_keys"] = sorted(cache)
    metadata["effective"] = {
        k: getattr(policy, k)
        for k in (
            "nfe",
            "guidance",
            "k_seeds",
            "parity_fixes",
            "persistent_noise",
            "drop_video",
            "task_text",
            "close_p",
            "action_time_origin",
        )
    }
    return policy, hw


def warmup(policy, hw):
    """One explicitly synthetic inference, then clear episode noise/CPK state."""
    import numpy as np

    from phantom.inference.policy import ObsSnapshot

    rng = np.random.default_rng(0)
    obs = ObsSnapshot(
        t=0.0,
        rgb=rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        wrist_window=rng.normal(size=(hw.wrist_ft.window_len, 6)).astype(np.float32),
        ur_state=rng.normal(size=hw.ur_state_dim).astype(np.float32),
        reactive=0.1,
    )
    if not policy.pm.mc.student:
        kd = hw.recording.keyframe_ds
        obs.gel = rng.integers(0, 255, (hw.n_fingers, 480, 640), dtype=np.uint8)
        obs.fields = rng.normal(
            size=(hw.n_fingers, kd.h, kd.w, hw.tactile.field_ch)
        ).astype(np.float32)
        obs.contact_state = rng.normal(
            size=(hw.n_fingers, hw.contact_state_dim)
        ).astype(np.float32)
    plan = policy.replan(obs, None, np.zeros(6))
    if not np.isfinite(plan.actions).all() or not np.isfinite(plan.sigma).all():
        raise RuntimeError("Synthetic warmup produced nonfinite actions or uncertainty")
    policy.reset_episode()
    return float(plan.latency_s)


def main():
    args = parser().parse_args()
    if not 1024 <= args.port <= 65535 or 7777 <= args.port <= 7783:
        raise ValueError("Choose a dedicated port outside reserved rig ports7777–7783")
    if args.nfe < 1 or args.k_seeds < 1 or not 0 <= args.guidance < float("inf"):
        raise ValueError("NFE/K must be positive; guidance finite and nonnegative")
    args.repo = args.repo.resolve()
    if not (args.repo / "phantom/inference/policy.py").is_file():
        raise ValueError("--repo must name the PHANTOM Python repository")
    args.ckpt = args.ckpt if args.ckpt.is_absolute() else args.repo / args.ckpt
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    from multiprocessing.connection import Listener

    from phantom.inference.remote import PHANTOM_AUTHKEY, PolicyServer

    metadata = {
        "schema_version": 1,
        "dedicated_simulation_server": True,
        "pid": os.getpid(),
        "port": args.port,
        "repo": str(args.repo),
        "status": "loading",
        "started_unix_s": time.time(),
        "hardware_connections": "none; only configuration/model modules imported",
    }
    ready = args.out / "ready.json"
    ready.unlink(missing_ok=True)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        # Reserve the port before allocating a model; never replace an owner.
        with Listener(("127.0.0.1", args.port), authkey=PHANTOM_AUTHKEY) as listener:
            write_json(args.out / "server.json", metadata)
            policy, hw = build_policy(args, metadata)
            server = PolicyServer(
                policy, str(args.ckpt.resolve()), ckpt_sha=metadata["checkpoint_sha256"]
            )
            metadata["status"] = "warming"
            write_json(args.out / "server.json", metadata)
            if not args.no_warmup:
                metadata["synthetic_warmup_latency_s"] = warmup(policy, hw)
                server.warmed = True
            forbidden = [
                name
                for name in sys.modules
                if name.startswith("phantom.drivers.real")
                or name in ("phantom.scripts.run_deploy", "phantom.deploy.runtime")
            ]
            if forbidden:
                raise RuntimeError(
                    f"Inference process unexpectedly imported hardware runtime modules: {forbidden}"
                )
            metadata.update(
                status="ready", warmed=server.warmed, ready_unix_s=time.time()
            )
            write_json(args.out / "server.json", metadata)
            write_json(ready, metadata)
            server.serve_forever(port=args.port, listener=listener)
    except KeyboardInterrupt:
        metadata["status"] = "stopped"
    except BaseException as error:
        metadata.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        ready.unlink(missing_ok=True)
        write_json(args.out / "server.json", metadata)


if __name__ == "__main__":
    main()
