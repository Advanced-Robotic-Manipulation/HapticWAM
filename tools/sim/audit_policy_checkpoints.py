#!/usr/bin/env python3
"""CPU-only provenance audit of trusted PHANTOM campaign checkpoints.

Reads checkpoint files without constructing a model or connecting any hardware.
Checkpoint loading uses pickle because these project-owned training artifacts
contain NumPy optimizer metadata. Run only on trusted checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

DEFAULT_CHECKPOINTS = {
    "ftA": "runs/teacher_v5_ftA/BEST.pt",
    "student_ftA_r1": "runs/student_ftA_r1/student_001200.pt",
    "control_ftA": "runs/control_ftA/teacher_001200.pt",
    "student_ftA": "runs/student_ftA/student_001200.pt",
}
SOURCE_FILES = (
    "phantom/inference/policy.py",
    "phantom/inference/remote.py",
    "phantom/train/common.py",
    "phantom/train/builder.py",
    "phantom/deploy/planner.py",
    "phantom/deploy/executor.py",
    "phantom/scripts/run_deploy.py",
    "phantom/scripts/policy_server.py",
    "phantom/config/model.py",
    "phantom/model/rf.py",
    "phantom/backbone/text_embedding.py",
    "configs/hardware.nuc.yaml",
    "configs/start_poses.yaml",
    "tools/rig/MODELS.tsv",
    "tools/rig/PICK.sh",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def tensor_inventory(tensors):
    import torch

    dtypes = Counter()
    nonfinite = []
    for name, value in tensors.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Expected checkpoint tensor at {name}")
        dtypes[str(value.dtype)] += value.numel()
        if value.is_floating_point() and not torch.isfinite(value).all():
            nonfinite.append(name)
    return {
        "tensor_count": len(tensors),
        "scalar_count": sum(dtypes.values()),
        "scalars_by_dtype": dict(dtypes),
        "nonfinite_tensors": nonfinite,
    }


def audit_checkpoint(path, label):
    import numpy as np
    import torch

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    configs = payload.get("configs", {})
    model = configs.get("model", {})
    raw = {**payload.get("lora", {}), **payload.get("phantom_modules", {})}
    ema = payload.get("ema", {})
    normalizers = payload.get("norm_stats", {})
    issues = []
    for name, value in normalizers.get("std", {}).items():
        a = np.asarray(value)
        if not np.isfinite(a).all() or (a <= 0).any():
            issues.append(f"normalizer std {name}: nonpositive/nonfinite")
    if not normalizers:
        issues.append("normalizer metadata absent")
    heads = {}
    for head in ("phantom_sigma_head", "phantom_event_head", "phantom_acc"):
        selected = {k: v for k, v in raw.items() if head in k}
        heads[head] = {
            "tensor_count": len(selected),
            "scalar_count": sum(v.numel() for v in selected.values()),
        }
    result = {
        "label": label,
        "requested_path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "format_version": payload.get("format_version"),
        "step": payload.get("step"),
        "base_checkpoint_training_path": payload.get("base_ckpt_path"),
        "architecture": "student" if model.get("student") is True else "teacher",
        "filename_is_architecture_authority": False,
        "configs": configs,
        "normalizers": normalizers,
        "normalizer_sha256": json_hash(normalizers),
        "raw_weight_inventory": tensor_inventory(raw),
        "ema_weight_inventory": tensor_inventory(ema),
        "ema_available": bool(ema),
        "default_inference_weights": "EMA" if ema else "raw",
        "head_inventory": heads,
        "issues": issues,
        "checkpoint_default_nfe": model.get("nfe"),
        "wrench_baseline_rows": int(
            configs.get("train", {}).get("wrench_baseline_rows", 0)
        ),
        "task_text_provenance": configs.get("text_conditioning", {}),
    }
    del payload, raw, ema
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="LABEL=PATH"
    )
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoints = (
        dict(value.split("=", 1) for value in args.checkpoint) or DEFAULT_CHECKPOINTS
    )
    rows = [audit_checkpoint(repo / path, label) for label, path in checkpoints.items()]
    report = {
        "schema_version": 1,
        "repo": str(repo),
        "repo_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "method": "CPU mmap checkpoint metadata/tensor finite scan and complete-file SHA256; no model construction, CUDA allocation, or hardware connection",
        "checkpoints": rows,
        "source_sha256": {
            name: sha256_file(repo / name)
            for name in SOURCE_FILES
            if (repo / name).exists()
        },
        "all_normalizers_identical": len({row["normalizer_sha256"] for row in rows})
        == 1,
        "runtime_options_not_recoverable_from_weights": [
            "guidance",
            "parity_fixes",
            "persistent_noise",
            "k_seeds",
            "task_text",
        ],
    }
    if args.text_cache:
        import torch

        cache = torch.load(
            args.text_cache, map_location="cpu", weights_only=True, mmap=True
        )
        report["text_cache"] = {
            "path": str(args.text_cache),
            "sha256": sha256_file(args.text_cache),
            "keys": sorted(k for k in cache if k != "__meta__"),
            "metadata": cache.get("__meta__", {}),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "checkpoints": [
                    {
                        k: row[k]
                        for k in (
                            "label",
                            "architecture",
                            "sha256",
                            "ema_available",
                            "checkpoint_default_nfe",
                            "issues",
                        )
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
