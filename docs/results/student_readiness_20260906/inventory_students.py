"""Inspect existing student payloads on CPU and print JSON; no remote writes."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/physicalai/phantom-icra-2027/phantom")
PATHS = (
    "runs/student_ftA_r1/student_001200.pt",
    "runs/student_v5_6/student_001200.pt",
    "runs/student_ftA/student_001200.pt",
    "runs/control_ftA/teacher_001200.pt",
    "runs/control_v5_6/teacher_001200.pt",
    "runs/teacher_v5_ftA/teacher_001500.pt",
)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


result = {
    "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
    "repo": str(REPO),
    "checkpoints": [],
}
torch.set_num_threads(1)
for name in PATHS:
    path = REPO / name
    p = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    norm = clean(p["norm_stats"])
    canonical = json.dumps(norm, sort_keys=True, separators=(",", ":"), allow_nan=False)
    norm_shapes = {
        kind: {key: list(np.asarray(value).shape) for key, value in group.items()}
        for kind, group in norm.items()
    }
    norm_finite = all(
        np.isfinite(np.asarray(v)).all()
        for group in norm.values()
        for v in group.values()
    )
    std_positive = all((np.asarray(v) > 0).all() for v in norm["std"].values())
    ema = p["ema"]
    result["checkpoints"].append(
        {
            "path": name,
            "absolute_path": str(path),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "step": p["step"],
            "architecture": "student"
            if p["configs"]["model"].get("student")
            else "teacher",
            "model_config": clean(p["configs"]["model"]),
            "training_config": clean(p["configs"]["train"]),
            "normalizers_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "normalizers_finite": bool(norm_finite),
            "normalizer_std_positive": bool(std_positive),
            "normalizer_shapes": norm_shapes,
            "normalizers": norm,
            "ema_tensor_count": len(ema),
            "ema_finite": all(
                bool(torch.isfinite(t).all())
                for t in ema.values()
                if isinstance(t, torch.Tensor)
            ),
            "ema_sigma_keys": [k for k in ema if "sigma" in k],
        }
    )
    del p
result["sources"] = {
    path: {"sha256": digest(REPO / path)}
    for path in (
        "tools/rig/MODELS.tsv",
        "tools/rig/GO_ANY.sh",
        "tools/rig/PICK.sh",
        "phantom/scripts/run_deploy.py",
    )
}
print(json.dumps(result, indent=2, allow_nan=False))
