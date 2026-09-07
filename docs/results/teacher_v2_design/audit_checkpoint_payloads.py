"""CPU-only teacher payload inspection. No model construction or GPU use."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
root = Path("/home/physicalai/phantom-icra-2027")
paths = [
    root / "phantom/runs/teacher_v5_ftA/teacher_001500.pt",
    root / "phantom/runs/teacher_v5_batch0822/v5_6.pt",
    root / "sim/waffles/checkpoints/teacher_v5_ftA/teacher_003000.pt",
]


def json_native(value):
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(x) for x in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


rows = []
for path in paths:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    ema = payload.get("ema", payload.get("ema_state_dict"))
    tensors = {k: v for k, v in ema.items() if isinstance(v, torch.Tensor)}
    norm = json_native(payload.get("norm_stats"))
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    rows.append(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": h.hexdigest(),
            "step": payload.get("step"),
            "configs": json_native(payload.get("configs")),
            "ema_present": ema is not None,
            "ema_tensor_count": len(tensors),
            "ema_elements": sum(v.numel() for v in tensors.values()),
            "ema_all_finite": all(
                bool(torch.isfinite(v).all()) for v in tensors.values()
            ),
            "normalizers": norm,
            "normalizers_canonical_json_sha256": hashlib.sha256(
                json.dumps(norm, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    del payload, ema, tensors
print(
    json.dumps(
        {
            "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": __doc__,
            "checkpoints": rows,
        },
        indent=2,
        allow_nan=False,
    )
)
