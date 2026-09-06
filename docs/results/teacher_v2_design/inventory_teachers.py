"""Read-only CPU inventory; no model construction, CUDA use, or downloads."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path("/home/physicalai/phantom-icra-2027/phantom")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


result = {
    "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
    "repo": str(REPO),
    "method": "Read existing checkpoints by CPU mmap; hash complete local teacher files; model-archive metadata only. No model construction, CUDA use, checkpoint downloads, or remote mutations.",
    "local_checkpoints": [],
    "teacher_directories": [
        str(p.relative_to(REPO)) for p in sorted((REPO / "runs").glob("*teacher*"))
    ],
    "model_registry": (REPO / "tools/rig/MODELS.tsv").read_text(),
    "active_training_processes": [],
}
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = (proc / "cmdline").read_bytes().split(b"\0")
        args = [a.decode(errors="replace") for a in argv if a]
        if any(
            a.startswith("phantom.train.") or a.endswith("/train_teacher.py")
            for a in args
        ):
            result["active_training_processes"].append(
                {"pid": int(proc.name), "argv": args}
            )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
for path in sorted((REPO / "runs").rglob("*.pt")):
    if not path.parent.name.startswith("teacher_"):
        continue
    if path.is_symlink() or "/dl/" in str(path) or path.name == "text_embeddings.pt":
        continue
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    configs = payload.get("configs", {})
    model, train = configs.get("model", {}), configs.get("train", {})
    architecture = "student" if model.get("student") else "teacher"
    aliases = [
        str(p.relative_to(REPO))
        for p in path.parent.iterdir()
        if p.is_symlink() and p.resolve() == path.resolve()
    ]
    ema = payload.get("ema", payload.get("ema_state_dict"))
    row = {
        "path": str(path.relative_to(REPO)),
        "bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "sha256": digest(path) if architecture == "teacher" else None,
        "architecture": architecture,
        "step": payload.get("step"),
        "keys": list(payload),
        "ema_present": ema is not None,
        "ema_keys": list(ema)[:8] if isinstance(ema, dict) else None,
        "aliases": aliases,
        "model_config": model,
        "training_config": train,
        "hardware_shapes": configs.get("hardware_shapes"),
        "normalizer_keys": list(payload.get("norm_stats", {})),
    }
    result["local_checkpoints"].append(row)
    del payload
try:
    from huggingface_hub import HfApi

    api = HfApi()
    name = "armteam/phantom-checkpoints"
    revision = api.model_info(name).sha
    top = list(api.list_repo_tree(name, revision=revision))
    teacher_dirs = [
        p.path for p in top if p.path.startswith("teacher") and not hasattr(p, "size")
    ]
    archive = {
        "repository": name,
        "revision": revision,
        "teacher_directories": teacher_dirs,
        "files": [],
    }
    for directory in teacher_dirs:
        for p in api.list_repo_tree(
            name, path_in_repo=directory, revision=revision, recursive=True
        ):
            if not hasattr(p, "size") or not p.path.endswith(
                (".pt", ".log", ".json", ".yaml", ".csv")
            ):
                continue
            lfs = getattr(p, "lfs", None)
            archive["files"].append(
                {
                    "path": p.path,
                    "bytes": p.size,
                    "sha256": getattr(lfs, "sha256", None),
                }
            )
    result["archive_metadata"] = archive
except Exception as exc:
    result["archive_metadata_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, indent=2, allow_nan=False))
