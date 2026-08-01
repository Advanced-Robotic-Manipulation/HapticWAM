"""Session-end offload: move episodes from local staging to the external
drive (configs/data_collect.yaml storage.external_drive).

Policy (requirement 5): episodes are recorded to a local staging directory
during the session (a USB drive cannot absorb the live recording throughput
safely), and MOVED to the drive when the session finishes — after a verified
offload nothing remains on the local disk (unless storage.keep_local).

Verification before any local delete: per-file size comparison (default) or
sha256 (storage.verify). A missing / full drive fails the offload LOUDLY and
leaves the local copy untouched — data is never at risk from an unplugged
disk.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class OffloadResult:
    ok: bool
    episodes: int = 0
    bytes_moved: int = 0
    seconds: float = 0.0
    error: str = ""
    moved: list = field(default_factory=list)   # destination paths (str)

    def describe(self) -> str:
        if not self.ok:
            return f"offload FAILED: {self.error}"
        return (f"offload OK: {self.episodes} episodes, "
                f"{self.bytes_moved / 1e9:.2f} GB in {self.seconds:.0f} s")


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_copy(src: Path, dst: Path, mode: str) -> str | None:
    """None if dst faithfully mirrors src, else a description of the mismatch."""
    src_files = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst) for p in dst.rglob("*") if p.is_file())
    if src_files != dst_files:
        return f"file list mismatch under {dst.name}"
    for rel in src_files:
        s, d = src / rel, dst / rel
        if s.stat().st_size != d.stat().st_size:
            return f"size mismatch: {rel}"
        if mode == "sha256" and _sha256(s) != _sha256(d):
            return f"sha256 mismatch: {rel}"
    return None


def offload_session(staging_dir: Path, drive_root: Path, *,
                    min_free_gb: float = 0.0, verify: str = "size",
                    keep_local: bool = False, require_separate_device: bool = False,
                    progress=None) -> OffloadResult:
    """Move every ep_* under staging_dir to drive_root/<staging_dir.name>/.

    progress: optional callback(done_episodes, total_episodes, current_name)
    — wired to the panel's progress bar."""
    t0 = time.time()
    staging_dir = Path(staging_dir)
    drive_root = Path(drive_root)
    episodes = sorted(p for p in staging_dir.glob("ep_*") if p.is_dir())
    if not episodes:
        return OffloadResult(ok=True, seconds=time.time() - t0)

    if not drive_root.exists():
        return OffloadResult(ok=False, error=(
            f"external drive not found at {drive_root} — plug it in (or fix "
            "storage.external_drive in configs/data_collect.yaml); episodes remain "
            f"in local staging {staging_dir}"))
    if require_separate_device:
        # an unmounted Linux mountpoint is a plain LOCAL directory — the
        # "offload" would land on the local disk and the local copy would be
        # deleted; refuse unless the target is really another filesystem
        try:
            if drive_root.stat().st_dev == staging_dir.stat().st_dev:
                return OffloadResult(ok=False, error=(
                    f"{drive_root} is on the SAME filesystem as the staging dir "
                    "— the external drive does not appear to be mounted "
                    "(storage.require_separate_device is on); episodes remain "
                    "in local staging"))
        except OSError as e:
            return OffloadResult(ok=False, error=f"cannot stat {drive_root}: {e}")
    total_bytes = sum(_dir_bytes(p) for p in episodes)
    try:
        free = shutil.disk_usage(drive_root).free
    except OSError as e:
        return OffloadResult(ok=False, error=f"cannot stat drive {drive_root}: {e}")
    need = total_bytes + int(min_free_gb * 1e9)
    if free < need:
        return OffloadResult(ok=False, error=(
            f"drive has {free / 1e9:.1f} GB free but the session needs "
            f"{total_bytes / 1e9:.1f} GB (+{min_free_gb:.0f} GB reserve) — free "
            "space on the drive; episodes remain in local staging"))

    dest_root = drive_root / staging_dir.name
    dest_root.mkdir(parents=True, exist_ok=True)
    result = OffloadResult(ok=True, episodes=len(episodes))
    for i, ep in enumerate(episodes):
        if progress is not None:
            progress(i, len(episodes), ep.name)
        dst = dest_root / ep.name
        if dst.exists():           # partial earlier offload — redo cleanly
            shutil.rmtree(dst)
        try:
            shutil.copytree(ep, dst)
        except OSError as e:
            return OffloadResult(ok=False, episodes=i, error=(
                f"copy failed at {ep.name}: {e} — local episodes untouched"))
        mismatch = _verify_copy(ep, dst, verify)
        if mismatch is not None:
            return OffloadResult(ok=False, episodes=i, error=(
                f"verification failed at {ep.name}: {mismatch} — local episodes "
                "untouched; check the drive"))
        result.bytes_moved += _dir_bytes(dst)
        result.moved.append(str(dst))
        if not keep_local:
            shutil.rmtree(ep)
    if progress is not None:
        progress(len(episodes), len(episodes), "")
    result.seconds = time.time() - t0
    log.info(result.describe())
    return result
