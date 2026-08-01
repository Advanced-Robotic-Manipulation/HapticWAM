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
import os
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


def _verify_copy(src: Path, dst: Path, mode: str, *,
                 allow_extra_dst: bool = False) -> str | None:
    """None if dst faithfully mirrors src, else a description of the mismatch.

    allow_extra_dst: accept a dst that is a SUPERSET of src (used to recognise
    an already-complete drive copy when the local side is truncated remains of
    an interrupted move)."""
    src_files = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst) for p in dst.rglob("*") if p.is_file())
    if allow_extra_dst:
        if not set(src_files) <= set(dst_files):
            return f"dst is missing files under {dst.name}"
    elif src_files != dst_files:
        return f"file list mismatch under {dst.name}"
    for rel in src_files:
        s, d = src / rel, dst / rel
        if s.stat().st_size != d.stat().st_size:
            return f"size mismatch: {rel}"
        if mode == "sha256" and _sha256(s) != _sha256(d):
            return f"sha256 mismatch: {rel}"
    return None


def _flush_to_disk(root: Path) -> None:
    """Best-effort fsync of every file under root, then a global sync barrier.

    The drive is NTFS over FUSE where directory fsync is unreliable, so after
    the per-file fsyncs a single os.sync() guarantees the copy is on the
    platter before the local original may be deleted (a 'verified' copy that
    exists only in the page cache dies with a power loss / yanked cable)."""
    for p in root.rglob("*"):
        if p.is_file():
            try:
                with open(p, "rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                pass
    os.sync()


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
        part = dest_root / (ep.name + ".part")
        if part.exists():          # stale interrupted copy — .part is never authoritative
            shutil.rmtree(part)
        if dst.exists():
            # NEVER blind-delete an existing drive copy: it may be the COMPLETE
            # episode from an earlier move whose local delete was interrupted
            # (in which case the local side is truncated remains).
            if _verify_copy(ep, dst, verify, allow_extra_dst=True) is None:
                # drive copy is a faithful superset — it is authoritative;
                # finish the interrupted move by clearing the local remains
                log.info("offload: %s already complete on the drive — "
                         "finishing the interrupted move", ep.name)
                result.bytes_moved += _dir_bytes(dst)
                result.moved.append(str(dst))
                if not keep_local:
                    shutil.rmtree(ep)
                continue
            aside = dest_root / f"{ep.name}.mismatch-{int(time.time())}"
            dst.rename(aside)
            log.error("offload: existing drive copy of %s does not match the "
                      "local episode — set aside as %s for manual review",
                      ep.name, aside.name)
        try:
            shutil.copytree(ep, part)
        except OSError as e:
            shutil.rmtree(part, ignore_errors=True)
            return OffloadResult(ok=False, episodes=i, error=(
                f"copy failed at {ep.name}: {e} — this and later episodes remain "
                f"in local staging ({i} already moved)"))
        _flush_to_disk(part)
        mismatch = _verify_copy(ep, part, verify)
        if mismatch is not None:
            shutil.rmtree(part, ignore_errors=True)
            return OffloadResult(ok=False, episodes=i, error=(
                f"verification failed at {ep.name}: {mismatch} — this and later "
                f"episodes remain in local staging ({i} already moved); check "
                "the drive"))
        os.replace(part, dst)      # atomic within the drive filesystem
        result.bytes_moved += _dir_bytes(dst)
        result.moved.append(str(dst))
        if not keep_local:
            shutil.rmtree(ep)
    if progress is not None:
        progress(len(episodes), len(episodes), "")
    result.seconds = time.time() - t0
    log.info(result.describe())
    return result
