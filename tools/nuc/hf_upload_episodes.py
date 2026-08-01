#!/usr/bin/env python3
"""Auto-upload PHANTOM episode sessions to hf.co/datasets/armteam/phantom-episodes.

v3 — hardened after the 2026-08-01 system review (Codex + Opus workflow):
 - single-instance flock (concurrent timer + manual runs cannot clobber the
   manifest or double-upload);
 - manifest keyed by session NAME with the uploaded episode list per session:
   every root copy (collect / episodes / drive) of a session uploads to the
   SAME hub path, so the hub converges to the UNION of episodes no matter how
   an offload was interrupted — a partial early upload can never freeze out
   episodes that surface later (hub content-dedup makes overlap free);
 - atomic manifest writes (tmp + fsync + os.replace) with a .bak of the last
   good state; a corrupt manifest is set aside and restored from .bak instead
   of crashing every future run;
 - legacy "prefix/name" manifest entries are migrated in place.

Safety rules kept from v1/v2:
 - a session is CLOSED when its newest file (across all copies) is older than
   --settle minutes (default 15) -- the live recording session is never touched;
 - if a phantom collection process is running, the sync defers entirely
   (unless --force), so uploads never steal CPU/IO from the 10 Hz rig loop;
 - per-session upload_folder = one commit per copy, resumable, content-dedup;
 - 90 s pause between session commits, 10 min cooldown on HTTP 429.

Usage:
  python hf_upload_episodes.py            # normal sync (used by the timer)
  python hf_upload_episodes.py --dry-run  # show what would upload
  python hf_upload_episodes.py --force    # ignore the running-collection guard
  python hf_upload_episodes.py --only NAME_SUBSTR   # restrict to matching sessions
"""
import argparse, fcntl, json, os, shutil, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO = "armteam/phantom-episodes"
HUB_PREFIX = "archive"          # single canonical hub prefix for all copies
ROOTS = [
    Path("/media/nuc/kostya_drive/phantom_episodes"),   # canonical, scanned first
    Path.home() / "phantom-data" / "collect",
    Path.home() / "phantom-data" / "episodes",
]
MANIFEST = Path.home() / ".phantom_hf_uploaded.json"
LOCK = Path.home() / ".phantom_hf_uploaded.lock"
LOG = Path.home() / "phantom-icra-2027" / "hf_upload.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def collection_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "phantom.scripts.(collect|record_episodes|panel)"],
                        capture_output=True, text=True)
    return out.returncode == 0


def load_manifest() -> dict:
    """Load + migrate the manifest; recover from corruption via .bak."""
    bak = MANIFEST.with_suffix(".json.bak")
    raw = None
    if MANIFEST.exists():
        try:
            raw = json.loads(MANIFEST.read_text())
        except (json.JSONDecodeError, OSError) as e:
            aside = MANIFEST.with_suffix(f".json.corrupt-{int(time.time())}")
            shutil.move(str(MANIFEST), str(aside))
            log(f"MANIFEST CORRUPT ({e}) -- set aside to {aside.name}")
            if bak.exists():
                try:
                    raw = json.loads(bak.read_text())
                    log("manifest restored from .bak")
                except (json.JSONDecodeError, OSError):
                    raw = None
            if raw is None:
                log("NO USABLE MANIFEST -- refusing to upload this run "
                    "(fix ~/.phantom_hf_uploaded.json manually)")
                raise SystemExit(1)
    if raw is None:
        return {}
    # migrate legacy "prefix/name" keys -> name-keyed entries (eps unknown ->
    # backfilled from the current on-disk union on first sight below)
    out: dict = {}
    for k, v in raw.items():
        name = k.split("/", 1)[-1] if "/" in k else k
        entry = out.setdefault(name, {"t": None, "mb": 0, "eps": []})
        if isinstance(v, dict):
            entry["t"] = entry["t"] or v.get("t")
            entry["mb"] = max(entry["mb"], int(v.get("mb", 0) or 0))
            eps = v.get("eps")
            if eps:
                entry["eps"] = sorted(set(entry["eps"]) | set(eps))
    return out


def save_manifest(done: dict) -> None:
    bak = MANIFEST.with_suffix(".json.bak")
    if MANIFEST.exists():
        try:
            shutil.copy2(str(MANIFEST), str(bak))
        except OSError:
            pass
    tmp = MANIFEST.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(done, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MANIFEST)


def newest_mtime(d: Path) -> float:
    newest = d.stat().st_mtime
    for p in d.rglob("*"):
        try:
            m = p.stat().st_mtime
            if m > newest:
                newest = m
        except OSError:
            pass
    return newest


def ep_names(d: Path) -> set:
    """Episode dir names in a session copy (flat or one level nested)."""
    return {p.name for p in d.glob("ep_*") if p.is_dir()} | \
           {p.name for p in d.glob("*/ep_*") if p.is_dir()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=float, default=15.0, help="minutes of quiescence")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--pause", type=float, default=90.0,
                    help="seconds between session commits (hub rate limit)")
    args = ap.parse_args()

    lock_f = open(LOCK, "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another uploader instance holds the lock -- exiting")
        return 0

    if collection_running() and not args.force:
        log("collection process running -- deferring (use --force to override)")
        return 0

    done = load_manifest()
    from huggingface_hub import HfApi
    api = HfApi()

    # session name -> list of on-disk copies that contain episodes
    copies: dict = {}
    for root in ROOTS:
        if not root.is_dir():
            continue
        for sess in sorted(p for p in root.iterdir() if p.is_dir()):
            if args.only and args.only not in sess.name:
                continue
            eps = ep_names(sess)
            if eps:
                copies.setdefault(sess.name, []).append((sess, eps))

    now = time.time()
    n_up = 0
    for name in sorted(copies):
        entry = done.get(name)
        union = set().union(*(e for _, e in copies[name]))
        if entry is not None and not entry.get("eps"):
            # legacy entry (pre-v3): assume what is on disk now was uploaded
            entry["eps"] = sorted(union)
            if not args.dry_run:
                save_manifest(done)
            continue
        have = set(entry["eps"]) if entry else set()
        missing = union - have
        if not missing:
            continue
        age_min = min((now - newest_mtime(p)) / 60.0 for p, _ in copies[name])
        if age_min < args.settle:
            log(f"SKIP {name}: active {age_min:.1f} min ago (< {args.settle})")
            continue
        target = f"{HUB_PREFIX}/{name}"
        for sess, eps in copies[name]:
            if not (eps - have):
                continue        # this copy adds nothing new
            size = sum(f.stat().st_size for f in sess.rglob("*") if f.is_file()) / 1e6
            if args.dry_run:
                log(f"WOULD UPLOAD {sess} -> {target} ({size:.0f} MB, "
                    f"{len(eps - have)} new eps)")
                have |= eps
                continue
            log(f"UPLOAD {sess} -> {target} ({size:.0f} MB, {len(eps - have)} new eps) ...")
            try:
                api.upload_folder(folder_path=str(sess), path_in_repo=target,
                                  repo_id=REPO, repo_type="dataset",
                                  commit_message=f"add {target} ({sess.parent.name})")
                have |= eps
                done[name] = {"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "mb": round(size), "eps": sorted(have)}
                save_manifest(done)
                log(f"OK {target} ({len(have)}/{len(union)} eps on hub)")
                n_up += 1
                if args.pause:
                    time.sleep(args.pause)
            except Exception as e:  # noqa: BLE001 -- log and continue with next session
                log(f"FAIL {target}: {e}")
                if "429" in str(e) or "rate" in str(e).lower():
                    log("hub rate limit -- cooling down 10 min")
                    time.sleep(600)
    log(f"sync done: {n_up} uploads, {len(done)} sessions in manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
