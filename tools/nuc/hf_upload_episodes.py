#!/usr/bin/env python3
"""Auto-upload PHANTOM episode sessions to hf.co/datasets/armteam/phantom-episodes.

Rules (safe by construction):
 - scans the roots below for session dirs containing at least one ep_*;
 - a session is CLOSED when its newest file is older than --settle minutes
   (default 15) -- the live recording session is never touched;
 - if a phantom collection process is running, the sync defers entirely
   (unless --force), so uploads never steal CPU/IO from the 10 Hz rig loop;
 - uploaded sessions are remembered in ~/.phantom_hf_uploaded.json and skipped
   forever after (delete an entry to force re-upload);
 - per-session upload_folder = one commit per session, resumable, hf_transfer
   accelerated when available.

Usage:
  python hf_upload_episodes.py            # normal sync (used by the timer)
  python hf_upload_episodes.py --dry-run  # show what would upload
  python hf_upload_episodes.py --force    # ignore the running-collection guard
  python hf_upload_episodes.py --only NAME_SUBSTR   # restrict to matching sessions
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO = "armteam/phantom-episodes"
ROOTS = [
    (Path.home() / "phantom-data" / "collect", "collect"),
    (Path.home() / "phantom-data" / "episodes", "episodes"),
    (Path("/media/nuc/kostya_drive/phantom_episodes"), "archive"),
]
MANIFEST = Path.home() / ".phantom_hf_uploaded.json"
LOG = Path.home() / "phantom-icra-2027" / "hf_upload.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def collection_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "phantom.scripts.(collect|record_episodes|panel)"],
                        capture_output=True, text=True)
    return out.returncode == 0


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


def has_episodes(d: Path) -> bool:
    return any(d.glob("ep_*")) or any(d.glob("*/ep_*"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=float, default=15.0, help="minutes of quiescence")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--pause", type=float, default=90.0,
                    help="seconds between session commits (hub rate limit)")
    args = ap.parse_args()

    if collection_running() and not args.force:
        log("collection process running -- deferring (use --force to override)")
        return 0

    done = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    # sessions are often mirrored (collect/ locally + archive/ on the drive);
    # one hub copy per session NAME, whichever root is scanned first
    done_names = {k.split("/", 1)[-1] for k in done}
    from huggingface_hub import HfApi
    api = HfApi()

    now = time.time()
    n_up = 0
    for root, prefix in ROOTS:
        if not root.is_dir():
            continue
        for sess in sorted(p for p in root.iterdir() if p.is_dir()):
            key = f"{prefix}/{sess.name}"
            if args.only and args.only not in sess.name:
                continue
            if key in done or sess.name in done_names:
                continue
            if not has_episodes(sess):
                continue
            age_min = (now - newest_mtime(sess)) / 60.0
            if age_min < args.settle:
                log(f"SKIP {key}: active {age_min:.1f} min ago (< {args.settle})")
                continue
            size = sum(f.stat().st_size for f in sess.rglob("*") if f.is_file()) / 1e6
            if args.dry_run:
                log(f"WOULD UPLOAD {key} ({size:.0f} MB)")
                done_names.add(sess.name)
                continue
            log(f"UPLOAD {key} ({size:.0f} MB) ...")
            try:
                api.upload_folder(folder_path=str(sess), path_in_repo=key,
                                  repo_id=REPO, repo_type="dataset",
                                  commit_message=f"add {key}")
                done[key] = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "mb": round(size)}
                done_names.add(sess.name)
                MANIFEST.write_text(json.dumps(done, indent=1))
                log(f"OK {key}")
                n_up += 1
                if args.pause:
                    time.sleep(args.pause)
            except Exception as e:  # noqa: BLE001 -- log and continue with next session
                log(f"FAIL {key}: {e}")
                if "429" in str(e) or "rate" in str(e).lower():
                    log("hub rate limit -- cooling down 10 min")
                    time.sleep(600)
    log(f"sync done: {n_up} uploaded, {len(done)} total on hub")
    return 0


if __name__ == "__main__":
    sys.exit(main())
