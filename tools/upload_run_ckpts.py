"""Egress a training run to the checkpoint hub — checkpoints + the train log.

    HF_TOKEN=hf_... python tools/upload_run_ckpts.py runs/teacher/teacher_v5_ftA
    HF_TOKEN=hf_... python tools/upload_run_ckpts.py runs/teacher/teacher_v5_ftA \
        --log train_ftA.log --dry-run

Why (validation 2026-08-30, §1 row 6e): nothing in the repo uploads a training
run. v5 was trained on a rented vast box that was destroyed; its hub folder
carries a `ckpt_watch.log` written by a watcher that does not exist in the
repo. A rental that dies between checkpoints loses the run.

Uploads `<run_dir>/teacher_*.pt` (and `student_*.pt` for the HID programs)
plus every `*.log` in the run dir and any `--log` given explicitly, to
`<repo>/<run_name>/`. `run_name` defaults to the run directory's own name —
which is why the launch line must say `--run-name teacher_v5_ftA` and never
`teacher_v5_batch0822` (that hub folder holds the six SHIPPED v5 checkpoints).

IDEMPOTENT: the destination folder is listed first (with `expand=True`) and a
file whose hub-side `lfs.sha256` matches the local file's sha256 is skipped, so
this is safe to re-run in a loop as checkpoints land. Byte SIZE is not a content
check here — every PHANTOM teacher checkpoint is exactly 393,115,861 bytes, so a
size-only rule made every same-name file "already present" whatever it held; the
size comparison survives only as the fallback for a file the hub reports no LFS
metadata for.

    while sleep 600; do HF_TOKEN=... python tools/upload_run_ckpts.py <run>; done

The token is read from HF_TOKEN / HUGGINGFACE_HUB_TOKEN only — never a flag
(a token on the command line lands in the shell history and in `ps`).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("upload_run_ckpts")

# The 2026-09 armteam restructure gave each run family its own PUBLIC model repo,
# and this script uploads whatever run dir it is handed — so the destination is a
# variable, not a guess. Callers (the ckpt_watch_*.sh watchers the provision
# scripts write) always pass --repo explicitly; the map is:
#     armteam/hapticwam-teacher     teacher_v6/, teacher_v6_simft/
#     armteam/hapticwam-student     hid_simft/, hid_ftA_r2_nowrist*/
#     armteam/hapticwam-baselines   diffusion_100k/, xvla_20k/, pi05_phantom_expert_v1/
#     armteam/hapticwam-ablations   every other teacher_*/hid_*/ctrl_*/cosmos_* run
#     armteam/hapticwam-ablations   also the superseded rounds (hid_r0_*, hid_r1_*, ctrl_r0_*,
#                                   teacher_v2..v5): those were never copied off the old repo,
#                                   so a NEW one goes here rather than back to it
# The default below is the ablations repo: an ad-hoc run with no --repo is an
# experiment, and a run that belongs in teacher/student/baselines must say so.
# PHANTOM_CKPT_REPO overrides it without touching the launch lines.
DEFAULT_REPO = os.environ.get("PHANTOM_CKPT_REPO", "armteam/hapticwam-ablations")
CKPT_GLOBS = ("teacher_*.pt", "student_*.pt", "student_hids_*.pt")


def local_files(run_dir: Path, extra_logs: list[Path]) -> list[Path]:
    """Checkpoints + logs of one run dir, de-duplicated, sorted by name."""
    found: dict[str, Path] = {}
    for pat in CKPT_GLOBS:
        for p in sorted(run_dir.glob(pat)):
            found[p.name] = p
    for p in sorted(run_dir.glob("*.log")):
        found[p.name] = p
    for p in extra_logs:
        p = Path(p)
        if not p.exists():
            raise SystemExit(f"--log {p} does not exist")
        found[p.name] = p
    return [found[k] for k in sorted(found)]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    """Streaming sha256 of a local file — the same digest the hub stores."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def remote_files(api, repo: str, prefix: str) -> dict[str, dict]:
    """{filename: {"sha256": str|None, "size": int}} under `<repo>/<prefix>/`.

    `expand=True` is what makes `e.lfs.sha256` available: without it the hub
    hands back sizes only, and EVERY PHANTOM teacher checkpoint is exactly
    393,115,861 bytes — so a same-name file was always "already present"
    regardless of its content, and a re-upload of a BETTER checkpoint over a
    worse one of the same step number was silently skipped (validation
    2026-08-31 #8). LFS metadata can still be missing (a small non-LFS file,
    or an older tree response); there the size comparison is the fallback.
    """
    from huggingface_hub.utils import HfHubHTTPError
    out: dict[str, dict] = {}
    try:
        for e in api.list_repo_tree(repo, path_in_repo=prefix, repo_type="model",
                                    recursive=False, expand=True):
            size = getattr(e, "size", None)
            if size is None:                      # a directory entry
                continue
            lfs = getattr(e, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs is not None else None
            if sha is None and isinstance(lfs, dict):
                sha = lfs.get("sha256")
            out[Path(e.path).name] = {"sha256": sha, "size": int(size)}
    except (HfHubHTTPError, OSError, ValueError) as e:   # folder not created yet
        log.info("no existing %s/%s on the hub (%s)", repo, prefix,
                 type(e).__name__)
    return out


def already_uploaded(remote: dict, path: Path, size: int) -> bool:
    """Is `path` byte-identical to what the hub already holds under its name?"""
    if remote is None:
        return False
    sha = remote.get("sha256")
    if sha is None:                               # no LFS metadata -> size only
        return int(remote.get("size", -1)) == size
    return sha == sha256_file(path)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, help="e.g. runs/teacher/teacher_v5_ftA")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--run-name", default="",
                    help="destination folder in the repo (default: the run "
                         "dir's own name)")
    ap.add_argument("--log", type=Path, nargs="*", default=[],
                    help="extra log files to ship (nohup writes train_*.log "
                         "next to the repo, not into the run dir)")
    ap.add_argument("--force", action="store_true",
                    help="re-upload even when an identical file is already there")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")
    run_name = args.run_name or run_dir.name
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is not set — a "
                         "write token for the checkpoint repo is required")

    files = local_files(run_dir, list(args.log))
    if not files:
        raise SystemExit(f"no checkpoints or logs under {run_dir} "
                         f"(looked for {', '.join(CKPT_GLOBS)}, *.log)")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    have = {} if args.force else remote_files(api, args.repo, run_name)

    up = skipped = 0
    for p in files:
        dest = f"{run_name}/{p.name}"
        size = p.stat().st_size
        if already_uploaded(have.get(p.name), p, size):
            log.info("skip %s (already on the hub, %.2f GB)", dest, size / 1e9)
            skipped += 1
            continue
        if args.dry_run:
            log.info("would upload %s -> %s/%s (%.2f GB)", p, args.repo, dest,
                     size / 1e9)
            up += 1
            continue
        log.info("uploading %s -> %s/%s (%.2f GB)", p, args.repo, dest, size / 1e9)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dest,
                        repo_id=args.repo, repo_type="model")
        up += 1
    log.info("%s: %d uploaded, %d already present -> %s/%s/",
             "DRY RUN" if args.dry_run else "done", up, skipped, args.repo, run_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
