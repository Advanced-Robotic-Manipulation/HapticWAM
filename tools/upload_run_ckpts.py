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

IDEMPOTENT: the destination folder is listed first and a file already there
with the same byte size is skipped, so this is safe to re-run in a loop as
checkpoints land:

    while sleep 600; do HF_TOKEN=... python tools/upload_run_ckpts.py <run>; done

The token is read from HF_TOKEN / HUGGINGFACE_HUB_TOKEN only — never a flag
(a token on the command line lands in the shell history and in `ps`).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("upload_run_ckpts")

DEFAULT_REPO = "armteam/phantom-checkpoints"
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


def remote_sizes(api, repo: str, prefix: str) -> dict[str, int]:
    """{filename: size} already under `<repo>/<prefix>/` (empty if absent)."""
    from huggingface_hub.utils import HfHubHTTPError
    out: dict[str, int] = {}
    try:
        for e in api.list_repo_tree(repo, path_in_repo=prefix, repo_type="model",
                                    recursive=False):
            size = getattr(e, "size", None)
            if size is None:                      # a directory entry
                continue
            out[Path(e.path).name] = int(size)
    except (HfHubHTTPError, OSError, ValueError) as e:   # folder not created yet
        log.info("no existing %s/%s on the hub (%s)", repo, prefix,
                 type(e).__name__)
    return out


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
                    help="re-upload even when a same-size file is already there")
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
    have = {} if args.force else remote_sizes(api, args.repo, run_name)

    up = skipped = 0
    for p in files:
        dest = f"{run_name}/{p.name}"
        size = p.stat().st_size
        if have.get(p.name) == size:
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
