#!/usr/bin/env python3
"""Upload the staged tasks/ view in FAT commits to stay inside the HF api
quota (1000 requests / 5 min): ~25 episodes (~9.5k files) per commit means
the whole 400-episode view fits in well under one quota window's worth of
requests. Content chunks are already on the hub -> dedup, no real transfer.

Resumable: completed chunks are recorded in ~/.phantom_reorg_chunks.json and
skipped on re-run; re-adding identical paths is a server-side no-op anyway.
Finishes with the manifests/README commit and a hub-side count verification.
"""
import json
import time
from pathlib import Path

from huggingface_hub import HfApi

STAGE = Path.home() / "phantom-hf-stage"
REPO = "armteam/hapticwam-teleop-raw"
EXPECTED = {"Carton": 180, "waffles": 180, "egg": 180, "Carton_fail": 20, "waffles_fail": 20, "egg_fail": 20}
CHUNK = 25                     # episodes per commit (~9.5k files)
PAUSE_S = 25                   # keeps request rate far below 1000/5min
PROGRESS = Path.home() / ".phantom_reorg_chunks.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    api = HfApi()
    done = set(json.loads(PROGRESS.read_text())) if PROGRESS.exists() else set()
    chunks = []
    for task in EXPECTED:
        eps = sorted(p.name for p in (STAGE / "tasks" / task).iterdir()
                     if p.is_dir())
        assert len(eps) == EXPECTED[task], f"{task}: {len(eps)} staged — ABORT"
        for i in range(0, len(eps), CHUNK):
            chunks.append((task, i // CHUNK, eps[i:i + CHUNK]))
    log(f"{len(chunks)} chunks total, {len(done)} already done")

    for task, idx, eps in chunks:
        key = f"{task}/{idx}"
        if key in done:
            continue
        patterns = [f"tasks/{task}/{ep}/**" for ep in eps]
        for attempt in range(1, 6):
            try:
                api.upload_folder(folder_path=str(STAGE), path_in_repo=".",
                                  repo_id=REPO, repo_type="dataset",
                                  allow_patterns=patterns,
                                  commit_message=f"tasks view: {task} chunk {idx} "
                                                 f"({len(eps)} eps)")
                break
            except Exception as e:  # noqa: BLE001
                wait = min(600, 60 * attempt)
                log(f"chunk {key} attempt {attempt} failed: {e} — retry in {wait}s")
                time.sleep(wait)
        else:
            log(f"chunk {key} FAILED after retries — aborting (resume later)")
            return 1
        done.add(key)
        PROGRESS.write_text(json.dumps(sorted(done)))
        log(f"OK {key} ({len(eps)} eps)")
        time.sleep(PAUSE_S)

    log("uploading manifests + README")
    api.upload_folder(folder_path=str(STAGE), path_in_repo=".",
                      repo_id=REPO, repo_type="dataset",
                      allow_patterns=["manifests/**", "README.md"],
                      commit_message="manifests + dataset card for tasks/ view")

    ok = True
    for task, n in EXPECTED.items():
        tree = list(api.list_repo_tree(REPO, repo_type="dataset",
                                       path_in_repo=f"tasks/{task}",
                                       recursive=False))
        got = sum(1 for t in tree if t.path.split("/")[-1].startswith("ep_"))
        ok &= got == n
        log(f"hub tasks/{task}: {got}/{n} {'OK' if got == n else 'MISMATCH'}")
    log("REORG " + ("COMPLETE — hub verified" if ok else "MISMATCH — investigate"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
