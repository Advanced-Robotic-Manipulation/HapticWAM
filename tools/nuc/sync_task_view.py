#!/usr/bin/env python3
"""Incrementally sync the hub's tasks/<task>/ view from the drive.

Uploads ONLY the episodes missing from tasks/<task> — straight from their
drive location, with no local staging copy (the previous reorg re-copied all
35 GB to re-register files that had not changed). Then regenerates the
manifests from episode metadata + the quality CSV and uploads those.

Usage: sync_task_view.py [--only TASK] [--dry-run]
"""
import argparse
import csv
import json
import time
from pathlib import Path

from huggingface_hub import HfApi

DRIVE = Path("/media/nuc/kostya_drive/phantom_episodes")
QUALITY_CSV = Path.home() / "phantom-icra-2027" / "episode_quality.csv"
REPO = "armteam/phantom-episodes"
EXPECTED = {"Carton": 180, "waffles": 180, "egg": 180, "whiteboard": 180,
            "Carton_fail": 20, "waffles_fail": 20, "egg_fail": 20,
            "whiteboard_fail": 10}
VAL_SESSIONS_PER_SUCCESS_TASK = 2
CHUNK = 25            # episodes per commit — well inside the api quota
PAUSE_S = 20


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def scan_drive():
    """task -> {ep_name: (episode_path, session_name)} for finalized episodes."""
    out = {t: {} for t in EXPECTED}
    for sess in sorted(p for p in DRIVE.iterdir() if p.is_dir()):
        task = sess.name.split("_", 2)[-1]
        if task not in EXPECTED:
            continue
        for ep in sorted(p for p in sess.glob("ep_*") if p.is_dir()):
            try:
                if json.loads((ep / "meta.json").read_text()).get("status") != "finalized":
                    continue
            except OSError:
                continue
            out[task][ep.name] = (ep, sess.name)
    return out


def hub_eps(api, task):
    try:
        tree = api.list_repo_tree(REPO, repo_type="dataset",
                                  path_in_repo=f"tasks/{task}", recursive=False)
        return {t.path.split("/")[-1] for t in tree
                if t.path.split("/")[-1].startswith("ep_")}
    except Exception:
        return set()


def build_manifests(drive, out_dir):
    quality = {}
    if QUALITY_CSV.exists():
        with open(QUALITY_CSV) as f:
            for r in csv.DictReader(f):
                quality[(r["session"], r["ep"])] = r
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task, eps in drive.items():
        sessions = sorted({s for _, s in eps.values()})
        val = set()
        if not task.endswith("_fail"):
            val = set(sessions[-VAL_SESSIONS_PER_SUCCESS_TASK:])
        rows = []
        for name, (path, sess) in sorted(eps.items()):
            meta = json.loads((path / "meta.json").read_text())
            q = quality.get((sess, name), {})
            contact = float(q.get("max_contact_mm2") or 0.0)
            rows.append({
                "episode": name, "task": task,
                "success": meta.get("success"),
                "failure_demo": task.endswith("_fail"),
                "tactile_contact": contact >= 1.0,
                "split": "train" if (task.endswith("_fail") or sess not in val) else "val",
                "session": sess, "operator": meta.get("operator"),
                "duration_s": float(q.get("dur_s") or 0.0),
                "peak_force_N": float(q.get("peak_force_N") or 0.0),
                "max_contact_mm2": contact,
                "path": f"tasks/{task}/{name}",
            })
        (out_dir / f"{task}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        all_rows += rows
        nval = sum(1 for r in rows if r["split"] == "val")
        log(f"manifest {task}: {len(rows)} eps ({nval} val, sessions {sorted(val)})")
    (out_dir / "all.jsonl").write_text(
        "".join(json.dumps(r) + "\n"
                for r in sorted(all_rows, key=lambda r: (r["task"], r["episode"]))))
    if QUALITY_CSV.exists():
        (out_dir / "quality_full.csv").write_text(QUALITY_CSV.read_text())
    return len(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = HfApi()

    drive = scan_drive()
    counts = {t: len(v) for t, v in drive.items()}
    log(f"drive counts: {counts}")
    assert counts == EXPECTED, f"count mismatch: {counts} != {EXPECTED} — ABORT"

    for task in EXPECTED:
        if args.only and args.only != task:
            continue
        have = hub_eps(api, task)
        missing = [n for n in sorted(drive[task]) if n not in have]
        stale = sorted(have - set(drive[task]))
        log(f"tasks/{task}: hub {len(have)} | missing {len(missing)} | stale {len(stale)}")
        if stale:
            from huggingface_hub import CommitOperationDelete
            log(f"  deleting {len(stale)} stale: {stale[:3]}{'...' if len(stale) > 3 else ''}")
            if not args.dry_run:
                api.create_commit(
                    repo_id=REPO, repo_type="dataset",
                    operations=[CommitOperationDelete(
                        path_in_repo=f"tasks/{task}/{n}/", is_folder=True) for n in stale],
                    commit_message=f"tasks/{task}: drop {len(stale)} stale episodes")
        # group by source session: upload_folder batches files into
        # right-sized commits internally (one giant create_commit with 10k
        # files makes the server hang up) and is resumable
        by_sess: dict[str, list[str]] = {}
        for n in missing:
            by_sess.setdefault(drive[task][n][1], []).append(n)
        for sess_name, names in sorted(by_sess.items()):
            sess_dir = drive[task][names[0]][0].parent
            if args.dry_run:
                log(f"  WOULD upload {len(names)} eps from {sess_name}")
                continue
            log(f"  uploading {len(names)} eps from {sess_name} -> tasks/{task} ...")
            for attempt in range(1, 6):
                try:
                    api.upload_folder(
                        folder_path=str(sess_dir), path_in_repo=f"tasks/{task}",
                        repo_id=REPO, repo_type="dataset",
                        allow_patterns=[f"{n}/**" for n in names],
                        commit_message=f"tasks/{task}: add {len(names)} eps from {sess_name}")
                    break
                except Exception as e:  # noqa: BLE001
                    wait = min(300, 30 * attempt)
                    log(f"    attempt {attempt} failed ({type(e).__name__}) — retry in {wait}s")
                    time.sleep(wait)
            else:
                log(f"    GIVING UP on {sess_name}")
                return 1
            time.sleep(PAUSE_S)

    mdir = Path.home() / "phantom-manifests"
    n = build_manifests(drive, mdir)
    if not args.dry_run:
        api.upload_folder(folder_path=str(mdir), path_in_repo="manifests",
                          repo_id=REPO, repo_type="dataset",
                          commit_message=f"manifests: {n} episodes, splits recomputed")
    ok = True
    for task, want in EXPECTED.items():
        got = len(hub_eps(api, task))
        ok &= got == want
        log(f"VERIFY tasks/{task}: {got}/{want} {'OK' if got == want else 'MISMATCH'}")
    log("SYNC " + ("COMPLETE" if ok else "MISMATCH — investigate"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
