#!/usr/bin/env python3
"""HF-only reorg of armteam/hapticwam-teleop-raw into the canonical tasks/ view.

Reads the external drive READ-ONLY, stages the flattened layout + manifests +
dataset card on the local disk, verifies exact counts (180/180/20/20 — aborts
on any mismatch), uploads with upload_large_folder (content-dedup: chunks are
already on the hub, so this is mostly metadata), then verifies the hub tree.
The drive and the existing archive/ + collect/ hub views are NOT touched.

Resumable: re-running skips already-staged episodes; upload_large_folder is
natively resumable.
"""
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

DRIVE = Path(os.environ.get("PHANTOM_NUC_DRIVE", "/media/nuc/phantom_drive")) / "phantom_episodes"
STAGE = Path.home() / "phantom-hf-stage"
QUALITY_CSV = Path.home() / "phantom-icra-2027" / "episode_quality.csv"
REPO = "armteam/hapticwam-teleop-raw"
EXPECTED = {"Carton": 180, "waffles": 180, "egg": 180, "Carton_fail": 20, "waffles_fail": 20, "egg_fail": 20}
VAL_SESSIONS_PER_SUCCESS_TASK = 2      # last N sessions chronologically -> val


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_quality():
    q = {}
    with open(QUALITY_CSV) as f:
        for r in csv.DictReader(f):
            q[(r["session"], r["ep"])] = r
    return q


def count_files(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.is_file())


def stage():
    quality = load_quality()
    rows = {t: [] for t in EXPECTED}
    sessions = {t: set() for t in EXPECTED}
    for sess in sorted(p for p in DRIVE.iterdir() if p.is_dir()):
        task = sess.name.split("_", 2)[-1]
        if task not in EXPECTED:
            continue
        for ep in sorted(p for p in sess.glob("ep_*") if p.is_dir()):
            meta = json.loads((ep / "meta.json").read_text())
            if meta.get("status") != "finalized":
                log(f"SKIP non-finalized {sess.name}/{ep.name}")
                continue
            dst = STAGE / "tasks" / task / ep.name
            if dst.exists() and count_files(dst) == count_files(ep):
                pass                              # already staged (resume)
            else:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(ep, dst)
            qr = quality.get((sess.name, ep.name), {})
            contact = float(qr.get("max_contact_mm2") or 0.0)
            rows[task].append({
                "episode": ep.name,
                "task": task,
                "success": meta.get("success"),
                "failure_demo": task.endswith("_fail"),
                "tactile_contact": contact >= 1.0,
                "session": sess.name,
                "operator": meta.get("operator"),
                "duration_s": float(qr.get("dur_s") or 0.0),
                "peak_force_N": float(qr.get("peak_force_N") or 0.0),
                "max_contact_mm2": contact,
                "n_camera_frames": int(qr.get("n_camera_scene_color") or 0),
                "n_tactile_samples": int(qr.get("n_tactile_left_wrench") or 0),
                "quality_notes": qr.get("problems") or "",
                "path": f"tasks/{task}/{ep.name}",
            })
            sessions[task].add(sess.name)
    # exact-count gate BEFORE anything uploads
    counts = {t: len(v) for t, v in rows.items()}
    log(f"staged counts: {counts}")
    assert counts == EXPECTED, f"count mismatch: {counts} != {EXPECTED} — ABORT"
    # split: last N sessions of each success task -> val; fail demos train-only
    for task, rws in rows.items():
        if task.endswith("_fail"):
            for r in rws:
                r["split"] = "train"
            continue
        val = set(sorted(sessions[task])[-VAL_SESSIONS_PER_SUCCESS_TASK:])
        for r in rws:
            r["split"] = "val" if r["session"] in val else "train"
        log(f"{task}: val sessions = {sorted(val)} "
            f"({sum(1 for r in rws if r['split'] == 'val')} eps)")
    mdir = STAGE / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    allrows = []
    for task, rws in rows.items():
        with open(mdir / f"{task}.jsonl", "w") as f:
            for r in sorted(rws, key=lambda r: r["episode"]):
                f.write(json.dumps(r) + "\n")
                allrows.append(r)
    with open(mdir / "all.jsonl", "w") as f:
        for r in sorted(allrows, key=lambda r: (r["task"], r["episode"])):
            f.write(json.dumps(r) + "\n")
    shutil.copy2(QUALITY_CSV, mdir / "quality_full.csv")
    write_readme(rows)
    return rows


def write_readme(rows):
    import statistics as st

    def med(task, key):
        return st.median(r[key] for r in rows[task])

    lines = [
        "# PHANTOM episodes",
        "",
        "Tactile manipulation episodes for the PHANTOM tactile world-action-model",
        "(UR3 + Robotiq 2F-85 + 2x Daimon DM-Tac W2L fingertip sensors +",
        "RealSense scene camera, teleoperated via the Echo exoskeleton leader).",
        "",
        "## Layout",
        "",
        "- `tasks/<task>/ep_<task>_<epoch>_<idx>/` — **canonical flattened view**:",
        "  one directory per episode, grouped by task. Use this + `manifests/`.",
        "- `manifests/<task>.jsonl`, `manifests/all.jsonl` — one JSON object per",
        "  episode: success, split (train/val), `tactile_contact`, duration, peak",
        "  force, contact area, source session, quality notes, hub path.",
        "- `manifests/quality_full.csv` — full per-episode quality audit.",
        "- `archive/`, `collect/` — raw as-recorded session layout (provenance;",
        "  superset that also contains pre-cleanup debug takes).",
        "",
        "## Episode format",
        "",
        "Each episode directory holds one zarr group per stream (`data` + `ts`",
        "arrays, timestamps in master-clock seconds) plus `meta.json`.",
        "Streams: `camera_scene_color` (JPEG, ~15 Hz), `tactile_{left,right}_`",
        "`{wrench (6D, N / 1e-2 Nm), area (mm^2), fields_ds (72x96x8 f16),",
        "keyframes (144x192x8 f16), infer_img (288x384 u8)}` (~6.5 Hz),",
        "`arm_{q,qd,tcp_pose,tcp_speed,ft}` (~125-140 Hz), `gripper` (pos,",
        "obj-detect, ~100 Hz), `actions` (delta-EE, ~10 Hz), `actions_abs`",
        "(absolute joint target + gripper).",
        "",
        "## Tasks",
        "",
        "| task | episodes | role | median dur | median peak |F| | median contact |",
        "|---|---|---|---|---|---|",
    ]
    for task in EXPECTED:
        role = "failure demos" if task.endswith("_fail") else "successes"
        lines.append(
            f"| {task} | {len(rows[task])} | {role} | "
            f"{med(task, 'duration_s'):.1f} s | {med(task, 'peak_force_N'):.1f} N | "
            f"{med(task, 'max_contact_mm2'):.1f} mm2 |")
    lines += [
        "",
        "## Notes",
        "",
        "- `success=true` everywhere in the success tasks; `*_fail` tasks are",
        "  DELIBERATE failure demonstrations (recorded as intended, higher",
        "  forces/contact — contrastive signal).",
        "- `tactile_contact=false` marks episodes where the grasp landed outside",
        "  the sensor pads (valid vision/proprio demos, no tactile signal).",
        "- Split rule: per success task, the last "
        f"{VAL_SESSIONS_PER_SUCCESS_TASK} sessions (chronological) are `val`,",
        "  the rest `train`; failure demos are train-only. Re-split freely via",
        "  the manifests — they are the source of truth, not the folder layout.",
        "- Peak forces briefly exceed the 30 N pad ceiling in a handful of",
        "  episodes (dynamic spikes, mostly failure demos) — flagged in",
        "  `quality_notes`.",
    ]
    (STAGE / "README.md").write_text("\n".join(lines) + "\n")


def upload():
    from huggingface_hub import HfApi
    api = HfApi()
    log("upload_large_folder starting (content-dedup: mostly metadata)")
    api.upload_large_folder(repo_id=REPO, folder_path=str(STAGE),
                            repo_type="dataset", num_workers=4,
                            print_report=False)
    log("upload_large_folder done")
    # post-verify: count episode dirs per task on the hub
    ok = True
    for task, n in EXPECTED.items():
        tree = list(api.list_repo_tree(REPO, repo_type="dataset",
                                       path_in_repo=f"tasks/{task}",
                                       recursive=False))
        got = sum(1 for t in tree if t.path.split("/")[-1].startswith("ep_"))
        status = "OK" if got == n else "MISMATCH"
        ok &= got == n
        log(f"hub tasks/{task}: {got}/{n} {status}")
    return ok


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    stage()
    if "--stage-only" in sys.argv:
        log("stage-only: done")
        return 0
    ok = upload()
    log("REORG " + ("COMPLETE — hub verified" if ok else "FINISHED WITH MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
