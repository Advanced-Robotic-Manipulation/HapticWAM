#!/usr/bin/env python3
"""Bounded, sequential CPU presentation watcher for the frozen 24-case confirmation."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v5/confirmation"
DEST = ROOT.parent / "video_reviews/confirmation"
HELPER = BASE / "source_teacher_anchor_driver_v5/tools/sim/make_policy_video.py"
HELPER_SHA = "e4e51ecb40ac12ff4dc2664c65a33a9a5a1560d5080bf44fb1a22fef95bef054"
CAMPAIGN_SHA = "fe62f45fcaf2e71e4df890ff7303bf70a2b49e35fc2a745f6f7492a5f200b855"
RUNTIME = BASE / "source_teacher_anchor_minimal_v5"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(array):
    array = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256_array_bytes": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def render(name):
    folder = ROOT / "rollouts" / name
    status_path = folder / "run_status.json"
    status = read(status_path)
    if status["status"] != "completed" or status.get("exit_code") != 0:
        raise ValueError("Run is not completed with a zero process exit")
    source = Path(status["command"][0]).parents[2]
    if source != RUNTIME:
        raise ValueError("Runtime source differs from frozen minimal v5")
    score_path = ROOT / "analysis/trials" / (name + ".json")
    score = read(score_path)
    score_hash = sha(score_path)
    for filename, digest in score["input_sha256"].items():
        if sha(folder / filename) != digest:
            raise ValueError(f"Scored input changed: {filename}")
    inputs = {
        filename: sha(folder / filename)
        for filename in (
            "run_status.json",
            "sim.mp4",
            "sim_trace.npz",
            "policy_tactile.npz",
            "execution_trace.jsonl",
            "planner_trace.json",
            "run.json",
            "effective_config.json",
            "case.json",
            "policy_info.json",
        )
    }
    with np.load(folder / "sim_trace.npz", allow_pickle=False) as z:
        state_t = np.asarray(z["t"], float)
        frame_t = np.asarray(z["frame_t"], float)
    with np.load(folder / "policy_tactile.npz", allow_pickle=False) as z:
        tactile_t = np.asarray(z["t"], float)
        gel = z["gel"]
        if gel.dtype != np.uint8 or gel.shape != (len(tactile_t), 2, 288, 384):
            raise ValueError("Expected exact saved native gel pixels")
        gel_evidence = {k: fingerprint(z[k]) for k in ("t", "gel", "gel_normal_force")}
    output = DEST / name / "policy_review.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(HELPER),
        "--run",
        str(folder),
        "--label",
        name,
        "--output",
        str(output),
    ]
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        OPENCV_FOR_THREADS_NUM="1",
        PHANTOM_REVIEW_SOURCE=str(source),
    )
    if not output.exists():
        with (output.parent / "render.log").open("x") as log:
            subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
    meta = read(output.with_suffix(".json"))
    trial = meta["trials"][0]
    if meta["presentation_source"]["sha256"] != HELPER_SHA:
        raise ValueError("Presentation helper hash changed")
    if trial["metrics"] != score["metrics"]:
        raise ValueError("Presentation metrics differ from primary score")
    if (
        trial["tactile_mapping"]["source"]
        != "policy_tactile.npz gel: saved pixels supplied at runtime"
    ):
        raise ValueError("Presentation used a tactile fallback")
    if (
        trial["tactile_mapping"]["display_force_source"]
        != "gel_normal_force: normal force used to deform gel proxy"
    ):
        raise ValueError("Presentation used a different gel force channel")
    if meta["horizon_s"] != float(state_t[-1]):
        raise ValueError("Presentation padded or truncated actual measured duration")
    mapping = trial["frame_mapping"]
    for row in mapping:
        for ts, index_key, age_key in (
            (state_t, "state_row", "state_age_s"),
            (frame_t, "scene_frame", "scene_age_s"),
            (tactile_t, "runtime_tactile_row", "runtime_tactile_age_s"),
        ):
            t = row["t"] + 1e-9
            index = int(np.searchsorted(ts, t, side="right") - 1)
            if row[index_key] != index:
                raise ValueError("Noncausal or mismatched input mapping")
            if index < 0:
                if row[age_key] is not None:
                    raise ValueError("Unexpected age before first input")
            elif abs(row[age_key] - (t - ts[index])) > 1e-9:
                raise ValueError("Incorrect causal age")
    cap = cv2.VideoCapture(str(output))
    if not cap.isOpened():
        raise ValueError("Cannot decode review MP4")
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded == 0 and not output.with_name("policy_review_first.png").exists():
            cv2.imwrite(str(output.with_name("policy_review_first.png")), frame)
        decoded += 1
    cap.release()
    if (
        decoded != meta["frames"]
        or decoded != math.floor(meta["horizon_s"] * meta["fps"]) + 1
    ):
        raise ValueError("Incomplete MP4 decode")
    if len(mapping) != decoded:
        raise ValueError("Frame mapping length differs from decoded video")
    if score_hash != sha(score_path) or any(
        sha(folder / f) != digest for f, digest in inputs.items()
    ):
        raise ValueError("Completed source inputs changed during presentation")
    return {
        "case_id": name,
        "video": str(output),
        "video_bytes": output.stat().st_size,
        "video_sha256": sha(output),
        "metadata": str(output.with_suffix(".json")),
        "metadata_sha256": sha(output.with_suffix(".json")),
        "decoded_frames": decoded,
        "fps": meta["fps"],
        "planned_horizon_s": 60,
        "actual_duration_s": meta["horizon_s"],
        "score_sha256": score_hash,
        "score_valid_for_scoring": score["metrics"]["valid_for_scoring"],
        "score_outcomes": score["metrics"]["outcomes"],
        "controller_completion": trial["controller_completion"],
        "gel_arrays": gel_evidence,
        "input_sha256": inputs,
        "score_inputs_sha256": score["input_sha256"],
        "mapping_sha256": hashlib.sha256(
            json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "full_decode_verified": True,
        "causal_mappings_verified": True,
        "stored_runtime_gel_verified": True,
        "metrics_exactly_match_primary_score": True,
        "raw_inputs_unchanged_after_render": True,
        "command": command,
    }


def main():
    os.nice(19)
    os.sched_setaffinity(0, {30, 31})
    cv2.setNumThreads(1)
    if (
        sha(HELPER) != HELPER_SHA
        or sha(ROOT / "campaign_snapshot.json") != CAMPAIGN_SHA
    ):
        raise ValueError("Frozen helper/campaign identity mismatch")
    design = read(ROOT / "campaign_snapshot.json")
    cases = [
        f"{p['id']}__{c['id']}__seed{s}"
        for p in design["policies"]
        for c in design["conditions"]
        for s in design["sampling_seeds"]
    ]
    if len(cases) != len(set(cases)) or len(cases) != 24:
        raise ValueError("Expected exactly24 frozen confirmation cases")
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / "video_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "Preserve existing watcher manifest; inspect before restarting"
        )
    manifest = {
        "status": "watching",
        "pid": os.getpid(),
        "started_at_unix_s": time.time(),
        "watch_timeout_s": 5400,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "expected_cases": cases,
        "helper_sha256": HELPER_SHA,
        "campaign_sha256": CAMPAIGN_SHA,
        "watcher_sha256": sha(Path(__file__)),
        "runtime_source": str(RUNTIME),
        "entries": [],
        "failures": [],
        "raw_writes": False,
    }
    save(manifest_path, manifest)
    deadline = time.monotonic() + 5400
    while time.monotonic() < deadline:
        progress = read(ROOT / "progress.json")
        done = {r["case_id"] for r in manifest["entries"] + manifest["failures"]}
        for name in cases:
            status = ROOT / "rollouts" / name / "run_status.json"
            score = ROOT / "analysis/trials" / (name + ".json")
            if name in done or not status.exists() or not score.exists():
                continue
            if read(status).get("status") != "completed":
                continue
            try:
                manifest["entries"].append(render(name))
                print(
                    json.dumps({"rendered": name, "count": len(manifest["entries"])}),
                    flush=True,
                )
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                subprocess.SubprocessError,
                cv2.error,
            ) as error:
                manifest["failures"].append({"case_id": name, "error": repr(error)})
                print(json.dumps({"failed": name, "error": repr(error)}), flush=True)
            manifest["last_progress_status"] = progress["status"]
            save(manifest_path, manifest)
        if len(manifest["entries"]) + len(manifest["failures"]) == 24:
            manifest["status"] = (
                "complete"
                if not manifest["failures"]
                else "complete_with_render_failures"
            )
            break
        if progress["status"] not in ("running", "all_trials_completed"):
            manifest["status"] = "campaign_terminal_incomplete"
            break
        time.sleep(20)
    else:
        manifest["status"] = "watch_timeout"
    manifest["finished_at_unix_s"] = time.time()
    manifest["total_video_bytes"] = sum(r["video_bytes"] for r in manifest["entries"])
    manifest["missing_cases"] = sorted(
        set(cases) - {r["case_id"] for r in manifest["entries"]}
    )
    save(manifest_path, manifest)
    print(
        json.dumps({"status": manifest["status"], "manifest": str(manifest_path)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
