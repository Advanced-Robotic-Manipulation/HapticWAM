#!/usr/bin/env python3
"""Read-only completion audit and low-priority CPU presentation of two trials."""

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v3/minimal_profile_diagnostic"
DEST = ROOT.parent / "video_reviews/minimal_profile_diagnostic"
HELPER = BASE / "source_teacher_anchor_driver_v3/tools/sim/make_policy_video.py"
HELPER_SHA = "e4e51ecb40ac12ff4dc2664c65a33a9a5a1560d5080bf44fb1a22fef95bef054"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def fingerprint(array):
    array = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256_array_bytes": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def main():
    os.nice(19)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    cv2.setNumThreads(1)
    assert sha(HELPER) == HELPER_SHA
    progress = read(ROOT / "progress.json")
    assert progress["status"] == "all_trials_completed" and len(progress["trials"]) == 2
    manifest = {
        "status": "rendering",
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "helper_sha256": HELPER_SHA,
        "entries": [],
    }
    DEST.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps({"pid": os.getpid(), "cpu_affinity": manifest["cpu_affinity"]}),
        flush=True,
    )
    for name, trial in sorted(progress["trials"].items()):
        assert trial["status"] == "completed" and trial["exit_code"] == 0
        folder = ROOT / "rollouts" / name
        source = Path(trial["command"][0]).parents[2]
        score_path = ROOT / "analysis/trials" / (name + ".json")
        score = read(score_path)
        assert score["metrics"]["valid_for_scoring"]
        for f, digest in score["input_sha256"].items():
            assert sha(folder / f) == digest
        execution = [
            json.loads(line)
            for line in (folder / "execution_trace.jsonl").read_text().splitlines()
        ]
        finish = [r for r in execution if r["diagnostics"].get("completed_reason")]
        completion = {
            "first_completion_execution_row": None if not finish else finish[0],
            "completion_report_rows": len(finish),
            "completion_hold_rows": sum(
                bool(r["diagnostics"].get("completion_hold")) for r in execution
            ),
            "run_json_completion_field": read(folder / "run.json").get(
                "policy_completed_reason"
            ),
            "meaning": "Completion is taken from execution diagnostics only; absence of a run.json field is not evidence of missing FINISH.",
        }
        output = DEST / name / "policy_review.mp4"
        command = [
            sys.executable,
            str(HELPER),
            "--run",
            str(folder),
            "--label",
            "minimal gel+FINISH " + name.split("__")[-1],
            "--output",
            str(output),
        ]
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PHANTOM_REVIEW_SOURCE=str(source),
        )
        if not output.exists():
            subprocess.run(command, env=env, check=True)
        meta = read(output.with_suffix(".json"))
        assert meta["presentation_source"]["sha256"] == HELPER_SHA
        assert meta["trials"][0]["metrics"] == score["metrics"]
        assert (
            meta["trials"][0]["controller_completion"] is None
            if not finish
            else meta["trials"][0]["controller_completion"] is not None
        )
        cap = cv2.VideoCapture(str(output))
        assert cap.isOpened()
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded == 0:
                cv2.imwrite(str(output.with_name("policy_review_first.png")), frame)
            decoded += 1
        cap.release()
        assert (
            decoded == meta["frames"] == math.floor(meta["horizon_s"] * meta["fps"]) + 1
        )
        mapping = meta["trials"][0]["frame_mapping"]
        for row in mapping:
            for key in ("scene_age_s", "state_age_s", "runtime_tactile_age_s"):
                assert row[key] is None or row[key] >= -1e-9
        with np.load(folder / "sim_trace.npz") as z:
            forces = {
                k: fingerprint(z[k])
                for k in (
                    "t",
                    "pad_force",
                    "pad_packet_normal_force",
                    "packet_robot_normal_force",
                    "packet_bin_normal_force",
                )
            }
        entry = {
            "case_id": name,
            "video": str(output),
            "video_bytes": output.stat().st_size,
            "video_sha256": sha(output),
            "metadata": str(output.with_suffix(".json")),
            "metadata_bytes": output.with_suffix(".json").stat().st_size,
            "metadata_sha256": sha(output.with_suffix(".json")),
            "decoded_frames": decoded,
            "fps": meta["fps"],
            "horizon_s": meta["horizon_s"],
            "score_sha256": sha(score_path),
            "score_outcomes": score["metrics"]["outcomes"],
            "score_events_s": score["metrics"]["event_times_s"],
            "execution_completion": completion,
            "physical_force_arrays": forces,
            "runtime_source": str(source),
            "runtime_metrics_sha256": sha(source / "phantom/sim/policy_metrics.py"),
            "source_execution_sha256": sha(folder / "execution_trace.jsonl"),
            "score_inputs_sha256": score["input_sha256"],
            "full_decode_verified": True,
            "metrics_exactly_match_frozen_score": True,
            "causal_mappings_verified": True,
            "command": command,
        }
        manifest["entries"].append(entry)
        (DEST / "video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps(
                {"case": name, "frames": decoded, "completion_rows": len(finish)}
            ),
            flush=True,
        )
    manifest["status"] = "complete"
    manifest["total_video_bytes"] = sum(e["video_bytes"] for e in manifest["entries"])
    (DEST / "video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"status": "complete", "manifest": str(DEST / "video_manifest.json")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
