#!/usr/bin/env python3
"""Sequential CPU-only presentation rendering; never writes raw trial folders."""

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
ROOT = BASE / "runs/teacher_success_anchor_v3"
HELPER = BASE / "source_teacher_anchor_driver_v3/tools/sim/make_policy_video.py"
HELPER_SHA = "e4e51ecb40ac12ff4dc2664c65a33a9a5a1560d5080bf44fb1a22fef95bef054"
DEST = ROOT / "video_reviews"
MANIFEST = DEST / "component_video_manifest.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_record(a):
    a = np.ascontiguousarray(a)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256_array_bytes": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def read(path):
    return json.loads(path.read_text())


def main():
    assert sha(HELPER) == HELPER_SHA, "Presentation helper drift"
    cv2.setNumThreads(1)
    affinity = sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0, affinity)
    os.nice(19)
    DEST.mkdir(parents=True, exist_ok=True)
    jobs, cases = [], {}
    for progress_path in sorted((ROOT / "components").glob("*/progress.json")):
        group = progress_path.parent
        progress = read(progress_path)
        assert progress["status"] == "all_trials_completed"
        for name, trial in sorted(progress["trials"].items()):
            assert trial["status"] == "completed" and trial["exit_code"] == 0
            folder = group / "rollouts" / name
            score_path = group / "analysis/trials" / (name + ".json")
            score = read(score_path)
            assert score["metrics"]["valid_for_scoring"]
            source = Path(trial["command"][0]).parents[2]
            provenance = {
                "run": str(folder),
                "variant": group.name,
                "case_id": name,
                "score_file": str(score_path),
                "score_sha256": sha(score_path),
                "runtime_source": str(source),
                "runtime_metrics_sha256": sha(source / "phantom/sim/policy_metrics.py"),
                "runtime_hardware_sha256": sha(source / "configs/hardware.nuc.yaml"),
                "raw_input_sha256": {
                    f: sha(folder / f)
                    for f in (
                        "sim_trace.npz",
                        "sim.mp4",
                        "execution_trace.jsonl",
                        "planner_trace.json",
                        "policy_tactile.npz",
                        "effective_config.json",
                        "case.json",
                    )
                },
            }
            with np.load(folder / "sim_trace.npz") as z:
                provenance["physics_force_arrays"] = {
                    k: array_record(z[k])
                    for k in (
                        "t",
                        "pad_force",
                        "pad_packet_normal_force",
                        "packet_robot_normal_force",
                        "packet_bin_normal_force",
                    )
                }
            with np.load(folder / "policy_tactile.npz") as z:
                provenance["runtime_tactile_arrays"] = {
                    k: array_record(z[k])
                    for k in ("t", "gel_normal_force", "pad_force")
                    if k in z
                }
            cases[(group.name, name)] = provenance
            jobs.append(
                {
                    "name": group.name + "/" + name,
                    "runs": [provenance],
                    "output": DEST / group.name / name / "policy_review.mp4",
                    "horizon": None,
                }
            )
    assert len(cases) == 12
    for name, members in (
        (
            "baseline_vs_gel_v2_seed904301",
            [
                ("baseline", "teacher__fixed_anchor__seed904301"),
                ("gel_v2_only", "teacher__fixed_anchor__seed904301"),
            ],
        ),
        (
            "gel_v2_both_seeds",
            [
                ("gel_v2_only", "teacher__fixed_anchor__seed904301"),
                ("gel_v2_only", "teacher__fixed_anchor__seed904302"),
            ],
        ),
    ):
        jobs.append(
            {
                "name": name,
                "runs": [cases[k] for k in members],
                "output": DEST / "comparisons" / (name + ".mp4"),
                "horizon": 60.0,
            }
        )
    result = {
        "schema_version": 1,
        "status": "rendering",
        "pid": os.getpid(),
        "cpu_affinity": affinity,
        "niceness": os.nice(0),
        "helper": str(HELPER),
        "helper_sha256": HELPER_SHA,
        "planned_trial_reviews": 12,
        "planned_comparisons": 2,
        "entries": [],
        "scope": "CPU cv2/libx264 presentation only; source and trial folders immutable. Full decode verification, causal frame mappings and force-array fingerprints. Comparisons freeze ended trials explicitly through60s.",
    }
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"pid": os.getpid(), "affinity": affinity, "jobs": len(jobs)}),
        flush=True,
    )
    for job in jobs:
        path = job["output"]
        assert len({r["runtime_metrics_sha256"] for r in job["runs"]}) == 1
        assert len({r["runtime_hardware_sha256"] for r in job["runs"]}) == 1
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            PHANTOM_REVIEW_SOURCE=job["runs"][0]["runtime_source"],
        )
        command = [
            sys.executable,
            str(HELPER),
            "--run",
            *[r["run"] for r in job["runs"]],
            "--label",
            *[r["variant"] + " " + r["case_id"].split("__")[-1] for r in job["runs"]],
            "--output",
            str(path),
        ]
        if job["horizon"] is not None:
            command.extend(["--horizon", str(job["horizon"])])
        if not path.exists():
            subprocess.run(command, env=env, check=True)
        metadata = read(path.with_suffix(".json"))
        assert metadata["presentation_source"]["sha256"] == HELPER_SHA
        expected = math.floor(metadata["horizon_s"] * metadata["fps"]) + 1
        assert metadata["frames"] == expected
        cap, decoded = cv2.VideoCapture(str(path)), 0
        assert cap.isOpened()
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded == 0:
                cv2.imwrite(str(path.with_name(path.stem + "_first.png")), frame)
            decoded += 1
        cap.release()
        assert decoded == expected, (path, decoded, expected)
        for trial, provenance in zip(metadata["trials"], job["runs"]):
            score = read(Path(provenance["score_file"]))["metrics"]
            assert trial["metrics"] == score, (
                "Presentation rescoring differs from frozen score"
            )
            assert (
                trial["tactile_mapping"]["source"]
                == "policy_tactile.npz gel: saved pixels supplied at runtime"
            )
            assert len(trial["frame_mapping"]) == expected
        entry = {
            "name": job["name"],
            "video": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "metadata": str(path.with_suffix(".json")),
            "metadata_bytes": path.with_suffix(".json").stat().st_size,
            "metadata_sha256": sha(path.with_suffix(".json")),
            "decoded_frames": decoded,
            "width": width,
            "height": height,
            "fps": metadata["fps"],
            "horizon_s": metadata["horizon_s"],
            "full_decode_verified": True,
            "frozen_score_exactly_equal": True,
            "command": command,
            "trial_provenance": job["runs"],
        }
        result["entries"].append(entry)
        MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "completed": len(result["entries"]),
                    "name": job["name"],
                    "frames": decoded,
                    "bytes": entry["bytes"],
                }
            ),
            flush=True,
        )
    assert sha(HELPER) == HELPER_SHA
    result["status"] = "complete"
    result["total_video_bytes"] = sum(e["bytes"] for e in result["entries"])
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(MANIFEST),
                "entries": len(result["entries"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
