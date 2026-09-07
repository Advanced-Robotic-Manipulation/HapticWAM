#!/usr/bin/env python3
"""Read-only final byte, causal-mapping and frozen-score-input audit."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(
    "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest_path = ROOT / "component_video_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete" and len(manifest["entries"]) == 14
    assert sha(Path(manifest["helper"])) == manifest["helper_sha256"]
    entries, seen, score_input_checks = [], set(), 0
    for entry in manifest["entries"]:
        video, metadata_path = Path(entry["video"]), Path(entry["metadata"])
        assert sha(video) == entry["sha256"] and video.stat().st_size == entry["bytes"]
        assert (
            sha(metadata_path) == entry["metadata_sha256"]
            and metadata_path.stat().st_size == entry["metadata_bytes"]
        )
        metadata = json.loads(metadata_path.read_text())
        mapping_checks = []
        for trial, provenance in zip(metadata["trials"], entry["trial_provenance"]):
            source = Path(provenance["runtime_source"])
            assert (
                sha(source / "phantom/sim/policy_metrics.py")
                == provenance["runtime_metrics_sha256"]
            )
            assert (
                sha(source / "configs/hardware.nuc.yaml")
                == provenance["runtime_hardware_sha256"]
            )
            score_path = Path(provenance["score_file"])
            assert sha(score_path) == provenance["score_sha256"]
            score = json.loads(score_path.read_text())
            assert trial["metrics"] == score["metrics"]
            if provenance["run"] not in seen:
                for name, expected in score["input_sha256"].items():
                    assert sha(Path(provenance["run"]) / name) == expected
                    score_input_checks += 1
                seen.add(provenance["run"])
            mapping = trial["frame_mapping"]
            assert len(mapping) == entry["decoded_frames"]
            assert np.allclose(
                [r["t"] for r in mapping],
                np.arange(len(mapping)) / metadata["fps"],
                atol=1e-12,
                rtol=0,
            )
            for row in mapping:
                for key in ("scene_age_s", "state_age_s", "runtime_tactile_age_s"):
                    assert row[key] is None or row[key] >= -1e-9, (
                        "Future sample displayed"
                    )
            if metadata["horizon_s"] == 60:
                assert mapping[-1]["frozen"]
                assert "FRAME FROZEN" in mapping[-1]["display_status"]
            mapping_checks.append(
                {
                    "case_id": provenance["case_id"],
                    "variant": provenance["variant"],
                    "causal_ages_verified": True,
                    "last_frame_mapping": mapping[-1],
                    "first_frozen_frame_t_s": next(
                        (r["t"] for r in mapping if r["frozen"]), None
                    ),
                }
            )
        entries.append(
            {
                "name": entry["name"],
                "decoded_frames": entry["decoded_frames"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "mappings": mapping_checks,
            }
        )
    assert len(seen) == 12
    print(
        json.dumps(
            {
                "status": "pass",
                "videos": len(entries),
                "unique_trials": len(seen),
                "comparisons": 2,
                "score_input_files_verified": score_input_checks,
                "total_video_bytes": manifest["total_video_bytes"],
                "manifest_sha256": sha(manifest_path),
                "presentation_helper_sha256": manifest["helper_sha256"],
                "decoding": "Every MP4 was decoded fully in the rendering pass; this final audit rechecks hashes/bytes without redundant decoding.",
                "entries": entries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
