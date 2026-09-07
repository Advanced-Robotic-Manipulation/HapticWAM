"""Read-only audit of all teacher-v2 tactile videos and a matched montage."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(
    "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_robustness_v2_delivery"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_video(path, expected_sha=None):
    meta = json.loads(path.with_suffix(".json").read_text())
    digest = sha(path)
    if expected_sha is not None:
        assert digest == expected_sha
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_read_frames,duration,r_frame_rate",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    assert probe["codec_name"] == "h264"
    assert int(probe["nb_read_frames"]) == meta["frames"]
    assert meta["presentation_source"]["source_root_override"] == str(
        ROOT.parent.parent / "source_teacher_v2_delivery"
    )
    mappings = 0
    for trial in meta["trials"]:
        tactile = trial["tactile_mapping"]
        assert tactile["actual_input_proxy"] is True
        assert (
            tactile["source"]
            == "policy_tactile.npz gel: saved pixels supplied at runtime"
        )
        assert tactile["calibrated"] is False
        rows = trial["frame_mapping"]
        assert len(rows) == meta["frames"]
        for row in rows:
            for key in ("scene_age_s", "state_age_s", "runtime_tactile_age_s"):
                assert row[key] is None or row[key] >= -1e-8
            if row["controller_completed"]:
                assert trial["controller_completion"] is not None
            mappings += 1
    return {
        "video": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "decoded_frames": int(probe["nb_read_frames"]),
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["r_frame_rate"],
        "physical_horizon_s": meta["horizon_s"],
        "causal_mappings_checked": mappings,
        "metadata_sha256": sha(path.with_suffix(".json")),
    }


def main():
    manifest = json.loads((ROOT / "video_manifest.json").read_text())
    assert manifest["complete"] is True and manifest["study_complete"] is True
    expected = {
        (stage, key)
        for stage in ("screen", "confirmation")
        for key in json.loads((ROOT / stage / "progress.json").read_text())["trials"]
    }
    assert len(expected) == len(manifest["videos"]) == 56
    assert {(row["stage"], row["trial"]) for row in manifest["videos"]} == expected
    videos = []
    for row in manifest["videos"]:
        assert row["exit_code"] == 0
        videos.append(inspect_video(Path(row["video"]), row["sha256"]))
    montage = inspect_video(ROOT / "review_media/paired_pickup_attempts_60s.mp4")
    assert montage["physical_horizon_s"] == 60 and montage["decoded_frames"] == 901
    print(
        json.dumps(
            {
                "status": "passed",
                "single_trial_videos": len(videos),
                "single_trial_bytes": sum(row["bytes"] for row in videos),
                "decoded_frames": sum(row["decoded_frames"] for row in videos),
                "videos": videos,
                "montage": montage,
                "scope": "Full ffprobe frame decoding, H264 identity, file hashes and causal frame maps; no score changes. Selected montage contains both confirmation lift cases and their matched counterparts.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
