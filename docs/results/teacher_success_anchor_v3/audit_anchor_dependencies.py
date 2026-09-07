#!/usr/bin/env python3
"""Read-only original successful-cell launch dependency and source identity audit."""

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_pick_place_v1"
SOURCE = BASE / "source_teacher_pick_place_v1"
CURRENT = BASE / "source_teacher_v2_delivery"
LIVE = Path("/home/physicalai/phantom-icra-2027/phantom")
EXPECTED_INFERENCE = {
    "phantom/inference/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "phantom/inference/policy.py": "e4d4c0797896f72bddb0d3c7383a1ebe6c8aeca02848b4456675c4bf7c35f435",
    "phantom/inference/remote.py": "6b72a5758817600520cc6a1025f6d83ab22b1394af13b65aca8e13dca418a557",
    "phantom/model/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "phantom/model/acc.py": "e817811f0bb9d90d10117f1f87986acbaa20651645a9511555b03c9729aa9254",
    "phantom/model/ace/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "phantom/model/ace/heads.py": "ca68b625a71cfdbd2fc16538e7988bc1609090423226be19e0b8f8afd4b02ab1",
    "phantom/model/ace/losses.py": "8841bb196d5ad24195f9f81bc50515b8f07cd3e137c20809b63f6a2b8e88a6f5",
    "phantom/model/ace/packing.py": "efa6f304f9608204f37f482a9b28b60d59f1d3e2953b788041791cfc9c93df2c",
    "phantom/model/attention_bias.py": "7e2fcce654ba077e521662081ad9045bbb82d8a5a374a0f4d8120bb7aa6ac740",
    "phantom/model/hht/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "phantom/model/hht/encoders.py": "cc0e36b7cd257c51c4504bdd1d93ce292bbafbe55c71527722dd7e9ba1c6db8e",
    "phantom/model/hht/hht.py": "b475785fb7b8715c788468fa5dfc4badd952975447cf40c082713f85ba3a740f",
    "phantom/model/hht/tactile_encoder.py": "8f7dc892e14c49876d08cb2094ec4f912d18cb5c425d82ee8a4f956dfb2f826b",
    "phantom/model/phantom_dit.py": "5640d756ee48818f38a9f32f7bee71c0191e2e7d4587981a0fd03370efded877",
    "phantom/model/rf.py": "674fcb35fc638de700d974ae1e97a068dafe6df01e509f4261e9a5a93ed49d04",
    "phantom/model/sequence.py": "4aedd86a75c42cc3b82d968acb85f9659e1c0e5ea978b404c99808750bbc73d0",
}


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    frozen = read(ROOT / "campaign/frozen_inputs.json")
    case_dir = ROOT / "campaign/rollouts/teacher__placement_xm10_ym10mm__seed4242"
    rows = []
    for group, base in [
        ("source_sha256", SOURCE),
        ("live_core_sha256", LIVE),
        ("episode_sha256", Path(frozen["prepared_episode"])),
    ]:
        for name, expected in frozen[group].items():
            p = base / name
            actual = sha(p) if p.is_file() else None
            rows.append(
                {
                    "path": str(p),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "match": actual == expected,
                    "group": group,
                }
            )
    for name, expected in EXPECTED_INFERENCE.items():
        p = LIVE / name
        actual = sha(p) if p.is_file() else None
        rows.append(
            {
                "path": str(p),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
                "group": "native_inference_manifest",
            }
        )
    for path_key, hash_key in [
        ("hardware_config", "hardware_sha256"),
        ("robot_usd", "robot_usd_sha256"),
    ]:
        p = Path(frozen[path_key])
        actual = sha(p) if p.is_file() else None
        expected = frozen[hash_key]
        rows.append(
            {
                "path": str(p),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
                "group": "runtime_input",
            }
        )
    for spec in frozen["adapter_profile_inputs"].values():
        p = Path(spec["path"])
        actual = sha(p) if p.is_file() else None
        rows.append(
            {
                "path": str(p),
                "expected_sha256": spec["sha256"],
                "actual_sha256": actual,
                "match": actual == spec["sha256"],
                "group": "runtime_input",
            }
        )
    old_command = read(case_dir / "run_status.json")["command"]
    path_flags = [
        "--episode",
        "--config",
        "--policy-config",
        "--hardware-config",
        "--robot-usd",
        "--policy-initial-state",
        "--tactile-baseline",
        "--terminal-veto-config",
        "--placement-release-config",
    ]
    launch = {}
    for flag in path_flags:
        p = Path(old_command[old_command.index(flag) + 1])
        launch[flag] = {
            "path": str(p),
            "exists": p.exists(),
            "sha256": sha(p) if p.is_file() else None,
        }
    condition = read(Path(launch["--config"]["path"]))
    effective = read(case_dir / "effective_config.json")
    assert condition == effective
    candidates = {
        "fta1500": (
            LIVE / "runs/teacher_v5_ftA/teacher_001500.pt",
            "67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e",
        ),
        "fta3000": (
            BASE / "checkpoints/teacher_v5_ftA/teacher_003000.pt",
            "ee0a448c00da2fe047b4bcba5036a8b79e9ae33e3a6ec47ea69d27466b4ff1dc",
        ),
        "v5_6": (
            LIVE / "runs/teacher_v5_batch0822/v5_6.pt",
            "7edcb8335681e19bead5ad39a2a80fe3fd8c5893b1761d2dd812c304881fe65c",
        ),
    }
    checkpoints = {}
    for key, (p, expected) in candidates.items():
        actual = sha(p) if p.is_file() else None
        checkpoints[key] = {
            "path": str(p),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
            "size_bytes": p.stat().st_size if p.exists() else None,
        }
    differences = []
    for name, expected in frozen["source_sha256"].items():
        if name.endswith((".py", ".sh")):
            p = CURRENT / name
            actual = sha(p) if p.is_file() else None
            if actual != expected:
                differences.append(
                    {
                        "relative_path": name,
                        "original_sha256": expected,
                        "v2_delivery_sha256": actual,
                    }
                )
    result = {
        "status": "passed"
        if all(r["match"] for r in rows)
        and all(r["match"] for r in checkpoints.values())
        and all(r["exists"] for r in launch.values())
        else "failed",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "original_case": str(case_dir),
        "original_source": str(SOURCE),
        "current_corrected_source": str(CURRENT),
        "original_launch_command": old_command,
        "source_and_input_checks": rows,
        "checks_count": len(rows),
        "launch_dependencies": launch,
        "checkpoints": checkpoints,
        "original_scene_equals_run_effective": True,
        "original_scene": effective,
        "original_policy_info": {
            k: read(case_dir / "policy_info.json").get(k)
            for k in [
                "effective",
                "wrist_model",
                "gel_contact_coverage",
                "terminal_veto",
                "placement_release",
                "policy_initial_state_provenance",
                "tactile_baseline_provenance",
            ]
        },
        "v2_executable_differences": differences,
        "limitations": [
            "Source/input hash parity does not guarantee bitwise closed-loop repeatability under native latency and nondeterministic physics/rendering.",
            "Cached robot USD root hash is verified; historical dependency closure beyond the original frozen manifest is not reconstructed by this audit.",
            "Reproduction may change only new output path, owned endpoint/PID and explicit episode seed/repeat identifier; all scientific inputs stay pinned.",
        ],
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
