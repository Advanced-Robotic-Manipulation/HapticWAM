#!/usr/bin/env python3
"""Read-only post-completion audit, run on compute3 via Python stdin.

Does not contact a policy socket, launch/stop a process, load a model, or write
remote files. It returns small saved JSON artifacts and audit data on stdout.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
OUT = BASE / "runs/teacher_rgb_transfer_v2"
STUDY = BASE / "runs/teacher_robustness_v2_delivery"
SHA = "67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e"
EXPECTED = [
    (903101, "rendered"),
    (903101, "real_rgb"),
    (903101, "rendered_repeat"),
    (903102, "real_rgb"),
    (903102, "rendered"),
    (903102, "rendered_repeat"),
]
SETTINGS = {
    "nfe": 1,
    "guidance": 1.0,
    "k_seeds": 4,
    "parity_fixes": True,
    "persistent_noise": True,
    "task_text": "waffles",
    "drop_video": False,
    "close_p": 0.5,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(value):
    value = np.asarray(value)
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def process(pid):
    path = Path("/proc") / str(pid)
    try:
        command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        return {"pid": pid, "exists": True, "command": command}
    except FileNotFoundError:
        return {"pid": pid, "exists": False, "command": None}


def difference(a, b):
    aa, bb = np.asarray(a["actions"]), np.asarray(b["actions"])
    return {
        "xyz_step_rmse_m": float(np.sqrt(np.mean((aa[:, :3] - bb[:, :3]) ** 2))),
        "rotvec_step_rmse_rad": float(np.sqrt(np.mean((aa[:, 3:6] - bb[:, 3:6]) ** 2))),
        "closure_rmse": float(np.sqrt(np.mean((aa[:, 6] - bb[:, 6]) ** 2))),
        "endpoint_10step_xyz_difference_m": float(
            np.linalg.norm(
                np.asarray(a["endpoint_10step_pose"])[:3]
                - np.asarray(b["endpoint_10step_pose"])[:3]
            )
        ),
        "endpoint_16step_xyz_difference_m": float(
            np.linalg.norm(
                np.asarray(a["endpoint_16step_pose"])[:3]
                - np.asarray(b["endpoint_16step_pose"])[:3]
            )
        ),
        "stored_action_values_equal": bool(np.array_equal(aa, bb)),
        "actions_max_abs_difference": float(np.max(np.abs(aa - bb))),
    }


def audit():
    status = json.loads((OUT / "orchestration/status.json").read_text())
    if status["status"] != "complete":
        return {"status": "not_ready", "orchestration": status}
    ex = OUT / "execution"
    manifest = json.loads((ex / "manifest.json").read_text())
    results = json.loads((ex / "results.json").read_text())
    ready = json.loads((ex / "dedicated_server_ready.json").read_text())
    live = json.loads((ex / "live_server_info.json").read_text())
    original = manifest["original_server"]
    checks = {}
    checks["completed_six_calls"] = status.get("calls") == results.get("calls") == 6
    checks["exact_six_output_names"] = {p.name for p in ex.glob("seed*.json")} == {
        f"seed{s}_{v}.json" for s, v in EXPECTED
    }
    checks["results_manifest_sha"] = results["manifest_sha256"] == digest(
        ex / "manifest.json"
    )
    checks["ema_teacher"] = (
        ready["weights"] == original["weights"] == "EMA"
        and ready["system"] == "teacher"
    )
    checks["checkpoint_identity"] = (
        ready["checkpoint_sha256"]
        == original["checkpoint_sha256"]
        == live["ckpt_sha"]
        == SHA
    )
    checks["settings"] = all(
        ready["effective"].get(k) == live["effective"].get(k) == v
        for k, v in SETTINGS.items()
    )
    identity_keys = (
        "normalizers_sha256",
        "backbone_sha256",
        "text_cache_sha256",
        "hardware_sha256",
        "inference_source_sha256",
    )
    checks["saved_request_loading_identity"] = all(
        ready[k] == original[k] for k in identity_keys
    )
    source_checks = {}
    for relative, expected in ready["inference_source_sha256"].items():
        current = digest(Path(ready["repo"]) / relative)
        source_checks[relative] = {
            "expected": expected,
            "current": current,
            "match": current == expected,
        }
    checks["six_current_native_source_hashes"] = len(source_checks) == 6 and all(
        r["match"] for r in source_checks.values()
    )
    checks["checkpoint_current_file_sha"] = digest(ready["checkpoint"]) == SHA
    checks["hardware_current_file_sha"] = (
        digest(ready["hardware_path"]) == ready["hardware_sha256"]
    )
    checks["client_entrypoint_hash"] = (
        digest(BASE / "review_tools/rgb_transfer_v2/tools/sim/diagnose_rgb_transfer.py")
        == manifest["diagnostic_entrypoint_sha256"]
    )
    checks["client_import_hash"] = (
        digest(manifest["remote_client_source"])
        == manifest["remote_client_source_sha256"]
    )
    checks["source_observation_file_sha"] = (
        digest(manifest["source_observation"]) == manifest["source_observation_sha256"]
    )
    arrays = {}
    with np.load(manifest["source_observation"], allow_pickle=False) as data:
        source = {k: data[k].copy() for k in data.files}
    for name in ("rendered_snapshot.npz", "real_rgb_snapshot.npz"):
        checks[name + "_file_sha"] = (
            digest(ex / name) == manifest["prepared_snapshot_file_sha256"][name]
        )
        with np.load(ex / name, allow_pickle=False) as data:
            arrays[name] = {k: data[k].copy() for k in data.files}
    non_rgb = {}
    for key, value in source.items():
        if key == "rgb":
            continue
        hashes = {
            "saved_request": array_digest(value),
            **{name: array_digest(data[key]) for name, data in arrays.items()},
        }
        non_rgb[key] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "hashes": hashes,
            "equal": len(set(hashes.values())) == 1,
            "manifest_match": hashes["saved_request"]
            == manifest["non_rgb_array_hashes"][key],
        }
    checks["all_eight_non_rgb_fields_identical"] = len(non_rgb) == 8 and all(
        v["equal"] and v["manifest_match"] for v in non_rgb.values()
    )
    checks["source_rendered_rgb_identical"] = (
        array_digest(source["rgb"])
        == array_digest(arrays["rendered_snapshot.npz"]["rgb"])
        == manifest["original_rgb_sha256"]
    )
    checks["real_rgb_hash"] = (
        array_digest(arrays["real_rgb_snapshot.npz"]["rgb"])
        == manifest["real_rgb_sha256"]
    )
    checks["real_rgb_differs"] = (
        manifest["original_rgb_sha256"] != manifest["real_rgb_sha256"]
    )
    rows = [json.loads((ex / f"seed{s}_{v}.json").read_text()) for s, v in EXPECTED]
    by_seed = {
        s: {r["variant"]: r["proposal"] for r in rows if r["seed"] == s}
        for s in (903101, 903102)
    }
    calls = []
    for (seed, variant), row in zip(EXPECTED, rows, strict=True):
        p = row["proposal"]
        a, pose, grid = (
            np.asarray(p[k]) for k in ("actions", "t0_pose", "action_times")
        )
        call_checks = {
            "identity": row["seed"] == seed and row["variant"] == variant,
            "finite_shape": a.shape == (16, 7)
            and pose.shape == (6,)
            and bool(np.isfinite(a).all()),
            "same_tcp": bool(
                np.array_equal(pose, source["ur_state"][12:18].astype(float))
            ),
            "same_request_time": float(p["t_created"]) == float(source["t"]),
            "head10_endpoint": bool(
                np.allclose(
                    p["endpoint_10step_pose"],
                    pose + a[:10, :6].sum(0),
                    rtol=0,
                    atol=1e-6,  # original native accumulation can be float32
                )
            ),
            "full16_endpoint": bool(
                np.allclose(
                    p["endpoint_16step_pose"],
                    pose + a[:, :6].sum(0),
                    rtol=0,
                    atol=1e-6,
                )
            ),
            "action_grid_10hz": grid.shape == (16,)
            and bool(np.allclose(np.diff(grid), 0.1, atol=1e-12)),
        }
        calls.append(
            {
                "seed": seed,
                "variant": variant,
                "checks": call_checks,
                "native_latency_s": p["latency_s"],
                "action_grid_start_s": float(grid[0]),
                "head10_xyz_delta_m": a[:10, :3].sum(0).tolist(),
                "head16_xyz_delta_m": a[:, :3].sum(0).tolist(),
                "head10_mean_closure": float(a[:10, 6].mean()),
                "closure_min_max": p["closure_min_max"],
                "k_pick": p["diag"].get("k_pick"),
            }
        )
    checks["all_call_semantics"] = all(all(r["checks"].values()) for r in calls)
    comparisons = []
    for seed, group in by_seed.items():
        swap = difference(group["rendered"], group["real_rgb"])
        repeat = difference(group["rendered"], group["rendered_repeat"])
        supplied = next(r for r in results["comparisons"] if r["seed"] == seed)
        for label, recomputed in [
            ("rgb_swap", swap),
            ("rendered_repeat_control", repeat),
        ]:
            checks[f"{seed}_{label}_metrics"] = all(
                np.isclose(supplied[label][k], v, atol=1e-12, rtol=0)
                for k, v in recomputed.items()
                if k in supplied[label]
            )
        comparisons.append(
            {
                "seed": seed,
                "rgb_swap": swap,
                "rendered_repeat_control": repeat,
                "swap_exceeds_repeat": {
                    k: swap[k] > repeat[k]
                    for k in (
                        "xyz_step_rmse_m",
                        "closure_rmse",
                        "endpoint_10step_xyz_difference_m",
                    )
                },
            }
        )
    cleanup = {"owned_server": process(status["owned_server_pid"])}
    orchestration = json.loads((OUT / "launcher.json").read_text())
    for k, v in orchestration.items():
        if "pid" in k and isinstance(v, int):
            cleanup[k] = process(v)
    checks["owned_server_exited"] = (
        not cleanup["owned_server"]["exists"]
        and status.get("owned_server_exited") is True
    )
    checks["owned_orchestrator_exited"] = all(
        not row["exists"] for row in cleanup.values()
    )
    before_claim = json.loads((ex / "preconnection_info.json").read_text())
    ownership = json.loads((ex / "ownership.json").read_text())
    checks["native_ownership_acknowledged"] = (
        before_claim.get("busy") is False
        and before_claim.get("owner") is None
        and live.get("busy") is True
        and live.get("owner") == ownership.get("owner_connection_id")
        and ownership.get("native_configure_acknowledged") is True
    )
    listeners = []
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in path.read_text().splitlines()[1:]:
            parts = line.split()
            if int(parts[1].split(":")[-1], 16) == ready["port"] and parts[3] == "0A":
                listeners.append({"table": str(path), "socket_inode": parts[9]})
    checks["owned_port_no_listener"] = not listeners
    stage_evidence = {}
    guard = json.loads((ex / "completed_study_guard.json").read_text())
    for stage, expected in [("screen", 32), ("confirmation", 24)]:
        root = STUDY / stage
        sel = json.loads((root / "selection/selection.json").read_text())
        progress = json.loads((root / "progress.json").read_text())
        stage_evidence[stage] = {
            "selection_status": sel["status"],
            "available_trials": sel["available_trials"],
            "controller": process(progress["controller_pid"]),
            "selection_sha_matches_guard": digest(root / "selection/selection.json")
            == guard[stage]["selection_sha256"],
            "progress_sha_matches_guard": digest(root / "progress.json")
            == guard[stage]["progress_sha256"],
        }
        checks[stage + "_completed_before_calls"] = (
            sel["status"] == "complete_valid_matched_stage"
            and sel["available_trials"] == expected
            and not stage_evidence[stage]["controller"]["exists"]
            and stage_evidence[stage]["selection_sha_matches_guard"]
            and stage_evidence[stage]["progress_sha_matches_guard"]
        )
    paths = [
        OUT / "orchestration/status.json",
        OUT / "launcher.json",
        OUT / "server/command.json",
        OUT / "server/server.json",
        OUT / "orchestration/client_command.json",
        ex / "manifest.json",
        ex / "results.json",
        ex / "dedicated_server_ready.json",
        ex / "live_server_info.json",
        ex / "preconnection_info.json",
        ex / "ownership.json",
        ex / "completed_study_guard.json",
    ] + [ex / f"seed{s}_{v}.json" for s, v in EXPECTED]
    copies = {
        str(p.relative_to(OUT)): {"sha256": digest(p), "text": p.read_text()}
        for p in paths
    }
    result = {
        "status": "passed" if all(checks.values()) else "issues_found",
        "audited_unix_s": time.time(),
        "remote_root": str(OUT),
        "checks": checks,
        "source_checks": source_checks,
        "non_rgb_fields": non_rgb,
        "calls": calls,
        "comparisons": comparisons,
        "cleanup": cleanup,
        "port_listeners": listeners,
        "stage_completion": stage_evidence,
        "identity": {
            k: ready[k]
            for k in (
                "checkpoint_sha256",
                "weights",
                "normalizers_sha256",
                "backbone_sha256",
                "text_cache_sha256",
                "hardware_sha256",
                "effective",
            )
        },
        "limits": manifest["limitations"],
        "backbone_text_normalizer_verification": "compared saved-request and dedicated-server metadata; no model reload",
        "small_artifacts": {
            name: {"sha256": v["sha256"], "bytes": len(v["text"].encode())}
            for name, v in copies.items()
        },
    }
    return {"audit": result, "copy_json": copies}


if __name__ == "__main__":
    print(json.dumps(audit(), allow_nan=False))
