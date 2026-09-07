#!/usr/bin/env python3
"""Isolated first-plan RGB swap; prepare on CPU, execute six calls only after study.

This never launches a server, physics or hardware. The caller supplies a newly
owned dedicated policy server after both primary study stages have finished.
Only the RGB array differs; proposals are not filtered or scored as task success.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

remote_module = importlib.import_module("phantom.sim.remote_policy")
OBS_FIELDS = remote_module.OBS_FIELDS
RemoteSimulationPolicy = remote_module.RemoteSimulationPolicy

CHECKPOINT_SHA = "67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e"
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
SEEDS = (903101, 903102)
PROC_ROOT = Path("/proc")


def probe_server_info(address, timeout_s=6.0):
    """Bounded, info-only native protocol request; never claims ownership."""
    from multiprocessing.connection import Client

    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Probe timeout must be finite and positive")
    box, ready, abandoned = {}, threading.Event(), threading.Event()
    lock = threading.Lock()

    def request():
        connection = None
        try:
            connection = Client(address, authkey=remote_module.AUTHKEY)
            with lock:
                if abandoned.is_set():
                    return
                box["connection"] = connection
            connection.send(("info",))
            if not connection.poll(timeout_s):
                raise TimeoutError("Dedicated policy info response timed out")
            status, info = connection.recv()
            if status != "ok":
                raise RuntimeError(f"Dedicated policy info refused: {info}")
            box["info"] = info
        except Exception as exc:  # noqa: BLE001 -- relay bounded worker failures
            box["error"] = exc
        finally:
            if connection is not None:
                connection.close()
            ready.set()

    threading.Thread(target=request, daemon=True).start()
    if not ready.wait(timeout_s):
        with lock:
            abandoned.set()
            if "connection" in box:
                box["connection"].close()
        raise TimeoutError("Dedicated policy info connection/response timed out")
    if "error" in box:
        raise box["error"]
    return box["info"]


@contextmanager
def acquire_diagnostic_policy(ready, out):
    """Reject foreign ownership before attach; native configure guards races."""
    address = ("127.0.0.1", ready["port"])
    info = probe_server_info(address)
    write_json(out / "preconnection_info.json", info)
    if info.get("busy") is not False or info.get("owner") is not None:
        raise RuntimeError("Dedicated server already has an active inference owner")
    if info.get("ckpt_sha") != CHECKPOINT_SHA:
        raise ValueError("Probed checkpoint differs from dedicated ready marker")
    # The constructor sends configure({}), which claims ownership without
    # changing settings. A competitor winning after the probe is refused by
    # the native server; do not suppress that exception or try another model.
    with RemoteSimulationPolicy(address) as policy:
        if policy.info.get("busy") is not True or policy.info.get("owner") is None:
            raise RuntimeError("Native server did not acknowledge owned connection")
        if policy.info.get("ckpt_sha") != CHECKPOINT_SHA or any(
            policy.info.get("effective", {}).get(k) != v for k, v in SETTINGS.items()
        ):
            raise ValueError("Live server identity/settings differ from ready marker")
        write_json(out / "live_server_info.json", policy.info)
        write_json(
            out / "ownership.json",
            {
                "preconnection_probe": "info only; idle",
                "native_configure_acknowledged": True,
                "owner_connection_id": policy.info["owner"],
                "busy_after_attach": True,
                "semantics": "busy means this connection owns the server after successful configure({}); competing claims are refused natively",
            },
        )
        yield policy


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    value = np.asarray(value)
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_native(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(v) for v in value]
    return value


def select_camera_index(camera_times, camera_target, request_time):
    """Closest frame to source camera time, constrained to precede the request."""
    ts = np.asarray(camera_times, float)
    if (
        ts.ndim != 1
        or not len(ts)
        or not np.isfinite(ts).all()
        or np.any(np.diff(ts) <= 0)
    ):
        raise ValueError("Camera timestamps must be finite and strictly increasing")
    available = np.flatnonzero(ts <= request_time)
    if not len(available):
        raise ValueError("No causal real camera frame for the request")
    return int(available[np.argmin(np.abs(ts[available] - camera_target))])


def replace_rgb(original, rgb):
    if set(original) - set(OBS_FIELDS):
        raise ValueError("Unexpected saved observation fields")
    missing = set(OBS_FIELDS) - set(original)
    if missing:
        raise ValueError(
            f"Exact full first snapshot required, missing {sorted(missing)}"
        )
    rgb = np.asarray(rgb)
    if rgb.shape != original["rgb"].shape or rgb.dtype != original["rgb"].dtype:
        raise ValueError("RGB replacement must preserve shape and dtype")
    swapped = {key: np.array(value, copy=True) for key, value in original.items()}
    swapped["rgb"] = rgb.copy()
    assert all(
        array_sha(original[k]) == array_sha(swapped[k]) for k in original if k != "rgb"
    )
    return swapped


def native_observation(arrays):
    """Restore snapshot wire types lost by np.savez's scalar array coercion.

    SimulationObservation.t and derived.reactive_score are Python floats;
    the remaining seven fields here are NumPy arrays in this saved
    full teacher request. Floating scalar values are preserved exactly.
    """
    values = {key: np.array(value, copy=True) for key, value in arrays.items()}
    for key in ("t", "reactive"):
        value = values[key]
        if value.shape != () or not np.isfinite(value):
            raise ValueError(f"Saved {key} must be one finite scalar")
        values[key] = float(value)
    return SimpleNamespace(**values)


def same_clock_arm_indices(q_times, tcp_times, target):
    """Join the selected native samples by time, not full-stream row number."""
    q_t, tcp_t = np.asarray(q_times, float), np.asarray(tcp_times, float)
    q_i = int(np.searchsorted(q_t, target, side="right") - 1)
    tcp_i = int(np.searchsorted(tcp_t, target, side="right") - 1)
    if q_i < 0 or tcp_i < 0 or q_t[q_i] != tcp_t[tcp_i]:
        raise ValueError("Selected measured arm q/TCP must share exact timestamps")
    return q_i, tcp_i


def prepare(observation, initial_state, episode, out):
    from scipy.spatial.transform import Rotation

    from phantom.data.episode_store import EpisodeReader

    meta_path = observation.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    if meta.get("replan_id") != 0:
        raise ValueError(
            "Diagnostic is limited to the saved first actual model request"
        )
    initial = json.loads(initial_state.read_text())
    with np.load(observation, allow_pickle=False) as source:
        arrays = {k: source[k].copy() for k in source.files}
    reader = EpisodeReader(episode)
    t0, request = float(initial["t_master"]), float(arrays["t"])
    camera_target = t0 + float(meta["sensor_times"]["camera_scene"])
    camera_times = reader.ts("camera_scene_color")
    index = select_camera_index(camera_times, camera_target, t0 + request)
    camera_t = float(camera_times[index])
    rgb = np.asarray(reader.data("camera_scene_color")[index])
    swapped = replace_rgb(arrays, rgb)
    case = observation.parent.parent
    original_server = json.loads((case / "server_ready.json").read_text())
    if original_server.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("Saved request must be from the declared ftA1500 checkpoint")
    arm_t = reader.ts("arm_q")
    arm_i, tcp_i = same_clock_arm_indices(arm_t, reader.ts("arm_tcp_pose"), camera_t)
    q = np.asarray(reader.data("arm_q")[arm_i])
    tcp = np.asarray(reader.data("arm_tcp_pose")[tcp_i])
    saved_tcp = arrays["ur_state"][12:18]
    non_rgb = {k: array_sha(v) for k, v in arrays.items() if k != "rgb"}
    manifest = {
        "schema_version": 1,
        "kind": "RGB-only first-plan diagnostic, not closed-loop task scoring",
        "source_observation": str(observation),
        "source_observation_sha256": file_sha(observation),
        "source_observation_metadata_sha256": file_sha(meta_path),
        "source_initial_state": str(initial_state),
        "source_initial_state_sha256": file_sha(initial_state),
        "source_episode": str(episode),
        "source_episode_meta_sha256": file_sha(episode / "meta.json"),
        "original_server": original_server,
        "original_server_file_sha256": file_sha(case / "server_ready.json"),
        "source_plan_path": str(case / "planner_trace.json"),
        "source_plan_sha256": file_sha(case / "planner_trace.json"),
        "source_case_manifest": json.loads((case / "case.json").read_text()),
        "source_case_manifest_sha256": file_sha(case / "case.json"),
        "diagnostic_entrypoint_sha256": file_sha(Path(__file__)),
        "remote_client_source": inspect.getfile(RemoteSimulationPolicy),
        "remote_client_source_sha256": file_sha(
            inspect.getfile(RemoteSimulationPolicy)
        ),
        "source_sensor_times_preserved": meta["sensor_times"],
        "request_t_s": request,
        "real_camera_index": index,
        "real_camera_t_master": camera_t,
        "native_initial_t_master": t0,
        "camera_target_t_master": camera_target,
        "signed_real_camera_snap_error_s": camera_t - camera_target,
        "real_frame_age_at_request_s": t0 + request - camera_t,
        "real_arm_sample_t_master": float(arm_t[arm_i]),
        "real_arm_max_q_change_from_initial_rad": float(
            np.max(np.abs(q - initial["q"]))
        ),
        "real_tcp_change_from_initial_m": float(
            np.linalg.norm(tcp[:3] - np.asarray(initial["measured_tcp_pose"])[:3])
        ),
        "real_image_pose_vs_saved_sim_translation_m": float(
            np.linalg.norm(tcp[:3] - saved_tcp[:3])
        ),
        "real_image_pose_vs_saved_sim_rotation_rad": float(
            (
                Rotation.from_rotvec(tcp[3:]).inv()
                * Rotation.from_rotvec(saved_tcp[3:])
            ).magnitude()
        ),
        "non_rgb_array_hashes": non_rgb,
        "native_wire_field_contract": {
            key: {
                "saved_dtype": str(value.dtype),
                "saved_shape": list(value.shape),
                "wire_type": "float" if key in ("t", "reactive") else "numpy.ndarray",
            }
            for key, value in arrays.items()
        },
        "scalar_reconstruction": "np.savez converts original Python floats t/reactive into zero-dimensional arrays; restore Python float values without modifying saved arrays or numerical values",
        "original_rgb_sha256": array_sha(arrays["rgb"]),
        "real_rgb_sha256": array_sha(rgb),
        "rgb_mae_0_255": float(
            np.mean(np.abs(arrays["rgb"].astype(float) - rgb.astype(float)))
        ),
        "rgb_mae_semantics": "Pixel appearance difference, not geometric calibration error",
        "seeds": list(SEEDS),
        "maximum_replan_calls": 6,
        "order": [
            "seed903101 rendered, real_rgb, rendered_repeat",
            "seed903102 real_rgb, rendered, rendered_repeat",
        ],
        "settings": SETTINGS,
        "limitations": [
            "One saved request from the halted original study, including its original non-RGB proxy/state limitations.",
            "Real image includes small arm motion and estimated pose/camera mismatch; not a calibrated optical intervention.",
            "Only raw RGB changes; teacher predictions can change jointly through video/contact/action heads.",
            "Paired proposed chunks are not executed, safety-filtered, or scored as physical success.",
            "Two seeds and repeat controls characterize this input only; no hardware causality or model selection claim.",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "rendered_snapshot.npz", **arrays)
    np.savez_compressed(out / "real_rgb_snapshot.npz", **swapped)
    manifest["prepared_snapshot_file_sha256"] = {
        p.name: file_sha(p) for p in out.glob("*.npz")
    }
    write_json(out / "manifest.json", manifest)
    return arrays, swapped, manifest


def validate_study_complete(root):
    evidence = {}
    for stage, count in (("screen", 32), ("confirmation", 24)):
        path = root / stage / "selection/selection.json"
        data = json.loads(path.read_text())
        if (
            data.get("status") != "complete_valid_matched_stage"
            or data.get("stage") != stage
            or data.get("expected_trials") != count
            or data.get("available_trials") != count
            or data.get("missing_trial_keys")
            or data.get("invalid_trial_keys")
        ):
            raise ValueError(
                "Both corrected study stages must be complete and valid before inference"
            )
        progress_path = root / stage / "progress.json"
        progress = json.loads(progress_path.read_text())
        if progress.get("status") != "all_trials_completed":
            raise ValueError("Primary controller has not completed")
        pid = progress.get("controller_pid")
        if not isinstance(pid, int) or (PROC_ROOT / str(pid)).exists():
            raise ValueError(
                "Primary controller must have exited before diagnostic inference"
            )
        evidence[stage] = {
            "selection_sha256": file_sha(path),
            "progress_sha256": file_sha(progress_path),
        }
    for path in root.glob("**/servers/**/server.json"):
        pid = json.loads(path.read_text()).get("pid")
        if isinstance(pid, int) and (PROC_ROOT / str(pid)).exists():
            raise ValueError(f"Primary model server still alive: {path}")
    return evidence


def validate_owned_server(path, pid, original):
    ready = json.loads(path.read_text())
    if (
        ready.get("dedicated_simulation_server") is not True
        or ready.get("status") != "ready"
        or ready.get("pid") != pid
        or not (PROC_ROOT / str(pid)).exists()
    ):
        raise ValueError(
            "A live newly owned dedicated server ready marker and explicit PID are required"
        )
    port = ready.get("port")
    if not isinstance(port, int) or port < 7784 or port > 65535:
        raise ValueError("Dedicated non-rig port required")
    argv = (PROC_ROOT / str(pid) / "cmdline").read_bytes().split(b"\0")
    argv = [x.decode() for x in argv if x]
    if not any(x.endswith("/tools/sim/policy_server.py") for x in argv):
        raise ValueError("PID is not the dedicated simulation policy server helper")
    if (
        "--out" not in argv
        or argv.index("--out") + 1 >= len(argv)
        or Path(argv[argv.index("--out") + 1]).resolve() != path.parent.resolve()
    ):
        raise ValueError("Ready marker does not belong to supplied server process")
    if (
        ready.get("system") != "teacher"
        or ready.get("weights") != "EMA"
        or ready.get("checkpoint_sha256") != CHECKPOINT_SHA
    ):
        raise ValueError("Teacher checkpoint/EMA mismatch")
    for key in (
        "normalizers_sha256",
        "backbone_sha256",
        "text_cache_sha256",
        "hardware_sha256",
        "inference_source_sha256",
    ):
        if not ready.get(key) or ready.get(key) != original.get(key):
            raise ValueError(f"Dedicated server differs from saved request in {key}")
    if any(ready.get("effective", {}).get(k) != v for k, v in SETTINGS.items()):
        raise ValueError(
            "Dedicated server must already have exact declared inference settings"
        )
    return ready


def proposal(plan):
    actions = np.asarray(plan.actions)
    pose = np.asarray(plan.t0_pose)
    if (
        actions.shape != (16, 7)
        or pose.shape != (6,)
        or not np.isfinite(actions).all()
        or not np.isfinite(pose).all()
    ):
        raise ValueError("Expected finite sixteen-step denormalized delta proposal")
    return json_native(
        {
            "t_created": plan.t_created,
            "t0_pose": pose,
            "actions": actions,
            "action_times": plan.action_times,
            "latency_s": plan.latency_s,
            "sigma": plan.sigma,
            "gate": plan.gate,
            "p_evt": plan.p_evt,
            "diag": plan.diag,
            "endpoint_10step_pose": pose + actions[:10, :6].sum(axis=0),
            "endpoint_16step_pose": pose + actions[:, :6].sum(axis=0),
            "closure_min_max": [float(actions[:, 6].min()), float(actions[:, 6].max())],
            "semantics": "Native cumulative per-step xyz/rotvec deltas, no extra dt multiplication; proposals before governor, veto, IK, contacts or closed-loop execution.",
        }
    )


def compare_proposals(a, b):
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
        "k_pick_a": a["diag"].get("k_pick"),
        "k_pick_b": b["diag"].get("k_pick"),
    }


def paired_calls(policy, rendered, real_rgb, persist=None):
    """Exactly six independent first replans; reset seed before every request."""
    source_hashes = {k: array_sha(v) for k, v in rendered.items()}
    replace_rgb(rendered, real_rgb["rgb"])
    if any(
        array_sha(rendered[k]) != array_sha(real_rgb[k]) for k in rendered if k != "rgb"
    ):
        raise ValueError("Non-RGB input changed")
    outputs = []
    for i, seed in enumerate(SEEDS):
        order = (
            ("rendered", "real_rgb", "rendered_repeat")
            if i == 0
            else ("real_rgb", "rendered", "rendered_repeat")
        )
        for variant in order:
            arrays = real_rgb if variant == "real_rgb" else rendered
            obs = native_observation(arrays)
            policy.remote_reset(seed)
            plan = policy.replan(
                obs, None, np.asarray(rendered["ur_state"][12:18], float)
            )
            row = {"seed": seed, "variant": variant, "proposal": proposal(plan)}
            outputs.append(row)
            if persist:
                persist(row)
    assert len(outputs) == 6
    assert source_hashes == {k: array_sha(v) for k, v in rendered.items()}
    comparisons = []
    for seed in SEEDS:
        group = {r["variant"]: r["proposal"] for r in outputs if r["seed"] == seed}
        comparisons.append(
            {
                "seed": seed,
                "rgb_swap": compare_proposals(group["rendered"], group["real_rgb"]),
                "rendered_repeat_control": compare_proposals(
                    group["rendered"], group["rendered_repeat"]
                ),
            }
        )
    return outputs, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow six first-plan requests after all completion/ownership guards",
    )
    parser.add_argument("--completed-study", type=Path)
    parser.add_argument("--server-ready", type=Path)
    parser.add_argument("--owned-server-pid", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(
            "Use a new diagnostic output directory; never overwrite evidence or retry silently"
        )
    for protected in (args.episode, args.observation.parent.parent, REPO):
        if args.out.resolve().is_relative_to(protected.resolve()):
            raise ValueError(
                "Diagnostic outputs must be outside source code, recordings and the source trial"
            )
    evidence, ready = None, None
    if args.execute:
        if (
            not args.completed_study
            or not args.server_ready
            or not args.owned_server_pid
        ):
            parser.error(
                "Execution requires completed-study, server-ready and owned-server-pid"
            )
        evidence = validate_study_complete(args.completed_study)
        original = json.loads(
            (args.observation.parent.parent / "server_ready.json").read_text()
        )
        ready = validate_owned_server(
            args.server_ready, args.owned_server_pid, original
        )
        if args.server_ready.resolve().is_relative_to(args.completed_study.resolve()):
            raise ValueError(
                "Diagnostic server must be outside the completed primary study"
            )
        if args.out.resolve().is_relative_to(args.completed_study.resolve()):
            raise ValueError(
                "Diagnostic outputs must be outside the completed primary study"
            )
    rendered, real_rgb, manifest = prepare(
        args.observation, args.initial_state, args.episode, args.out
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "prepared_only_no_server_contact",
                    "manifest": str(args.out / "manifest.json"),
                }
            )
        )
        return
    write_json(args.out / "dedicated_server_ready.json", ready)
    write_json(args.out / "completed_study_guard.json", evidence)
    with acquire_diagnostic_policy(ready, args.out) as policy:
        rows, comparisons = paired_calls(
            policy,
            rendered,
            real_rgb,
            lambda r: write_json(args.out / f"seed{r['seed']}_{r['variant']}.json", r),
        )
    write_json(
        args.out / "results.json",
        {
            "status": "six_native_first_plan_proposals_completed",
            "calls": len(rows),
            "comparisons": comparisons,
            "manifest_sha256": file_sha(args.out / "manifest.json"),
            "limitations": manifest["limitations"],
            "closed_loop_task_score": None,
        },
    )
    print(json.dumps({"status": "completed", "calls": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
