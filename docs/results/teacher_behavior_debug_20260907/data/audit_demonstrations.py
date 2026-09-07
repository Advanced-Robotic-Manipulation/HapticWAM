#!/usr/bin/env python3
"""Read-only native demo trajectory/tactile audit; run with live repo Python.

No scene/model/hardware construction. Prints JSON; never writes to recordings.
Events below are explicitly TCP/gripper/tactile proxies, not object truth.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

from phantom.config.hardware import load_hardware
from phantom.data.derived import derive_timestep, pose_delta

BASE = Path("/home/physicalai/phantom-icra-2027")
REPO = BASE / "phantom"
DATA = BASE / "data/full/tasks/waffles"
HW = load_hardware(REPO / "configs/hardware.nuc.yaml", quiet=True)
NUMERIC = (
    "arm_q",
    "arm_qd",
    "arm_tcp_pose",
    "gripper",
    "arm_ft",
    "actions",
    "actions_abs",
)
SOURCE_FILES = [
    "phantom/data/derived.py",
    "phantom/data/windows.py",
    "phantom/data/schema.py",
    "phantom/train/common.py",
    "phantom/train/train_teacher.py",
    "phantom/scripts/record_episodes.py",
    "phantom/drivers/record_episodes.py",
    "phantom/drivers/real/dmtac.py",
    "phantom/drivers/base.py",
    "phantom/deploy/safety.py",
    "phantom/deploy/executor.py",
    "configs/hardware.nuc.yaml",
    "tools/rig/MODELS.tsv",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, stream):
    p = path / f"{stream}.zarr"
    if not p.exists():
        return None
    g = zarr.open_group(str(p), mode="r")
    a = np.asarray(g["data"][:])
    t = np.asarray(g["ts"][:], dtype=np.float64) if "ts" in g else None
    return t, a


def stats(a):
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return None
    return {
        "n": len(a),
        "min": np.min(a, axis=0).tolist(),
        "p50": np.median(a, axis=0).tolist(),
        "p95": np.quantile(a, 0.95, axis=0).tolist(),
        "max": np.max(a, axis=0).tolist(),
    }


def radius(q):
    a2, a3, d4 = HW.safety.ur_dh_a2_a3_d4_m
    return np.sqrt(a2 * a2 + a3 * a3 + 2 * a2 * a3 * np.cos(q[:, 2]) + d4 * d4)


def sample(t, a, targets):
    i = np.searchsorted(t, targets, side="right") - 1
    return a[np.clip(i, 0, len(a) - 1)]


def first_hold(t, mask, hold):
    start = None
    for i in range(len(t)):
        if mask[i]:
            if start is None:
                start = i
            if t[i] - t[start] >= hold:
                return float(t[start])
        else:
            start = None
    return None


result = {
    "schema_version": 1,
    "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
    "live_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip(),
    "source_sha256": {x: sha(REPO / x) for x in SOURCE_FILES},
    "checkpoint": {
        "path": str(REPO / "runs/teacher_v5_ftA/teacher_001500.pt"),
        "sha256": sha(REPO / "runs/teacher_v5_ftA/teacher_001500.pt"),
    },
    "phase_method": {
        "time_zero": "first native arm_q timestamp",
        "loaded": "both abs(raw SDK wrench fz)>2.5, sustained .15s on native TCP grid",
        "lift": "TCP z at least30mm above post-contact minimum, sustained .5s",
        "release": "after lift; first measured closure decrease .15 from running post-contact maximum",
        "meaning": "sensor/kinematic proxies, metadata success is not re-adjudicated object outcome",
        "sampling": "latest causal values; native timestamps retained; no RGB or highres tactile images loaded",
    },
    "hardware_derived": HW.derived.model_dump(),
    "hardware_force_units": {
        "wrench_force_to_n": HW.tactile.force_unit_to_N,
        "torque_to_nm": HW.tactile.torque_unit_to_Nm,
        "distributed_force_to_n": HW.tactile.dist_force_unit_to_N,
        "sdk_area_unit": "mm^2, per native TactileFrame.contact_area_mm2",
    },
    "full": [],
    "thin": [],
}
for path in sorted(DATA.iterdir()):
    if not path.is_dir() or not (path / "arm_q.zarr").exists():
        continue
    q = read(path, "arm_q")
    tp = read(path, "arm_tcp_pose")
    grip = read(path, "gripper")
    meta = json.loads((path / "meta.json").read_text())
    row = {
        "episode": path.name,
        "path": str(path),
        "meta_sha256": sha(path / "meta.json"),
        "success_metadata": meta.get("success"),
        "policy": meta.get("policy"),
        "text": meta.get("text"),
        "config_hash": meta.get("config_hash"),
        "tags": meta.get("tags"),
        "samples": len(q[1]),
        "wrist_radius_m": stats(radius(q[1])),
        "wrist_radius_over_468mm_samples": int(np.count_nonzero(radius(q[1]) > 0.468)),
        "elbow_abs_min_rad": float(np.min(np.abs(q[1][:, 2]))),
        "tcp_xyz_range": stats(tp[1][:, :3]),
        "initial_q": q[1][0].tolist(),
        "initial_tcp": tp[1][0].tolist(),
        "initial_gripper": float(grip[1][0, 0]),
    }
    if q[0] is None:
        # Thin inspection mirrors have absent later chunks: Zarr fills them
        # with zeros. Those are not observed trajectory samples.
        for key in (
            "wrist_radius_m",
            "wrist_radius_over_468mm_samples",
            "elbow_abs_min_rad",
            "tcp_xyz_range",
        ):
            row.pop(key)
        row["timestamp_status"] = (
            "missing; first stored row only, no trajectory inference"
        )
        row["storage"] = {}
        for name in ("arm_q", "arm_tcp_pose", "gripper"):
            arr = zarr.open_group(str(path / f"{name}.zarr"), mode="r")["data"]
            row["storage"][name] = {
                "expected_chunks": int(arr.nchunks),
                "initialized_chunks": int(arr.nchunks_initialized),
                "chunks_shape": list(arr.chunks),
                "fill_value": float(arr.fill_value),
            }
        row["historical_training_input_status"] = (
            "not established; incomplete retained inspection mirror, not evidence optimizer saw fill values"
        )
        result["thin"].append(row)
        continue
    streams = {name: read(path, name) for name in NUMERIC}
    streams = {k: v for k, v in streams.items() if v is not None}
    row["streams"] = {
        k: {
            "shape": list(a.shape),
            "t_start_master": float(t[0]),
            "t_end_master": float(t[-1]),
            "median_dt_s": float(np.median(np.diff(t))),
            "p95_dt_s": float(np.quantile(np.diff(t), 0.95)),
            "payload_sha256": hashlib.sha256(t.tobytes() + a.tobytes()).hexdigest(),
        }
        for k, (t, a) in streams.items()
    }
    t0 = q[0][0]
    t, tcp = tp
    row["t0_master"] = float(t0)
    row["duration_s"] = float(q[0][-1] - t0)
    row["max_measured_joint_speed_rad_s"] = float(np.max(np.abs(streams["arm_qd"][1])))
    row["wrist_force_norm_n"] = stats(
        np.linalg.norm(streams["arm_ft"][1][:, :3], axis=1)
    )
    sens = {}
    loaded = np.ones(len(t), dtype=bool)
    for side in ("left", "right"):
        low = {
            kind: read(path, f"tactile_{side}_{kind}")
            for kind in ("area", "wrench", "fields_ds")
        }
        ft, fields = low["fields_ds"]
        ds = [
            derive_timestep(
                frame,
                fields[i - 1] if i else None,
                ft[i] - ft[i - 1] if i else 1 / HW.recording.field_ds_rate_hz,
                HW,
            )
            for i, frame in enumerate(fields)
        ]
        d = {
            "t": ft,
            "mask_frac": np.asarray([x["mask_frac"] for x in ds]),
            "slip": np.asarray([x["slip"] for x in ds]),
            "depth_abs_p95_mm": np.quantile(np.abs(fields[..., 2]), 0.95, axis=(1, 2)),
            "depth_abs_max_mm": np.max(np.abs(fields[..., 2]), axis=(1, 2)),
            "area_mm2": sample(*low["area"], ft),
            "wrench_sdk": sample(*low["wrench"], ft),
        }
        sens[side] = d
        loaded &= np.abs(sample(*low["wrench"], t)[:, 2]) > 2.5
        for kind, (st, arr) in low.items():
            row["streams"][f"tactile_{side}_{kind}"] = {
                "shape": list(arr.shape),
                "t_start_master": float(st[0]),
                "t_end_master": float(st[-1]),
                "payload_sha256": hashlib.sha256(
                    st.tobytes() + arr.tobytes()
                ).hexdigest(),
            }
    tc = first_hold(t, loaded, 0.15)
    g = sample(*grip, t)[:, 0]
    tl = tr = None
    if tc is not None:
        post = t >= tc
        minz = np.minimum.accumulate(np.where(post, tcp[:, 2], np.inf))
        tl = first_hold(t, post & (tcp[:, 2] >= minz + 0.03), 0.5)
        if tl is not None:
            gmax = np.maximum.accumulate(np.where(post, g, -np.inf))
            cand = np.where((t > tl) & (gmax - g >= 0.15))[0]
            tr = float(t[cand[0]]) if len(cand) else None
    events = {"bilateral_load": tc, "tcp_lift_30mm": tl, "gripper_release": tr}
    row["events"] = {}
    for name, et in events.items():
        if et is None:
            row["events"][name] = None
            continue
        vals = {
            k: {
                "value": sample(st, arr, et).tolist(),
                "t_master": float(
                    st[
                        np.clip(
                            np.searchsorted(st, et, side="right") - 1, 0, len(st) - 1
                        )
                    ]
                ),
            }
            for k, (st, arr) in streams.items()
            if k in ("arm_q", "arm_tcp_pose", "gripper", "arm_ft")
        }
        row["events"][name] = {"t_s": float(et - t0), "native_samples": vals}
    phases = {
        "initial_0_1s": (t0, t0 + 1),
        "approach": (t0 + 1, tc),
        "loaded_pre_lift": (tc, tl),
        "lift_carry": (tl, tr),
        "after_release": (tr, t[-1] + 1e-6),
    }
    row["phases"] = {}
    for name, (lo, hi) in phases.items():
        if lo is None or hi is None or hi <= lo:
            row["phases"][name] = None
            continue
        m = (t >= lo) & (t < hi)
        qm = (q[0] >= lo) & (q[0] < hi)
        srow = {
            "interval_s": [float(lo - t0), float(hi - t0)],
            "tcp_pose": stats(tcp[m]),
            "wrist_radius_m": stats(radius(q[1][qm])),
            "min_elbow_abs_rad": float(np.min(np.abs(q[1][qm, 2])))
            if qm.any()
            else None,
        }
        srow["tactile"] = {
            side: {
                k: stats(a[(d["t"] >= lo) & (d["t"] < hi)])
                for k, a in d.items()
                if k != "t"
            }
            for side, d in sens.items()
        }
        row["phases"][name] = srow
    # At actions timestamps use closest measured TCP. This tests action convention
    # only approximately: the recording loop stamps after a state read.
    ta, a = streams["actions"]
    i = np.searchsorted(t, ta)
    i = np.clip(i, 1, len(t) - 1)
    i = np.where(abs(t[i - 1] - ta) < abs(t[i] - ta), i - 1, i)
    deltas = np.stack([pose_delta(tcp[i[j - 1]], tcp[i[j]]) for j in range(1, len(i))])
    err = a[1:, :6] - deltas
    row["action_audit"] = {
        "action_dt_s": stats(np.diff(ta)),
        "xyz_step_norm_m": stats(np.linalg.norm(a[:, :3], axis=1)),
        "rotation_component_step_norm_rad": stats(np.linalg.norm(a[:, 3:6], axis=1)),
        "absolute_closure_range": [float(a[:, 6].min()), float(a[:, 6].max())],
        "consecutive_action_timestamp_tcp_delta_rmse_components": np.sqrt(
            np.mean(err**2, axis=0)
        ).tolist(),
        "interpretation": "nearest measured state timing diagnostic, not exact recording-loop reconstruction",
    }
    # Decimate numeric trajectory to .1s for downstream plots; exact source hashes above.
    idx = np.unique(np.searchsorted(t, np.arange(t[0], t[-1], 0.1)))
    row["trajectory_10hz"] = {
        "t_s": (t[idx] - t0).tolist(),
        "tcp": tcp[idx].tolist(),
        "q": sample(*q, t[idx]).tolist(),
        "closure": g[idx].tolist(),
        "bilateral_fz_gt2p5": loaded[idx].astype(int).tolist(),
    }
    result["full"].append(row)
result["counts"] = {
    "full_timestamped": len(result["full"]),
    "thin_missing_timestamps": len(result["thin"]),
}
print(json.dumps(result, allow_nan=False))
