"""Measured trajectory references for physics reconstruction, without drivers."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from phantom.sim.kinematics import JOINT_NAMES


class MeasuredArmVelocityReference:
    """Pair measured qd with measured q in the reconstruction's force drive.

    A zero velocity reference damps the intended motion as well as tracking
    error, adding approximately D/K seconds of lag. This reference cancels
    that term without changing gains, force limits, or measured positions.
    It is an offline trajectory-tracking treatment, not recorded motor effort
    or a reconstruction of the real servo controller.
    """

    def __init__(self, data, *, mode, duration):
        if mode != "dynamics":
            raise ValueError("Measured arm velocity references require dynamics mode")
        required = ("native_arm_qd_t", "native_arm_qd", "joint_names")
        if any(key not in data for key in required):
            raise ValueError("Measured velocity replay requires native arm_qd and joint_names")
        if list(data["joint_names"]) != list(JOINT_NAMES):
            raise ValueError("Measured velocity joint ordering differs from the arm")
        self.t = np.asarray(data["native_arm_qd_t"], dtype=float)
        self.qd = np.asarray(data["native_arm_qd"], dtype=float)
        self.duration = float(duration)
        if (self.t.ndim != 1 or len(self.t) < 2
                or self.qd.shape != (len(self.t), 6)
                or not np.isfinite(self.t).all() or not np.isfinite(self.qd).all()
                or np.any(np.diff(self.t) <= 0)
                or not np.isfinite(self.duration) or self.duration <= 0):
            raise ValueError("Measured velocity references need finite increasing native timestamps and qd[N,6]")
        if self.t[0] > 1e-9 or self.t[-1] < self.duration - 1e-9:
            raise ValueError("Native arm_qd must cover the full replay interval")
        self.metadata = {
            "source": "native_arm_qd", "units": "rad/s",
            "joint_names": list(JOINT_NAMES), "samples": len(self.t),
            "reference_interval_s": [float(self.t[0]), float(self.t[-1])],
            "interpolation": "linear offline measured-state reconstruction; no endpoint extrapolation",
            "controller": "Unchanged force-drive K/D/limits, q target plus measured qd velocity target; all finger velocity targets stay zero",
            "scope": "Dynamics reconstruction only; not a policy setting or recorded hardware effort",
        }

    def at(self, t):
        if not np.isfinite(t) or t < -1e-9 or t > self.duration + 1e-9:
            raise ValueError("Measured velocity query is outside the replay interval")
        return np.array([np.interp(t, self.t, self.qd[:, j]) for j in range(6)])


class SampledGripperCommands:
    """Causal hold of explicitly audited teleop last-sent command samples.

    The collector's actions_abs column is a sampled command echo, not an
    exact timestamped socket-write log. Keep that limitation and its binding
    to the measured episode visible; never substitute deployment proposals.
    """

    def __init__(self, path, *, reference_path, reference_t0, initial_closure, mode):
        if mode != "dynamics":
            raise ValueError("Sampled gripper command replay requires dynamics mode")
        payload = Path(path).read_bytes()
        record = json.loads(payload)
        if (record.get("format") != "phantom_teleop_gripper_commands_v1"
                or record.get("acquisition") != "teleop"
                or record.get("semantics") != "sampled_last_sent_gripper_target"):
            raise ValueError("Gripper sidecar requires audited teleop last-sent command provenance")
        source = record.get("source_episode")
        meta_hash = record.get("source_meta_sha256")
        limitations = record.get("sampling_limitations")
        if (not isinstance(source, str) or not source.strip()
                or not isinstance(meta_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", meta_hash) is None
                or not isinstance(limitations, list) or not limitations
                or any(not isinstance(item, str) or not item.strip() for item in limitations)):
            raise ValueError("Gripper sidecar requires source_episode, valid source_meta_sha256 and nonempty sampling_limitations")
        reference_hash = hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()
        if record.get("reference_sha256") != reference_hash:
            raise ValueError("Gripper sidecar is bound to a different measured replay")
        origin = float(record["t0_master"])
        if not np.isfinite(origin) or not np.isclose(origin, reference_t0, rtol=0, atol=1e-7):
            raise ValueError("Gripper command time origin differs from measured replay")
        self.t = np.asarray(record["timestamps_s"], float)
        requested = np.asarray(record["requested_closure"], float)
        self.initial = float(initial_closure)
        if (self.t.ndim != 1 or len(self.t) < 2 or requested.shape != self.t.shape
                or not np.isfinite(self.t).all() or not np.isfinite(requested).all()
                or np.any(np.diff(self.t) <= 0) or np.any((requested < 0) | (requested > 1))
                or not np.isfinite(self.initial) or not 0 <= self.initial <= 1):
            raise ValueError("Gripper commands need finite increasing timestamps and normalized positions")
        self.command = np.rint(requested * 255.) / 255.
        self.metadata = {
            **{key: value for key, value in record.items()
               if key not in ("timestamps_s", "requested_closure")},
            "sidecar_path": str(Path(path).resolve()),
            "sidecar_sha256": hashlib.sha256(payload).hexdigest(),
            "samples": len(self.t),
            "first_last_s": [float(self.t[0]), float(self.t[-1])],
            "max_sample_gap_s": float(np.diff(self.t).max()),
            "quantization": "Nearest integer 0..255, matching the Robotiq driver round(position*255)",
            "hold": "Last sample at or before t; initial measured closure before first command; last request persists after final sample",
            "scope": "Sampled teleop command diagnostic; missing intermediate sends and exact hardware send times are unknown",
        }

    def at(self, t):
        if not np.isfinite(t) or t < 0:
            raise ValueError("Gripper replay time must be finite and nonnegative")
        index = int(np.searchsorted(self.t, t, side="right")) - 1
        return self.initial if index < 0 else float(self.command[index])
