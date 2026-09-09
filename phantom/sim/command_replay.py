"""Causal replay of recorded drive submissions for mechanics diagnostics only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class RecordedDriveCommands:
    """Hold the last submitted joint targets, never interpolate future commands.

    This does not run a policy or re-evaluate safety at a new physics timestep.
    It isolates the contact/drive response to an identical recorded command stream.
    """

    def __init__(self, path, *, finger_limit_m):
        path = Path(path)
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if not rows:
            raise ValueError("Command trace is empty")
        self.t = np.asarray([row["t"] for row in rows], dtype=float)
        self.q = np.asarray([row["target_q"] for row in rows], dtype=float)
        self.fingers = np.asarray([row["target_finger_q"] for row in rows], dtype=float)
        if (
            self.t.shape != (len(rows),)
            or self.q.shape != (len(rows), 6)
            or self.fingers.shape != (len(rows), 2)
            or not all(np.isfinite(x).all() for x in (self.t, self.q, self.fingers))
            or self.t[0] < 0
            or np.any(np.diff(self.t) <= 0)
            or np.any(self.fingers < 0)
            or np.any(self.fingers > finger_limit_m + 1e-9)
        ):
            raise ValueError("Command trace has invalid timestamps or drive targets")
        if any(row.get("status") != "drive_submitted" for row in rows):
            raise ValueError("Replay requires actual drive-submitted records")
        self.metadata = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(rows),
            "first_command_s": float(self.t[0]),
            "last_command_s": float(self.t[-1]),
            "semantics": "Causal zero-order hold of recorded articulation targets; no inference, no object pose replay, no new safety decisions; not a policy score",
        }

    def at(self, t):
        if not np.isfinite(t) or t < 0:
            raise ValueError("Replay time must be finite and nonnegative")
        index = int(np.searchsorted(self.t, t + 1e-10, side="right") - 1)
        if index < 0:
            return None
        return self.q[index].copy(), self.fingers[index].copy()


def native_drive_manifest(path, cfg):
    """Declare named, hash-pinned native drive arrays for a mechanics replay.

    The caller must verify the source runtime's joint ordering before signing
    this declaration. The trace itself stays unchanged. Physics timestep is
    intentionally outside the gripper hash so paired numerical probes can vary it.
    """
    from phantom.sim.gripper_adaptive import JOINT_NAMES as FINGER_NAMES
    from phantom.sim.gripper_adaptive import MODEL
    from phantom.sim.kinematics import JOINT_NAMES as ARM_NAMES

    if cfg["gripper"].get("model") != MODEL:
        raise ValueError("Native command manifest requires the adaptive gripper")
    return {
        "format": "phantom_native_drive_commands_v1",
        "trace_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "gripper_configuration_sha256": hashlib.sha256(
            json.dumps(cfg["gripper"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "gripper_model": MODEL,
        "arm_joint_names": list(ARM_NAMES),
        "finger_joint_names": list(FINGER_NAMES),
        "arm_target_units": "rad",
        "finger_target_units": "rad",
        "finger_target_semantics": "native_motor_and_passive_drive_references",
    }


class NativeRecordedDriveCommands(RecordedDriveCommands):
    """Replay all eight named native drive references, never measured angles.

    Only master/mirrored motor targets vary. Spring preload references and
    undriven follower/coupler placeholders must match the pinned configuration.
    Spring references can exceed physical joint limits because they specify
    spring torque, not a desired passive joint pose. Actual joint guards remain
    the simulator's responsibility on every physics step.
    """

    def __init__(self, path, *, manifest_path, cfg):
        from phantom.sim.gripper_adaptive import drive_targets

        path, manifest_path = Path(path), Path(manifest_path)
        data, manifest_data = path.read_bytes(), manifest_path.read_bytes()
        manifest = json.loads(manifest_data)
        if not isinstance(manifest, dict):
            raise ValueError("Native command manifest must be an object")  # noqa: TRY004 - external file validation
        expected = native_drive_manifest(path, cfg)
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Native command manifest hash, joint ordering, units or semantics mismatch")
        # Detect a concurrent trace change during manifest validation as well.
        if hashlib.sha256(data).hexdigest() != expected["trace_sha256"]:
            raise ValueError("Native command trace changed while loading")
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if not rows:
            raise ValueError("Command trace is empty")
        try:
            self.t = np.asarray([row["t"] for row in rows], dtype=float)
            self.q = np.asarray([row["target_q"] for row in rows], dtype=float)
            self.fingers = np.asarray([row["target_finger_q"] for row in rows], dtype=float)
            closure = np.asarray([row["gripper_command"] for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Native replay requires explicit arm/finger drive targets and closure commands") from error
        if (
            self.t.shape != (len(rows),)
            or self.q.shape != (len(rows), 6)
            or self.fingers.shape != (len(rows), 8)
            or closure.shape != (len(rows),)
            or not all(np.isfinite(x).all() for x in (self.t, self.q, self.fingers, closure))
            or self.t[0] < 0
            or np.any(np.diff(self.t) <= 0)
            or np.any((closure < 0) | (closure > 1))
        ):
            raise ValueError("Native command trace has invalid timestamps or drive targets")
        if any(row.get("status") != "drive_submitted" for row in rows):
            raise ValueError("Replay requires actual drive-submitted records")
        references = np.stack([drive_targets(value, cfg) for value in closure])
        if not np.allclose(self.fingers, references, rtol=0, atol=1e-12):
            raise ValueError("Native finger targets differ from pinned motor/passive drive references")
        self.metadata = {
            **expected,
            "path": str(path.resolve()),
            "sha256": expected["trace_sha256"],
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
            "rows": len(rows),
            "first_command_s": float(self.t[0]),
            "last_command_s": float(self.t[-1]),
            "semantics": "Causal zero-order hold of recorded articulation targets; no inference, no object pose replay, no new safety decisions; not a policy score",
            "after_final_command": "Hold final recorded targets; subsequent physics has no recorded policy or safety decisions",
            "actual_joint_guards": "Unchanged native coupling, joint-limit and loop-closure checks on every physics step",
        }
