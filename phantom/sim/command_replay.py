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
