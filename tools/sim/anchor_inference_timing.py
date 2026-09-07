"""Recorded per-request simulator latency override; no model or clock patching."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class RecordedInferenceLatencies:
    """Replay recorded adapter delays by replan ordinal, never silently extend.

    Native inference still computes its actual latency before K-seed selection.
    This controls the simulator delivery/action grid and the latency returned as
    the next previous-plan CPK offset. It is not full native timing replay.
    """

    def __init__(self, path):
        path = Path(path)
        data = path.read_bytes()
        rows = json.loads(data)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Recorded planner trace must be a nonempty list")
        self.rows = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("replan_id") != index:
                raise ValueError("Recorded replan IDs must be contiguous from zero")
            try:
                t, delay = float(row["t"]), float(row["latency_s"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Every recorded inference must have a valid latency"
                ) from error
            if not all(math.isfinite(x) and x >= 0 for x in (t, delay)):
                raise ValueError(
                    "Recorded inference times/delays must be finite and nonnegative"
                )
            if self.rows and t <= self.rows[-1]["request_t_s"]:
                raise ValueError("Recorded requests must have increasing times")
            if row.get("status") in ("inference_error", "inference_started"):
                raise ValueError(
                    "Incomplete/failed inference cannot supply timing truth"
                )
            self.rows.append({"request_t_s": t, "latency_s": delay})
        self.used = []
        self.metadata = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(self.rows),
            "semantics": "Adapter latency_s replay by replan ordinal; action grid and simulated delivery use this value; native K-seed selection still uses freshly measured inference latency; next-plan CPK offset receives the overridden previous latency",
            "exhaustion": "raise before additional inference; no final-value padding",
            "used": self.used,
        }

    def next(self, replan_id, request_t):
        if replan_id != len(self.used):
            raise ValueError(
                "Latency schedule requests must be consumed exactly once in order"
            )
        if not math.isfinite(request_t) or request_t < 0:
            raise ValueError("Request time must be finite and nonnegative")
        if replan_id >= len(self.rows):
            raise RuntimeError("Recorded inference latency schedule exhausted")
        row = self.rows[replan_id]
        selected = {
            "replan_id": replan_id,
            "recorded_request_t_s": row["request_t_s"],
            "actual_request_t_s": request_t,
            "request_time_error_s": request_t - row["request_t_s"],
            "latency_s": row["latency_s"],
        }
        self.used.append(selected)
        return row["latency_s"]
