"""Bounded-memory, every-physics-step evidence; never changes control or score."""

from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


class BoundaryPhysicsAudit:
    def __init__(self, guard, dt, config_sha256):
        self.guard = guard
        hw = guard.hw
        self.report = {
            "schema": "boundary_physics_audit_v1", "enabled": True, "finalized": False,
            "config": guard.config.to_dict(), "config_sha256": config_sha256,
            "hardware_effective_sha256": hashlib.sha256(json.dumps(
                hw.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "thresholds": {"hitbox_m": hw.safety.hitbox_m.model_dump(mode="json"),
                           "workspace_m": hw.safety.workspace_m.model_dump(mode="json"),
                           "reach_clamp_m": hw.safety.reach_clamp_m,
                           "wrist_extension_stop_m": hw.safety.wrist_extension_stop_m,
                           "joint_speed_stop_rad_s": hw.safety.joint_speed_stop_rad_s,
                           "selected_ceiling_y_m": guard.plane,
                           "accepted_ceiling_y_m": guard.ceiling},
            "physics_dt_s": float(dt), "samples": 0,
            "first_step": None, "last_step": None, "first_t_s": None, "last_t_s": None,
            "max_gap_s": 0.0, "contiguous": True, "phase_counts": {},
            "violations": {}, "first_violation": None, "worst_violation": None,
            "nonfinite_samples": 0, "per_face": {},
            "measured_y_min_m": None, "measured_y_max_m": None,
            "accepted_fk_y_max_m": None, "selected_y_max_m": None,
            "selected_continuing_y_max_m": None, "selected_terminal_y_max_m": None,
            "selected_by_kind": {},
            "selected_samples": 0,
            "tracking_y_max_positive_m": 0.0, "tracking_y_max_abs_m": 0.0,
            "tracking_xyz_max_m": 0.0, "measured_outward_y_speed_max_m_s": 0.0,
            "meaning": "Read-only measured/FK envelope evidence at every physics tick including terminal hold; no success inference",
        }
        self.counts, self.phases = Counter(), Counter()
        self.selected = None
        self.selected_terminal = False
        self.previous_measurement = None

    def note_selected(self, value, *, kind="continuation"):
        if kind not in ("continuation", "finish_transition", "sealed_terminal_hold"):
            raise ValueError("unknown selected target kind")
        self.selected = None if value is None else np.asarray(value, dtype=float).copy()
        self.selected_terminal = kind != "continuation"
        self.selected_kind = kind

    def _violate(self, name, excess, step, t, measured, accepted, phase):
        self.counts[name] += 1
        sample = {"kind": name, "excess": float(excess), "step": int(step), "t_s": float(t),
                  "measured_tcp": np.asarray(measured).tolist(),
                  "accepted_fk_tcp": np.asarray(accepted).tolist(), "phase": phase}
        if self.report["first_violation"] is None:
            self.report["first_violation"] = sample
        worst = self.report["worst_violation"]
        if worst is None or excess > worst["excess"]:
            self.report["worst_violation"] = sample

    def sample(self, step, t, measured, accepted_fk, q, qd, *, phase):
        r = self.report
        if r["last_step"] is not None:
            r["contiguous"] &= step == r["last_step"] + 1
            gap = t - r["last_t_s"]
            r["max_gap_s"] = max(r["max_gap_s"], float(gap))
            r["contiguous"] &= abs(gap - r["physics_dt_s"]) <= 1e-9
        else:
            r["first_step"], r["first_t_s"] = int(step), float(t)
        r["samples"] += 1
        r["last_step"], r["last_t_s"] = int(step), float(t)
        self.phases[phase] += 1
        measured, accepted_fk, q, qd = (np.asarray(v, dtype=float) for v in (measured, accepted_fk, q, qd))
        if any(v.shape != (6,) or not np.isfinite(v).all() for v in (measured, accepted_fk, q, qd)):
            r["nonfinite_samples"] += 1
            self.counts["nonfinite"] += 1
            return
        for key, value, choose in (("measured_y_min_m", measured[1], min),
                                  ("measured_y_max_m", measured[1], max),
                                  ("accepted_fk_y_max_m", accepted_fk[1], max)):
            r[key] = float(value) if r[key] is None else float(choose(r[key], value))
        residual = measured[:3] - accepted_fk[:3]
        r["tracking_y_max_positive_m"] = max(r["tracking_y_max_positive_m"], float(residual[1]))
        r["tracking_y_max_abs_m"] = max(r["tracking_y_max_abs_m"], float(abs(residual[1])))
        r["tracking_xyz_max_m"] = max(r["tracking_xyz_max_m"], float(np.linalg.norm(residual)))
        if self.previous_measurement is not None and t > self.previous_measurement[0]:
            outward = (measured[1] - self.previous_measurement[1]) / (t - self.previous_measurement[0])
            r["measured_outward_y_speed_max_m_s"] = max(r["measured_outward_y_speed_max_m_s"], float(outward))
        self.previous_measurement = (float(t), float(measured[1]))
        for box_name, box in (("hitbox", self.guard.hb), ("workspace", self.guard.hw.safety.workspace_m)):
            for i, axis in enumerate("xyz"):
                for side, clearance in (("low", measured[i] - getattr(box, axis)[0]),
                                        ("high", getattr(box, axis)[1] - measured[i])):
                    key = f"{box_name}_{axis}_{side}"
                    old = r["per_face"].setdefault(key, {"min_clearance_m": float(clearance), "max_violation_m": 0.0})
                    old["min_clearance_m"] = min(old["min_clearance_m"], float(clearance))
                    old["max_violation_m"] = max(old["max_violation_m"], float(-clearance))
                    if clearance < 0:
                        self._violate(key, -clearance, step, t, measured, accepted_fk, phase)
        sf = self.guard.hw.safety
        for name, measured_value, limit in (
            ("measured_reach", float(np.linalg.norm(measured[:3])), sf.reach_clamp_m),
            ("measured_joint_speed", float(np.max(np.abs(qd))), sf.joint_speed_stop_rad_s),
            ("accepted_fk_y", float(accepted_fk[1]), self.guard.ceiling),
        ):
            if limit is not None and measured_value > limit:
                self._violate(name, measured_value - limit, step, t, measured, accepted_fk, phase)
        a2, a3, d4 = sf.ur_dh_a2_a3_d4_m
        wrist = float(np.sqrt(a2 * a2 + a3 * a3 + 2 * a2 * a3 * np.cos(q[2]) + d4 * d4))
        if sf.wrist_extension_stop_m is not None and wrist > sf.wrist_extension_stop_m:
            self._violate("measured_wrist_extension", wrist - sf.wrist_extension_stop_m,
                          step, t, measured, accepted_fk, phase)
        if self.selected is not None:
            r["selected_samples"] += 1
            if self.selected.shape != (6,) or not np.isfinite(self.selected).all():
                self.counts["nonfinite_selected"] += 1
            else:
                y = float(self.selected[1])
                r["selected_y_max_m"] = y if r["selected_y_max_m"] is None else max(r["selected_y_max_m"], y)
                key = "selected_terminal_y_max_m" if self.selected_terminal else "selected_continuing_y_max_m"
                r[key] = y if r[key] is None else max(r[key], y)
                ceiling = self.guard.ceiling if self.selected_terminal else self.guard.plane
                by_kind = r["selected_by_kind"].setdefault(self.selected_kind, {
                    "samples": 0, "y_max_m": y, "ceiling_y_m": ceiling})
                by_kind["samples"] += 1
                by_kind["y_max_m"] = max(by_kind["y_max_m"], y)
                if y > ceiling:
                    self._violate("selected_y", y - ceiling, step, t, measured, accepted_fk, phase)

    def finalize(self, *, completed, expected_end_s, stopped_reason, completed_reason, completed_at_s):
        self.report.update(finalized=bool(completed), expected_end_s=float(expected_end_s),
                           stopped_reason=stopped_reason, completed_reason=completed_reason,
                           completed_at_s=completed_at_s, violations=dict(self.counts),
                           phase_counts=dict(self.phases))
        return self.report
