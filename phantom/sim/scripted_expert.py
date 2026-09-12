"""Privileged scripted expert for the simulated waffle task (data generation, not a policy).

Produces deployment-frame plans (same fields as ``phantom.inference.policy.Plan``, as a
SimpleNamespace so the Isaac Python, which has no zarr, never imports the data stack: t0_pose + per-step
Δ-EE(6) at the action rate, absolute gripper channel) from the sim's ground-truth packet
pose, so the recorded executed actions live in exactly the frame the teacher is trained
on. The executor path (bounded limiter, rate limits, latch) is unchanged: the expert only
replaces the policy server. Phases: pre-grasp above the packet, slow descent, close, hold,
lift, carry at demo height, descent into the bin, open on the floor band, retreat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from types import SimpleNamespace

import numpy as np

EVENTS = ("none", "onset", "hold", "slip", "release")


@dataclass(frozen=True)
class ExpertParams:
    action_rate_hz: float = 16.0
    horizon: int = 16
    latency_s: float = 0.05
    grip_open: float = 0.15
    grip_close: float = 0.62
    z_pregrasp_m: float = 0.20          # hover height before the descent
    grasp_height_above_center_m: float = 0.035   # TCP z above the packet centre at closure
    z_carry_m: float = 0.33
    z_place_m: float = 0.11
    z_retreat_m: float = 0.25
    v_travel: float = 0.12              # m/s in free space
    v_descend: float = 0.06
    v_contact: float = 0.03             # last 3 cm before contact / floor
    slow_band_m: float = 0.03
    close_s: float = 0.5
    hold_after_close_s: float = 0.4
    open_s: float = 0.4
    hold_after_open_s: float = 0.5
    reach_tol_m: float = 0.006
    place_offset_xy_m: tuple[float, float] = (0.0, -0.015)   # from the bin centre, toward the near wall


PHASES = ("pregrasp", "descend", "close", "settle", "lift", "carry", "place_descend", "open", "retreat", "done")


class ScriptedExpertPolicy:
    """Same surface as RemoteSimulationPolicy for the sim adapter (replan / reset_episode / close)."""

    policy_kind = "scripted"
    wrench_baseline_rows = 0
    k_seeds = 1
    nfe = 0
    guidance = 1.0
    action_time_origin = "inference_ready"

    def __init__(self, packet_pose_fn, bin_center_xy, params: ExpertParams | None = None, *, rng_seed: int = 0):
        self.packet_pose_fn = packet_pose_fn          # () -> (xyz base frame)
        self.bin_center_xy = np.asarray(bin_center_xy, dtype=float)[:2]
        self.p = params or ExpertParams()
        self.info = {"ckpt": "scripted_expert_v1", "ckpt_sha": "scripted", "warmed": True, "student": False,
                     "policy_kind": self.policy_kind, "wrench_baseline_rows": 0,
                     "effective": {"nfe": 0, "guidance": 1.0, "k_seeds": 1, "parity_fixes": False,
                                   "persistent_noise": False, "task_text": "scripted", "drop_video": False,
                                   "close_p": 0.5, "action_time_origin": "inference_ready"},
                     "expert_params": self.p.__dict__}
        self._rng = np.random.default_rng(rng_seed)
        self.reset_episode()

    # -- lifecycle ----------------------------------------------------------
    def reset_episode(self):
        self.phase = "pregrasp"
        self.phase_t0 = None
        self.grasp_xyz = None
        self.events = []

    def remote_reset(self, seed=None):
        self.reset_episode()

    def close(self):
        pass

    # -- helpers ------------------------------------------------------------
    def _event(self, t, name, **kw):
        self.events.append(dict(t=float(t), phase=self.phase, event=name, **kw))

    def _advance(self, t, phase):
        self._event(t, "phase", to=phase)
        self.phase, self.phase_t0 = phase, float(t)

    @staticmethod
    def _p_evt(kind):
        v = np.zeros(len(EVENTS), dtype=np.float32); v[EVENTS.index(kind)] = 1.0
        return v

    def _target(self, tcp, packet):
        """Cartesian target for the current phase; None = hold in place."""
        p = self.p
        if self.phase == "pregrasp":
            return np.array([packet[0], packet[1], max(p.z_pregrasp_m, packet[2] + p.grasp_height_above_center_m + 0.10)])
        if self.phase == "descend":
            return np.array([packet[0], packet[1], packet[2] + p.grasp_height_above_center_m])
        if self.phase in ("close", "settle"):
            return None
        if self.phase == "lift":
            g = self.grasp_xyz
            return np.array([g[0], g[1], p.z_carry_m])
        if self.phase == "carry":
            return np.array([self.bin_center_xy[0] + p.place_offset_xy_m[0], self.bin_center_xy[1] + p.place_offset_xy_m[1], p.z_carry_m])
        if self.phase == "place_descend":
            return np.array([self.bin_center_xy[0] + p.place_offset_xy_m[0], self.bin_center_xy[1] + p.place_offset_xy_m[1], p.z_place_m])
        if self.phase == "open":
            return None
        if self.phase == "retreat":
            return np.array([tcp[0], tcp[1], p.z_retreat_m])
        return None

    def _speed(self, tcp, target):
        p = self.p
        if self.phase in ("descend", "place_descend"):
            return p.v_contact if abs(tcp[2] - target[2]) <= p.slow_band_m else p.v_descend
        return p.v_travel

    def _grip(self, t):
        p = self.p
        el = 0.0 if self.phase_t0 is None else float(t) - self.phase_t0
        if self.phase in ("pregrasp", "descend"):
            return p.grip_open
        if self.phase == "close":
            return p.grip_open + (p.grip_close - p.grip_open) * min(1.0, el / p.close_s)
        if self.phase in ("settle", "lift", "carry", "place_descend"):
            return p.grip_close
        if self.phase == "open":
            return p.grip_close - (p.grip_close - p.grip_open) * min(1.0, el / p.open_s)
        return p.grip_open

    def _step_phase(self, t, tcp, packet):
        """Phase transitions from measured TCP and elapsed time (no object feedback beyond its pose)."""
        p = self.p
        if self.phase_t0 is None:
            self.phase_t0 = float(t)
        el = float(t) - self.phase_t0
        target = self._target(tcp, packet)
        reached = target is not None and np.linalg.norm(tcp[:3] - target) <= p.reach_tol_m
        if self.phase == "pregrasp" and reached:
            self._advance(t, "descend")
        elif self.phase == "descend" and reached:
            self.grasp_xyz = tcp[:3].copy(); self._advance(t, "close")
        elif self.phase == "close" and el >= p.close_s:
            self._advance(t, "settle")
        elif self.phase == "settle" and el >= p.hold_after_close_s:
            self._advance(t, "lift")
        elif self.phase == "lift" and reached:
            self._advance(t, "carry")
        elif self.phase == "carry" and reached:
            self._advance(t, "place_descend")
        elif self.phase == "place_descend" and reached:
            self._advance(t, "open")
        elif self.phase == "open" and el >= p.open_s + p.hold_after_open_s:
            self._advance(t, "retreat")
        elif self.phase == "retreat" and reached:
            self._advance(t, "done")

    # -- the policy call ----------------------------------------------------
    def replan(self, obs, prev_plan, tcp_pose):
        p = self.p
        t = float(obs.t)
        tcp = np.asarray(tcp_pose, dtype=float)
        packet = np.asarray(self.packet_pose_fn(), dtype=float)[:3]
        self._step_phase(t, tcp, packet)
        H, dt = p.horizon, 1.0 / p.action_rate_hz
        actions = np.zeros((H, 7), dtype=np.float32)
        pos = tcp[:3].copy()
        target = self._target(tcp, packet)
        for k in range(H):
            if target is not None:
                delta = target - pos
                dist = float(np.linalg.norm(delta))
                v = self._speed(pos, target)
                step = delta if dist <= v * dt else delta * (v * dt / dist)
                actions[k, :3] = step
                pos = pos + step
            actions[k, 6] = self._grip(t + p.latency_s + k * dt)
        kind = {"close": "onset", "settle": "hold", "lift": "hold", "carry": "hold", "place_descend": "hold",
                "open": "release"}.get(self.phase, "none")
        return SimpleNamespace(t_created=t, t0_pose=tcp.copy(), actions=actions,
                    action_times=t + p.latency_s + np.arange(H) / p.action_rate_hz,
                    sigma=np.zeros(H, dtype=np.float32), gate=1.0, p_evt=self._p_evt(kind), cpk=None,
                    latency_s=p.latency_s, _cpk_token=None,
                    diag={"expert_phase": self.phase, "packet_xyz": packet.tolist(),
                          "target_xyz": None if target is None else [float(v) for v in target]})
