"""Privileged scripted expert for the simulated waffle task (data generation, not a policy).

Produces deployment-frame plans (same fields as ``phantom.inference.policy.Plan``, as a
SimpleNamespace so the Isaac Python, which has no zarr, never imports the data stack):
t0_pose + per-step Δ-EE(6) at the action rate (translation in metres, rotation as a
rotation-vector difference, the executor's linear cumsum-in-rotvec model) and an absolute
gripper channel. Targets come from the sim's ground-truth packet pose, so the recorded
executed actions live in exactly the frame the teacher is trained on. The executor path
(bounded limiter, rate limits, latch) is unchanged: the expert only replaces the policy
server.

Orientation matters for clearance: descending straight down in the start orientation drives
the wrist links into the bin wall (expert smoke 2026-09-12, wrist_2 at 307 N on the bin
while over the packet; wrist_1 at 234 N inside the bin). The real demonstrations rotate the
tool during the descent and again over the bin; the per-phase rotation targets below are
the val-demo means (20 teleop episodes, rotvec, base frame) and are jittered per episode.

Phases: pre-grasp above the packet, slow descent, close, hold, lift, carry at demo height,
descent into the bin, open on the floor band, retreat.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np

EVENTS = ("none", "onset", "hold", "slip", "release")


@dataclass(frozen=True)
class ExpertParams:
    action_rate_hz: float = 10.0
    horizon: int = 16
    latency_s: float = 0.05
    grip_open: float = 0.15
    grip_close: float = 0.62
    z_pregrasp_m: float = 0.20          # hover height before the descent
    grasp_height_above_center_m: float = 0.035   # TCP z above the packet centre at closure
    z_lift_m: float = 0.20              # straight lift above the grasp; the arm is near full reach at the packet
    z_carry_m: float = 0.33              # reached over the bin while translating (demo apex 0.29-0.37 over the bin)
    z_place_m: float = 0.10
    z_retreat_m: float = 0.25
    v_travel: float = 0.15              # m/s in free space (demos p99 0.33-0.56)
    v_descend: float = 0.07             # demos 0.06-0.10 on the last 15 cm
    v_contact: float = 0.03             # last 3 cm before contact / floor
    slow_band_m: float = 0.03
    w_max_rad_s: float = 0.8            # rotation-vector rate cap while empty (executor cap 1.0)
    w_max_gripped_rad_s: float = 0.3    # while holding the packet: turning at 0.8 rad/s during the carry shook it out (both physics rates, 2026-09-12)
    close_s: float = 0.5
    hold_after_close_s: float = 0.4
    open_s: float = 0.4
    hold_after_open_s: float = 0.5
    reach_tol_m: float = 0.006
    rot_tol_rad: float = 0.06
    place_offset_xy_m: tuple[float, float] = (0.0, -0.015)   # from the bin centre, toward the near wall
    # rotation-vector targets (base frame), val-demo means
    rot_descend: tuple[float, float, float] = (-1.50, -2.00, 1.17)   # first sample below z 0.15
    rot_grasp: tuple[float, float, float] = (-1.70, -1.92, 1.26)     # at closure
    rot_carry: tuple[float, float, float] = (-1.35, -1.50, 0.95)     # at carry apex
    rot_release: tuple[float, float, float] = (-2.04, -1.22, 0.61)   # at opening
    # per-episode jitter (drawn once in reset_episode from the rng)
    rot_jitter_rad: float = 0.08
    place_jitter_m: float = 0.015
    z_place_jitter_m: float = 0.01
    # recovery case (campaign-drawn): the FIRST grasp aims this far off the object, closes on
    # air or a corner, the lift finds the object still down, the expert reopens and re-grasps
    # on target. (0, 0) = no deliberate miss.
    miss_offset_xy_m: tuple[float, float] = (0.0, 0.0)
    miss_lift_check_m: float = 0.01     # object must have risen this much by the end of the lift
    # touchdown release: during the place descent, open as soon as the object has stopped
    # descending (it is supported) while the TCP still moves down — before the target height
    # if the support comes early. Egg lane 2026-09-14: descending to a fixed height pressed
    # the egg onto the holder ring (20 N, mechanics stop); a fixed height 1.5 cm higher
    # bounced it off. Real rig 09-12: the in-box wrench stops were the same thing.
    touchdown_release: bool = True
    touchdown_tcp_travel_m: float = 0.006   # TCP descent over which the object must not follow
    touchdown_object_tol_m: float = 0.0015
    touchdown_stall_s: float = 0.6          # no descent progress for this long during the place descent = touchdown


PHASES = ("pregrasp", "descend", "close", "settle", "lift", "regrasp_open", "carry", "place_descend", "open", "retreat", "done")


def rotvec_nearest(reference: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """The representation of `rotvec` (same rotation, angle shifted by 2πk) closest to
    `reference` — the executor's continuity guard, so the linear rotvec delta stays short."""
    r = np.asarray(rotvec, dtype=float)
    ref = np.asarray(reference, dtype=float)
    angle = float(np.linalg.norm(r))
    if angle < 1e-9:
        return r
    axis = r / angle
    best, best_d = r, float(np.linalg.norm(r - ref))
    for k in (-2, -1, 1, 2):
        cand = axis * (angle + 2.0 * np.pi * k)
        d = float(np.linalg.norm(cand - ref))
        if d < best_d:
            best, best_d = cand, d
    return best


class ScriptedExpertPolicy:
    """Same surface as RemoteSimulationPolicy for the sim adapter (replan / reset_episode / close)."""

    policy_kind = "scripted"
    wrench_baseline_rows = 0
    k_seeds = 1
    nfe = 0
    guidance = 1.0
    action_time_origin = "inference_ready"

    def __init__(self, packet_pose_fn, bin_center_xy, params: ExpertParams | None = None, *, rng_seed: int = 0,
                 pad_midpoint_fn=None):
        self.packet_pose_fn = packet_pose_fn          # () -> (xyz base frame)
        # () -> xyz of the midpoint between the two pads (base frame). The pads sit off the TCP
        # by up to ~3 cm laterally depending on the tool orientation and the finger closure
        # (egg smoke 2026-09-14: pads 2.7 cm beside a 48 mm egg at the demo orientation), so
        # the expert steers the PAD midpoint onto the object and derives the TCP target from
        # the currently measured offset. None = aim the TCP itself (waffle behaviour).
        self.pad_midpoint_fn = pad_midpoint_fn
        self._pad_offset_xy = np.zeros(2)
        self.bin_center_xy = np.asarray(bin_center_xy, dtype=float)[:2]
        self.base_params = params or ExpertParams()
        self.p = self.base_params
        self.info = {"ckpt": "scripted_expert_v2", "ckpt_sha": "scripted", "warmed": True, "student": False,
                     "policy_kind": self.policy_kind, "wrench_baseline_rows": 0,
                     "effective": {"nfe": 0, "guidance": 1.0, "k_seeds": 1, "parity_fixes": False,
                                   "persistent_noise": False, "task_text": "scripted", "drop_video": False,
                                   "close_p": 0.5, "action_time_origin": "inference_ready"},
                     "expert_params": dict(self.base_params.__dict__)}
        self._rng = np.random.default_rng(rng_seed)
        self.reset_episode()

    # -- lifecycle ----------------------------------------------------------
    def reset_episode(self):
        """Draw this episode's jitter (rotation targets, place offset, place height)."""
        b = self.base_params
        j = self._rng.normal
        self.p = replace(
            b,
            rot_descend=tuple(np.asarray(b.rot_descend) + j(0.0, b.rot_jitter_rad, 3)),
            rot_grasp=tuple(np.asarray(b.rot_grasp) + j(0.0, b.rot_jitter_rad, 3)),
            rot_carry=tuple(np.asarray(b.rot_carry) + j(0.0, b.rot_jitter_rad, 3)),
            rot_release=tuple(np.asarray(b.rot_release) + j(0.0, b.rot_jitter_rad, 3)),
            place_offset_xy_m=tuple(np.asarray(b.place_offset_xy_m) + j(0.0, b.place_jitter_m, 2)),
            z_place_m=float(b.z_place_m + j(0.0, b.z_place_jitter_m)),
        ) if b.rot_jitter_rad > 0 or b.place_jitter_m > 0 or b.z_place_jitter_m > 0 else b
        self.info["episode_params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.p.__dict__.items()}
        self.phase = "pregrasp"
        self.phase_t0 = None
        self.grasp_xyz = None
        self.events = []
        self.attempt = 1
        self.packet_z_at_close = None
        self._td_ref = None                 # (tcp_z, packet_z) reference for the touchdown check
        self._td_stall = None               # (t, tcp_z) of the last real descent progress
        self._aim_offset = np.asarray(self.p.miss_offset_xy_m, dtype=float)

    def remote_reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
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

    def _place_xy(self):
        return np.array([self.bin_center_xy[0] + self.p.place_offset_xy_m[0],
                         self.bin_center_xy[1] + self.p.place_offset_xy_m[1]])

    def _target(self, tcp, packet):
        """Cartesian target for the current phase; None = hold in place."""
        p = self.p
        c = self._pad_offset_xy          # TCP target = object - (pad midpoint - TCP), xy
        if self.phase == "pregrasp":
            return np.array([packet[0] + self._aim_offset[0] - c[0], packet[1] + self._aim_offset[1] - c[1],
                             max(p.z_pregrasp_m, packet[2] + p.grasp_height_above_center_m + 0.10)])
        if self.phase == "descend":
            return np.array([packet[0] + self._aim_offset[0] - c[0], packet[1] + self._aim_offset[1] - c[1],
                             packet[2] + p.grasp_height_above_center_m])
        if self.phase in ("close", "settle"):
            return None
        if self.phase == "lift":
            # only a clearance lift here: lifting to the carry height straight above the
            # packet (y -0.27, tool tilted) straightens the elbow past the limiter's 0.40 rad
            # bound (servo_constraint_hold_timeout, smokes 2026-09-12); the demos rise
            # while translating toward the bin, where the arm is less extended
            g = self.grasp_xyz
            return np.array([g[0], g[1], p.z_lift_m])
        if self.phase == "carry":
            xy = self._place_xy()
            return np.array([xy[0] - c[0], xy[1] - c[1], p.z_carry_m])
        if self.phase == "place_descend":
            xy = self._place_xy()
            return np.array([xy[0] - c[0], xy[1] - c[1], p.z_place_m])
        if self.phase in ("open", "regrasp_open"):
            return None
        if self.phase == "retreat":
            return np.array([tcp[0], tcp[1], p.z_retreat_m])
        return None

    def _rot_target(self, tcp):
        """Rotation-vector target for the current phase (nearest representation to the
        measured one); None = hold the current orientation."""
        p = self.p
        # lift keeps the grasp orientation: turning to the carry orientation straight above
        # the packet parks the arm on the elbow constraint (limiter hold timeout, 1 ms
        # physics smoke 2026-09-12); the demos turn while translating toward the bin
        # the demos' profile: the tool turns gradually while translating — toward the
        # carry orientation during the carry, toward the release orientation during the
        # descent into the bin. Holding the grasp orientation all the way to the bin
        # wrapped the arm into a branch that parks on the elbow limit (v4 smoke), and a
        # fast turn (0.8 rad/s) shook the packet out (v3); the gripped rate cap keeps
        # the turn slow instead
        goal = {"pregrasp": p.rot_descend, "descend": p.rot_grasp, "lift": p.rot_grasp,
                "carry": p.rot_carry, "place_descend": p.rot_release,
                "retreat": p.rot_release}.get(self.phase)
        if goal is None:
            return None
        return rotvec_nearest(tcp[3:6], np.asarray(goal, dtype=float))

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
        if self.phase in ("open", "regrasp_open"):
            return p.grip_close - (p.grip_close - p.grip_open) * min(1.0, el / p.open_s)
        return p.grip_open

    def _step_phase(self, t, tcp, packet):
        """Phase transitions from measured TCP and elapsed time (no object feedback beyond its pose)."""
        p = self.p
        if self.phase_t0 is None:
            self.phase_t0 = float(t)
        el = float(t) - self.phase_t0
        target = self._target(tcp, packet)
        rot = self._rot_target(tcp)
        pos_ok = target is not None and np.linalg.norm(tcp[:3] - target) <= p.reach_tol_m
        rot_ok = rot is None or np.linalg.norm(tcp[3:6] - rot) <= p.rot_tol_rad
        reached = pos_ok and rot_ok
        if self.phase == "pregrasp" and reached:
            self._advance(t, "descend")
        elif self.phase == "descend" and pos_ok:
            self.grasp_xyz = tcp[:3].copy(); self.packet_z_at_close = float(packet[2]); self._advance(t, "close")
        elif self.phase == "close" and el >= p.close_s:
            self._advance(t, "settle")
        elif self.phase == "settle" and el >= p.hold_after_close_s:
            self._advance(t, "lift")
        elif self.phase == "lift" and pos_ok:
            lifted = self.packet_z_at_close is None or (float(packet[2]) - self.packet_z_at_close) >= p.miss_lift_check_m
            if lifted or self.attempt >= 2:
                self._advance(t, "carry")
            else:
                # missed (deliberate recovery case or a real slip): reopen, re-aim on target
                self._event(t, "miss", packet_dz=float(packet[2]) - self.packet_z_at_close)
                self.attempt += 1
                self._aim_offset = np.zeros(2)
                self._advance(t, "regrasp_open")
        elif self.phase == "regrasp_open" and el >= p.open_s + 0.2:
            self._advance(t, "pregrasp")
        elif self.phase == "carry" and pos_ok:
            self._advance(t, "place_descend")
        elif self.phase == "place_descend":
            touched = False
            if p.touchdown_release and self.phase_t0 is not None and el > 0.3:
                if self._td_ref is None:
                    self._td_ref = (float(tcp[2]), float(packet[2]))
                tcp_drop = self._td_ref[0] - float(tcp[2]); obj_drop = self._td_ref[1] - float(packet[2])
                if tcp_drop >= p.touchdown_tcp_travel_m:
                    touched = obj_drop < tcp_drop - p.touchdown_tcp_travel_m + p.touchdown_object_tol_m and \
                        (obj_drop < p.touchdown_object_tol_m or obj_drop < 0.5 * tcp_drop)
                    if not touched:
                        self._td_ref = (float(tcp[2]), float(packet[2]))   # slide the window
                # a rigidly held object (Carton) stops the whole arm when it meets the floor:
                # no TCP travel, no object travel, target not reached -> that is the touchdown
                if not touched:
                    if self._td_stall is None or float(tcp[2]) < self._td_stall[1] - 0.002:
                        self._td_stall = (float(t), float(tcp[2]))
                    elif float(t) - self._td_stall[0] >= p.touchdown_stall_s:
                        touched = True
            if pos_ok or touched:
                if touched:
                    self._event(t, "touchdown", tcp_z=float(tcp[2]), packet_z=float(packet[2]))
                self._advance(t, "open")
        elif self.phase == "open" and el >= p.open_s + p.hold_after_open_s:
            self._advance(t, "retreat")
        elif self.phase == "retreat" and pos_ok:
            self._advance(t, "done")

    # -- the policy call ----------------------------------------------------
    def replan(self, obs, prev_plan, tcp_pose):
        p = self.p
        t = float(obs.t)
        tcp = np.asarray(tcp_pose, dtype=float)
        packet = np.asarray(self.packet_pose_fn(), dtype=float)[:3]
        if self.pad_midpoint_fn is not None:
            mid = np.asarray(self.pad_midpoint_fn(), dtype=float)[:3]
            if np.isfinite(mid).all():
                self._pad_offset_xy = mid[:2] - tcp[:2]
        self._step_phase(t, tcp, packet)
        H, dt = p.horizon, 1.0 / p.action_rate_hz
        actions = np.zeros((H, 7), dtype=np.float32)
        pos = tcp[:3].copy()
        rot = tcp[3:6].copy()
        target = self._target(tcp, packet)
        rot_target = self._rot_target(tcp)
        for k in range(H):
            if target is not None:
                delta = target - pos
                dist = float(np.linalg.norm(delta))
                v = self._speed(pos, target)
                step = delta if dist <= v * dt else delta * (v * dt / dist)
                actions[k, :3] = step
                pos = pos + step
            if rot_target is not None:
                rdelta = rot_target - rot
                rdist = float(np.linalg.norm(rdelta))
                gripped = self.phase in ("settle", "lift", "carry", "place_descend", "regrasp_open")
                cap = (p.w_max_gripped_rad_s if gripped else p.w_max_rad_s) * dt
                rstep = rdelta if rdist <= cap else rdelta * (cap / rdist)
                actions[k, 3:6] = rstep
                rot = rot + rstep
            actions[k, 6] = self._grip(t + p.latency_s + k * dt)
        kind = {"close": "onset", "settle": "hold", "lift": "hold", "carry": "hold", "place_descend": "hold",
                "open": "release", "regrasp_open": "release"}.get(self.phase, "none")
        return SimpleNamespace(t_created=t, t0_pose=tcp.copy(), actions=actions,
                               action_times=t + p.latency_s + np.arange(H) / p.action_rate_hz,
                               sigma=np.zeros(H, dtype=np.float32), gate=1.0, p_evt=self._p_evt(kind), cpk=None,
                               latency_s=p.latency_s, _cpk_token=None,
                               diag={"expert_phase": self.phase, "attempt": self.attempt, "packet_xyz": packet.tolist(),
                                     "target_xyz": None if target is None else [float(v) for v in target],
                                     "pad_offset_xy": [float(v) for v in self._pad_offset_xy],
                                     "rot_target": None if rot_target is None else [float(v) for v in rot_target]})
