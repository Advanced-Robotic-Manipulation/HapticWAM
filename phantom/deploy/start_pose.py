"""Per-task start-pose homing + episode-start gate (rig postmortem 2026-08-20).

Every episode of the first v4 rig session launched 2-6 sigma outside the demo
start distribution (y 6-13 cm inside the workspace, z low, gripper
half-closed). The policy read that mid-approach state as "approach done" and
executed place-phase behavior. This module makes the demo start distribution
an explicit precondition:

- ``move_to_start``: move the arm/gripper to a jittered sample of the task's
  demo start distribution (jitter <= 1 sigma, so the rig matches the
  DISTRIBUTION the model trained on, not a single memorized point).
- ``start_sigma_report``: how far the live state is from the distribution,
  per axis, for the operator + the >max_sigma refuse gate.
- ``wait_gripper_settled``: block until the gripper reports OBJ==3 (at rest).
  3/7 postmortem episodes started while the gripper was still homing
  (OBJ==0 at frame 0) - a state that occurs mid-grasp in training data.

Known limits (reviewed 2026-08-20): rotation sigma is per-axis on the
axis-angle vector, canonicalized to the representation nearest the task mean
(rotvec_nearest) - the task clusters sit at rotvec norm 2.55-2.65 rad, close
enough to pi that live readings can flip to the antipodal representation. The homing moveL runs BEFORE the
episode SafetyMonitor exists - by design it is slow (0.1 m/s), prompted
("clear the arm's path"), and covered by UR firmware limits + the operator's
E-stop, not by the deploy safety layer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)

AXES = ("x", "y", "z", "rx", "ry", "rz")


@dataclass(frozen=True)
class TaskStartStats:
    task: str
    n: int
    tcp_mean: np.ndarray          # (6,) m / axis-angle rad
    tcp_std: np.ndarray           # (6,)
    gripper_mean: float
    gripper_std: float
    # optional (start_poses.yaml >= 2026-08-28): demo START joint configuration
    # and the lowest TCP height any demo reached. The policy's proprio input is
    # the RAW joint vector, so a wrist wrapped by a full turn or an elbow-flipped
    # IK branch is hundreds of sigma OOD even when the TCP pose gates fine
    # (rig 2026-08-28: 16/26 episodes ran like that after manual jogging).
    q_mean: np.ndarray | None = None      # (dof,) rad
    q_std: np.ndarray | None = None       # (dof,) rad
    tcp_z_min: float | None = None        # m
    tcp_min: np.ndarray | None = None     # (3,) m — demo TCP envelope (all frames)
    tcp_max: np.ndarray | None = None     # (3,) m


# a demo joint std below this is treated as this (the taught start pose is
# repeated within ~0.5 deg on some joints — that is not a useful tolerance)
Q_STD_FLOOR_RAD = np.radians(5.0)


def load_start_stats(path: str | Path | None = None) -> dict[str, TaskStartStats]:
    """configs/start_poses.yaml -> {task: TaskStartStats}."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "start_poses.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    out = {}
    for task, d in raw["tasks"].items():
        st = TaskStartStats(
            task=task, n=int(d["n"]),
            tcp_mean=np.asarray(d["tcp_mean"], dtype=np.float64),
            tcp_std=np.asarray(d["tcp_std"], dtype=np.float64),
            gripper_mean=float(d["gripper_mean"]),
            gripper_std=float(d["gripper_std"]),
            q_mean=(np.asarray(d["q_mean"], dtype=np.float64) if "q_mean" in d else None),
            q_std=(np.asarray(d["q_std"], dtype=np.float64) if "q_std" in d else None),
            tcp_z_min=(float(d["tcp_z_min"]) if "tcp_z_min" in d else None),
            tcp_min=(np.asarray(d["tcp_min"], dtype=np.float64) if "tcp_min" in d else None),
            tcp_max=(np.asarray(d["tcp_max"], dtype=np.float64) if "tcp_max" in d else None),
        )
        if st.tcp_min is not None or st.tcp_max is not None:
            assert st.tcp_min is not None and st.tcp_max is not None, task
            assert st.tcp_min.shape == (3,) and st.tcp_max.shape == (3,), task
            assert np.all(np.isfinite(st.tcp_min)) and np.all(np.isfinite(st.tcp_max)), task
            assert np.all(st.tcp_max > st.tcp_min), task
        if st.q_mean is not None or st.q_std is not None:
            assert st.q_mean is not None and st.q_std is not None, task
            assert st.q_mean.shape == st.q_std.shape and st.q_mean.ndim == 1, task
            assert np.all(np.isfinite(st.q_mean)) and np.all(np.isfinite(st.q_std)), task
            assert np.all(st.q_std >= 0), task
        if st.tcp_z_min is not None:
            assert np.isfinite(st.tcp_z_min) and 0.0 < st.tcp_z_min < 1.0, task
        # a gate built on malformed stats fails OPEN (NaN > x is False) —
        # refuse to load instead
        assert st.tcp_mean.shape == (6,) and st.tcp_std.shape == (6,), task
        assert np.all(np.isfinite(st.tcp_mean)) and np.all(np.isfinite(st.tcp_std)), task
        assert np.all(st.tcp_std > 0) and st.gripper_std > 0, task
        assert np.isfinite(st.gripper_mean) and 0.0 <= st.gripper_mean <= 1.0, task
        out[task] = st
    return out


def sample_start_pose(stats: TaskStartStats, rng: np.random.Generator | None = None,
                      jitter_sigma: float = 1.0) -> tuple[np.ndarray, float]:
    """(tcp_target, gripper_target): task mean + truncated <=1-sigma jitter.

    Jitter matches the demo distribution instead of overfitting one point;
    truncation keeps every sampled start well inside what the model has seen.
    """
    rng = rng or np.random.default_rng()
    j = np.clip(rng.standard_normal(6), -1.0, 1.0) * stats.tcp_std * jitter_sigma
    grip = float(np.clip(
        stats.gripper_mean
        + float(np.clip(rng.standard_normal(), -1.0, 1.0)) * stats.gripper_std * jitter_sigma,
        0.0, 1.0))
    return stats.tcp_mean + j, grip


def start_sigma_report(stats: TaskStartStats, tcp_pose: np.ndarray,
                       gripper_pos: float | None = None,
                       q: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Per-axis |sigma| distances + a printable table line-set.

    `q` (live joint vector, rad) is gated too when the stats carry a demo
    joint distribution: on the RAW difference, deliberately unwrapped — the
    model consumes raw joint angles, so wrist-3 at +183 deg IS ~150 sigma
    away from the demos' -179 deg even though the flange points the same way."""
    from phantom.data.derived import rotvec_nearest
    tcp_pose = np.asarray(tcp_pose, dtype=np.float64).copy()
    # UR reports the ||r|| <= pi rotvec representation; a pose ~2 sigma out
    # radially (whiteboard mean is at 2.646 rad) crosses pi and comes back
    # antipodal, which would read as ~20 false sigma component-wise
    tcp_pose[3:6] = rotvec_nearest(stats.tcp_mean[3:6], tcp_pose[3:6])
    std = np.where(stats.tcp_std > 1e-9, stats.tcp_std, np.inf)
    sig = np.abs((tcp_pose - stats.tcp_mean) / std)
    # a non-finite live reading must gate OUT, not slip past a NaN comparison
    sig = np.where(np.isfinite(sig), sig, np.inf)
    lines = []
    for i, ax in enumerate(AXES):
        unit = "mm" if i < 3 else "rad"
        cur = tcp_pose[i] * (1000 if i < 3 else 1)
        mean = stats.tcp_mean[i] * (1000 if i < 3 else 1)
        std_u = stats.tcp_std[i] * (1000 if i < 3 else 1)
        lines.append(f"  {ax:>2}: {cur:8.1f} {unit}  (demo {mean:.1f} +/- {std_u:.1f}, "
                     f"{sig[i]:.1f} sigma)")
    if gripper_pos is not None and stats.gripper_std > 1e-9:
        gs = abs(gripper_pos - stats.gripper_mean) / stats.gripper_std
        if not np.isfinite(gs):
            gs = np.inf
        lines.append(f"  grip: {gripper_pos:6.2f}      (demo {stats.gripper_mean:.2f} "
                     f"+/- {stats.gripper_std:.2f}, {gs:.1f} sigma)")
        sig = np.append(sig, gs)      # the gripper GATES, not just prints:
                                      # half-closed 0.43-0.47 starts were part
                                      # of the postmortem OOD state
    if q is not None and stats.q_mean is not None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != stats.q_mean.shape:
            lines.append(f"  joints: live vector has {q.shape}, stats {stats.q_mean.shape} — GATED OUT")
            sig = np.append(sig, np.inf)
        else:
            raw = q - stats.q_mean
            std = np.maximum(stats.q_std, Q_STD_FLOOR_RAD)
            sq = np.abs(raw) / std
            sq = np.where(np.isfinite(sq), sq, np.inf)
            for i in range(len(q)):
                lines.append(f"  q{i + 1}: {np.degrees(q[i]):8.1f} deg  (demo {np.degrees(stats.q_mean[i]):.1f} "
                             f"+/- {np.degrees(std[i]):.1f}, {sq[i]:.1f} sigma)")
                if abs(raw[i]) > np.pi:
                    lines.append(f"  !!! q{i + 1} is {np.degrees(raw[i]):+.0f} deg from the demos — a FULL-TURN "
                                 f"wrap or a flipped IK branch. The policy reads raw joint angles: "
                                 f"unwind it on the pendant (joint jog) before running; the TCP pose "
                                 f"gate cannot see this.")
            sig = np.append(sig, sq)
    return sig, "\n".join(lines)


def wait_gripper_settled(gripper, timeout_s: float = 10.0) -> bool:
    """Block until the gripper reports at-rest (OBJ==3). False on timeout."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        st = gripper.get_state()
        if float(getattr(st, "obj", 0.0)) == 3.0:
            return True
        time.sleep(0.1)
    return False


def move_to_start(arm, gripper, hw, stats: TaskStartStats,
                  rng: np.random.Generator | None = None,
                  speed: float = 0.10, accel: float = 0.30,
                  home_joints: bool = False, joint_speed: float = 0.20,
                  joint_accel: float = 0.50) -> tuple[np.ndarray, float]:
    """Move arm+gripper to a jittered demo start. Returns (tcp_target, grip_target).

    Uses moveL at a deliberately slow speed (0.1 m/s). The caller owns safety:
    call this BEFORE the executor starts (single arm user), operator at the
    E-stop.
    """
    tcp_target, grip_target = sample_start_pose(stats, rng)
    log.info("homing to %s demo start: xyz=[%.0f, %.0f, %.0f]mm grip=%.2f "
             "(task mean + <=1sigma jitter)", stats.task,
             *(tcp_target[:3] * 1000), grip_target)
    gripper.move(grip_target, hw.gripper.default_speed, hw.gripper.default_force)
    if home_joints:
        if stats.q_mean is None:
            log.warning("--home-joints: no demo joint stats for %s — skipping the joint move",
                        stats.task)
        else:
            # moveJ to the demo joint configuration FIRST: moveL keeps whatever
            # IK branch / wrist wrap the arm is currently in, and that is
            # invisible to the TCP gate (rig 2026-08-28). Slow, operator at
            # the E-stop, path must be clear (a base rotation can sweep the bin).
            log.info("homing JOINTS to the %s demo configuration: %s deg (moveJ %.2f rad/s)",
                     stats.task, np.round(np.degrees(stats.q_mean), 0).tolist(), joint_speed)
            arm.move_j(np.asarray(stats.q_mean, dtype=np.float64), joint_speed, joint_accel)
    arm.move_l(tcp_target, speed, accel)
    if not wait_gripper_settled(gripper):
        log.warning("gripper did not settle (OBJ!=3) within timeout")
    return tcp_target, grip_target
