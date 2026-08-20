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


def load_start_stats(path: str | Path | None = None) -> dict[str, TaskStartStats]:
    """configs/start_poses.yaml -> {task: TaskStartStats}."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "start_poses.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    out = {}
    for task, d in raw["tasks"].items():
        out[task] = TaskStartStats(
            task=task, n=int(d["n"]),
            tcp_mean=np.asarray(d["tcp_mean"], dtype=np.float64),
            tcp_std=np.asarray(d["tcp_std"], dtype=np.float64),
            gripper_mean=float(d["gripper_mean"]),
            gripper_std=float(d["gripper_std"]),
        )
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
                       gripper_pos: float | None = None) -> tuple[np.ndarray, str]:
    """Per-axis |sigma| distances + a printable table line-set."""
    std = np.where(stats.tcp_std > 1e-9, stats.tcp_std, np.inf)
    sig = np.abs((np.asarray(tcp_pose, dtype=np.float64) - stats.tcp_mean) / std)
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
        lines.append(f"  grip: {gripper_pos:6.2f}      (demo {stats.gripper_mean:.2f} "
                     f"+/- {stats.gripper_std:.2f}, {gs:.1f} sigma)")
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
                  speed: float = 0.10, accel: float = 0.30) -> tuple[np.ndarray, float]:
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
    arm.move_l(tcp_target, speed, accel)
    if not wait_gripper_settled(gripper):
        log.warning("gripper did not settle (OBJ!=3) within timeout")
    return tcp_target, grip_target
