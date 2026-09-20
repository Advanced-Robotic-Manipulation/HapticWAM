"""Re-derive a deploy rollout's `actions.zarr` so it means what training reads.

    python tools/rederive_rollout_actions.py <episode-or-root> [--hardware configs/hardware.yaml]
    python tools/rederive_rollout_actions.py data/episodes/deploy --dry-run

Why (P9):

  A TELEOP demo's `actions` stream is the MEASURED delta-EE pose, derived from
  two consecutive measured TCP poses on the `control.action_rate_hz` (10 Hz)
  command grid, plus the gripper (`phantom/data_collect/session.py:733-750`).
  `WindowSampler` (`phantom/data/windows.py:235-239,296-297`) reads exactly
  that stream, nearest-neighbour on the same grid, for both `action_chunk`
  (the target) and `prev_chunk` (the intent conditioning).

  A DEPLOY rollout's `actions` stream is something else entirely: the executor
  records `plan.actions[k]` — the policy's raw chunk PROPOSAL — once per
  action-grid step it enters (`phantom/deploy/executor.py:232-241`). Those
  rows are stamped at the loop's wall time, and the step index advances on the
  GOVERNOR-WARPED clock (`self._play_time += dt * scale`), so their cadence is
  not 10 Hz. They are also pre-clamp and pre-rate-limit: whatever the safety
  monitor clamped, the workspace box cut, or the kinematic rate limiter
  refused never appears in them.

  So training a DAgger round on rollouts as recorded would imitate what the
  policy WANTED at times it did not happen — including exactly the commands
  the safety layer refused. This tool rewrites `actions` to the same measured
  quantity a demo carries and keeps the proposal stream under
  `actions_plan.zarr` (it is the record of what the policy asked for, and the
  only way to decompose policy error from executor error post-hoc).

  The GRIPPER channel is the one exception to "measured": a demo records the
  COMMANDED aperture, so the re-derived rows take channel 6 from the proposal
  (the executor's command stream) at each grid time, not from
  `gripper.zarr` — the measured position is the grasp OUTCOME and would leak
  it into the action target (validation 2026-08-30 F20).

Idempotent: an episode that already has `actions_plan.zarr` is skipped, so the
tool is safe to re-run over a whole deploy root as new sessions land.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phantom.config.hardware import HardwareConfig, load_hardware  # noqa: E402
from phantom.data.derived import pose_delta  # noqa: E402
from phantom.data.episode_store import EpisodeReader  # noqa: E402
from phantom.data.schema import (REDERIVED_TAG, STREAM_ACTIONS,  # noqa: E402
                                 STREAM_ACTIONS_PLAN, STREAM_ARM_TCP_POSE,
                                 STREAM_GRIPPER, EpisodeMeta, is_policy_rollout)

log = logging.getLogger("rederive_rollout_actions")

_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

# re-exported: the tag and the rollout test now live in phantom.data.schema so
# the intake gate and WindowSampler can check them too (2026-08-30 F13)
is_rollout = is_policy_rollout


def _write_stream(path: Path, ts: np.ndarray, data: np.ndarray) -> None:
    """Write one {data, ts} zarr group, matching EpisodeWriter's layout."""
    g = zarr.open_group(str(path), mode="w")
    rows = max(1, min(120, len(ts)))
    g.create_dataset("data", shape=(0, *data.shape[1:]), dtype=data.dtype,
                     chunks=(rows, *data.shape[1:]), compressor=_COMPRESSOR)
    g.create_dataset("ts", shape=(0,), dtype=np.float64, chunks=(rows,),
                     compressor=_COMPRESSOR)
    g["data"].append(data)
    g["ts"].append(np.asarray(ts, dtype=np.float64))


def _latest_at_or_before(ts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the newest sample at or before each grid time.

    This is what the teleop loop sees: it reads `ring.latest(1)` at each 10 Hz
    tick (`session.py:_tcp_pose`), i.e. the most recent sample WRITTEN, never
    an interpolation and never a future one."""
    idx = np.searchsorted(ts, grid, side="right") - 1
    return np.clip(idx, 0, len(ts) - 1)


def action_grid(tcp_ts: np.ndarray, grip_ts: np.ndarray,
                rate: float) -> np.ndarray:
    """The uniform `rate` Hz command grid over the span both streams cover.

    Uniform on purpose: the teleop loop commands on a fixed-period tick and
    WindowSampler resamples the recorded rows at exactly `k / rate` offsets, so
    an episode whose action ts are governor-warped (as the executor's are) is
    read as if it were on this grid anyway."""
    t0 = max(float(tcp_ts[0]), float(grip_ts[0]))
    t1 = min(float(tcp_ts[-1]), float(grip_ts[-1]))
    n = int(np.floor((t1 - t0) * rate + 1e-9)) + 1
    return t0 + np.arange(max(n, 0)) / rate


def _gripper_commands(reader: EpisodeReader, grid: np.ndarray,
                      grip: np.ndarray, grip_ts: np.ndarray,
                      ep_name: str = "") -> np.ndarray:
    """Channel 6 of the re-derived actions: the COMMANDED aperture in effect at
    each grid time — never the measured one (validation 2026-08-30 F20).

    A teleop demo records `grip_cmd = pilot.last_sent`
    (`data_collect/session.py:735-745`), and the executor's own parity
    `prev_chunk` history stores the commanded `a[6]`
    (`deploy/executor.py:241`). `gripper.zarr[:,0]` is `GripperState.position`
    — what the fingers actually reached. On the rig that difference IS the
    grasp outcome: a command of 1.0 that closes on an object reads back as the
    object's width, a close on air reads ~1.0. Writing the measured position
    into the ACTION target both leaks the outcome into what the policy is
    trained to predict and disagrees with demos on the single channel the
    terminal commit is about.

    The command stream here is the executor proposal still sitting in
    `actions.zarr` (it becomes `actions_plan.zarr` a few lines later). Falls
    back to the measured position only when no command stream exists."""
    if reader.has(STREAM_ACTIONS):
        cmd = np.asarray(reader.data(STREAM_ACTIONS)[:], dtype=np.float64)
        cmd_ts = reader.ts(STREAM_ACTIONS)
        if cmd.ndim == 2 and cmd.shape[1] >= 7 and len(cmd_ts) >= 1:
            return np.clip(cmd[_latest_at_or_before(cmd_ts, grid), 6], 0.0, 1.0)
    log.warning("%s: no usable command stream — gripper channel falls back to "
                "the MEASURED aperture", ep_name)
    return grip[_latest_at_or_before(grip_ts, grid), 0]


def rederive_actions(ep_path: Path, hw: HardwareConfig, *,
                     dry_run: bool = False) -> str:
    """Rebuild one episode's `actions` stream. Returns a status word:
    'skipped' (already done), 'no-actions', 'too-short', or 'rederived'."""
    ep = Path(ep_path)
    plan_path = ep / f"{STREAM_ACTIONS_PLAN}.zarr"
    src_path = ep / f"{STREAM_ACTIONS}.zarr"
    if plan_path.exists():
        return "skipped"
    if not src_path.exists():
        return "no-actions"

    reader = EpisodeReader(ep)
    for stream in (STREAM_ARM_TCP_POSE, STREAM_GRIPPER):
        if not reader.has(stream):
            log.error("%s: no %s.zarr — cannot re-derive", ep.name, stream)
            return "no-actions"
    tcp_ts = reader.ts(STREAM_ARM_TCP_POSE)
    tcp = np.asarray(reader.data(STREAM_ARM_TCP_POSE)[:], dtype=np.float64)
    grip_ts = reader.ts(STREAM_GRIPPER)
    grip = np.asarray(reader.data(STREAM_GRIPPER)[:], dtype=np.float64)
    if len(tcp_ts) < 2 or len(grip_ts) < 1:
        return "too-short"

    rate = float(hw.control.action_rate_hz)
    grid = action_grid(tcp_ts, grip_ts, rate)
    n = len(grid)
    if n < 2:
        return "too-short"

    poses = tcp[_latest_at_or_before(tcp_ts, grid)]
    grips = _gripper_commands(reader, grid, grip, grip_ts, ep.name)

    # One row per grid step from the SECOND one on: the first tick has no
    # previous measured pose, exactly as session.py's `prev_tcp is None` guard.
    out = np.empty((n - 1, 7), dtype=np.float32)
    for k in range(1, n):
        out[k - 1, :6] = pose_delta(poses[k - 1], poses[k])
        out[k - 1, 6] = grips[k]
    out_ts = grid[1:]

    if dry_run:
        log.info("%s: would write %d rows at %.1f Hz (proposal had %d)",
                 ep.name, len(out), rate, reader.n(STREAM_ACTIONS))
        return "rederived"

    # Move the proposal aside FIRST: if anything below fails, the episode is
    # left with actions_plan.zarr and no actions.zarr — visibly broken, rather
    # than silently half-converted (and a re-run would then skip it).
    src_path.rename(plan_path)
    _write_stream(ep / f"{STREAM_ACTIONS}.zarr", out_ts, out)

    meta_path = ep / "meta.json"
    if meta_path.exists():
        meta = EpisodeMeta.load(meta_path)
        if REDERIVED_TAG not in meta.tags:
            meta.tags.append(REDERIVED_TAG)   # training-inert provenance
        meta.save(meta_path)
    log.info("%s: actions re-derived (%d rows on the %.1f Hz grid); proposal "
             "kept as %s.zarr", ep.name, len(out), rate, STREAM_ACTIONS_PLAN)
    return "rederived"


def _episodes(target: Path) -> list[Path]:
    target = Path(target)
    if (target / "meta.json").exists():
        return [target]
    return sorted(p.parent for p in target.rglob("ep_*/meta.json"))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", type=Path,
                    help="one episode directory, or a root to walk for ep_*/")
    ap.add_argument("--hardware", default="configs/hardware.yaml",
                    help="config the action grid rate comes from")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="also convert non-rollout (teleop) episodes — normally "
                         "refused: a demo's actions stream is already measured")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    eps = _episodes(args.target)
    if not eps:
        log.error("no ep_*/meta.json under %s", args.target)
        return 2
    counts: dict[str, int] = {}
    for ep in eps:
        meta = EpisodeMeta.load(ep / "meta.json")
        if not args.all and not is_rollout(meta):
            counts["not-a-rollout"] = counts.get("not-a-rollout", 0) + 1
            continue
        status = rederive_actions(ep, hw, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1
    log.info("%d episodes: %s", len(eps),
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
