"""Export a scripted-expert Isaac trial into the training episode layout.

One sim trial directory (tools/sim/run_waffles.py --policy-server scripted, with
--record-gel-contacts) becomes one `ep_sim_<task>_...` episode that WindowSampler
reads like a teleop demo:

  * arm_q / arm_qd / arm_tcp_pose / arm_ft / gripper  <- execution_trace.jsonl (125 Hz,
    the executor's measured state; wrist F/T is the sim's gripper-contact proxy)
  * arm_tcp_speed                                       <- finite difference of the TCP pose
  * actions                                             <- MEASURED delta-EE on the action grid
    plus the commanded aperture (the same construction as
    tools/rederive_rollout_actions.py, so the episode is tagged `actions_rederived`)
  * camera_scene_color                                  <- sim.mp4 frames (render rate), JPEG
  * tactile_<pad>_{fields_ds,infer_img,keyframes,wrench,area} <- IDLE pad rows copied from a
    real episode's untouched start and tiled at the recorded rates: the sim has no gel
    model, so the pads are masked (tag `pads_masked`) and the sampler takes the contact
    events / gate from `contact_gt` instead and zeroes the tactile reconstruction losses
  * contact_gt                                          <- (T, 3) [any pad in contact,
    left N, right N] from gel_contact_trace.json (physical pad-packet contact)

Episodes that did not place are exported only with --include-failed (they would train
at action weight 0 with no tactile supervision, i.e. contribute nothing).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from phantom.config.hardware import load_hardware  # noqa: E402
from phantom.data.derived import pose_delta  # noqa: E402
from phantom.data.episode_store import EpisodeReader, EpisodeWriter  # noqa: E402
from phantom.data.schema import (REDERIVED_TAG, STREAM_ACTIONS, STREAM_ARM_FT,  # noqa: E402
                                 STREAM_ARM_Q, STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                                 STREAM_ARM_TCP_SPEED, STREAM_CAMERA_SCENE, STREAM_GRIPPER,
                                 EpisodeMeta, tactile_stream)

log = logging.getLogger("export_expert_episode")

STREAM_CONTACT_GT = "contact_gt"
PADS_MASKED_TAG = "pads_masked"
SIM_TAG = "sim"
IDLE_KINDS = ("fields_ds", "infer_img", "keyframes", "wrench", "area")


# ---------------------------------------------------------------------------
def _latest_at_or_before(ts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts, grid, side="right") - 1
    return np.clip(idx, 0, len(ts) - 1)


def load_execution_trace(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: empty execution trace")
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    keep = np.concatenate([[True], np.diff(t) > 0])            # strictly increasing
    rows = [r for r, k in zip(rows, keep) if k]
    t = t[keep]
    q = np.array([r["measured_q"] for r in rows], dtype=np.float64)
    qd = np.array([r["measured_qd"] for r in rows], dtype=np.float64)
    tcp = np.array([r["measured_tcp"] for r in rows], dtype=np.float64)
    grip = np.array([[float(r["measured_gripper"][0]), float(r["measured_gripper"][1])] for r in rows],
                    dtype=np.float32)
    ft = np.array([r["measured_wrist_ft"] for r in rows], dtype=np.float64)
    cmd = np.array([np.nan if r.get("gripper_command") is None else float(r["gripper_command"]) for r in rows],
                   dtype=np.float64)
    stopped = np.array([bool(r.get("stopped")) for r in rows])
    return {"t": t, "q": q, "qd": qd, "tcp": tcp, "gripper": grip, "ft": ft, "grip_cmd": cmd, "stopped": stopped}


def tcp_speed(t: np.ndarray, tcp: np.ndarray, taps: int = 5) -> np.ndarray:
    """Finite-difference TCP twist (6,) at each sample, lightly smoothed (real rig: RTDE
    actual_TCP_speed)."""
    if len(t) < 2:
        return np.zeros_like(tcp)
    d = np.gradient(tcp, t, axis=0)
    if taps > 1 and len(t) > taps:
        k = np.ones(taps) / taps
        d = np.stack([np.convolve(d[:, i], k, mode="same") for i in range(d.shape[1])], axis=1)
    return d


def measured_actions(tr: dict[str, np.ndarray], rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Measured delta-EE on the uniform `rate` grid + commanded aperture (rederive semantics)."""
    t = tr["t"]
    t0, t1 = float(t[0]), float(t[-1])
    n = int(np.floor((t1 - t0) * rate + 1e-9)) + 1
    grid = t0 + np.arange(max(n, 0)) / rate
    if len(grid) < 2:
        raise ValueError("trial too short for one action step")
    idx = _latest_at_or_before(t, grid)
    poses = tr["tcp"][idx]
    cmd = tr["grip_cmd"][idx]
    meas = tr["gripper"][idx, 0]
    grips = np.where(np.isfinite(cmd), cmd, meas)          # before the first plan: measured aperture
    out = np.empty((len(grid) - 1, 7), dtype=np.float32)
    for k in range(1, len(grid)):
        out[k - 1, :6] = pose_delta(poses[k - 1], poses[k])
        out[k - 1, 6] = float(np.clip(grips[k], 0.0, 1.0))
    return grid[1:], out


def video_frames(mp4: Path, expected: int | None):
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {mp4}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    cap.release()
    if expected is not None and len(frames) != expected:
        log.warning("%s: %d frames decoded, %d frame timestamps", mp4.name, len(frames), expected)
    return frames


def idle_tactile_rows(src: Path, sensors: list[str], seconds: float = 1.0) -> dict[tuple[str, str], np.ndarray]:
    """First `seconds` of every tactile stream of a real episode (pads untouched at the
    start pose): the sim's stand-in for an idle sensor."""
    reader = EpisodeReader(src)
    out = {}
    for s in sensors:
        for kind in IDLE_KINDS:
            stream = tactile_stream(s, kind)
            ts = reader.ts(stream)
            n = max(1, int((ts - ts[0] < seconds).sum()))
            out[(s, kind)] = np.asarray(reader.data(stream)[:n])
    return out


def tile_rows(rows: np.ndarray, times: np.ndarray) -> np.ndarray:
    idx = np.arange(len(times)) % len(rows)
    return rows[idx]


def contact_gt(trial: Path, tr_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gel = trial / "gel_contact_trace.json"
    if gel.exists():
        samples = json.loads(gel.read_text())
        samples = samples.get("samples", samples) if isinstance(samples, dict) else samples
        t = np.array([s["t"] for s in samples], dtype=np.float64)
        f = np.array([s.get("normal_force_n", [0.0, 0.0]) for s in samples], dtype=np.float32)
        c = np.array([s.get("has_gel_contact", [False, False]) for s in samples], dtype=bool)
        data = np.column_stack([c.any(axis=1).astype(np.float32), f[:, 0], f[:, 1]]).astype(np.float32)
        order = np.argsort(t)
        return t[order], data[order]
    sim = np.load(trial / "sim_trace.npz")
    t = np.asarray(sim["t"], dtype=np.float64)
    force = np.asarray(sim["pad_packet_normal_force"], dtype=np.float32)      # (T, 2)
    data = np.column_stack([(force.max(axis=1) > 0.05).astype(np.float32), force[:, 0], force[:, 1]])
    return t, data.astype(np.float32)


# ---------------------------------------------------------------------------
def export_trial(trial: Path, out_root: Path, hw, *, idle_src: Path, task: str = "waffles",
                 text: str | None = None, include_failed: bool = False, name: str | None = None,
                 spike_rad_s: float = 1.0) -> Path | None:
    trial = Path(trial)
    result = json.loads((trial / "trial_result.json").read_text()) if (trial / "trial_result.json").exists() else {}
    placed = result.get("stage_name") == "placed"
    if not placed and not include_failed:
        log.info("%s: stage %r, not placed — skipped (use --include-failed)", trial.name, result.get("stage_name"))
        return None
    info = json.loads((trial / "policy_info.json").read_text()) if (trial / "policy_info.json").exists() else {}
    tr = load_execution_trace(trial / "execution_trace.jsonl")
    sim = np.load(trial / "sim_trace.npz")
    frame_t = np.asarray(sim["frame_t"], dtype=np.float64)

    # spikes: one-tick PhysX joint-velocity excursions (no contact) — logged for the sampler
    qd_max = np.abs(tr["qd"]).max(axis=1)
    spikes = tr["t"][qd_max > spike_rad_s]

    ep_name = name or f"ep_sim_{task}_{trial.name}"
    ep_dir = out_root / "tasks" / task / ep_name
    if ep_dir.exists():
        raise FileExistsError(ep_dir)
    tags = [SIM_TAG, PADS_MASKED_TAG, REDERIVED_TAG, "expert:" + str(info.get("ckpt", "scripted")),
            f"stage:{result.get('stage_name')}", f"stop:{result.get('stop_reason')}", f"spikes:{len(spikes)}"]
    meta = EpisodeMeta(task=task, text=text or task, operator="sim", tags=tags, policy="scripted_expert",
                       success=bool(placed),
                       notes=json.dumps({"trial": str(trial), "spike_t": [round(float(x), 4) for x in spikes],
                                         "duration_s": result.get("duration_s")}),
                       driver_modes={"drivers": "isaac_sim"},
                       deploy_overrides={"sim_trial": str(trial), "expert_params": info.get("episode_params"),
                                         "pads_masked": True})
    w = EpisodeWriter(ep_dir, hw, meta)
    try:
        t = tr["t"]
        w.append(STREAM_ARM_Q, t, tr["q"])
        w.append(STREAM_ARM_QD, t, tr["qd"])
        w.append(STREAM_ARM_TCP_POSE, t, tr["tcp"])
        w.append(STREAM_ARM_TCP_SPEED, t, tcp_speed(t, tr["tcp"]))
        w.append(STREAM_ARM_FT, t, tr["ft"])
        w.append(STREAM_GRIPPER, t, tr["gripper"])
        a_ts, a = measured_actions(tr, float(hw.control.action_rate_hz))
        w.append(STREAM_ACTIONS, a_ts, a)
        frames = video_frames(trial / "sim.mp4", len(frame_t))
        n = min(len(frames), len(frame_t))
        w.append(STREAM_CAMERA_SCENE, frame_t[:n], np.stack(frames[:n]))
        # idle pads tiled at the recorded rates over the sim span
        sensors = [s.name for s in hw.tactile.sensors]
        idle = idle_tactile_rows(idle_src, sensors)
        span = float(t[-1] - t[0])
        rates = {"fields_ds": hw.recording.field_ds_rate_hz, "infer_img": hw.recording.infer_img_rate_hz,
                 "keyframes": hw.recording.keyframe_rate_hz, "wrench": hw.recording.field_ds_rate_hz,
                 "area": hw.recording.field_ds_rate_hz}
        for s in sensors:
            for kind in IDLE_KINDS:
                times = t[0] + np.arange(0.0, span, 1.0 / rates[kind])
                w.append(tactile_stream(s, kind), times, tile_rows(idle[(s, kind)], times))
        c_t, c = contact_gt(trial, t)
        w.append(STREAM_CONTACT_GT, c_t, c)
        w.finalize(success=bool(placed))
    except Exception:
        w.abort()
        raise
    log.info("%s -> %s (%d ticks, %d actions, %d frames, %d spikes, placed=%s)", trial.name, ep_dir,
             len(t), len(a), n, len(spikes), placed)
    return ep_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", type=Path, nargs="+", required=True, help="sim trial directories")
    ap.add_argument("--out-root", type=Path, required=True, help="data root (episodes go under tasks/<task>/)")
    ap.add_argument("--hardware", default="configs/hardware.nuc.yaml",
                    help="config the REAL episodes were recorded under (shapes, rates, hash)")
    ap.add_argument("--idle-tactile-from", type=Path, required=True,
                    help="a real episode whose first second supplies the idle pad rows")
    ap.add_argument("--task", default="waffles")
    ap.add_argument("--text", default=None)
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--name", default=None, help="episode directory name (single --trial only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    hw = load_hardware(args.hardware, quiet=True)
    n = 0
    for trial in args.trial:
        if args.name and len(args.trial) != 1:
            raise SystemExit("--name applies to a single --trial")
        if export_trial(trial, args.out_root, hw, idle_src=args.idle_tactile_from, task=args.task,
                        text=args.text, include_failed=args.include_failed, name=args.name) is not None:
            n += 1
    log.info("exported %d/%d trials", n, len(args.trial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
