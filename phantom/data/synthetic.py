"""SyntheticEpisodeGenerator: schema-conforming episodes scripted by the SAME
ContactScenario + render_fields the mock drivers use — so mock recordings,
synthetic training data, and the derived-channel acceptance tests share one
signal model (pipeline.md §7 smoke path; --synthetic in every train program).

No hardware, no torch. Rates come from the hardware config (scaled by
rate_scale); shapes come from the config only.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from phantom.data.episode_store import EpisodeWriter
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ARM_FT, STREAM_ARM_Q,
                                 STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                                 STREAM_ARM_TCP_SPEED, STREAM_GRIPPER,
                                 EpisodeMeta, tactile_stream)
from phantom.drivers.mock.dmtac import render_fields
from phantom.drivers.mock.scenario import ContactScenario

log = logging.getLogger(__name__)

TASKS = ("approach_contact", "grasp_slip")


def _pool_ds(stack: np.ndarray, hw_out: tuple[int, int]) -> np.ndarray:
    """(H, W, C) -> (h, w, C) mean-pool; integer ratios are config-validated."""
    H, W, C = stack.shape
    h, w = hw_out
    return stack.reshape(h, H // h, w, W // w, C).mean(axis=(1, 3))


class SyntheticEpisodeGenerator:
    def __init__(self, hw, seed: int = 0, rate_scale: float = 1.0):
        self.hw = hw
        self.seed = seed
        self.rate_scale = float(rate_scale)
        self._n = 0

    # ------------------------------------------------------------------
    def _rate(self, hz: float, cap: float = 250.0) -> float:
        """Config rate, scaled, and capped so synthetic generation stays fast
        (a 500 Hz RTDE stream adds nothing to synthetic training data)."""
        return min(hz * self.rate_scale, cap)

    def generate(self, root: Path, task: str, duration_s: float) -> Path:
        hw = self.hw
        rng = np.random.default_rng(self.seed + 7919 * self._n)
        scenario = ContactScenario(seed=self.seed + self._n)
        ep_path = Path(root) / f"ep_synth_{self.seed:03d}_{self._n:04d}"
        self._n += 1
        meta = EpisodeMeta(task=task, text=f"synthetic {task}", operator="synthetic",
                           tags=["synthetic"])
        writer = EpisodeWriter(ep_path, hw, meta)

        field_dtype = np.float16 if hw.recording.field_dtype == "float16" else np.float32

        # ---- tactile: fields_ds + wrench + area at field rate; keyframes;
        #      infer_img — all per sensor, sharing the scenario timeline
        for f_idx, s in enumerate(hw.tactile.sensors):
            r_field = self._rate(hw.recording.field_ds_rate_hz)
            ts = np.arange(0.0, duration_s, 1.0 / r_field)
            ds_rows, wrench_rows, area_rows = [], [], []
            kf_ts, kf_rows = [], []
            kf_period = 1.0 / self._rate(hw.recording.keyframe_rate_hz)
            next_kf = 0.0
            for t in ts:
                fr = render_fields(scenario.state(float(t)), hw.tactile, rng,
                                   finger_phase=float(f_idx))
                stack = np.concatenate(
                    [fr["deformation"], fr["depth"][..., None], fr["shear"],
                     fr["dist_force"]], axis=-1, dtype=np.float32)
                ds_rows.append(_pool_ds(stack, hw.recording.field_ds.hw).astype(field_dtype))
                wrench_rows.append(fr["wrench"])
                area_rows.append(fr["area_mm2"])
                if t >= next_kf:
                    kf_ts.append(t)
                    kf_rows.append(stack.astype(field_dtype))
                    next_kf += kf_period
            writer.append(tactile_stream(s.name, "fields_ds"), ts, np.stack(ds_rows))
            writer.append(tactile_stream(s.name, "wrench"), ts,
                          np.stack(wrench_rows).astype(np.float32))
            writer.append(tactile_stream(s.name, "area"), ts,
                          np.asarray(area_rows, dtype=np.float32))
            writer.append(tactile_stream(s.name, "keyframes"),
                          np.asarray(kf_ts), np.stack(kf_rows))
            if hw.recording.save_infer_img:
                r_img = self._rate(hw.recording.infer_img_rate_hz)
                ts_img = np.arange(0.0, duration_s, 1.0 / r_img)
                shape = ((hw.tactile.infer_img.h, hw.tactile.infer_img.w)
                         if hw.tactile.infer_img.c == 1 else
                         (hw.tactile.infer_img.h, hw.tactile.infer_img.w,
                          hw.tactile.infer_img.c))
                imgs = np.stack([
                    (rng.random(shape) * 30
                     + scenario.state(float(t)).press_depth * 200).astype(np.uint8)
                    for t in ts_img])
                writer.append(tactile_stream(s.name, "infer_img"), ts_img, imgs)

        # ---- arm streams (q/qd/tcp/ft) — slow scripted motion + the
        #      scenario-correlated wrist wrench (the ACC precursor)
        r_arm = self._rate(hw.arm.rtde_receive_hz, cap=100.0)
        ts_arm = np.arange(0.0, duration_s, 1.0 / r_arm)
        dof = hw.arm.dof
        base_q = rng.uniform(-1.0, 1.0, size=dof)
        amp = 0.1
        q = base_q[None, :] + amp * np.sin(
            2 * math.pi * 0.2 * ts_arm[:, None] + np.arange(dof)[None, :])
        qd = amp * 2 * math.pi * 0.2 * np.cos(
            2 * math.pi * 0.2 * ts_arm[:, None] + np.arange(dof)[None, :])
        tcp = np.zeros((len(ts_arm), 6), dtype=np.float32)
        tcp[:, 0] = 0.3 + 0.05 * np.sin(2 * math.pi * 0.1 * ts_arm)
        tcp[:, 1] = 0.05 * np.cos(2 * math.pi * 0.1 * ts_arm)
        tcp[:, 2] = 0.2
        tcp_speed = np.gradient(tcp, ts_arm, axis=0).astype(np.float32) \
            if len(ts_arm) > 1 else np.zeros_like(tcp)
        ft_dir = np.array([0.1, 0.05, -1.0, 0.02, -0.02, 0.01], dtype=np.float32)
        env = np.array([scenario.wrist_ft_scale(float(t)) for t in ts_arm],
                       dtype=np.float32)
        ft = env[:, None] * ft_dir[None, :] * 8.0 \
            + rng.normal(0, 0.05, size=(len(ts_arm), 6)).astype(np.float32)
        writer.append(STREAM_ARM_Q, ts_arm, q.astype(np.float32))
        writer.append(STREAM_ARM_QD, ts_arm, qd.astype(np.float32))
        writer.append(STREAM_ARM_TCP_POSE, ts_arm, tcp)
        writer.append(STREAM_ARM_TCP_SPEED, ts_arm, tcp_speed)
        writer.append(STREAM_ARM_FT, ts_arm, ft.astype(np.float32))

        # ---- gripper: closes through contact phases
        r_grip = self._rate(hw.gripper.feedback_rate_hz, cap=60.0)
        ts_grip = np.arange(0.0, duration_s, 1.0 / r_grip)
        pos = np.array([0.7 if scenario.state(float(t)).in_contact else 0.1
                        for t in ts_grip], dtype=np.float32)
        # first-order smoothing so it looks like a real gripper
        for i in range(1, len(pos)):
            pos[i] = pos[i - 1] + 0.2 * (pos[i] - pos[i - 1])
        cur = 0.3 * pos + rng.normal(0, 0.01, len(pos)).astype(np.float32)
        writer.append(STREAM_GRIPPER, ts_grip, np.stack([pos, cur], axis=-1))

        # ---- scene camera: noise canvas + blob-tracking square
        cam = hw.cameras.scene
        r_cam = self._rate(cam.fps, cap=30.0)
        ts_cam = np.arange(0.0, duration_s, 1.0 / r_cam)
        h, wd = cam.color.h, cam.color.w
        frames = np.empty((len(ts_cam), h, wd, 3), dtype=np.uint8)
        for i, t in enumerate(ts_cam):
            img = (rng.random((h, wd, 3)) * 40).astype(np.uint8)
            st = scenario.state(float(t))
            cy = int((st.blob_center_uv[1] * 0.5 + 0.5) * (h - 1))
            cx = int((st.blob_center_uv[0] * 0.5 + 0.5) * (wd - 1))
            r = max(2, h // 16)
            img[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = \
                (60 + 180 * st.press_depth, 80, 200)
            frames[i] = img
        writer.append("camera_scene_color", ts_cam, frames)

        # ---- actions: small Δ-EE strokes + gripper command from the phase
        r_act = self._rate(hw.control.action_rate_hz, cap=30.0)
        ts_act = np.arange(0.0, duration_s, 1.0 / r_act)
        act = np.zeros((len(ts_act), hw.control.action_dim), dtype=np.float32)
        act[:, 0] = 0.01 * np.sin(2 * math.pi * 0.3 * ts_act)
        act[:, 1] = 0.01 * np.cos(2 * math.pi * 0.3 * ts_act)
        act[:, 2] = -0.005 * np.sin(2 * math.pi * 0.15 * ts_act)
        act[:, 6] = np.array([1.0 if scenario.state(float(t)).in_contact else 0.0
                              for t in ts_act], dtype=np.float32)
        act[:, :6] += rng.normal(0, 5e-4, size=(len(ts_act), 6)).astype(np.float32)
        writer.append(STREAM_ACTIONS, ts_act, act)

        writer.finalize(success=True, notes="synthetic")
        return ep_path

    # ------------------------------------------------------------------
    def generate_dataset(self, root: Path, n_episodes: int = 4,
                         duration_s: float = 8.0) -> list[Path]:
        out = []
        for i in range(n_episodes):
            task = TASKS[i % len(TASKS)]
            out.append(self.generate(Path(root), task, duration_s))
        log.info("synthetic dataset: %d episodes at %s", n_episodes, root)
        return out
