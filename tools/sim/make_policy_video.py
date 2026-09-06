#!/usr/bin/env python3
"""Render policy trials with sampled scene/contact panels on one physical clock.

Examples::

    python tools/sim/make_policy_video.py --run RUN --output RUN/policy_review.mp4
    python tools/sim/make_policy_video.py --run A B C D --label Teacher Revised Control Original \
        --horizon 30 --output comparison.mp4

When policy_tactile.npz contains gel frames, panels use those saved runtime
pixels. Older logs reconstruct the grayscale proxy from causal pad-force
samples using the original runtime formula. Otherwise panels are
explicitly labelled packet-contact visualizations, not policy input pixels.
Neither is real tactile footage or a calibrated sensor reconstruction. Ended
trials explicitly freeze while other trials continue on the physical clock.
For a separately copied presentation tool, set PHANTOM_REVIEW_SOURCE to the
frozen runtime source tree used for metric code and hardware image shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(
    os.environ.get("PHANTOM_REVIEW_SOURCE", Path(__file__).resolve().parents[2])
).resolve()
sys.path.insert(0, str(REPO))

from phantom.sim.policy_metrics import evaluate_policy_trace
from tools.sim.tactile_panels import (
    force_proxy,
    previous_sample,
    timestamps,
)

WIDTH, HEIGHT = 960, 690
COLORS = [(95, 215, 255), (255, 170, 90)]


def _read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _label(image, text, xy, *, color=(235, 239, 242), scale=0.48):
    import cv2

    cv2.putText(
        image, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _controller_completion(execution):
    """First causally observable FINISH event; never infer it from task scores."""
    completed = []
    for index, row in enumerate(execution):
        diag = row.get("diagnostics") or {}
        reason = diag.get("completed_reason")
        if not reason:
            continue
        observed_t = float(row["t"])
        reported_t = diag.get("completed_at_s")
        if reported_t is not None:
            reported_t = float(reported_t)
        if not np.isfinite(observed_t) or (
            reported_t is not None and not np.isfinite(reported_t)
        ):
            raise ValueError("Controller completion has a nonfinite timestamp")
        completed.append(
            {
                "completed_reason": str(reason),
                "completed_at_s": reported_t,
                "first_execution_row": index,
                "first_execution_t_s": observed_t,
                "display_from_s": max(
                    observed_t, observed_t if reported_t is None else reported_t
                ),
                "source": "execution_trace.jsonl diagnostics; visible no earlier than both event and first reporting row",
                "is_object_task_success": False,
            }
        )
    return min(completed, key=lambda event: event["display_from_s"], default=None)


class PolicyPanel:
    def __init__(self, directory, label=None):
        import cv2

        self.directory = Path(directory).resolve()
        self.label = label or self.directory.name
        with np.load(self.directory / "sim_trace.npz", allow_pickle=False) as source:
            self.trace = dict(source)
        self.t = timestamps(self.trace["t"], "policy state time")
        self.frame_t = timestamps(self.trace.get("frame_t", self.t), "scene frame time")
        self.run = _read_json(self.directory / "run.json", {})
        config = _read_json(self.directory / "effective_config.json", None)
        if config is None:
            raise ValueError(f"Missing effective_config.json in {self.directory}")
        self.execution = []
        execution_path = self.directory / "execution_trace.jsonl"
        if execution_path.exists():
            self.execution = [
                json.loads(line)
                for line in execution_path.read_text().splitlines()
                if line.strip()
            ]
        self.metrics = evaluate_policy_trace(
            self.trace,
            config,
            _read_json(self.directory / "case.json", {}).get("scoring_thresholds"),
            run=self.run,
            execution_trace=self.execution,
            planner_trace=_read_json(self.directory / "planner_trace.json", []),
        )
        self.normal = np.asarray(self.trace["pad_packet_normal_force"])
        if self.normal.shape != (len(self.t), 2) or not np.isfinite(self.normal).all():
            raise ValueError("Expected finite packet contact normal forces (N,2)")
        self.uv = np.asarray(
            self.trace.get("pad_contact_uv", np.full((len(self.t), 2, 2), np.nan))
        )
        if self.uv.shape != (len(self.t), 2, 2):
            raise ValueError("Expected contact UV shape (N,2,2)")
        self.actual_input_proxy = (self.directory / "policy_tactile.npz").exists()
        self.input_t = np.zeros(0)
        self.input_force = np.zeros((0, 2, 3))
        self.input_gel = None
        self.input_gel_normal = None
        self.tactile_mapping = {
            "actual_input_proxy": self.actual_input_proxy,
            "calibrated": False,
            "normalization_transfer_validated": False,
        }
        if self.actual_input_proxy:
            from phantom.config.hardware import load_hardware

            with np.load(
                self.directory / "policy_tactile.npz", allow_pickle=False
            ) as source:
                self.input_t = np.asarray(source["t"], float)
                if "pad_force" in source:
                    self.input_force = np.asarray(source["pad_force"], float)
                else:
                    self.input_force = np.zeros((len(self.input_t), 2, 3))
                    if "gel_normal_force" not in source:
                        raise ValueError("Runtime tactile log has no force samples")
                if "gel" in source:
                    self.input_gel = np.asarray(source["gel"])
                if "gel_normal_force" in source:
                    self.input_gel_normal = np.asarray(
                        source["gel_normal_force"], float
                    )
            if not len(self.input_t):
                self.input_force = self.input_force.reshape(0, 2, 3)
            if (
                self.input_t.ndim != 1
                or not np.isfinite(self.input_t).all()
                or not (np.diff(self.input_t) > 0).all()
                or self.input_force.shape != (len(self.input_t), 2, 3)
                or not np.isfinite(self.input_force).all()
            ):
                raise ValueError("Invalid exact runtime tactile sample log")
            hw = load_hardware(REPO / "configs/hardware.nuc.yaml", quiet=True)
            self.field_hw = tuple(hw.recording.field_ds.hw)
            self.infer_hw = tuple(hw.tactile.infer_img.hwc[:2])
            if self.input_gel is not None and (
                self.input_gel.dtype != np.uint8
                or self.input_gel.shape != (len(self.input_t), 2, *self.infer_hw)
            ):
                raise ValueError("Expected native uint8 gel frames (N,2,H,W)")
            if self.input_gel_normal is not None and (
                self.input_gel_normal.shape != (len(self.input_t), 2)
                or not np.isfinite(self.input_gel_normal).all()
                or np.any(self.input_gel_normal < 0)
            ):
                raise ValueError("Expected finite nonnegative gel normal force (N,2)")
            self.tactile_mapping.update(
                source="policy_tactile.npz effective net pad force, including non-packet contacts",
                field_hw=self.field_hw,
                infer_hw=self.infer_hw,
                model=self.run.get("tactile_model"),
                formula="pressure=exp(-(x*x+y*y)/.18)*min(norm(force)/15,.8) on field grid; cv2 bilinear resize to infer_hw; uint8(clip(70+150*pressure)); zero_ablation gives black pixels",
                note="Exact runtime grayscale proxy, resized for display; student architectures may omit tactile model inputs while shared safety still consumes proxy wrench.",
            )
            if self.input_gel is not None:
                self.tactile_mapping.update(
                    source="policy_tactile.npz gel: saved pixels supplied at runtime",
                    formula=None,
                    native_pixel_shape=list(self.input_gel.shape[2:]),
                    native_pixel_dtype=str(self.input_gel.dtype),
                    note="Stored native grayscale pixels selected causally and resized only for display. These are simulated policy inputs, not real tactile footage or calibrated sensor predictions.",
                )
            self.tactile_mapping["display_force_source"] = (
                "gel_normal_force: normal force used to deform gel proxy"
                if self.input_gel_normal is not None
                else "pad_force: norm of net world-frame pad force"
            )
            if self.run.get("tactile_model") == "measured_baseline_proxy":
                policy_info = _read_json(self.directory / "policy_info.json", {})
                self.tactile_mapping.update(
                    model_description="Static measured no-contact baseline plus physics-driven deformation; uncalibrated",
                    baseline_provenance=policy_info.get("tactile_baseline_provenance"),
                    proxy_parameters=policy_info.get("tactile_proxy_parameters"),
                )
        self.cap = cv2.VideoCapture(str(self.directory / "sim.mp4"))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open scene video in {self.directory}")
        self.video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.video_frames != len(self.frame_t):
            self.cap.release()
            raise ValueError(
                f"Scene frame count {self.video_frames} != timestamp count {len(self.frame_t)}"
            )
        self.cached_index, self.cached_frame = -1, None
        self.stop_reason = self.metrics["control"]["stop_reason"]
        stop_rows = [row["t"] for row in self.execution if row.get("stopped")]
        stop_rows += [
            event["t"]
            for event in self.run.get("events", [])
            if event.get("event") == "policy_stop"
        ]
        self.stop_time = min(stop_rows) if stop_rows else None
        self.completion = _controller_completion(self.execution)
        self.mapping = []

    def close(self):
        self.cap.release()

    def _runtime_force(self, index, side):
        if index < 0:
            return 0.0
        if self.input_gel_normal is not None:
            return float(self.input_gel_normal[index, side])
        return float(np.linalg.norm(self.input_force[index, side]))

    def _runtime_gray(self, index, side):
        """Native supplied pixels, or the documented formula for older logs."""
        import cv2

        if index < 0:
            return np.zeros(self.infer_hw, np.uint8)
        if self.input_gel is not None:
            return self.input_gel[index, side]
        # Older runtime formulas used net force even if a later sidecar adds
        # a normal-force channel. Preserve their pixel reconstruction exactly.
        force = float(np.linalg.norm(self.input_force[index, side]))
        h, w = self.field_hw
        gh, gw = self.infer_hw
        yy, xx = np.mgrid[-1 : 1 : complex(h), -1 : 1 : complex(w)]
        pressure = np.exp(-(xx * xx + yy * yy) / 0.18) * min(force / 15, 0.8)
        gray = np.clip(cv2.resize(pressure, (gw, gh)) * 150 + 70, 0, 255).astype(
            np.uint8
        )
        if self.run.get("tactile_model") == "zero_ablation":
            gray[:] = 0
        return gray

    def _scene(self, index):
        import cv2

        if index < 0:
            return np.zeros((480, 640, 3), np.uint8)
        if index != self.cached_index:
            if index != self.cached_index + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError(
                    f"Failed reading scene frame {index} from {self.directory}"
                )
            self.cached_index = index
            self.cached_frame = cv2.resize(
                frame, (640, 480), interpolation=cv2.INTER_AREA
            )
        return self.cached_frame

    def render(self, t, horizon, force_max):
        import cv2

        state_index, state_age = previous_sample(self.t, t + 1e-9)
        frame_index, frame_age = previous_sample(self.frame_t, t + 1e-9)
        input_index, input_age = previous_sample(self.input_t, t + 1e-9)
        frozen = bool(t > self.frame_t[-1] + 1e-6)
        canvas = np.full((HEIGHT, WIDTH, 3), (22, 27, 33), np.uint8)
        _label(canvas, self.label, (12, 23), scale=0.62)
        _label(canvas, f"simulation t={t:05.2f}s", (710, 23))
        status = "RUNNING"
        color = (175, 220, 175)
        completed = bool(
            self.completion is not None
            and t + 1e-9 >= self.completion["display_from_s"]
        )
        if completed:
            status = (
                "RELEASE COMPLETE / HOLD"
                if self.completion["completed_reason"] == "placement_release_finished"
                else "CONTROLLER COMPLETE / HOLD"
            )
            color = (220, 210, 125)
        if self.stop_time is not None and t >= self.stop_time:
            status, color = f"STOP: {self.stop_reason}", (90, 160, 255)
        if frozen:
            status = f"{status if self.stop_reason or completed else 'HORIZON / TRACE END'} | FRAME FROZEN at {self.frame_t[-1]:.2f}s"
        if not self.metrics["valid_for_scoring"]:
            status = "INELIGIBLE TRACE | " + status
        _label(canvas, status, (12, 45), color=color, scale=0.43)
        canvas[52:532, :640] = self._scene(frame_index)
        for side in range(2):
            force = float(self.normal[state_index, side]) if state_index >= 0 else 0.0
            uv = self.uv[state_index, side] if state_index >= 0 else np.full(2, np.nan)
            known = np.isfinite(uv).all()
            if self.actual_input_proxy:
                force = self._runtime_force(input_index, side)
                gray = self._runtime_gray(input_index, side)
                panel = cv2.cvtColor(
                    cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_GRAY2BGR,
                )
                location = (
                    f"8 Hz causal sample age {input_age:.3f}s"
                    if input_age is not None
                    else "no runtime sample yet"
                )
                title = (
                    "SAVED INPUT PIXELS"
                    if self.input_gel is not None
                    else "RUNTIME TACTILE PROXY"
                )
                force_label = (
                    f"gel normal force {force:.2f} N"
                    if self.input_gel_normal is not None
                    else f"net pad force {force:.2f} N"
                )
            else:
                panel = force_proxy(
                    [0, 0, force],
                    uv=np.clip(uv, -1, 1) if known else None,
                    size=(320, 240),
                )
                location = (
                    "measured contact centroid"
                    if known
                    else "no location; centered display prior"
                )
                title = "CONTACT DIAGNOSTIC"
                force_label = f"packet normal force {force:.2f} N"
            panel[:54] = (29, 35, 42)
            panel[-27:] = (29, 35, 42)
            _label(
                panel,
                f"{'LEFT' if side == 0 else 'RIGHT'} {title}",
                (10, 22),
                color=COLORS[side],
                scale=0.44,
            )
            _label(panel, force_label, (10, 44))
            _label(panel, location, (8, 230), scale=0.4)
            canvas[52 + side * 240 : 292 + side * 240, 640:] = panel
        footer = (
            "UNCALIBRATED RUNTIME PROXY - exact supplied grayscale; student tactile modality may be omitted"
            if self.actual_input_proxy
            else "UNCALIBRATED CONTACT VISUALIZATION - no real tactile recording; not policy input pixels"
        )
        if (
            self.input_gel is not None
            and self.run.get("tactile_model") == "measured_baseline_proxy"
        ):
            footer = "UNCALIBRATED: static measured baseline + physics deformation | saved supplied pixels"
        _label(
            canvas,
            footer,
            (12, 551),
            color=(170, 195, 220),
            scale=0.43,
        )
        x0, x1, y0, y1 = 58, 945, 575, 643
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (75, 82, 90), 1)
        for value in [0, force_max]:
            yy = int(y1 - value / force_max * (y1 - y0))
            _label(canvas, f"{value:.0f}N", (6, yy + 3), scale=0.4)
        if state_index >= 0:
            for side in range(2):
                tt = self.t[: state_index + 1]
                force = self.normal[: state_index + 1, side]
                points = np.c_[
                    x0 + np.clip(tt / horizon, 0, 1) * (x1 - x0),
                    y1 - np.clip(force / force_max, 0, 1) * (y1 - y0),
                ].astype(np.int32)
                if len(points) > 1:
                    cv2.polylines(canvas, [points], False, COLORS[side], 1, cv2.LINE_AA)
        cursor = int(x0 + min(t / horizon, 1) * (x1 - x0))
        cv2.line(canvas, (cursor, y0), (cursor, y1), (160, 165, 170), 1)
        _label(canvas, "0s", (x0, 660), scale=0.38)
        _label(
            canvas, "Physical packet normal force: left / right", (320, 660), scale=0.38
        )
        _label(canvas, f"{horizon:g}s", (x1 - 30, 660), scale=0.38)
        phase = "no acquisition"
        for key, text in [
            ("acquisition", "acquired"),
            ("lift", "lifted"),
            ("carry", "carried"),
            ("release_in_bin", "released in bin"),
            ("full_task", "PLACED AND SETTLED"),
        ]:
            onset = self.metrics["event_times_s"].get(key)
            if onset is not None and onset <= t:
                phase = text
        if state_index >= 0:
            dz = (
                self.trace["waffle_position"][state_index, 2]
                - self.trace["waffle_position"][0, 2]
            )
            _label(
                canvas,
                f"Object: {phase} | rise {1000 * dz:.0f} mm | trace sample {self.t[state_index]:.3f}s",
                (12, 683),
                scale=0.45,
            )
        self.mapping.append(
            {
                "t": float(t),
                "scene_frame": frame_index,
                "scene_age_s": frame_age,
                "state_row": state_index,
                "state_age_s": state_age,
                "frozen": frozen,
                "runtime_tactile_row": input_index,
                "runtime_tactile_age_s": input_age,
                "controller_completed": completed,
                "controller_completed_reason": self.completion["completed_reason"]
                if completed
                else None,
                "display_status": status,
            }
        )
        return canvas


def make_video(runs, output, *, labels=None, fps=15.0, horizon=None):
    """Write one trial or a synchronized two-by-two comparison and JSON audit."""
    if not 1 <= len(runs) <= 4:
        raise ValueError("Provide one to four trial directories")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if labels is not None and len(labels) != len(runs):
        raise ValueError("Provide one label per run")
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        raise RuntimeError("ffmpeg is required for browser-compatible H.264 output")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing video: {output}")
    panels = []
    process = None
    temporary = output.with_name(output.stem + ".partial.mp4")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        for i, run in enumerate(runs):
            panels.append(PolicyPanel(run, labels[i] if labels else None))
        horizon = (
            max(panel.t[-1] for panel in panels) if horizon is None else float(horizon)
        )
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be finite and positive")
        cols = 1 if len(panels) == 1 else 2
        rows = math.ceil(len(panels) / cols)
        force_max = max(15.0, max(float(panel.normal.max()) for panel in panels))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as error_log:
            process = subprocess.Popen(
                [
                    encoder,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{WIDTH * cols}x{HEIGHT * rows}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "21",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                stdin=subprocess.PIPE,
                stderr=error_log,
            )
            count = math.floor(horizon * fps) + 1
            for frame in range(count):
                t = frame / fps
                canvas = np.zeros((HEIGHT * rows, WIDTH * cols, 3), np.uint8)
                for i, panel in enumerate(panels):
                    y, x = (i // cols) * HEIGHT, (i % cols) * WIDTH
                    canvas[y : y + HEIGHT, x : x + WIDTH] = panel.render(
                        t, horizon, force_max
                    )
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                error_log.seek(0)
                raise RuntimeError(error_log.read().decode(errors="replace"))
        temporary.replace(output)
        metadata = {
            "schema_version": 1,
            "video": str(output),
            "fps": fps,
            "horizon_s": horizon,
            "frames": count,
            "presentation_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "runtime_source_root": str(REPO),
                "source_root_override": os.environ.get("PHANTOM_REVIEW_SOURCE"),
            },
            "time_alignment": "same physical simulation time; previous sampled frame/state, never future; ended trials explicitly freeze",
            "contact_panels": "Prefer saved native runtime gel pixels. Older policy_tactile.npz files reconstruct the original grayscale formula; absent sidecars use packet-contact diagnostics, not policy inputs. See per-trial tactile_mapping.",
            "force_axis_max_n": force_max,
            "trials": [
                {
                    "run": str(panel.directory),
                    "label": panel.label,
                    "metrics": panel.metrics,
                    "tactile_mapping": panel.tactile_mapping,
                    "controller_completion": panel.completion,
                    "run_reported_completed_reason": panel.run.get(
                        "policy_completed_reason"
                    ),
                    "frame_mapping": panel.mapping,
                }
                for panel in panels
            ],
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        return metadata
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        for panel in panels:
            panel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", type=Path, required=True)
    parser.add_argument("--label", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--horizon", type=float)
    args = parser.parse_args()
    result = make_video(
        args.run, args.output, labels=args.label, fps=args.fps, horizon=args.horizon
    )
    print(json.dumps({key: result[key] for key in ("video", "frames", "horizon_s")}))


if __name__ == "__main__":
    main()
