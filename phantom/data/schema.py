"""Episode schema: stream names, EpisodeMeta, NormStats.

Layout on disk (one directory per episode, name starts with "ep_"):

    <episode>/meta.json                 EpisodeMeta + provenance
    <episode>/<stream>.zarr/            zarr group: "data" (T, ...) + "ts" (T,)

"ts" is t_master seconds (MasterClock domain). Stream names are the constants
below; per-sensor tactile streams come from tactile_stream(sensor_name, kind).

No torch imports here — the recording stack must run with base deps only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# stream names
# ---------------------------------------------------------------------------

STREAM_ACTIONS = "actions"                    # (T, 7)  teleop / executor commands
STREAM_ARM_Q = "arm_q"                        # (T, dof)
STREAM_ARM_QD = "arm_qd"                      # (T, dof)
STREAM_ARM_TCP_POSE = "arm_tcp_pose"          # (T, 6)
STREAM_ARM_TCP_SPEED = "arm_tcp_speed"        # (T, 6)
STREAM_ARM_FT = "arm_ft"                      # (T, 6)  wrist wrench
STREAM_GRIPPER = "gripper"                    # (T, 2)  [pos, current]
# camera streams follow f"camera_{name}_color" (recorder.py writes the literal)
STREAM_CAMERA_SCENE = "camera_scene_color"    # (T, h, w, 3) uint8
STREAM_CAMERA_WRIST = "camera_wrist_color"    # (T, h, w, 3) uint8


def tactile_stream(sensor_name: str, kind: str) -> str:
    """Per-sensor tactile stream name.

    Recorded kinds: fields_ds, keyframes, wrench, area, infer_img, raw_img.
    Derived kinds (written by recording/postprocess.py): mask_frac, cop,
    slip, events.
    """
    return f"tactile_{sensor_name}_{kind}"


# ---------------------------------------------------------------------------
# episode metadata
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMeta:
    """meta.json contents. Only `task` is required; every caller passes a
    different subset (mock_smoke: operator; deploy runtime: tags/policy/
    dagger_round), so everything else defaults."""
    task: str
    text: str = ""
    operator: str = ""
    tags: list = field(default_factory=list)
    policy: str = ""
    dagger_round: int = -1
    success: bool | None = None
    notes: str = ""
    # filled by EpisodeRecorder.start:
    driver_modes: dict = field(default_factory=dict)
    clock_calibration: dict = field(default_factory=dict)
    # filled by EpisodeWriter (provenance for WindowSampler warnings):
    config_hash: str = ""
    hardware_shapes: dict = field(default_factory=dict)
    status: str = "recording"        # recording | finalized | aborted

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeMeta":
        """Tolerant loader: ignores unknown keys, fills missing with defaults."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, default=str))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "EpisodeMeta":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# normalization statistics
# ---------------------------------------------------------------------------

def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


@dataclass
class NormStats:
    """Per-key channel-wise (last axis) normalization. Keys written by
    scripts/dump_norm_stats.py: fields, wrench, area, wrist_ft, ur_state,
    action, cpk_d_disp, cpk_d_fz. Unknown keys pass through unchanged, so an
    identity/partial stats file degrades gracefully."""
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)

    @classmethod
    def identity(cls) -> "NormStats":
        return cls()

    # ------------------------------------------------------------------
    def _mom(self, key: str, arr):
        m, s = self.mean[key], self.std[key]
        if _is_torch(arr):
            import torch
            m = torch.as_tensor(np.asarray(m), dtype=arr.dtype, device=arr.device)
            s = torch.as_tensor(np.asarray(s), dtype=arr.dtype, device=arr.device)
        return m, s

    def normalize(self, key: str, arr):
        if key not in self.mean:
            return arr
        m, s = self._mom(key, arr)
        return (arr - m) / s

    def denormalize(self, key: str, arr):
        if key not in self.mean:
            return arr
        m, s = self._mom(key, arr)
        return arr * s + m

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        payload = {"mean": {k: np.asarray(v).tolist() for k, v in self.mean.items()},
                   "std": {k: np.asarray(v).tolist() for k, v in self.std.items()}}
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = json.loads(Path(path).read_text())
        return cls(mean={k: np.asarray(v, dtype=np.float32) for k, v in d["mean"].items()},
                   std={k: np.asarray(v, dtype=np.float32) for k, v in d["std"].items()})
