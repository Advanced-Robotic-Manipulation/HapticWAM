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
STREAM_ACTIONS_QTARGET = "actions_qtarget"    # (T, dof) raw joint-space teleop targets
STREAM_ACTIONS_ABS = "actions_abs"            # (T, 7)  ABSOLUTE action: q_target(6) + gripper cmd(1)
                                              #   (Echo leader; Δ-EE stays canonical)
STREAM_ACTIONS_PLAN = "actions_plan"          # (T, 7)  deploy only: the policy's raw chunk
                                              #   PROPOSAL as entered by the executor, at
                                              #   governor-warped times, before clamp/rate
                                              #   limit. Written by the executor as `actions`
                                              #   and moved aside by
                                              #   tools/rederive_rollout_actions.py, which
                                              #   rebuilds `actions` as the MEASURED delta-EE
                                              #   on the action grid (what teleop demos hold
                                              #   and what WindowSampler trains on).
STREAM_ARM_Q = "arm_q"                        # (T, dof)
STREAM_ARM_QD = "arm_qd"                      # (T, dof)
STREAM_ARM_TCP_POSE = "arm_tcp_pose"          # (T, 6)
STREAM_ARM_TCP_SPEED = "arm_tcp_speed"        # (T, 6)
STREAM_ARM_FT = "arm_ft"                      # (T, 6)  wrist wrench
STREAM_GRIPPER = "gripper"                    # (T, 2)  [pos, obj 0..3]
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
    # operator verdict on collateral damage (object broken, gripper/table hit).
    # Independent of `success`: a run can fail cleanly or succeed destructively.
    # Also mirrored as the training-inert tag 'damaged' (eval/trial_runner).
    damage: bool = False
    notes: str = ""
    # filled by EpisodeRecorder.start:
    driver_modes: dict = field(default_factory=dict)
    clock_calibration: dict = field(default_factory=dict)
    # filled by EpisodeWriter (provenance for WindowSampler warnings) — unless
    # the caller pre-set it. Deploy pre-sets the BASE (as-loaded-from-yaml)
    # hash so run-time safety overrides do not make every rollout look like it
    # came from another rig; see `deploy_overrides`.
    config_hash: str = ""
    hardware_shapes: dict = field(default_factory=dict)
    # run-time hardware overrides applied on top of the config identified by
    # `config_hash` (deploy only): {"z_floor_m", "hitbox_m", "tcp_speed_m_s"}.
    # Recorded so the envelope an episode actually ran under stays auditable.
    deploy_overrides: dict = field(default_factory=dict)
    status: str = "recording"        # recording | finalized | aborted
    # per-episode ACTION-loss multiplier (D8 self-improvement, added
    # 2026-08-30 F19). 1.0 = an ordinary demo; failure demos are forced to 0
    # by `is_failure_demo` regardless of this value; intake writes >1 to
    # oversample a small on-policy rollout pool
    # (`clip(1/(p_task+0.2), 1, 3)`). Default 1.0 keeps every existing
    # meta.json (and every EpisodeMeta() call site) behaving exactly as
    # before — `from_dict` is tolerant, so old files simply take the default.
    weight: float = 1.0

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


def is_failure_demo(meta: "EpisodeMeta") -> bool:
    """True for a DELIBERATE failure demonstration (over-squeeze / induced slip).

    Three encodings exist in the wild and any of them counts:
      * an explicit failure verdict (`success is False`);
      * the SOP tag `deliberate_failure`;
      * a `<task>_fail` task name — how the rig actually recorded them, with
        success=True meaning "the EPISODE succeeded at capturing the intended
        failure", NOT that the manipulation succeeded.
    These episodes are boundary examples for the contact/event/slip heads and
    must never supervise action imitation.
    """
    if meta.success is False:
        return True
    if any("deliberate_failure" in str(t) for t in (meta.tags or [])):
        return True
    return str(meta.task or "").endswith("_fail")


# Tags that disqualify an episode from training entirely (not "train it at
# weight 0" like a deliberate failure demo — it must not be indexed at all).
#   unlabeled    — stopped but never judged (collect: session._file_unlabeled;
#                  deploy: runtime.run_episode's no-verdict guard)
#   contaminated — hand in frame / bumped scene / sensor glitch. NOT a failure
#                  demonstration (success stays None so is_failure_demo() does
#                  not misread it as a deliberate failure), so the tag is the
#                  only thing standing between it and full-weight training.
NON_TRAINING_TAGS: tuple[str, ...] = ("contaminated", "unlabeled")


def is_trainable_episode(meta: "EpisodeMeta") -> bool:
    """False for any episode that must never enter a training index.

    Three independent disqualifiers (P9, review 2026-08-28):
      * a NON_TRAINING_TAGS tag (`unlabeled` / `contaminated`);
      * status != 'finalized' (crashed, in flight, or aborted);
      * a POLICY ROLLOUT with no verdict — `policy` set to anything but the
        teleop marker and `success is None`. A deploy rollout the operator
        never judged is exactly the on-policy state where the model is already
        wrong; training it at action_weight 1.0 reinforces the bug.

    Deliberate failure demos stay trainable here on purpose: they carry real
    supervision for the contact/event/slip heads and WindowSampler already
    zeroes their action weight via is_failure_demo().
    """
    tags = {str(t) for t in (meta.tags or [])}
    if tags & set(NON_TRAINING_TAGS):
        return False
    if str(meta.status or "finalized") != "finalized":
        return False
    policy = str(meta.policy or "").strip()
    if policy and policy != "teleop" and meta.success is None:
        return False
    return True
