"""Offline teacher relabeling of student rollout episodes (pipeline.md §5:
'the teacher can, since the rig still wears the sensors during training').

For each rollout episode, iterates its replan-rate anchor grid, rebuilds the
TEACHER's full-sensor observation windows from the recorded streams, runs the
teacher, and stores its action chunks + contact packages + confidences in
<episode>/relabels/teacher.pt. Pure offline batch job (8xH100 or the 5090).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import HardwareConfig
from phantom.config.paths import PathsConfig
from phantom.data.episode_store import list_episodes
from phantom.data.schema import NormStats
from phantom.data.windows import WindowSampler
from phantom.model.sequence import FrameGroup
from phantom.train import common as C
from phantom.train.builder import build_model

log = logging.getLogger(__name__)


def relabel_episode(ep: Path, teacher_rf, sampler: WindowSampler,
                    replan_period_s: float) -> int:
    lo, hi = sampler.valid_range(ep)
    anchors = np.arange(lo, hi, replan_period_s)
    if not len(anchors):
        log.warning("%s too short to relabel", ep.name)
        return 0
    records = []
    for t0 in anchors:
        batch = C.collate_windows([sampler.sample(ep, float(t0))])
        batch = C.to_device(batch, str(teacher_rf.device))
        with torch.no_grad():
            pred = teacher_rf.sample(batch)
        records.append({
            "t0": float(t0),
            "actions": pred.actions_B_H_A[0].float().cpu(),
            "event_logits": pred.event_logits_B_Tc_E[0].float().cpu(),
            "log_sigma": pred.log_sigma_B_Tc_K[0].float().cpu(),
            "cpk_packed": pred.x_final_B_C_T_H_W[
                0, :, teacher_rf.layout.frame_slice(FrameGroup.CONTACT)].float().cpu(),
        })
    out_dir = ep / "relabels"
    out_dir.mkdir(exist_ok=True)
    torch.save({"records": records}, out_dir / "teacher.pt")
    return len(records)


def relabel_root(root: Path, teacher_ckpt: Path, hw: HardwareConfig,
                 paths: PathsConfig, *, device: str = "cuda",
                 tiny: bool = False) -> int:
    dtype = torch.bfloat16 if (device == "cuda" and not tiny) else torch.float32
    teacher = build_model(hw, paths, student=False, tiny=tiny,
                          load_base=not tiny, device=device, dtype=dtype)
    payload = C.load_phantom_checkpoint(Path(teacher_ckpt), teacher.rf, hw=hw)
    teacher.rf.eval()
    norm = NormStats.identity()
    if payload.get("norm_stats"):
        ns = payload["norm_stats"]
        norm = NormStats(
            mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
            std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})
    sampler = WindowSampler(hw, teacher.bb, norm, student=False)
    replan_period = hw.control.chunk_horizon / hw.control.action_rate_hz

    n = 0
    for ep in list_episodes(root):
        try:
            k = relabel_episode(ep, teacher.rf, sampler, replan_period)
            log.info("relabeled %s: %d replans", ep.name, k)
            n += 1
        except Exception:
            log.exception("relabel failed for %s", ep.name)
    return n
