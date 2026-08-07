"""PhantomPolicy: the runtime-facing inference API — one replan() call per
model tick group (pipeline.md §6e).

Consumes an ObsSnapshot (built by deploy/planner from the ring buffers),
produces a Plan: denormalized action chunk on the action-rate grid, the
generated contact package (next replan's ACC self-anticipation input), the
governor sigma, and the ACC gate diagnostics.

AR grounding is structural: every replan reads REAL sensors from the rings;
imagination only re-enters through prev_plan.cpk (the designed one-step
staleness of ACC).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from phantom.data.schema import NormStats
from phantom.data.windows import bilinear_resize
from phantom.model.ace.packing import ContactPackage
from phantom.model.rf import PhantomPrediction
from phantom.train.builder import PhantomModel


@dataclass
class ObsSnapshot:
    """Everything the planner read from the rings at replan time (t_master)."""
    t: float
    rgb: np.ndarray                      # (h, w, 3) uint8 current scene frame
    wrist_window: np.ndarray             # (L, 6) float
    ur_state: np.ndarray                 # (ur_state_dim,)
    gel: np.ndarray | None = None        # (F, hg, wg[, c]) uint8 (teacher mode)
    fields: np.ndarray | None = None     # (F, Hf, Wf, 8) float (teacher mode)
    contact_state: np.ndarray | None = None  # (F, contact_state_dim)
    reactive: float = 0.0


@dataclass
class Plan:
    t_created: float
    t0_pose: np.ndarray                  # TCP pose at plan creation (delta reference)
    actions: np.ndarray                  # (H, A) DENORMALIZED: Δ-EE (6) + gripper (1)
    action_times: np.ndarray             # (H,) t_master execution grid
    sigma: np.ndarray                    # (Tc,) governor uncertainty per future step
    gate: float
    p_evt: np.ndarray                    # (E,)
    cpk: ContactPackage | None
    latency_s: float = 0.0
    diag: dict = field(default_factory=dict)


class PhantomPolicy:
    def __init__(self, pm: PhantomModel, norm: NormStats, *, nfe: int | None = None,
                 drop_video: bool = False, task_text: str = ""):
        # task_text: the per-episode instruction (pipeline.md input l). Must
        # match a key of the text-embedding cache the teacher trained with;
        # empty keeps the v2 empty-string conditioning.
        self.task_text = task_text
        self.pm = pm
        self.rf = pm.rf
        self.hw = pm.hw
        self.bb = pm.bb
        self.norm = norm
        self.nfe = nfe or pm.mc.nfe
        self.drop_video = drop_video
        self.rf.eval()

    # ------------------------------------------------------------------
    def _batch_from_obs(self, obs: ObsSnapshot, prev_plan: Plan | None) -> dict:
        hw, bb = self.hw, self.bb
        dev, dt = self.rf.device, self.rf.dtype
        # video: current frame tiled across the pixel window (only the first
        # latent frame is conditioning; VIDEO_GEN slots start from pure noise)
        frame = bilinear_resize(obs.rgb.astype(np.float32), (bb.res_h, bb.res_w))
        video = np.repeat(frame[None], bb.frames_pix, axis=0)
        batch: dict = {
            "video": torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0)
            / 127.5 - 1.0,
            "wrist": torch.from_numpy(
                self.norm.normalize("wrist_ft", obs.wrist_window.astype(np.float32))
            ).unsqueeze(0),
            "ur_state": torch.from_numpy(
                self.norm.normalize("ur_state", obs.ur_state.astype(np.float32))
            ).unsqueeze(0),
            "reactive": torch.tensor([obs.reactive], dtype=torch.float32),
            # the instruction the teacher was conditioned on during training;
            # empty -> the provider's empty-string embedding (v2-compatible)
            "text": [self.task_text],
        }
        if prev_plan is not None and prev_plan.actions is not None:
            prev = self.norm.normalize("action", prev_plan.actions.astype(np.float32))
        else:
            prev = np.zeros((hw.control.chunk_horizon, hw.control.action_dim),
                            dtype=np.float32)
        batch["prev_chunk"] = torch.from_numpy(prev).unsqueeze(0)

        if not self.pm.layout.student:
            assert obs.gel is not None and obs.fields is not None, \
                "teacher mode needs tactile observations in the snapshot"
            gel = obs.gel.astype(np.float32)
            if gel.ndim == 3:
                gel = np.repeat(gel[..., None], 3, axis=-1)
            gel = np.stack([bilinear_resize(g, (bb.res_h, bb.res_w)) for g in gel])
            batch["gel"] = torch.from_numpy(gel).permute(0, 3, 1, 2).unsqueeze(0) \
                / 127.5 - 1.0
            batch["fields"] = torch.from_numpy(
                self.norm.normalize("fields", obs.fields.astype(np.float32))
            ).unsqueeze(0)
            batch["contact_state"] = torch.from_numpy(
                obs.contact_state.astype(np.float32)).unsqueeze(0)

        # zero contact-package placeholders (gen-frame x0 content is unused;
        # ACC's real prev-cpk is passed separately to sample())
        Tc = bb.t_video - 1
        Fn = hw.n_fingers
        cph, cpw = hw.cpk_shape
        batch.update({
            "events": torch.zeros(1, Tc, dtype=torch.long),
            "cpk_d_disp": torch.zeros(1, Tc, Fn, 3, cph, cpw),
            "cpk_d_fz": torch.zeros(1, Tc, Fn, cph, cpw),
            "cpk_mask": torch.zeros(1, Tc, Fn, cph, cpw),
            "cpk_cop": torch.zeros(1, Tc, Fn, 2),
            "cpk_slip": torch.zeros(1, Tc, Fn),
            "cpk_wrench": torch.zeros(1, Tc, Fn, 6),
            "cpk_wrist": torch.zeros(1, Tc, 6),
            "action_chunk": torch.zeros(1, hw.control.chunk_horizon,
                                        hw.control.action_dim),
            "gate_label": torch.zeros(1),
        })
        return batch

    # ------------------------------------------------------------------
    @torch.no_grad()
    def replan(self, obs: ObsSnapshot, prev_plan: Plan | None,
               tcp_pose: np.ndarray) -> Plan:
        t_start = time.perf_counter()
        batch = self._batch_from_obs(obs, prev_plan)
        pred: PhantomPrediction = self.rf.sample(
            batch, nfe=self.nfe,
            prev_cpk=prev_plan.cpk if prev_plan is not None else None,
            drop_video=self.drop_video)
        actions = self.norm.denormalize(
            "action", pred.actions_B_H_A[0].float().cpu()).numpy()
        rate = self.hw.control.action_rate_hz
        latency = time.perf_counter() - t_start
        t_exec0 = obs.t + latency
        return Plan(
            t_created=obs.t,
            t0_pose=np.asarray(tcp_pose, dtype=np.float64).copy(),
            actions=actions,
            action_times=t_exec0 + np.arange(actions.shape[0]) / rate,
            sigma=pred.governor_sigma_B_Tc[0].float().cpu().numpy(),
            gate=float(pred.acc.g[0]) if pred.acc is not None else 0.0,
            p_evt=(pred.acc.p_evt[0].float().cpu().numpy()
                   if pred.acc is not None else np.zeros(5)),
            cpk=pred.cpk.detach(),
            latency_s=latency,
            diag={"nfe": self.nfe,
                  "event_logits": pred.event_logits_B_Tc_E[0].float().cpu().numpy()},
        )
