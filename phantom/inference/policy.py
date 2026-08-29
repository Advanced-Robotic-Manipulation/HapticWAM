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
from dataclasses import fields as dc_fields

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
    # (H, A) DENORMALIZED executed-past action grid, built by SnapshotBuilder
    # under --parity-fixes (P2). None => the legacy behaviour: condition on the
    # policy's own previous PROPOSAL (prev_plan.actions).
    prev_chunk: np.ndarray | None = None


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


def _cpk_row(cpk: ContactPackage, j: int) -> ContactPackage:
    """Batch row j of a contact package, kept as a (1, ...) package.

    Under K-seed sampling the denoise returns K packages; the one that must be
    fed back as the next replan's ACC self-anticipation input is the one
    belonging to the chunk that was actually SELECTED."""
    return ContactPackage(**{f.name: getattr(cpk, f.name)[j:j + 1].detach()
                             for f in dc_fields(cpk)})


class PhantomPolicy:
    def __init__(self, pm: PhantomModel, norm: NormStats, *, nfe: int | None = None,
                 drop_video: bool = False, task_text: str = "",
                 persistent_noise: bool = False, guidance: float = 1.0,
                 parity_fixes: bool = False, k_seeds: int = 1,
                 close_p: float = 0.5):
        # task_text: the per-episode instruction (pipeline.md input l). Must
        # match a key of the text-embedding cache the teacher trained with;
        # empty keeps the v2 empty-string conditioning.
        self.task_text = task_text
        # fixed initial noise per episode: fresh randn each replan re-rolled
        # the plan direction (rig-measured consec-replan cosine 0.16-0.35)
        self.persistent_noise = persistent_noise
        # observation-guidance (classifier-free) weight; >1 sharpens the
        # obs->action coupling cond_dropout trained for, at ~2x denoise cost
        self.guidance = float(guidance)
        self.pm = pm
        self.rf = pm.rf
        self.hw = pm.hw
        self.bb = pm.bb
        self.norm = norm
        self.nfe = nfe or pm.mc.nfe
        self.drop_video = drop_video
        # --parity-fixes (P2): also realign the ACC self-anticipation input.
        # `ContactPacker.flatten_summary` summarises step `s` of the package,
        # default 0. Training's two_pass proxy predicts the package at the SAME
        # t0 as the window, so step 0 there means "contact at t0 + latent_dt";
        # at deploy the package handed back was produced one replan (latency L)
        # EARLIER, so its step 0 means "contact at t0 - L + latent_dt". The
        # aligned index is round(L / latent_dt).
        self.parity_fixes = bool(parity_fixes)
        self.latent_dt = pm.bb.temporal_comp / pm.bb.fps
        # K-seed sampling with contact-consistent selection (P6 / BID 2408.17355)
        self.k_seeds = max(1, int(k_seeds))
        self.close_p = float(close_p)
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
        if obs.prev_chunk is not None:
            # --parity-fixes: the EXECUTED past (measured pose_delta chain +
            # the gripper command actually sent), the way WindowSampler builds
            # prev_chunk in training. On the first replan of an episode the arm
            # is holding the start pose, so this is the zero-physical-action
            # chunk the branch below constructs — normalized identically.
            prev = self.norm.normalize("action", obs.prev_chunk.astype(np.float32))
        elif prev_plan is not None and prev_plan.actions is not None:
            prev = self.norm.normalize("action", prev_plan.actions.astype(np.float32))
        else:
            # "no previous chunk" = zero PHYSICAL action, which must be
            # normalized like any other action — raw zeros in normalized space
            # decode to the per-dim demo MEAN offset (gripper channel: -1.5
            # sigma), an off-distribution first-replan conditioning
            prev = self.norm.normalize(
                "action", np.zeros((hw.control.chunk_horizon,
                                    hw.control.action_dim), dtype=np.float32))
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

    def reset_episode(self) -> None:
        """Per-episode state reset (held sampling noise)."""
        self.rf.reset_episode_noise()

    # ------------------------------------------------------------------
    # K-seed selection (P6). Head = the steps the executor actually reaches
    # before the next replan: at the rig's ~0.9-1.0 s cadence and 10 Hz action
    # grid, steps 0-8 (the tail beyond that has never executed — P4).
    HEAD_STEPS = 9
    HEAD_DZ_KEEP = 0.5      # reject a chunk descending < 50% of the K-max ...
                            # ... but only while p_contact says we are not there

    def _select_seed(self, actions_K_H_A: np.ndarray, p_contact: float,
                     prev_plan: Plan | None, t_exec0: float,
                     rate: float) -> tuple[int, dict]:
        """Contact-consistent chunk selection over K sampled chunks.

        1. head descent = cumulative -dz over steps 0..HEAD_STEPS-1;
        2. while p_contact is LOW (< close_p), reject any chunk descending less
           than HEAD_DZ_KEEP of the K-max — the under-commit mode this whole
           lever exists for (BID 2408.17355's "reject the timid mode");
        3. among the survivors take the one nearest the previous ACCEPTED plan
           over the steps the two chunks share in time (L2 on the 6 pose
           channels) — mode-consistency across replans, which the rig's
           0.16-0.35 consecutive-replan cosine says is otherwise absent;
        4. fall back to the first chunk.
        """
        K = actions_K_H_A.shape[0]
        n = min(self.HEAD_STEPS, actions_K_H_A.shape[1])
        head_dz = -actions_K_H_A[:, :n, 2].sum(axis=1)      # + = descending
        keep = np.arange(K)
        dmax = float(head_dz.max()) if K else 0.0
        if K > 1 and dmax > 0.0 and p_contact < self.close_p:
            surv = np.nonzero(head_dz >= self.HEAD_DZ_KEEP * dmax)[0]
            if surv.size:
                keep = surv
        pick = int(keep[0])
        if prev_plan is not None and prev_plan.actions is not None and keep.size > 1:
            off = int(np.clip(round((t_exec0 - float(prev_plan.action_times[0])) * rate),
                              0, prev_plan.actions.shape[0] - 1))
            ref = prev_plan.actions[off:, :6]
            m = min(ref.shape[0], actions_K_H_A.shape[1])
            if m > 0:
                d = np.linalg.norm(
                    actions_K_H_A[keep, :m, :6] - ref[None, :m], axis=(1, 2))
                pick = int(keep[int(np.argmin(d))])
        return pick, {"k_seeds": K,
                      "head_dz_mm": [round(float(v) * 1000, 2) for v in head_dz],
                      "head_dz_spread_mm": round(float(head_dz.max() - head_dz.min())
                                                 * 1000, 2) if K else 0.0,
                      "k_rejected": int(K - keep.size), "k_pick": pick}

    @torch.no_grad()
    def replan(self, obs: ObsSnapshot, prev_plan: Plan | None,
               tcp_pose: np.ndarray) -> Plan:
        t_start = time.perf_counter()
        batch = self._batch_from_obs(obs, prev_plan)
        cpk_step = 0
        if self.parity_fixes and prev_plan is not None:
            cpk_step = int(round(float(prev_plan.latency_s) / self.latent_dt))
        pred: PhantomPrediction = self.rf.sample(
            batch, nfe=self.nfe, guidance_scale=self.guidance,
            prev_cpk=prev_plan.cpk if prev_plan is not None else None,
            prev_cpk_step=cpk_step,
            drop_video=self.drop_video,
            reuse_noise=self.persistent_noise,
            k_seeds=self.k_seeds)
        rate = self.hw.control.action_rate_hz
        latency = time.perf_counter() - t_start
        t_exec0 = obs.t + latency
        acts_K = self.norm.denormalize(
            "action", pred.actions_B_H_A.float().cpu()).numpy()
        p_evt0 = (pred.acc.p_evt[0].float().cpu().numpy()
                  if pred.acc is not None else np.zeros(5))
        j, sel = self._select_seed(acts_K, 1.0 - float(p_evt0[0]),
                                   prev_plan, t_exec0, rate)
        actions = acts_K[j]
        diag = {"nfe": self.nfe, "guidance": self.guidance,
                "event_logits": pred.event_logits_B_Tc_E[j].float().cpu().numpy()}
        if self.k_seeds > 1:
            diag.update(sel)
        return Plan(
            t_created=obs.t,
            t0_pose=np.asarray(tcp_pose, dtype=np.float64).copy(),
            actions=actions,
            action_times=t_exec0 + np.arange(actions.shape[0]) / rate,
            sigma=pred.governor_sigma_B_Tc[j].float().cpu().numpy(),
            gate=float(pred.acc.g[j]) if pred.acc is not None else 0.0,
            p_evt=(pred.acc.p_evt[j].float().cpu().numpy()
                   if pred.acc is not None else np.zeros(5)),
            cpk=_cpk_row(pred.cpk, j) if self.k_seeds > 1 else pred.cpk.detach(),
            latency_s=latency,
            diag=diag,
        )
