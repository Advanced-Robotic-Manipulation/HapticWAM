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

from phantom.inference.action_timing import (
    validate_action_time_origin, observation_cpk_history,
)
from phantom.data.schema import NormStats
from phantom.data.windows import bilinear_resize
from phantom.model.ace.packing import ContactPackage
from phantom.model.rf import PhantomPrediction
from phantom.model.sequence import FrameGroup
from phantom.train.builder import PhantomModel

#: K-candidate selectors `replan` can rank the sampled chunks with.
#: "default"         — the contact-consistent head-descent rule (`_select_seed`)
#: "video_agreement" — the imagined-future agreement rule below
SELECTORS = ("default", "video_agreement")

#: Deploy-time causal probe on the IMAGINED contact package — the rig-side
#: twin of `tools/terminal_eval.py --null`, with identical mechanics (the
#: evaluator's `zero_package` / `contact_pinned_layout` are the same objects).
#: The student has no tactile pads, so its contact package is pure imagination;
#: corrupting it at inference and pairing cell-by-cell against the intact
#: student is the deploy answer to "does the imagination carry the actions?".
#:
#: "none"         — the shipped path, byte-identical (the default)
#: "prev_cpk"     — every replan hands ACC a ZERO ContactPackage as the
#:                  previous replan's package, at the usual `prev_cpk_step`.
#:                  Including the FIRST replan, so the two-pass inner
#:                  anticipation that would otherwise predict one is bypassed
#:                  too: the ACC intent channel carries no contact all episode.
#: "contact_zero" — the CONTACT frames are cond-PINNED to the zero package at
#:                  every denoise step (`rf.sample(pin_contact_x0=True)`), so
#:                  the ACTION queries attend a contact block the model was not
#:                  allowed to imagine anything into.
NULL_IMAGINATION_MODES = ("none", "prev_cpk", "contact_zero")


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


def video_gen_latents(x_final_B_C_T_H_W, layout) -> list[np.ndarray]:
    """Per candidate: the denoised VIDEO_GEN x0 latents, (C, T_gen, h, w).

    `rf.sample`'s Euler loop ends at t=0, so `x_final` IS the x0 estimate, and
    under K-seed sampling its rows are the K candidates in the SAME order as
    `actions_B_H_A` (rf.sample repeat_interleaves every conditioning tensor).
    Identical to tools/terminal_eval.py's `video_gen_latents`, which reads row
    0 because there each seed is its own single-sample call."""
    if not layout.has(FrameGroup.VIDEO_GEN):
        raise ValueError("select_by='video_agreement' needs VIDEO_GEN frames: "
                         "this layout imagines nothing (drop_video)")
    sl = layout.frame_slice(FrameGroup.VIDEO_GEN)
    x = x_final_B_C_T_H_W
    return [x[j, :, sl].float().cpu().numpy() for j in range(x.shape[0])]


def video_agreement_scores(vids) -> list[float]:
    """Per candidate: MSE between its imagined future and the ELEMENTWISE
    MEDIAN of all K imagined futures.

    Needs no ground truth, so unlike the hindsight `video_err` it is computable
    at deploy. THE definition is tools/terminal_eval.py's `seed_agreement`, the
    column the offline sweep called `video_err_to_median`; this function must
    stay byte-identical to it (tests/test_video_agreement_selector.py pins the
    two against each other on random latents).

    With a single candidate every distance is 0 by construction."""
    a = np.stack([np.asarray(v, dtype=np.float64) for v in vids])
    med = np.median(a, axis=0)
    return [float(((v - med) ** 2).mean()) for v in a]


def select_by_video_agreement(scores) -> int:
    """The candidate closest to the K-median imagination (ties -> lowest index,
    matching the offline `min(window, key=...)`)."""
    return int(np.argmin(np.asarray(scores, dtype=np.float64)))


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
                 close_p: float = 0.5, select_by: str = "default",
                 agreement_veto: float | None = None,
                 agreement_shadow: bool = False,
                 action_time_origin: str = "inference_ready",
                 null_imagination: str = "none"):
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
        # which rule ranks the K candidates (SELECTORS). "video_agreement" is
        # the opt-in imagined-future rule: offline over 124 val windows x 4
        # seeds it moved mean endpoint error 20.04 -> 19.63 mm (v6), 16.25 ->
        # 15.36 (v6 nfe2), 19.57 -> 19.20 (ftA), 18.68 -> 18.43 (sensor-free
        # student), and flags catastrophic actions (> 28.2 mm) at AUC 0.64-0.81
        # where the learned gate and the governor sigma are at chance.
        self.select_by = select_by
        # optional plan-level gate on the SELECTED candidate's agreement
        # distance: above it the plan is marked for rejection and the planner
        # keeps the previous plan (no new arm behaviour).
        self.agreement_veto = agreement_veto
        # opt-in SHADOW scoring (--agreement-shadow): with the default rule
        # still choosing the chunk, also compute the K agreement distances and
        # record which candidate the imagined-future rule WOULD have kept
        # (diag only, never the pick) — the rig-side record of how often the
        # two rules disagree and how agreement relates to the outcome.
        self.agreement_shadow = agreement_shadow
        self.action_time_origin = action_time_origin
        # deploy-time causal probe on the student's IMAGINED contact package
        # (the ACC channel). See NULL_IMAGINATION_MODES.
        self.null_imagination = null_imagination
        self.rf.eval()

    @property
    def k_seeds(self):
        return self._k_seeds

    @k_seeds.setter
    def k_seeds(self, value):
        if getattr(self, "_action_time_origin", "inference_ready") == "observation" and value != 1:
            raise ValueError("observation action epoch candidate supports K1 only")
        self._k_seeds = value

    @property
    def select_by(self):
        # policies built without the selector setting (tests, legacy loaders)
        # rank with the default contact-consistent rule
        return getattr(self, "_select_by", "default")

    @select_by.setter
    def select_by(self, value):
        value = "default" if value is None else str(value)
        if value not in SELECTORS:
            raise ValueError(f"select_by must be one of {SELECTORS}, got {value!r}")
        self._select_by = value

    @property
    def agreement_shadow(self) -> bool:
        # absent on policies built before the flag existed => off
        return bool(getattr(self, "_agreement_shadow", False))

    @agreement_shadow.setter
    def agreement_shadow(self, value):
        self._agreement_shadow = bool(value)

    @property
    def agreement_veto(self):
        return getattr(self, "_agreement_veto", None)

    @agreement_veto.setter
    def agreement_veto(self, value):
        if value is None:
            self._agreement_veto = None
            return
        value = float(value)
        if not (value > 0.0):
            raise ValueError("agreement_veto must be a positive distance "
                             f"(the K-median MSE threshold), got {value!r}")
        self._agreement_veto = value

    @property
    def null_imagination(self) -> str:
        # policies built before the probe existed => off
        return getattr(self, "_null_imagination", "none")

    @null_imagination.setter
    def null_imagination(self, value):
        value = "none" if value is None else str(value)
        if value not in NULL_IMAGINATION_MODES:
            raise ValueError("null_imagination must be one of "
                             f"{NULL_IMAGINATION_MODES}, got {value!r}")
        self._null_imagination = value

    @property
    def action_time_origin(self):
        # policies built without the timing settings (tests, legacy loaders) are inference_ready
        return getattr(self, "_action_time_origin", "inference_ready")

    @action_time_origin.setter
    def action_time_origin(self, value):
        validate_action_time_origin(value)
        if value == "observation" and self.k_seeds != 1:
            raise ValueError("observation action epoch candidate supports K1 only")
        self._action_time_origin = value

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

    def _check_video_agreement(self) -> None:
        """Refuse the combinations for which the score does not exist, rather
        than ranking constants: K=1 makes every distance 0 by construction, and
        --drop-video deletes the VIDEO_GEN frames the score is computed on (the
        sampler then denoises a SHORTER sequence, so this layout's frame slice
        would not even address the imagined frames)."""
        if self.k_seeds < 2:
            raise ValueError("select_by='video_agreement' needs --k-seeds >= 2:"
                             " with one candidate the agreement distance is 0 "
                             "by construction and ranks nothing")
        if self.drop_video:
            raise ValueError("select_by='video_agreement' is incompatible with "
                             "drop_video: there is no imagined future to score")

    def _select_video_agreement(self, pred: PhantomPrediction) -> tuple[int, dict]:
        """Agreement selection over the K imagined futures (GT-free).

        Take each candidate's denoised VIDEO_GEN x0 latents, form their
        elementwise median, and keep the candidate closest to it — the seed
        whose imagined future the other seeds agree with. Offline this is the
        only one of the deploy-time scores (learned ACC gate, governor sigma)
        that separates catastrophic actions from the rest at all."""
        self._check_video_agreement()
        scores = video_agreement_scores(
            video_gen_latents(pred.x_final_B_C_T_H_W, self.pm.layout))
        pick = select_by_video_agreement(scores)
        diag = {"k_seeds": len(scores), "k_pick": pick,
                "k_selection": "video_agreement",
                # the K distances: the only record of how much the seeds
                # disagreed on this replan, and what a veto threshold is read
                # off (kept by the planner trace's numeric-list filter)
                "video_agreement": [float(v) for v in scores],
                "video_agreement_pick": float(scores[pick]),
                "video_agreement_spread": float(max(scores) - min(scores))}
        thr = self.agreement_veto
        if thr is not None:
            diag["agreement_veto_threshold"] = float(thr)
            # read by PlannerLoop.run: True => do not submit this plan
            diag["agreement_vetoed"] = bool(scores[pick] > float(thr))
        return pick, diag

    def _shadow_video_agreement(self, pred: PhantomPrediction, pick: int) -> dict:
        """Diag-only companion of the default rule (opt-in, --agreement-shadow):
        the K agreement distances and the candidate the imagined-future rule
        would have kept, next to the default rule's actual pick. Nothing here
        touches the chosen chunk; a failure to score is recorded, not raised,
        so the shadow can never cost a replan."""
        try:
            scores = video_agreement_scores(
                video_gen_latents(pred.x_final_B_C_T_H_W, self.pm.layout))
            spick = select_by_video_agreement(scores)
            return {"video_agreement_shadow": [float(v) for v in scores],
                    "video_agreement_shadow_pick": int(spick),
                    "video_agreement_shadow_agrees": int(spick == int(pick)),
                    "video_agreement_shadow_of_pick": float(scores[int(pick)]),
                    "video_agreement_shadow_spread": float(max(scores) - min(scores))}
        except Exception as e:  # noqa: BLE001 — diag only
            return {"video_agreement_shadow_error": str(e)[:160]}

    @torch.no_grad()
    def replan(self, obs: ObsSnapshot, prev_plan: Plan | None,
               tcp_pose: np.ndarray) -> Plan:
        t_start = time.perf_counter()
        batch = self._batch_from_obs(obs, prev_plan)
        cpk_step = 0
        previous_cpk = prev_plan.cpk if prev_plan is not None else None
        cpk_timing = {}
        if self.action_time_origin == "observation":
            previous_cpk, cpk_step, cpk_timing = observation_cpk_history(
                obs.t, prev_plan, self.latent_dt)
        elif self.parity_fixes and prev_plan is not None:
            cpk_step = int(round(float(prev_plan.latency_s) / self.latent_dt))
        null_imag = self.null_imagination
        if null_imag == "prev_cpk":
            # EVERY replan, the first included: a zero package is still a
            # package, so rf.sample takes the cheap branch and skips the inner
            # anticipation sample it would otherwise run on replan 1. The step
            # index is left exactly as computed above — the probe nulls the
            # package's CONTENT, not the parity alignment.
            previous_cpk = self.rf.c_pack.zero_package(
                batch=1, device=self.rf.device, dtype=torch.float32)
        pred: PhantomPrediction = self.rf.sample(
            batch, nfe=self.nfe, guidance_scale=self.guidance,
            prev_cpk=previous_cpk,
            prev_cpk_step=cpk_step,
            drop_video=self.drop_video,
            reuse_noise=self.persistent_noise,
            k_seeds=self.k_seeds,
            pin_contact_x0=(null_imag == "contact_zero"))
        rate = self.hw.control.action_rate_hz
        latency = time.perf_counter() - t_start
        t_exec0 = (obs.t if self.action_time_origin == "observation"
                   else obs.t + latency)
        acts_K = self.norm.denormalize(
            "action", pred.actions_B_H_A.float().cpu()).numpy()
        p_evt0 = (pred.acc.p_evt[0].float().cpu().numpy()
                  if pred.acc is not None else np.zeros(5))
        if self.action_time_origin == "observation":
            # K>1 needs a separately bound capped selector. This candidate
            # deliberately permits one sample only; no new ranking heuristic.
            j, sel = 0, {"k_seeds": 1, "k_pick": 0,
                         "k_selection": "single_sample_no_selection"}
        elif self.select_by == "video_agreement":
            j, sel = self._select_video_agreement(pred)
        else:
            j, sel = self._select_seed(acts_K, 1.0 - float(p_evt0[0]),
                                       prev_plan, t_exec0, rate)
            if self.agreement_shadow and self.k_seeds > 1 and not self.drop_video:
                sel.update(self._shadow_video_agreement(pred, j))
        actions = acts_K[j]
        diag = {"nfe": self.nfe, "guidance": self.guidance,
                "event_logits": pred.event_logits_B_Tc_E[j].float().cpu().numpy()}
        if self.k_seeds > 1 or self.action_time_origin == "observation":
            diag.update(sel)
        if self.action_time_origin == "observation":
            diag.update(cpk_timing)
            diag.update(action_time_origin="observation",
                        action_interval_convention="start_grid_end_labeled_delta",
                        action_observation_epoch_s=float(obs.t))
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
