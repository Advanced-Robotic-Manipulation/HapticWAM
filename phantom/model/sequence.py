"""SequenceLayout — the single source of truth for the extended latent-frame
axis of the PHANTOM diffusion sequence (pipeline.md §6b/§6e).

Teacher (repo defaults):                     Student (student=True):
    [VIDEO_COND | VIDEO_GEN x3 |                 [VIDEO_COND | VIDEO_GEN x3 |
     OBS_GEL | OBS_MECH | OBS_PROPRIO |           OBS_PROPRIO |
     CONTACT x3 | ACTION x4]                      CONTACT x3 | ACTION x4]
    T=14 -> 4480 tokens                          T=12 -> 3840 tokens

    video_cond  [0:1)                            video_cond  [0:1)
    video_gen   [1:4)                            video_gen   [1:4)
    obs_gel     [4:5)                            obs_proprio [4:5)
    obs_mech    [5:6)                            contact     [5:8)
    obs_proprio [6:7)                            action      [8:12)
    contact     [7:10)
    action      [10:14)

(OBS_MECH is mc.contact_obs_frames frames — 1 by default; ACTION is
hw.control.chunk_horizon // bb.actions_per_latent_frame — 4 on this rig.)

All frames are (lat_ch, lat_h, lat_w) latents through the pretrained patch
embedder. Everything that indexes the T axis or the flattened token axis goes
through this class — no slicing arithmetic anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from enum import Enum

import numpy as np

from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import HardwareConfig
from phantom.config.model import PhantomModelConfig


class FrameGroup(str, Enum):
    VIDEO_COND = "video_cond"
    VIDEO_GEN = "video_gen"
    OBS_GEL = "obs_gel"
    OBS_MECH = "obs_mech"
    OBS_PROPRIO = "obs_proprio"
    CONTACT = "contact"
    ACTION = "action"


# groups that are pinned conditioning (FRAME_REPLACE mask = 1, never denoised)
COND_GROUPS = frozenset({FrameGroup.VIDEO_COND, FrameGroup.OBS_GEL,
                         FrameGroup.OBS_MECH, FrameGroup.OBS_PROPRIO})
# teacher-only observation groups (deleted in the student)
TEACHER_ONLY_GROUPS = frozenset({FrameGroup.OBS_GEL, FrameGroup.OBS_MECH})


@dataclass(frozen=True)
class GroupSlot:
    group: FrameGroup
    t_start: int
    t_len: int

    @property
    def t_slice(self) -> slice:
        return slice(self.t_start, self.t_start + self.t_len)


@dataclass(frozen=True)
class SequenceLayout:
    slots: tuple[GroupSlot, ...]
    lat_c: int
    lat_h: int
    lat_w: int
    tokens_per_frame: int
    t_video_gen: int                 # future latent steps (CONTACT horizon too)
    actions_per_frame: int
    student: bool
    drop_video: bool
    # physical span of one ACTION latent frame in latent-video-frame units
    # (rope "time_true" mode); 0.0 only for hand-built test layouts
    action_pos_step: float = 0.0
    # mc.video_attend: drop the CONTACT/ACTION -> VIDEO_GEN block of the
    # structural mask (world-action coupling). Default False = shipped mask.
    video_attend: bool = False

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, bb: BackboneConfig, mc: PhantomModelConfig, hw: HardwareConfig,
              *, student: bool | None = None, drop_video: bool = False) -> "SequenceLayout":
        student = mc.student if student is None else student
        t_gen = bb.t_video - 1
        apf = bb.actions_per_latent_frame
        assert hw.control.chunk_horizon % apf == 0, (
            f"control.chunk_horizon ({hw.control.chunk_horizon}) must be a multiple of "
            f"{apf} (backbone actions per latent frame)")
        n_action_frames = hw.control.chunk_horizon // apf

        order: list[tuple[FrameGroup, int]] = [(FrameGroup.VIDEO_COND, 1)]
        if not drop_video:
            order.append((FrameGroup.VIDEO_GEN, t_gen))
        if not student:
            order.append((FrameGroup.OBS_GEL, 1))
            order.append((FrameGroup.OBS_MECH, mc.contact_obs_frames))
        order.append((FrameGroup.OBS_PROPRIO, 1))
        order.append((FrameGroup.CONTACT, t_gen))
        order.append((FrameGroup.ACTION, n_action_frames))

        slots, t = [], 0
        for g, n in order:
            slots.append(GroupSlot(g, t, n))
            t += n
        # physical duration of one ACTION latent frame, in latent-video-frame
        # units (the RoPE time axis): with apf=4 @ 10 Hz on the 4 Hz backbone
        # (13 pixel frames -> 3 gen frames over 3 s) this is 0.4
        latent_frame_s = ((bb.frames_pix - 1) / bb.fps) / max(t_gen, 1)
        action_pos_step = (apf / hw.control.action_rate_hz) / latent_frame_s
        return cls(slots=tuple(slots), lat_c=bb.lat_ch, lat_h=bb.lat_h, lat_w=bb.lat_w,
                   tokens_per_frame=bb.tokens_per_frame, t_video_gen=t_gen,
                   actions_per_frame=apf, student=student, drop_video=drop_video,
                   action_pos_step=action_pos_step,
                   video_attend=bool(mc.video_attend))

    # ------------------------------------------------------------------
    @property
    def t_total(self) -> int:
        return sum(s.t_len for s in self.slots)

    @property
    def n_tokens(self) -> int:
        return self.t_total * self.tokens_per_frame

    def has(self, g: FrameGroup) -> bool:
        return any(s.group == g for s in self.slots)

    def slot(self, g: FrameGroup) -> GroupSlot:
        for s in self.slots:
            if s.group == g:
                return s
        raise KeyError(f"group {g} not in this layout (student={self.student}, "
                       f"drop_video={self.drop_video})")

    def frame_slice(self, g: FrameGroup) -> slice:
        return self.slot(g).t_slice

    def token_slice(self, g: FrameGroup) -> slice:
        s = self.slot(g)
        return slice(s.t_start * self.tokens_per_frame,
                     (s.t_start + s.t_len) * self.tokens_per_frame)

    # ------------------------------------------------------------------
    def cond_mask_T(self) -> np.ndarray:
        """(T,) bool: FRAME_REPLACE conditioning mask (1 = pinned, never denoised)."""
        m = np.zeros(self.t_total, dtype=bool)
        for s in self.slots:
            if s.group in COND_GROUPS:
                m[s.t_slice] = True
        return m

    def group_of_frame(self) -> np.ndarray:
        """(T,) int: FrameGroup index per frame (for the frame-type embedding)."""
        ids = np.zeros(self.t_total, dtype=np.int64)
        order = list(FrameGroup)
        for s in self.slots:
            ids[s.t_slice] = order.index(s.group)
        return ids

    def rope_frame_positions(self, mode: str = "aligned") -> np.ndarray:
        """(T,) temporal RoPE position per frame (int for aligned/append —
        bit-compatible with v3 checkpoints; float for time_true).

        aligned:   OBS frames share position 0 (current time); CONTACT/ACTION
                   frame k sits at 1+k (the video frame it predicts) — stays
                   inside the pretrained temporal range [0, t_video).
                   DEFECT kept for v3 compat: with more action frames than
                   t_video_gen this clamps to e.g. [1,2,3,3] — the last
                   frames alias one temporal phase (v4 audit).
        time_true: ACTION frame k at its PHYSICAL future time in latent-frame
                   units ((k+1)*action_pos_step, e.g. [0.4,0.8,1.2,1.6]) —
                   distinct, order-true, inside the pretrained range. CONTACT
                   stays on the latent grid 1..t_gen (its labels live there).
        append:    every frame a fresh position 0..T-1 (ablation).
        """
        if mode == "append":
            return np.arange(self.t_total, dtype=np.int64)
        if mode not in ("aligned", "time_true"):
            raise ValueError(f"unknown rope_time_mode {mode!r}")
        pos = np.zeros(self.t_total, dtype=np.float64)
        for s in self.slots:
            if s.group == FrameGroup.VIDEO_COND:
                pos[s.t_slice] = 0
            elif s.group == FrameGroup.VIDEO_GEN:
                pos[s.t_slice] = np.arange(1, 1 + s.t_len)
            elif s.group in COND_GROUPS:
                pos[s.t_slice] = 0
            elif s.group == FrameGroup.ACTION and mode == "time_true":
                assert self.action_pos_step > 0, \
                    "time_true rope needs a layout built with action_pos_step"
                pos[s.t_slice] = (1 + np.arange(s.t_len)) * self.action_pos_step
            else:  # CONTACT (both modes) / ACTION (aligned)
                pos[s.t_slice] = 1 + np.arange(s.t_len) * max(1, (self.t_video_gen // s.t_len))
                pos[s.t_slice] = np.minimum(pos[s.t_slice], self.t_video_gen)
        if mode == "aligned":
            return pos.astype(np.int64)      # exact v3 values + dtype
        return pos

    # ------------------------------------------------------------------
    def structural_attn_bias(self) -> np.ndarray:
        """(T, T) additive {0, -1e4} frame-level mask, Fast-WAM style: CONTACT
        and ACTION queries never attend VIDEO_GEN keys, so video frames are
        droppable at inference without a train/test attention mismatch.
        Expanded to token level by attention_bias.structural_bias_tokens.

        `video_attend` (ablation) returns the ALL-ZERO bias instead: the
        CONTACT/ACTION queries do read the imagined future, and the frames stop
        being droppable (`rf.sample` refuses drop_video for such a model)."""
        bias = np.zeros((self.t_total, self.t_total), dtype=np.float32)
        if self.video_attend or not self.has(FrameGroup.VIDEO_GEN):
            return bias
        vid = self.frame_slice(FrameGroup.VIDEO_GEN)
        for g in (FrameGroup.CONTACT, FrameGroup.ACTION):
            if self.has(g):
                bias[self.frame_slice(g), vid] = -1e4
        return bias

    def haptic_group_token_mask(self) -> np.ndarray:
        """(n_tokens,) bool: the ACC bias targets. Teacher: tactile tokens
        (OBS_MECH + OBS_GEL) + CONTACT frames; student: CONTACT + OBS_PROPRIO
        (pipeline.md §3)."""
        m = np.zeros(self.n_tokens, dtype=bool)
        groups = ((FrameGroup.CONTACT, FrameGroup.OBS_PROPRIO) if self.student
                  else (FrameGroup.OBS_MECH, FrameGroup.OBS_GEL, FrameGroup.CONTACT))
        for g in groups:
            if self.has(g):
                m[self.token_slice(g)] = True
        return m

    def describe(self) -> str:
        rows = [f"  {s.group.value:<12} frames [{s.t_start}:{s.t_start + s.t_len})"
                for s in self.slots]
        return (f"SequenceLayout(student={self.student}, drop_video={self.drop_video}, "
                f"video_attend={self.video_attend}, "
                f"T={self.t_total}, tokens={self.n_tokens})\n" + "\n".join(rows))


def contact_pinned_layout(layout: SequenceLayout) -> SequenceLayout:
    """`layout` with the CONTACT frames added to the FRAME_REPLACE cond mask,
    i.e. held at their x0 through every denoise step instead of being
    co-denoised from noise.

    The one mechanism behind BOTH pinned arms of finding P7 — offline
    (`tools/terminal_eval.py --null contact_zero|contact_gt`) and at deploy
    (`run_deploy --null-imagination contact_zero`, via `rf.sample(
    pin_contact_x0=True)`). What the frames are pinned TO is whatever the
    batch's `cpk_*` keys hold: the GT package offline under `--null
    contact_gt`, the zero package everywhere else (the deploy batch's
    placeholders are zeros, so deploy can only ever pin to zeros).

    Idempotent: wrapping an already-pinned layout returns it unchanged."""
    if getattr(layout, "contact_pinned", False):
        return layout

    class _ContactPinned(type(layout)):
        #: not a dataclass field (no annotation) — a marker for the guard above
        contact_pinned = True

        def cond_mask_T(self):
            m = super().cond_mask_T()
            if self.has(FrameGroup.CONTACT):
                m[self.frame_slice(FrameGroup.CONTACT)] = True
            return m

    vals = {f.name: getattr(layout, f.name) for f in dc_fields(layout)}
    return _ContactPinned(**vals)
