"""SequenceLayout invariants: slices partition the axis; student/drop-video
variants; masks; RoPE positions stay in the pretrained range in aligned mode."""

import numpy as np
import pytest

from phantom.config.backbone import BackboneConfig
from phantom.config.model import PhantomModelConfig
from phantom.model.sequence import COND_GROUPS, FrameGroup, SequenceLayout


@pytest.fixture(params=[(False, False), (True, False), (False, True), (True, True)],
                ids=["teacher", "student", "teacher-dropvid", "student-dropvid"])
def layout(request, small_hw):
    student, drop_video = request.param
    return SequenceLayout.build(BackboneConfig.tiny(), PhantomModelConfig(),
                                small_hw, student=student, drop_video=drop_video)


def test_frames_partition(layout):
    covered = np.zeros(layout.t_total, dtype=int)
    for s in layout.slots:
        covered[s.t_slice] += 1
    assert (covered == 1).all()


def test_token_slices_partition(layout):
    covered = np.zeros(layout.n_tokens, dtype=int)
    for s in layout.slots:
        covered[layout.token_slice(s.group)] += 1
    assert (covered == 1).all()


def test_student_has_no_tactile_obs(layout):
    if layout.student:
        assert not layout.has(FrameGroup.OBS_GEL)
        assert not layout.has(FrameGroup.OBS_MECH)
    else:
        assert layout.has(FrameGroup.OBS_GEL) and layout.has(FrameGroup.OBS_MECH)
    assert layout.has(FrameGroup.OBS_PROPRIO)
    assert layout.has(FrameGroup.CONTACT) and layout.has(FrameGroup.ACTION)


def test_cond_mask(layout):
    m = layout.cond_mask_T()
    for s in layout.slots:
        expected = s.group in COND_GROUPS
        assert (m[s.t_slice] == expected).all()


def test_rope_aligned_stays_in_range(layout):
    pos = layout.rope_frame_positions("aligned")
    assert pos.min() >= 0 and pos.max() <= layout.t_video_gen
    append = layout.rope_frame_positions("append")
    assert (append == np.arange(layout.t_total)).all()


def test_structural_bias(layout):
    bias = layout.structural_attn_bias()
    if layout.has(FrameGroup.VIDEO_GEN):
        vid = layout.frame_slice(FrameGroup.VIDEO_GEN)
        con = layout.frame_slice(FrameGroup.CONTACT)
        act = layout.frame_slice(FrameGroup.ACTION)
        assert (bias[con, vid] < 0).all() and (bias[act, vid] < 0).all()
        assert (bias[vid, :] == 0).all()          # video queries unrestricted
        assert (bias[con, con.start] == 0).all()  # contact->contact allowed
    else:
        assert (bias == 0).all()


def test_haptic_group(layout):
    m = layout.haptic_group_token_mask()
    contact = layout.token_slice(FrameGroup.CONTACT)
    assert m[contact].all()
    if layout.student:
        assert m[layout.token_slice(FrameGroup.OBS_PROPRIO)].all()
    else:
        assert m[layout.token_slice(FrameGroup.OBS_MECH)].all()
        assert not m[layout.token_slice(FrameGroup.OBS_PROPRIO)].any()


def test_action_frames_from_config(small_hw):
    lay = SequenceLayout.build(BackboneConfig.tiny(), PhantomModelConfig(), small_hw)
    slot = lay.slot(FrameGroup.ACTION)
    assert slot.t_len * lay.actions_per_frame == small_hw.control.chunk_horizon
