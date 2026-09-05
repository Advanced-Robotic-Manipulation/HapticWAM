"""v6 data fix B.1: the model's contact_state must not carry a pad's
per-session zero offset (the left pad idles 0.7-2.2 N above zero). Gated by
`TeacherTrainConfig.wrench_baseline_rows` (0 = raw, every checkpoint before
v6): training, deploy and replay read the same number from the checkpoint
and subtract the same statistic — the median of the first N wrench rows /
the ring's latest N rows at the first build."""
from __future__ import annotations

import types

import numpy as np
import pytest
import zarr

from phantom.config.backbone import BackboneConfig
from phantom.data.schema import NormStats, tactile_stream
from phantom.data.windows import WindowSampler
from phantom_test_utils import make_hw


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    hw = make_hw(
        tactile={"field": {"h": 48, "w": 64},
                 "raw_img": {"h": 60, "w": 80, "c": 1},
                 "infer_img": {"h": 60, "w": 80, "c": 1}, "rate_hz": 30.0},
        cameras={"scene": {"color": {"h": 60, "w": 80, "c": 3}, "fps": 10.0}},
        recording={"field_ds": {"h": 24, "w": 32}, "field_ds_rate_hz": 30.0,
                   "keyframe_rate_hz": 5.0, "keyframe_ds": {"h": 24, "w": 32},
                   "infer_img_rate_hz": 10.0, "zarr_chunk_frames": 16},
        derived={"cpk_downsample": 4},
        wrist_ft={"window_s": 0.1},
    )
    root = tmp_path_factory.mktemp("eps")
    SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0).generate(
        root, task="grasp_slip", duration_s=8.0)
    return hw, root


def _sample(hw, ep, rows):
    sampler = WindowSampler(hw, BackboneConfig.tiny(), NormStats.identity(),
                            student=False, seed=0, wrench_baseline_rows=rows)
    w = sampler.sample(ep)
    return w["contact_state"].numpy().copy(), w["cpk_wrench"].numpy().copy()


def test_training_contact_state_and_target_invariant_to_offset_when_enabled(synthetic_root):
    hw, root = synthetic_root
    ep = sorted(root.glob("ep_*"))[0]
    cs0, cw0 = _sample(hw, ep, 8)
    raw0, _ = _sample(hw, ep, 0)
    pad = hw.tactile.sensors[0].name
    g = zarr.open_group(str(ep / f"{tactile_stream(pad, 'wrench')}.zarr"), mode="r+")
    off = np.array([0.3, -0.2, 1.7, 0.0, 0.05, 0.0], dtype=np.float32)
    g["data"][:] = np.asarray(g["data"][:]) + off
    cs1, cw1 = _sample(hw, ep, 8)
    assert np.allclose(cs0, cs1, atol=1e-5), "input: offset must not reach the model"
    assert np.allclose(cw0, cw1, atol=1e-4), "target: the model must not predict a session constant"
    # rows=0 (every pre-v6 checkpoint) keeps the RAW value — the shift is visible
    raw1, _ = _sample(hw, ep, 0)
    assert not np.allclose(raw0, raw1, atol=1e-3)


class _ArrRing:
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.float32)

    def latest(self, k=1):
        m = min(k, len(self.rows))
        return np.arange(m, dtype=float), {"wrench": self.rows[-m:].copy()}


def test_deploy_baseline_off_by_default_and_median_of_first_build_when_on():
    from phantom.deploy.planner import SnapshotBuilder
    hw = make_hw()
    rows = np.zeros((20, 6), np.float32)
    rows[:, 2] = np.linspace(1.5, 2.1, 20)
    ring = _ArrRing(rows)
    off = SnapshotBuilder(hw, types.SimpleNamespace(rings={}), "vision_only")
    assert off.wrench_baseline_rows == 0
    assert np.all(off._baseline_for("left", ring) == 0), "v4/v5 checkpoints: raw input"
    sb = SnapshotBuilder(hw, types.SimpleNamespace(rings={}), "vision_only",
                         wrench_baseline_rows=8)
    base = sb._baseline_for("left", ring)
    assert base[2] == pytest.approx(float(np.median(rows[-8:, 2])))
    ring.rows = rows + 5.0                          # contact later in the episode
    assert sb._baseline_for("left", ring)[2] == pytest.approx(base[2]), \
        "captured once per episode, not re-read under load"
    sb.reset_baseline()
    assert sb._baseline_for("left", ring)[2] == pytest.approx(base[2] + 5.0)


def test_rows_flow_from_checkpoint_to_deploy_and_server():
    """A pre-v6 payload (no key) -> 0; a v6 payload -> its N; the policy
    server mirrors it into `info` so the robot process can read it."""
    from phantom.inference.remote import PolicyServer
    from phantom.train.common import wrench_baseline_rows_of
    assert wrench_baseline_rows_of({"configs": {"train": {"grasp_frac": 0.3}}}) == 0
    assert wrench_baseline_rows_of({"configs": {"train": {"wrench_baseline_rows": 8}}}) == 8
    assert wrench_baseline_rows_of({}) == 0
    pol = types.SimpleNamespace(wrench_baseline_rows=8)
    srv = PolicyServer(pol, ckpt="x.pt")
    assert srv.status()["wrench_baseline_rows"] == 8


def test_apply_grasp_labels_makes_a_rollout_trainable(tmp_path):
    import json
    from phantom.data.schema import EpisodeMeta, is_trainable_episode
    from tools.apply_grasp_labels import apply
    ep = tmp_path / "ep_teacher_waffles_1_000"; ep.mkdir()
    m = EpisodeMeta(task="waffles", policy="teacher"); m.status = "aborted"; m.tags = ["unlabeled"]
    m.save(ep / "meta.json")
    assert not is_trainable_episode(EpisodeMeta.load(ep / "meta.json"))
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([{"episode": ep.name, "grasp_ok": True, "c_hold": 1.0,
                                   "lift_mm": 300, "hold_truncated": False, "inconclusive": False}]))
    st = apply(labels, [tmp_path], dry_run=False, overwrite=False)
    assert st["written"] == 1
    meta = EpisodeMeta.load(ep / "meta.json")
    assert meta.success is True and meta.status == "finalized" and "unlabeled" not in meta.tags
    assert is_trainable_episode(meta)
    # re-applying is idempotent and NEVER mistakes its own label for an operator verdict
    st = apply(labels, [tmp_path], dry_run=False, overwrite=False)
    assert st["kept_operator"] == 0 and st["unchanged"] == 1
    # a human verdict is kept
    m3 = EpisodeMeta.load(ep / "meta.json"); m3.tags = []; m3.success = False; m3.save(ep / "meta.json")
    st = apply(labels, [tmp_path], dry_run=False, overwrite=False)
    assert st["kept_operator"] == 1 and EpisodeMeta.load(ep / "meta.json").success is False
    # a truncated hold abstains and stays untrainable rather than becoming a False
    ep2 = tmp_path / "ep_teacher_waffles_2_000"; ep2.mkdir()
    m2 = EpisodeMeta(task="waffles", policy="teacher"); m2.status = "aborted"; m2.tags = ["unlabeled"]
    m2.save(ep2 / "meta.json")
    labels.write_text(json.dumps([{"episode": ep2.name, "grasp_ok": False, "hold_truncated": True,
                                   "inconclusive": True}]))
    st = apply(labels, [tmp_path], dry_run=False, overwrite=False)
    assert st["abstained"] == 1 and not is_trainable_episode(EpisodeMeta.load(ep2 / "meta.json"))


def test_runtime_hands_the_policy_rows_to_the_snapshot_builder():
    """The number lives on the policy (from its checkpoint); the runtime must
    hand exactly that to SnapshotBuilder."""
    import inspect
    from phantom.deploy import runtime as rt
    text = inspect.getsource(rt.DeploymentRuntime.run_episode)
    assert 'getattr(self.policy, "wrench_baseline_rows", 0)' in text
    assert "SnapshotBuilder(hw, self.session, self.mode" in text


def test_student_programs_inherit_and_persist_the_rows():
    """A student distilled from a v6 teacher must carry the teacher's rows in
    ITS checkpoint, or the next program / replay re-grounds it on raw targets."""
    import inspect
    from phantom.config.training import HIDConfig, HIDSConfig, TeacherTrainConfig
    for cls in (TeacherTrainConfig, HIDConfig, HIDSConfig):
        assert "wrench_baseline_rows" in cls().to_dict(), cls.__name__
    from phantom.dagger import relabel
    from phantom.train import distill_hid, finetune_hids
    for mod in (distill_hid, finetune_hids):
        assert "wrench_baseline_rows=C.wrench_baseline_rows_of(payload)" in inspect.getsource(mod)
    assert "wrench_baseline_rows_of(payload)" in inspect.getsource(relabel)
