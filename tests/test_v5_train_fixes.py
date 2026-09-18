"""2026-08-26 audit fixes on the fine-tune path: EMA keyed by canonical names
(so --ema / --init-ema actually reach the LoRA tensors), per-worker numpy
streams, fine-tune CLI knobs, list_episodes status filter, EpisodeWriter
duplicate guard."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from phantom_test_utils import make_hw
from phantom.config.model import PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.config.training import TeacherTrainConfig
from phantom.train import common as C
from phantom.train.builder import build_model


@pytest.fixture(scope="module")
def tiny():
    hw = make_hw()
    mc = PhantomModelConfig(student=False)
    pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False, mc=mc)
    return hw, pm


@pytest.mark.requires_cosmos_repo
def test_ema_keys_are_state_dict_keys(tiny):
    hw, pm = tiny
    ema = C.EMA(pm.rf, 0.999)
    lora, phantom = C.trainable_state_dicts(pm.rf)
    trainable = set(lora) | set(phantom)
    assert set(ema.shadow) == trainable, (
        f"EMA keys drift from checkpoint keys: only-ema={list(set(ema.shadow) - trainable)[:3]} "
        f"only-ckpt={list(trainable - set(ema.shadow))[:3]}")
    assert any("lora_" in k for k in ema.shadow)
    assert not any(C._SAC_INFIX in k for k in ema.shadow)


@pytest.mark.requires_cosmos_repo
def test_load_ema_reaches_lora_tensors(tiny):
    hw, pm = tiny
    ema = C.EMA(pm.rf, 0.999)
    for k, v in ema.shadow.items():
        v.fill_(0.25)
    p = Path(tempfile.mkdtemp(prefix="ema")) / "ck.pt"
    C.save_phantom_checkpoint(p, pm.rf, hw=hw, bb=pm.bb, mc=pm.mc,
                              train_cfg=TeacherTrainConfig(), step=1, ema=ema)
    C.load_phantom_checkpoint(p, pm.rf, hw=hw, load_ema=True)
    sd = pm.rf.state_dict()
    lora = [k for k in sd if "lora_" in k]
    assert lora and all(torch.all(sd[k] == 0.25) for k in lora), "EMA not applied to LoRA"
    assert all(torch.all(sd[k] == 0.25) for k in sd if "phantom_" in k)


@pytest.mark.requires_cosmos_repo
def test_ema_resume_accepts_prefixed_legacy_keys(tiny):
    hw, pm = tiny
    ema = C.EMA(pm.rf, 0.999)
    legacy = {}
    for k, v in ema.shadow.items():
        kk = k.replace("net.blocks.", "net.blocks.").replace(".self_attn", C._SAC_INFIX + ".self_attn", 1) \
            if ".self_attn" in k else k
        legacy[kk] = torch.full_like(v, 0.5)
    assert any(C._SAC_INFIX in k for k in legacy)
    ema.load_state_dict(legacy)
    assert all(torch.all(v == 0.5) for v in ema.shadow.values())


@pytest.mark.requires_cosmos_repo
def test_load_ema_refuses_unmapped_tensors(tiny):
    hw, pm = tiny
    ema = C.EMA(pm.rf, 0.999)
    p = Path(tempfile.mkdtemp(prefix="ema")) / "ck.pt"
    C.save_phantom_checkpoint(p, pm.rf, hw=hw, bb=pm.bb, mc=pm.mc,
                              train_cfg=TeacherTrainConfig(), step=1, ema=ema)
    payload = torch.load(str(p), map_location="cpu", weights_only=False)
    payload["ema"] = {"net.bogus.lora_A.weight": torch.zeros(1), **payload["ema"]}
    with pytest.raises(AssertionError, match="EMA tensors do not map"):
        C.load_phantom_checkpoint(p, pm.rf, hw=hw, load_ema=True, payload=payload)


class _RngDataset(torch.utils.data.Dataset):
    """Mimics WindowDataset's per-item numpy draws."""
    def __init__(self):
        self.seed, self.epoch = 7, 0
        self._rng = np.random.default_rng(7)

    def __len__(self):
        return 8

    def __getitem__(self, i):
        info = torch.utils.data.get_worker_info()
        return torch.tensor([float(self._rng.uniform()), float(info.id if info else -1)])


def test_worker_streams_differ_across_workers_and_epochs():
    ds = _RngDataset()
    ld = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2, shuffle=False,
                                     worker_init_fn=C._seed_worker_rng)
    e0 = torch.cat([b for b in ld])
    ds.epoch = 1
    e1 = torch.cat([b for b in ld])
    by_worker = {w: e0[e0[:, 1] == w][:, 0] for w in (0, 1)}
    n = min(len(by_worker[0]), len(by_worker[1]))
    assert not torch.allclose(by_worker[0][:n], by_worker[1][:n]), "workers share one stream"
    assert not torch.allclose(e0[:, 0], e1[:, 0]), "epochs replay the same draws"


def test_finetune_cli_knobs_override_config():
    from phantom.train.train_teacher import apply_overrides
    ns = argparse.Namespace(synthetic=False, tiny=False, device="cpu", run_name=None,
                            max_steps=3000, batch_size=None, grad_accum=None, num_workers=None,
                            lr=2e-5, lr_new_modules=6e-5, warmup_steps=150,
                            ckpt_every=500, eval_every=5000)
    cfg = apply_overrides(TeacherTrainConfig(), ns)
    assert (cfg.lr, cfg.lr_new_modules, cfg.warmup_steps) == (2e-5, 6e-5, 150)
    assert cfg.ckpt_every == 500 and cfg.eval_every == 3000     # clamped to max_steps
    ns2 = argparse.Namespace(**{**vars(ns), "lr": None, "warmup_steps": None,
                               "ckpt_every": None, "eval_every": None})
    cfg2 = apply_overrides(TeacherTrainConfig(), ns2)
    assert cfg2.lr == 1e-4 and cfg2.warmup_steps == 300 and cfg2.ckpt_every == 1000


def _ep(root: Path, name: str, status: str):
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"episode_id": name, "task": "waffles",
                                             "status": status, "tags": []}))
    return d


def test_list_episodes_skips_recording_and_aborted(tmp_path):
    from phantom.data.episode_store import list_episodes
    _ep(tmp_path, "ep_a", "finalized")
    _ep(tmp_path, "ep_b", "recording")
    _ep(tmp_path, "ep_c", "aborted")
    assert [p.name for p in list_episodes(tmp_path)] == ["ep_a"]
    assert len(list_episodes(tmp_path, include_unfinalized=True)) == 3


def test_episode_writer_refuses_existing_streams(tmp_path):
    import zarr
    from phantom.data.episode_store import EpisodeWriter
    from phantom.data.schema import EpisodeMeta
    d = tmp_path / "ep_dup"
    d.mkdir()
    zarr.open_group(str(d / "gripper.zarr"), mode="a")
    with pytest.raises(FileExistsError):
        EpisodeWriter(d, make_hw(), EpisodeMeta(task="waffles"))
