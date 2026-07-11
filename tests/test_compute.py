"""Compute-target config (configs/compute.yaml): the rtx5090 default must be
provably identical to the historical single-GPU behavior, cluster profiles
must parse with their sub-overrides, precedence must be
dataclass defaults < profile < explicit CLI, and the DistributedSampler-based
sharding must be disjoint and covering. All CPU, no process group."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from phantom.config.compute import (ComputeConfig, ComputeProfile,
                                    ProgramOverrides, _OVERRIDE_FIELDS,
                                    load_compute, load_compute_config)
from phantom.config.training import (HIDConfig, HIDSConfig,
                                     TactilePretrainConfig, TeacherTrainConfig)
from phantom.train import common as C
from phantom.train.train_teacher import add_common_args, apply_overrides

PROGRAM_CFGS = {
    "pretrain_tactile": TactilePretrainConfig,
    "train_teacher": TeacherTrainConfig,
    "distill_hid": HIDConfig,
    "finetune_hids": HIDSConfig,
}


# ---------------------------------------------------------------------------
# the committed default file
# ---------------------------------------------------------------------------

def test_default_file_loads():
    profile = load_compute()          # configs/compute.yaml target: rtx5090
    assert profile.launcher == "python"
    assert profile.nproc_per_node == 1
    for f in _OVERRIDE_FIELDS:
        assert getattr(profile, f) is None, f"rtx5090 must not override {f}"


def test_default_profile_is_identity():
    profile = load_compute("rtx5090")
    for program, cfg_cls in PROGRAM_CFGS.items():
        comp = profile.for_program(program)
        cfg = cfg_cls()
        assert comp.apply_to(cfg) == cfg, f"{program}: default profile mutated cfg"
    # pick_dtype reproduces the historical inline expression on the full grid
    for device in ("cuda", "cpu"):
        for tiny in (False, True):
            legacy = torch.bfloat16 if (device == "cuda" and not tiny) else torch.float32
            assert C.pick_dtype(device, tiny, profile.for_program("train_teacher")) == legacy


def test_cluster_profiles():
    for name in ("h100x8", "a100x8"):
        p = load_compute(name)
        assert p.launcher == "torchrun" and p.nproc_per_node == 8
        assert p.dtype == "bf16"
        teacher = p.for_program("train_teacher")
        assert (teacher.batch_size, teacher.grad_accum, teacher.num_workers) == (1, 1, 8)
        tactile = p.for_program("pretrain_tactile")
        assert tactile.batch_size == 4, "programs: sub-override must beat profile level"
        assert tactile.grad_accum == 1  # inherited from profile level


def test_effective_batch_preserved():
    """The cluster recipes deliberately keep effective batch equal to the
    single-5090 defaults (no silent lr-scaling questions)."""
    for name in ("h100x8", "a100x8"):
        p = load_compute(name)
        for program, cfg_cls in PROGRAM_CFGS.items():
            default = cfg_cls()
            comp = p.for_program(program)
            eff = ((comp.batch_size or default.batch_size)
                   * (comp.grad_accum or default.grad_accum) * p.nproc_per_node)
            assert eff == default.batch_size * default.grad_accum, (name, program)


# ---------------------------------------------------------------------------
# schema strictness + selectors
# ---------------------------------------------------------------------------

def test_unknown_profile_key_rejected():
    with pytest.raises(Exception):
        ComputeProfile(batch_sise=2)          # typo field -> extra="forbid"


def test_unknown_target_rejected():
    with pytest.raises(ValueError):
        ComputeConfig(target="nope", profiles={"rtx5090": ComputeProfile()})
    with pytest.raises(KeyError):
        load_compute("nope")


def test_path_selector(tmp_path: Path):
    f = tmp_path / "mine.yaml"
    f.write_text("target: big\nprofiles:\n  big:\n    launcher: torchrun\n"
                 "    nproc_per_node: 4\n    batch_size: 2\n")
    p = load_compute(str(f))
    assert p.nproc_per_node == 4 and p.batch_size == 2


def test_local_override(tmp_path: Path):
    base = tmp_path / "compute.yaml"
    base.write_text("target: a\nprofiles:\n  a: {}\n  b:\n    nproc_per_node: 8\n")
    local = tmp_path / "compute.local.yaml"
    local.write_text("target: b\n")
    cfg = load_compute_config(base, local_path=local)
    assert cfg.target == "b" and cfg.profiles["b"].nproc_per_node == 8
    local.write_text("targt: b\n")               # typo -> rejected
    with pytest.raises(KeyError):
        load_compute_config(base, local_path=local)


def test_precedence_cli_beats_profile():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    comp = ProgramOverrides(batch_size=2, grad_accum=1)
    args = ap.parse_args([])                      # no explicit CLI values
    cfg = apply_overrides(TeacherTrainConfig(), args, compute=comp)
    assert (cfg.batch_size, cfg.grad_accum) == (2, 1)
    args = ap.parse_args(["--batch-size", "4"])   # CLI wins over profile
    cfg = apply_overrides(TeacherTrainConfig(), args, compute=comp)
    assert (cfg.batch_size, cfg.grad_accum) == (4, 1)
    args = ap.parse_args(["--grad-accum", "8"])
    cfg = apply_overrides(TeacherTrainConfig(), args, compute=comp)
    assert (cfg.batch_size, cfg.grad_accum) == (2, 8)


# ---------------------------------------------------------------------------
# sharding math (explicit num_replicas/rank -> no process group needed)
# ---------------------------------------------------------------------------

def _toy_ds(n: int = 103) -> TensorDataset:
    return TensorDataset(torch.arange(n))


def test_distributed_sampler_disjoint_cover():
    ds = _toy_ds()
    shards = [set(iter(DistributedSampler(ds, num_replicas=8, rank=r,
                                          shuffle=True, seed=0, drop_last=True)))
              for r in range(8)]
    sizes = {len(s) for s in shards}
    assert len(sizes) == 1, "ranks must get equal shards"
    union: set[int] = set()
    for s in shards:
        assert not (union & s), "shards must be pairwise disjoint"
        union |= s
    assert union <= set(range(len(ds)))
    # without drop_last the (padded) union covers every index
    full = set()
    for r in range(8):
        full |= set(iter(DistributedSampler(ds, num_replicas=8, rank=r,
                                            shuffle=True, seed=0, drop_last=False)))
    assert full == set(range(len(ds)))


def test_sampler_set_epoch_reshuffles():
    ds = _toy_ds()
    s = DistributedSampler(ds, num_replicas=8, rank=0, shuffle=True, seed=0)
    s.set_epoch(0)
    order0 = list(iter(s))
    s.set_epoch(1)
    order1 = list(iter(s))
    assert order0 != order1, "train_loop's set_epoch must actually reshuffle"


def test_make_loader_world1_identity():
    """No RANK env -> make_loader must reproduce the historical DataLoader."""
    ds = _toy_ds(16)
    cfg = TeacherTrainConfig(batch_size=2, synthetic=True)
    loader = C.make_loader(ds, cfg, collate_fn=None)
    assert loader.batch_size == 2
    assert loader.drop_last is True
    assert loader.num_workers == 0                # synthetic -> 0
    assert isinstance(loader.sampler, RandomSampler)   # shuffle preserved
    cfg2 = TeacherTrainConfig(batch_size=2, synthetic=False, num_workers=3)
    assert C.make_loader(ds, cfg2, collate_fn=None).num_workers == 3
    assert C.make_loader(ds, cfg).collate_fn is C.collate_windows
