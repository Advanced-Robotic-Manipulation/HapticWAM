"""Shared training infrastructure: DDP setup, param groups, EMA, checkpoint
format, dataset wrappers, and the generic training loop used by all four
programs. Plain PyTorch — no imaginaire trainer (user decision)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import HardwareConfig
from phantom.config.model import PhantomModelConfig
from phantom.config.training import CommonTrainConfig
from phantom.data.schema import NormStats

log = logging.getLogger(__name__)

CKPT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# distributed
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int]:
    """(rank, world_size); initializes the process group under torchrun.
    Idempotent — call it EARLY in a program's main() so a bare "cuda" resolves
    to this rank's device for model/loader construction (the VAE and the text
    provider are unregistered attrs the loop's model.to() never moves), and
    again (no-op) inside train_loop."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = ("nccl" if torch.cuda.is_available() and dist.is_nccl_available()
                   else "gloo")
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK",
                                       rank % torch.cuda.device_count()))
            torch.cuda.set_device(local)
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)  # rank 0 keeps INFO
        return rank, world
    return 0, 1


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def pick_dtype(device: str, tiny: bool, compute=None) -> torch.dtype:
    """--tiny is always fp32 (CPU smoke, even with a cluster --compute); else
    the compute profile's dtype if set; else the legacy rule (bf16 on cuda,
    fp32 otherwise) — bit-identical to the historical inline expression when
    no profile dtype is given."""
    if tiny:
        return torch.float32
    if compute is not None and compute.dtype is not None:
        return torch.bfloat16 if compute.dtype == "bf16" else torch.float32
    return torch.bfloat16 if device == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# optimizer / EMA
# ---------------------------------------------------------------------------

def make_optimizer(model: torch.nn.Module, cfg: CommonTrainConfig) -> torch.optim.Optimizer:
    lora, new = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora if "lora_" in name else new).append(p)
    return torch.optim.AdamW(
        [{"params": lora, "lr": cfg.lr},
         {"params": new, "lr": cfg.lr_new_modules}],
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95))


def make_scheduler(opt: torch.optim.Optimizer, cfg: CommonTrainConfig):
    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


class EMA:
    """EMA over trainable params only."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone().float()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.detach().float(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow


# ---------------------------------------------------------------------------
# checkpoints — never re-save the frozen 2B base
# ---------------------------------------------------------------------------

def trainable_state_dicts(model: torch.nn.Module) -> tuple[dict, dict]:
    """(lora, phantom) — the only weights PHANTOM checkpoints carry."""
    lora, phantom = {}, {}
    for name, t in model.state_dict().items():
        if "lora_" in name:
            lora[name] = t
        elif "phantom_" in name:
            phantom[name] = t
    return lora, phantom


def save_phantom_checkpoint(path: Path, model: torch.nn.Module, *,
                            hw: HardwareConfig, bb: BackboneConfig,
                            mc: PhantomModelConfig, train_cfg: CommonTrainConfig,
                            step: int, base_ckpt_path: str = "",
                            norm_stats: NormStats | None = None,
                            optimizer=None, scheduler=None, ema: EMA | None = None
                            ) -> None:
    lora, phantom = trainable_state_dicts(model)
    payload = {
        "format_version": CKPT_FORMAT_VERSION,
        "base_ckpt_path": base_ckpt_path,
        "lora": lora,
        "phantom_modules": phantom,
        "ema": ema.state_dict() if ema else None,
        "configs": {
            "hardware_shapes": hw.shape_relevant_fields(),
            "hardware_hash": hw.config_hash(),
            "backbone": bb.to_dict(),
            "model": mc.to_dict(),
            "train": train_cfg.to_dict(),
        },
        "norm_stats": ({"mean": {k: v.tolist() for k, v in norm_stats.mean.items()},
                        "std": {k: v.tolist() for k, v in norm_stats.std.items()}}
                       if norm_stats else None),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    log.info("checkpoint saved: %s (step %d, lora %d keys, phantom %d keys)",
             path, step, len(lora), len(phantom))


def load_phantom_checkpoint(path: Path, model: torch.nn.Module, *,
                            hw: HardwareConfig, load_ema: bool = False,
                            allow_missing: bool = False) -> dict:
    """Load lora+phantom weights into a built model; asserts hardware
    shape-compat (value-only drift warns via hash)."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    assert payload["format_version"] == CKPT_FORMAT_VERSION
    saved_shapes = payload["configs"]["hardware_shapes"]
    cur_shapes = hw.shape_relevant_fields()
    mismatches = {k: (saved_shapes.get(k), v) for k, v in cur_shapes.items()
                  if _norm_shape(saved_shapes.get(k)) != _norm_shape(v)}
    assert not mismatches, (
        f"hardware shape-relevant fields changed since this checkpoint was trained: "
        f"{mismatches} — retrain or restore configs/hardware.yaml")
    if payload["configs"]["hardware_hash"] != hw.config_hash():
        log.warning("hardware config VALUES differ from checkpoint provenance "
                    "(shape-compatible — proceeding)")
    weights = dict(payload["lora"])
    weights.update(payload["phantom_modules"])
    if load_ema and payload.get("ema"):
        weights.update({k: v for k, v in payload["ema"].items() if k in weights})
    missing, unexpected = model.load_state_dict(weights, strict=False)
    unexpected = [k for k in unexpected]
    if unexpected and not allow_missing:
        raise RuntimeError(f"checkpoint keys unknown to this model: {unexpected[:8]}")
    not_loaded = [k for k in missing if "lora_" in k or "phantom_" in k]
    if not_loaded and not allow_missing:
        raise RuntimeError(f"model trainable keys absent from checkpoint: {not_loaded[:8]} "
                           "(pass allow_missing=True for teacher->student init)")
    if allow_missing and (unexpected or not_loaded):
        log.info("partial init (teacher->student): %d checkpoint keys dropped "
                 "(teacher-only paths), %d model keys left fresh",
                 len(unexpected), len(not_loaded))
    return payload


def _norm_shape(v):
    if isinstance(v, (list, tuple)):
        return tuple(_norm_shape(x) for x in v)
    return v


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def manifest_split(data_root: Path, split: str) -> list[Path] | None:
    """Episode dirs for `split` from manifests/all.jsonl, or None if absent.

    The dataset ships a manifest next to the task folders (data_root is
    <root>/tasks, so the manifest is ../manifests/all.jsonl). Without this the
    training set silently includes the held-out episodes and no validation
    number means anything."""
    if split == "all":
        return None
    mf = Path(data_root).parent / "manifests" / "all.jsonl"
    if not mf.exists():
        log.warning("no manifest at %s — using every episode under %s "
                    "(NO held-out set)", mf, data_root)
        return None
    eps: list[Path] = []
    for line in mf.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != split:
            continue
        p = Path(data_root).parent / r["path"]
        if (p / "meta.json").exists():
            eps.append(p)
    log.info("manifest split %r: %d episodes", split, len(eps))
    return eps


class WindowDataset(Dataset):
    """Wraps WindowSampler over an episode root for DataLoader consumption."""

    def __init__(self, root: Path, sampler, windows_per_episode: int = 8,
                 episodes: list[Path] | None = None, resample: bool = True,
                 seed: int = 0):
        self.sampler = sampler
        self.index = sampler.build_index(root, windows_per_episode, episodes)
        # A frozen index replays the same anchors every epoch (~35x over a long
        # run) — redraw t0 within the episode's admissible range on each access
        # so the run keeps seeing fresh windows from the same episodes. Held-out
        # sets pass resample=False to stay comparable across evals.
        self.resample = resample
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        wi = self.index[i]
        t0 = wi.t0
        if self.resample and wi.hi > wi.lo:
            t0 = float(self._rng.uniform(wi.lo, wi.hi))
        return self.sampler.sample(wi.episode, t0)


def collate_windows(items: list[dict]) -> dict:
    out: dict = {}
    for k in items[0]:
        v = items[0][k]
        if torch.is_tensor(v):
            out[k] = torch.stack([it[k] for it in items])
        else:
            out[k] = [it[k] for it in items]
    return out


def ensure_synthetic_dataset(root: Path, hw: HardwareConfig, n_episodes: int = 4,
                             duration_s: float = 8.0, seed: int = 0) -> Path:
    """Generate (once per hardware-config hash) a synthetic episode set for
    --synthetic smoke runs. Episodes live under root/<hash>/ so that changing
    hardware.yaml never silently reuses data generated under old values.
    Under torchrun only rank 0 generates (the barrier holds the other ranks
    until the episodes exist); identical behavior single-process."""
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    root = Path(root) / hw.config_hash()[:10]
    if is_main() and not list(root.rglob("meta.json")):
        root.mkdir(parents=True, exist_ok=True)
        SyntheticEpisodeGenerator(hw, seed=seed).generate_dataset(
            root, n_episodes=n_episodes, duration_s=duration_s)
    if dist.is_initialized():
        dist.barrier()
    return root


def make_loader(ds: Dataset, cfg: CommonTrainConfig, *,
                collate_fn=collate_windows, shuffle: bool | None = None) -> DataLoader:
    """The one training DataLoader for all four programs. world==1 reproduces
    the historical construction exactly (shuffle=True, drop_last=True,
    synthetic -> workers 0); under torchrun a DistributedSampler shards the
    window index disjointly per rank (train_loop's re-iteration calls
    set_epoch for the reshuffle). Construct AFTER any ds.index mutation
    (distill_hid --extra-data) — the sampler snapshots len(ds)."""
    rank, world = setup_ddp()          # idempotent
    sampler = None
    if world > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=True, seed=cfg.seed, drop_last=True)
        assert len(sampler) >= cfg.batch_size, (
            f"dataset too small to shard: {len(ds)} windows over {world} ranks "
            f"gives {len(sampler)}/rank < batch_size {cfg.batch_size}")
    return DataLoader(ds, batch_size=cfg.batch_size,
                      shuffle=((sampler is None) if shuffle is None else shuffle),
                      sampler=sampler,
                      num_workers=0 if cfg.synthetic else cfg.num_workers,
                      collate_fn=collate_fn, drop_last=True)


# ---------------------------------------------------------------------------
# generic loop
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, val_loader: DataLoader, step_fn,
             max_batches: int = 24) -> dict[str, float]:
    """Mean losses over a held-out loader (no grad). Keeps the run honest:
    without this the only signal for days is the training loss."""
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            parts = step_fn(batch)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
    if was_training:
        model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def train_loop(cfg: CommonTrainConfig, model: torch.nn.Module, loader: DataLoader,
               step_fn, *, on_checkpoint=None, val_loader: DataLoader | None = None,
               eval_step_fn=None) -> int:
    """step_fn(batch) -> dict with 'total' loss tensor. Handles grad accum,
    clipping, EMA, logging; returns final step.

    val_loader: optional held-out loader evaluated every cfg.eval_every steps
    (eval_step_fn defaults to step_fn)."""
    rank, world = setup_ddp()
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    model = model.to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg)
    ema = EMA(model.module if world > 1 else model, cfg.ema_decay)

    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    step, t0 = 0, time.perf_counter()
    epoch = 0
    it = iter(loader)
    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}
        for _ in range(cfg.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                if hasattr(getattr(loader, "sampler", None), "set_epoch"):
                    loader.sampler.set_epoch(epoch)   # DistributedSampler reshuffle
                it = iter(loader)
                batch = next(it)
            parts = step_fn(batch)
            (parts["total"] / cfg.grad_accum).backward()
            for k, v in parts.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    logs[k] = logs.get(k, 0.0) + float(v.detach()) / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()
        sched.step()
        ema.update(model.module if world > 1 else model)
        step += 1
        if step % cfg.log_every == 0:
            if world > 1 and logs:
                # true cross-rank mean; SUM/world (ReduceOp.AVG is NCCL-only)
                keys = sorted(logs)
                t = torch.tensor([logs[k] for k in keys],
                                 device=device if device != "cpu" else None)
                dist.all_reduce(t)
                logs = {k: float(v) / world for k, v in zip(keys, t.tolist())}
            if is_main():
                rate = step / (time.perf_counter() - t0)
                log.info("step %d/%d  %s  (%.2f it/s)", step, cfg.max_steps,
                         "  ".join(f"{k}={v:.4f}" for k, v in sorted(logs.items())), rate)
        if (val_loader is not None and is_main() and cfg.eval_every
                and step % cfg.eval_every == 0):
            vm = evaluate(model, val_loader, eval_step_fn or step_fn)
            log.info("EVAL step %d  %s", step,
                     "  ".join(f"val_{k}={v:.4f}" for k, v in sorted(vm.items())))
        if is_main() and on_checkpoint and step % cfg.ckpt_every == 0:
            on_checkpoint(step, opt, sched, ema)
    if is_main() and on_checkpoint:
        on_checkpoint(step, opt, sched, ema)
    if world > 1:
        dist.destroy_process_group()
    return step


def to_device(batch: dict, device, dtype=None) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if dtype is not None and v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out
