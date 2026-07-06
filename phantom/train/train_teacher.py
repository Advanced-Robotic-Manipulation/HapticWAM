"""Program (2): teacher PHANTOM fine-tune (pipeline.md §4/§6) — LoRA on the
frozen 2B DiT + all phantom modules (HHT, ACC, ACE heads, frame-type
embeddings), joint grouped RF objective, all tasks co-trained.

Launch (8xH100):
    torchrun --nproc_per_node 8 -m phantom.train.train_teacher --data <episodes_root> \
        [--tactile-pretrain <ckpt>] [--run-name teacher_v1]
Smoke (any machine):
    python -m phantom.train.train_teacher --tiny --synthetic --max-steps 2
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.config.training import TeacherTrainConfig
from phantom.data.schema import NormStats
from phantom.data.windows import WindowSampler
from phantom.train import common as C
from phantom.train.builder import build_model

log = logging.getLogger("train_teacher")


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="tiny backbone preset (CPU smoke)")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def apply_overrides(cfg, args):
    updates = {"synthetic": args.synthetic, "tiny": args.tiny, "device": args.device}
    if args.run_name:
        updates["run_name"] = args.run_name
    if args.max_steps is not None:
        updates["max_steps"] = args.max_steps
        updates["warmup_steps"] = min(cfg.warmup_steps, max(1, args.max_steps // 10))
        updates["ckpt_every"] = min(cfg.ckpt_every, args.max_steps)
        updates["log_every"] = 1
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        updates["grad_accum"] = args.grad_accum
    return dataclasses.replace(cfg, **updates)


def resolve_data(args, hw, paths) -> tuple[Path, NormStats]:
    if args.synthetic:
        root = C.ensure_synthetic_dataset(paths.data_root / "synthetic", hw)
    else:
        assert args.data, "--data required (or --synthetic)"
        root = Path(args.data)
    stats = root / "norm_stats.json"
    return root, (NormStats.load(stats) if stats.exists() else NormStats.identity())


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--tactile-pretrain", default="", help="program-1 checkpoint")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    cfg = apply_overrides(TeacherTrainConfig(), args)
    out_dir = Path(cfg.out_dir) if cfg.out_dir else paths.runs_root / "teacher" / cfg.run_name
    dtype = torch.bfloat16 if (args.device == "cuda" and not args.tiny) else torch.float32

    pm = build_model(hw, paths, student=False, tiny=cfg.tiny,
                     load_base=not cfg.tiny, device=args.device, dtype=dtype)
    if args.tactile_pretrain:
        pm.rf.hht.load_pretrained_tactile(Path(args.tactile_pretrain))
        if cfg.freeze_tactile_steps > 0:
            for p in pm.rf.hht.phantom_tactile_enc.parameters():
                p.requires_grad = False
            log.info("tactile encoder frozen for warmup (%d steps configured)",
                     cfg.freeze_tactile_steps)

    data_root, norm = resolve_data(args, hw, paths)
    sampler = WindowSampler(hw, pm.bb, norm, student=False, seed=cfg.seed)
    ds = C.WindowDataset(data_root, sampler)
    log.info("dataset: %d windows from %s", len(ds), data_root)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0 if cfg.synthetic else cfg.num_workers,
                        collate_fn=C.collate_windows, drop_last=True)

    def step_fn(batch: dict) -> dict:
        return pm.rf.training_step(C.to_device(batch, args.device))

    def on_ckpt(step, opt, sched, ema):
        C.save_phantom_checkpoint(
            out_dir / f"teacher_{step:06d}.pt", pm.rf, hw=hw, bb=pm.bb, mc=pm.mc,
            train_cfg=cfg, step=step, base_ckpt_path=str(paths.cosmos_checkpoint),
            norm_stats=norm, optimizer=opt, scheduler=sched, ema=ema)

    C.train_loop(cfg, pm.rf, loader, step_fn, on_checkpoint=on_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
