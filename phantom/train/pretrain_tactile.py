"""Program (1): contact-play self-supervised pretraining of the tactile field
encoder (pipeline.md §7) — masked-patch reconstruction + cross-channel
force-from-displacement prediction on 2-4 h of unscripted contact play.

Launch (single GPU is enough at ~5M params; DDP-capable):
    python -m phantom.train.pretrain_tactile --data <contact_play_root> [--synthetic]
Output: <out>/tactile_encoder_pretrain.pt  {encoder, hw_snapshot, norm_stats}
consumed by train_teacher via HHT.load_pretrained_tactile.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import torch

from phantom.config.compute import load_compute
from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.config.training import TactilePretrainConfig
from phantom.data.contact_play import ContactPlayDataset
from phantom.data.schema import NormStats
from phantom.model.hht.tactile_encoder import TactilePretrainModel
from phantom.train import common as C

log = logging.getLogger("pretrain_tactile")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="contact-play episode root")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compute", default=None, help="see configs/compute.yaml")
    args = ap.parse_args(argv)

    rank, world = C.setup_ddp()   # EARLY: "cuda" resolves per-rank from here on
    profile = load_compute(args.compute)
    profile.check_world(world)
    comp = profile.for_program("pretrain_tactile")

    hw = load_hardware(args.hardware)
    paths = load_paths()
    cfg = TactilePretrainConfig(synthetic=args.synthetic, device=args.device)
    cfg = comp.apply_to(cfg)      # profile beats defaults; CLI lines below win
    if args.max_steps is not None:
        cfg = dataclasses.replace(cfg, max_steps=args.max_steps)
    if args.batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=args.batch_size)
    out_dir = Path(args.out) if args.out else paths.runs_root / "tactile_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.synthetic:
        data_root = C.ensure_synthetic_dataset(paths.data_root / "synthetic", hw)
    else:
        assert args.data, "--data required (or use --synthetic)"
        data_root = Path(args.data)

    norm = NormStats.identity()
    stats_path = data_root / "norm_stats.json"
    if stats_path.exists():
        norm = NormStats.load(stats_path)

    ds = ContactPlayDataset(data_root, hw, norm)
    log.info("contact-play dataset: %d frames", len(ds))
    loader = C.make_loader(ds, cfg, collate_fn=None)   # plain tensors -> default_collate

    model = TactilePretrainModel(hw).to(args.device)

    def step_fn(batch: torch.Tensor) -> dict:
        batch = batch.to(args.device)
        parts = model.training_losses(batch, cfg.mask_ratio, cfg.mask_patch)
        parts["total"] = (cfg.w_masked_recon * parts["masked_recon"]
                          + cfg.w_cross_channel * parts["cross_channel"])
        return parts

    def on_ckpt(step, opt, sched, ema):
        torch.save({"encoder": model.encoder.state_dict(),
                    "full": model.state_dict(),
                    "hw_snapshot": hw.snapshot_yaml(),
                    "step": step},
                   out_dir / "tactile_encoder_pretrain.pt")

    C.train_loop(cfg, model, loader, step_fn, on_checkpoint=on_ckpt)
    log.info("done -> %s", out_dir / "tactile_encoder_pretrain.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
