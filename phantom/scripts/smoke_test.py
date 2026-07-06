"""Model-side smoke test: build -> (load) -> forward -> one training step of
every program's loss -> 5-NFE sample -> pack/unpack round trip.

Tiny CPU smoke (any machine, needs only the cloned cosmos repo):
    python -m phantom.scripts.smoke_test --tiny --synthetic
Full-checkpoint smoke (GPU box; measures the extended-layout VRAM footprint —
week-1 checklist item 3):
    python -m phantom.scripts.smoke_test --synthetic
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data.schema import NormStats
from phantom.data.windows import WindowSampler
from phantom.train import common as C
from phantom.train.builder import build_model

log = logging.getLogger("smoke_test")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--data", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nfe", type=int, default=2)
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    dtype = torch.bfloat16 if (args.device == "cuda" and not args.tiny) else torch.float32

    if args.data:
        data_root = (C.ensure_synthetic_dataset(Path(args.data), hw)
                     if args.synthetic else Path(args.data))
    else:
        data_root = C.ensure_synthetic_dataset(paths.data_root / "synthetic", hw)

    results: dict[str, str] = {}

    def stage(name: str):
        log.info(">>> %s", name)
        results[name] = "..."

    # ---- teacher build + forward + training step + sample ----
    stage("build teacher")
    teacher = build_model(hw, paths, student=False, tiny=args.tiny,
                          load_base=not args.tiny, device=args.device, dtype=dtype)
    results["build teacher"] = f"{teacher.layout.t_total} frames, {teacher.layout.n_tokens} tokens"

    stage("dataset window")
    sampler = WindowSampler(hw, teacher.bb, NormStats.identity(), student=False)
    ds = C.WindowDataset(data_root, sampler, windows_per_episode=2)
    loader = DataLoader(ds, batch_size=1, collate_fn=C.collate_windows)
    batch = next(iter(loader))
    results["dataset window"] = f"{len(ds)} windows, keys={sorted(batch.keys())[:6]}..."

    stage("teacher training_step")
    t0 = time.perf_counter()
    parts = teacher.rf.training_step(C.to_device(batch, args.device))
    parts["total"].backward()
    grads = sum(1 for p in teacher.rf.parameters()
                if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    teacher.rf.zero_grad(set_to_none=True)
    results["teacher training_step"] = (
        f"total={float(parts['total']):.3f} ({time.perf_counter() - t0:.1f}s, "
        f"{grads} tensors got gradient)")

    stage("teacher sample")
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = teacher.rf.sample(C.to_device(batch, args.device), nfe=args.nfe)
    results["teacher sample"] = (
        f"actions {tuple(pred.actions_B_H_A.shape)}, "
        f"sigma {tuple(pred.governor_sigma_B_Tc.shape)}, "
        f"gate={float(pred.acc.g[0]) if pred.acc else -1:.2f} "
        f"({time.perf_counter() - t0:.1f}s)")

    stage("drop-video sample")
    with torch.no_grad():
        pred_dv = teacher.rf.sample(C.to_device(batch, args.device), nfe=args.nfe,
                                    drop_video=True)
    results["drop-video sample"] = f"actions {tuple(pred_dv.actions_B_H_A.shape)}"

    # ---- student build + HID step + HID-S step ----
    stage("build student")
    student = build_model(hw, paths, student=True, tiny=args.tiny,
                          load_base=not args.tiny, device=args.device, dtype=dtype)
    results["build student"] = f"{student.layout.t_total} frames (no tactile obs)"

    stage("HID distill step")
    from phantom.config.training import HIDConfig
    from phantom.train.distill_hid import distill_step
    teacher.rf.eval()
    parts = distill_step(student.rf, teacher.rf, batch,
                         HIDConfig(tiny=args.tiny, synthetic=True), args.device)
    parts["total"].backward()
    student.rf.zero_grad(set_to_none=True)
    results["HID distill step"] = f"total={float(parts['total']):.3f}"

    stage("HID-S step")
    from phantom.config.training import HIDSConfig
    from phantom.train.finetune_hids import hids_step
    parts = hids_step(student.rf, student.rf, batch,
                      HIDSConfig(tiny=args.tiny, synthetic=True), {}, [0.0],
                      args.device)
    results["HID-S step"] = f"total={float(parts['total']):.3f}"

    stage("tactile pretrain step")
    from phantom.config.training import TactilePretrainConfig
    from phantom.model.hht.tactile_encoder import TactilePretrainModel
    tpm = TactilePretrainModel(hw).to(args.device)
    fields = batch["fields"][:, 0].permute(0, 3, 1, 2).to(args.device)
    tp_cfg = TactilePretrainConfig()
    losses = tpm.training_losses(fields, tp_cfg.mask_ratio, tp_cfg.mask_patch)
    results["tactile pretrain step"] = \
        f"recon={float(losses['masked_recon']):.3f} cross={float(losses['cross_channel']):.3f}"

    stage("checkpoint round trip")
    import tempfile
    from phantom.config.training import TeacherTrainConfig
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ckpt.pt"
        C.save_phantom_checkpoint(p, teacher.rf, hw=hw, bb=teacher.bb, mc=teacher.mc,
                                  train_cfg=TeacherTrainConfig(), step=1)
        C.load_phantom_checkpoint(p, teacher.rf, hw=hw)
        C.load_phantom_checkpoint(p, student.rf, hw=hw, allow_missing=True)
    results["checkpoint round trip"] = "teacher reload + student init OK"

    if args.device == "cuda":
        results["VRAM"] = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak"

    print("\n===== SMOKE TEST RESULTS =====")
    for k, v in results.items():
        print(f"  {k:<28} {v}")
    print("ALL STAGES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
