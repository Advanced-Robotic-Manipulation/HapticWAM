"""Program (4): HID-S — force-safety fine-tune of the post-HID student
(pipeline.md §5, what survives of RA-HID).

    r = -lambda_F * max(0, ||f_z||_inf - tau_obj)

on the recorded calibrated normal-force field (deformation fallback while
force units are uncalibrated — automatic via hw.tactile.force_calibrated),
tau_obj auto-calibrated per task as the cfg.tau_quantile percentile of peak
normal force over SUCCESSFUL teacher episodes. Exponentiated-advantage
weighted regression on the ACTION-frame RF loss with a velocity-MSE KL leash
to the frozen post-HID reference.

Launch:
    python -m phantom.train.finetune_hids --student-ckpt runs/hid/.../student_030000.pt \
        --data <episodes_root>
Smoke: python -m phantom.train.finetune_hids --tiny --synthetic --max-steps 2
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.config.training import HIDSConfig
from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader, list_episodes
from phantom.data.schema import tactile_stream
from phantom.data.windows import WindowSampler
from phantom.model.ace import losses as L
from phantom.model.sequence import FrameGroup
from phantom.train import common as C
from phantom.train.builder import build_model
from phantom.train.train_teacher import add_common_args, apply_overrides, resolve_data

log = logging.getLogger("finetune_hids")


# ---------------------------------------------------------------------------

def calibrate_tau_per_task(root: Path, hw, quantile: float) -> dict[str, float]:
    """tau_obj per task from SUCCESSFUL episodes' peak normal force —
    'the dataset defines gentle enough'."""
    peaks: dict[str, list[float]] = {}
    for ep in list_episodes(root):
        r = EpisodeReader(ep)
        if not r.meta.success:
            continue
        ep_peak = 0.0
        for s in hw.tactile.sensors:
            stream = tactile_stream(s.name, "fields_ds")
            if not r.has(stream):
                continue
            fields = np.asarray(r._g(stream)["data"][:], dtype=np.float32)
            ep_peak = max(ep_peak, dv.peak_normal_force(fields, hw))
        peaks.setdefault(r.meta.task, []).append(ep_peak)
    taus = {task: dv.calibrate_tau_obj(np.asarray(v), quantile)
            for task, v in peaks.items() if v}
    log.info("auto-calibrated tau_obj per task: %s", taus)
    return taus


def window_peak_force(batch: dict, hw) -> torch.Tensor:
    """(B,) peak recorded normal force (or depth fallback) over the window's
    future contact-package steps — the recorded quantity the reward penalizes."""
    if hw.tactile.force_calibrated:
        # cpk_d_fz is a delta; use the packed absolute proxy: cumulative sum
        dfz = batch["cpk_d_fz"]                         # (B, Tc, F, h, w)
        fz = dfz.cumsum(dim=1)
        return fz.abs().amax(dim=(1, 2, 3, 4)) * hw.tactile.dist_force_unit_to_N
    disp = batch["cpk_d_disp"]                          # (B, Tc, F, 3, h, w)
    depth = disp[:, :, :, 2].cumsum(dim=1)
    return depth.abs().amax(dim=(1, 2, 3, 4))


def hids_step(student_rf, ref_rf, batch: dict, cfg: HIDSConfig, taus: dict,
              running_mean: list, device: str) -> dict:
    batch = C.to_device(batch, device)
    hw = student_rf.hw
    B = batch["video"].shape[0]

    # reward + advantage weights
    tau_default = float(np.mean(list(taus.values()))) if taus else 1.0
    tau = torch.tensor([taus.get(t, tau_default) for t in batch["task"]],
                       device=device)
    peak = window_peak_force(batch, hw).to(device)
    r = -cfg.lambda_F * torch.clamp(peak - tau, min=0.0)
    running_mean[0] = 0.99 * running_mean[0] + 0.01 * float(r.mean())
    adv = r - running_mean[0]
    w = torch.exp(adv / cfg.beta_awr).clamp(max=cfg.max_weight).detach()

    # AWR-weighted action-frame RF loss + KL leash at shared (x_t, t)
    t_B = student_rf._sample_t(B, student_rf.device)
    x0, x_t, cond, acc_in, tBT, eps_used = student_rf.prepare_denoise(batch, t_B)
    out = student_rf.velocity_at(batch, x_t, tBT, cond, acc_in)
    v = out.velocity_B_C_T_H_W
    v_target = torch.cat([eps_used[s.group] for s in student_rf.layout.slots], dim=2) - x0
    sl = student_rf.layout.frame_slice(FrameGroup.ACTION)
    per_sample = ((v[:, :, sl].float() - v_target[:, :, sl].float()) ** 2) \
        .mean(dim=(1, 2, 3, 4))
    parts = {"awr_action": (w * per_sample).mean(),
             "reward_mean": r.mean().detach(),
             "weight_mean": w.mean().detach()}

    with torch.no_grad():
        v_ref = ref_rf.velocity_at(batch, x_t, tBT, cond, acc_in).velocity_B_C_T_H_W
    parts["kl_leash"] = F.mse_loss(v[:, :, sl].float(), v_ref[:, :, sl].float())

    parts["total"] = parts["awr_action"] + cfg.kl_coef * parts["kl_leash"]
    return parts


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--student-ckpt", default="")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    cfg = apply_overrides(HIDSConfig(), args)
    cfg = dataclasses.replace(cfg, student_ckpt=args.student_ckpt)
    out_dir = paths.runs_root / "hids" / cfg.run_name
    dtype = torch.bfloat16 if (args.device == "cuda" and not args.tiny) else torch.float32

    student = build_model(hw, paths, student=True, tiny=cfg.tiny,
                          load_base=not cfg.tiny, device=args.device, dtype=dtype)
    reference = build_model(hw, paths, student=True, tiny=cfg.tiny,
                            load_base=not cfg.tiny, device=args.device, dtype=dtype)
    if cfg.student_ckpt:
        C.load_phantom_checkpoint(Path(cfg.student_ckpt), student.rf, hw=hw)
        C.load_phantom_checkpoint(Path(cfg.student_ckpt), reference.rf, hw=hw)
    reference.rf.eval()
    for p in reference.rf.parameters():
        p.requires_grad = False

    data_root, norm = resolve_data(args, hw, paths)
    taus = calibrate_tau_per_task(data_root, hw, cfg.tau_quantile)
    sampler = WindowSampler(hw, student.bb, norm, student=False, seed=cfg.seed)
    ds = C.WindowDataset(data_root, sampler)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0 if cfg.synthetic else cfg.num_workers,
                        collate_fn=C.collate_windows, drop_last=True)

    running_mean = [0.0]

    def step_fn(batch: dict) -> dict:
        return hids_step(student.rf, reference.rf, batch, cfg, taus,
                         running_mean, args.device)

    def on_ckpt(step, opt, sched, ema):
        C.save_phantom_checkpoint(
            out_dir / f"student_hids_{step:06d}.pt", student.rf, hw=hw,
            bb=student.bb, mc=student.mc, train_cfg=cfg, step=step,
            base_ckpt_path=str(paths.cosmos_checkpoint), norm_stats=norm,
            optimizer=opt, scheduler=sched, ema=ema)

    C.train_loop(cfg, student.rf, loader, step_fn, on_checkpoint=on_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
