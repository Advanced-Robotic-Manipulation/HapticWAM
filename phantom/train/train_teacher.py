"""Program (2): teacher PHANTOM fine-tune (pipeline.md §4/§6) — LoRA on the
frozen 2B DiT + all phantom modules (HHT, ACC, ACE heads, frame-type
embeddings), joint grouped RF objective, all tasks co-trained.

Launch (single 5090 — the default configs/compute.yaml target):
    python -m phantom.train.train_teacher --data <episodes_root> \
        [--tactile-pretrain <ckpt>] [--run-name teacher_v1]
Launch (8xH100 / 8xA100 — set target: h100x8|a100x8 in configs/compute.yaml
or configs/compute.local.yaml, or pass --compute h100x8):
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

from phantom.config.compute import load_compute
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
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compute", default=None,
                    help="compute profile name in configs/compute.yaml, or a path "
                         "to a compute yaml (default: the file's target:)")


def apply_overrides(cfg, args, compute=None):
    if compute is not None:
        cfg = compute.apply_to(cfg)   # profile beats dataclass defaults;
                                      # the CLI updates below beat the profile
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
    if getattr(args, "num_workers", None) is not None:
        updates["num_workers"] = args.num_workers
    # fine-tune knobs (2026-08-26): a fresh cosine at the from-scratch peak
    # gives a converged checkpoint a per-element Adam displacement budget of
    # ~2.7x the LoRA-B weight scale — a rewrite, not a fine-tune. Explicit
    # CLI values beat the compute profile.
    for name in ("lr", "lr_new_modules", "warmup_steps", "ckpt_every", "eval_every"):
        v = getattr(args, name, None)
        if v is not None:
            updates[name] = v
    if args.max_steps is not None:
        for name in ("ckpt_every", "eval_every"):
            if name in updates:
                updates[name] = min(updates[name], args.max_steps)
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
    ap.add_argument("--allow-config-drift", action="store_true",
                    help="train even though episodes were recorded under a "
                         "different hardware config (checkpoint may not load "
                         "on the rig)")
    ap.add_argument("--split", default="train", choices=["train", "val", "all"],
                    help="episode subset from manifests/all.jsonl (default train; "
                         "'all' reproduces the pre-split behaviour)")
    ap.add_argument("--tactile-pretrain", default="", help="program-1 checkpoint")
    ap.add_argument("--grasp-frac", type=float, default=0.0,
                    help="fraction of training windows anchored in the 1.5 s "
                         "before the first gripper close (terminal-phase "
                         "fine-tune; 0 = uniform)")
    ap.add_argument("--photo-aug", type=float, default=0.0,
                    help="photometric jitter strength on the scene camera "
                         "(0 disables; 1.0 = brightness +-30%%, contrast "
                         "+-25%%, per-channel +-8%% — one draw per window)")
    ap.add_argument("--init-weights", default="",
                    help="teacher checkpoint to initialize WEIGHTS from, with a "
                         "fresh optimizer + schedule (fine-tuning; contrast "
                         "--resume which restores optimizer/step too)")
    ap.add_argument("--init-ema", dest="init_ema", action="store_true", default=True,
                    help="(default) --init-weights starts from the checkpoint's EMA "
                         "weights — the artifact eval/deploy actually use")
    ap.add_argument("--no-init-ema", dest="init_ema", action="store_false",
                    help="start the fine-tune from the raw (last-step) weights")
    ap.add_argument("--lr", type=float, default=None, help="peak LoRA LR (config default 1e-4)")
    ap.add_argument("--lr-new-modules", type=float, default=None,
                    help="peak LR for HHT/ACC/heads (config default 3e-4)")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="linear warmup steps (default min(500, max_steps//10))")
    ap.add_argument("--ckpt-every", type=int, default=None, help="checkpoint cadence (default 1000)")
    ap.add_argument("--eval-every", type=int, default=None, help="val cadence (default 1000)")
    ap.add_argument("--resume", default="",
                    help="teacher checkpoint to resume from: restores lora+phantom "
                         "weights, optimizer, scheduler, EMA and the step counter, "
                         "then continues to --max-steps. Model flags must match the "
                         "original run (checked against the checkpoint's saved "
                         "model config).")
    ap.add_argument("--acc-two-pass", action="store_true",
                    help="train ACC on the TRUE previous-replan prediction (two "
                         "forward passes) instead of the gt_noised proxy. REQUIRED "
                         "for the final teacher whose gate lead-time is reported "
                         "(RQ2) — gt_noised trains the gate against leaked GT.")
    ap.add_argument("--rope-time-mode", default="time_true",
                    choices=["aligned", "append", "time_true"],
                    help="ACTION-frame RoPE positions. time_true (v4 default) "
                         "places each action frame at its physical future "
                         "time; aligned reproduces v3's [1,2,3,3] clamping "
                         "(last frames alias one temporal phase).")
    ap.add_argument("--cond-dropout", type=float, default=0.1,
                    help="p of nulling ALL observation inputs for a training "
                         "sample (classifier-free) — makes obs-guidance "
                         "available at sampling and sharpens obs->action "
                         "coupling. 0 disables.")
    ap.add_argument("--action-t-max-of-two", action="store_true", default=True,
                    help="ACTION frames train at max of two timestep draws "
                         "(high-noise-biased — the band few-NFE sampling "
                         "visits first). --no-action-t-max-of-two disables.")
    ap.add_argument("--no-action-t-max-of-two", dest="action_t_max_of_two",
                    action="store_false")
    args = ap.parse_args(argv)

    rank, world = C.setup_ddp()   # EARLY: "cuda" resolves per-rank from here on
    profile = load_compute(args.compute)
    profile.check_world(world)
    comp = profile.for_program("train_teacher")

    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    cfg = apply_overrides(TeacherTrainConfig(), args, compute=comp)
    out_dir = Path(cfg.out_dir) if cfg.out_dir else paths.runs_root / "teacher" / cfg.run_name
    dtype = C.pick_dtype(args.device, args.tiny, comp)

    from phantom.config.model import AccConfig, PhantomModelConfig
    acc = (AccConfig(self_anticipation="two_pass") if args.acc_two_pass
           else AccConfig())
    if not args.acc_two_pass:
        log.warning(
            "ACC self-anticipation = gt_noised (fast proxy). Do NOT report the "
            "RQ2 gate lead-time from this run — retrain the final teacher with "
            "--acc-two-pass so the gate never sees leaked GT contact.")
    mc = PhantomModelConfig(student=False, acc=acc,
                            rope_time_mode=args.rope_time_mode,
                            cond_dropout_p=args.cond_dropout,
                            action_t_max_of_two=args.action_t_max_of_two)

    pm = build_model(hw, paths, student=False, tiny=cfg.tiny, mc=mc,
                     load_base=not cfg.tiny, device=args.device, dtype=dtype)

    resume_payload = None
    if args.init_weights:
        assert not args.resume, "--init-weights and --resume are exclusive"
        assert not args.tactile_pretrain, "--init-weights already carries the tactile encoder"
        init_payload = C.load_phantom_checkpoint(Path(args.init_weights), pm.rf, hw=hw,
                                                 load_ema=args.init_ema)
        log.info("--init-weights %s from %s weights", args.init_weights,
                 "EMA" if args.init_ema else "RAW")
        saved_mc = init_payload["configs"]["model"]
        if saved_mc != mc.to_dict():
            # same shapes can hide a silent behavioral change (acc two_pass ->
            # gt_noised, rope mode, cond-dropout): a fine-tune must inherit the
            # checkpoint's model config unless the operator says otherwise
            drift = {k: (saved_mc.get(k), v) for k, v in mc.to_dict().items()
                     if saved_mc.get(k) != v}
            raise SystemExit(f"--init-weights model-config drift vs checkpoint: "
                             f"{drift} — pass the flags the checkpoint was "
                             f"trained with (e.g. --acc-two-pass)")
        log.info("weights initialized from %s (fresh optimizer/schedule)",
                 args.init_weights)
    if args.resume:
        assert not args.tactile_pretrain, "--resume already carries trained weights"
        resume_payload = C.load_phantom_checkpoint(Path(args.resume), pm.rf, hw=hw)
        saved_mc = resume_payload["configs"]["model"]
        if saved_mc != mc.to_dict():
            drift = {k: (saved_mc.get(k), v) for k, v in mc.to_dict().items()
                     if saved_mc.get(k) != v}
            raise SystemExit(f"--resume model-config drift vs checkpoint: {drift} "
                             f"— pass the flags the original run used")
        log.info("resuming from %s at step %d", args.resume, resume_payload["step"])

    if args.tactile_pretrain:
        pm.rf.hht.load_pretrained_tactile(Path(args.tactile_pretrain))
        if cfg.freeze_tactile_steps > 0:
            for p in pm.rf.hht.phantom_tactile_enc.parameters():
                p.requires_grad = False
            log.info("tactile encoder frozen for warmup (%d steps configured)",
                     cfg.freeze_tactile_steps)

    data_root, norm = resolve_data(args, hw, paths)
    if args.init_weights:
        ns = init_payload.get("norm_stats")
        if ns:
            import numpy as _np
            # every normalized key, mean AND std (an action-mean-only check let
            # a recomputed norm_stats through with different stds / obs keys)
            keys = set(ns["mean"]) | set(getattr(norm, "mean", {}))
            for k in sorted(keys):
                if k not in ns["mean"] or k not in norm.mean:
                    raise SystemExit(f"--init-weights norm_stats key set differs from the "
                                     f"data root's norm_stats.json (missing {k!r})")
                for which, a, b in (("mean", ns["mean"][k], norm.mean[k]),
                                    ("std", ns["std"][k], norm.std[k])):
                    if not _np.allclose(a, b, atol=1e-6):
                        raise SystemExit(f"--init-weights norm_stats[{k}].{which} differ from "
                                         f"the fine-tune data root's norm_stats.json — the "
                                         f"checkpoint's normalization would not match the data")
    sampler = WindowSampler(hw, pm.bb, norm, student=False, seed=cfg.seed)
    train_eps = C.manifest_split(data_root, args.split)
    ds = C.WindowDataset(data_root, sampler, episodes=train_eps, seed=cfg.seed,
                         grasp_frac=args.grasp_frac, photo_aug=args.photo_aug)
    if args.grasp_frac > 0:
        cov = ds.grasp_coverage()
        log.info("terminal-phase weighting: %.0f%% of windows anchored before "
                 "the first gripper close — coverage %s", 100 * args.grasp_frac, cov)
        if cov["weightable"] < 0.5 * max(cov["episodes"], 1):
            raise SystemExit(f"grasp weighting would silently degrade: only "
                             f"{cov['weightable']}/{cov['episodes']} episodes "
                             f"weightable ({cov}) — check gripper.zarr ts/pos")
    log.info("dataset: %d windows from %s (split=%s)", len(ds), data_root, args.split)
    # A training run under a config the data was NOT recorded under silently
    # changes window semantics and produces a checkpoint the rig rejects on
    # load (shape assert). Shapes are checked separately; this catches VALUE
    # drift, which is the failure mode that survives to deployment.
    if sampler.n_config_drift:
        msg = (f"{sampler.n_config_drift}/{len(ds.index) // 8 or 1} episodes were "
               f"recorded under a DIFFERENT hardware config than "
               f"{args.hardware!r}. Pass the config the data was recorded with "
               f"(e.g. --hardware configs/hardware.nuc.yaml) or the resulting "
               f"checkpoint may not load on the rig.")
        if args.allow_config_drift:
            log.warning("CONFIG DRIFT (allowed): %s", msg)
        else:
            raise SystemExit(f"CONFIG DRIFT: {msg}\n"
                             f"Re-run with --allow-config-drift to override.")
    loader = C.make_loader(ds, cfg)

    val_loader = None
    if args.split == "train":
        val_eps = C.manifest_split(data_root, "val")
        if val_eps:
            # frozen anchors on the held-out set so successive evals compare
            val_ds = C.WindowDataset(data_root, sampler, episodes=val_eps,
                                     resample=False, seed=cfg.seed)
            val_loader = C.make_loader(val_ds, cfg, shuffle=False)
            log.info("val: %d windows from %d episodes", len(val_ds), len(val_eps))

    if not cfg.synthetic and not cfg.tiny:
        # LOUD label-degeneracy gate (issue #1: v3 trained its entire
        # contact/anticipation stack on 100%-positive gate labels and 96%
        # "hold" events without anything noticing — teacher capacity spent
        # memorizing constants). Probes real windows through the exact
        # training path before any GPU time is spent.
        C.assert_label_sanity(ds, log)

    def step_fn(batch: dict) -> dict:
        return pm.rf.training_step(C.to_device(batch, args.device))

    tc_prov = {"cache_path": str(getattr(paths, "cosmos_text_embedding_cache", "") or ""),
               "active": bool(getattr(pm.rf.text, "_cache", None))}
    if tc_prov["active"]:
        tc_prov["texts"] = sorted(pm.rf.text._cache.keys())

    def on_ckpt(step, opt, sched, ema):
        C.save_phantom_checkpoint(
            out_dir / f"teacher_{step:06d}.pt", pm.rf, hw=hw, bb=pm.bb, mc=pm.mc,
            train_cfg=cfg, step=step, base_ckpt_path=str(paths.cosmos_checkpoint),
            norm_stats=norm, optimizer=opt, scheduler=sched, ema=ema,
            text_conditioning=tc_prov)

    sampled_eval = None
    if val_loader is not None and not cfg.tiny:
        val_ds_ref = val_loader.dataset
        sampled_eval = lambda: C.evaluate_sampled(  # noqa: E731
            pm.rf, val_ds_ref, norm.mean["action"], norm.std["action"],
            n_windows=8, nfe=pm.mc.nfe)

    C.train_loop(cfg, pm.rf, loader, step_fn, on_checkpoint=on_ckpt,
                 val_loader=val_loader, sampled_eval_fn=sampled_eval,
                 resume_payload=resume_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
