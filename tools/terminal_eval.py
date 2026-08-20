"""Terminal-phase offline eval: does the sampled chunk COMMIT to the grasp?

The openloop dir_cos metric is transport-dominated and could not see the
rig's 3-6 cm grasp miss (13/13 episodes closed 35-80 mm above grasp height).
This evaluates windows anchored in the last 1.5 s before the demo's first
gripper close, sampled exactly like deploy, and reports:

  endpoint_err_mm   |cum(pred dpos) - cum(gt dpos)| at chunk end (xyz)
  z_end_err_mm      pred z-at-end - gt z-at-end  (POSITIVE = ends HIGH)
  close_step_err    (first step pred gripper > 0.45) - (same for gt); +ve = late
  commit_ratio      |pred descent| / |gt descent| over the chunk

    python tools/terminal_eval.py --ckpt <pt> --data <root>/tasks \
        --hardware configs/hardware.nuc.yaml [--nfe 5] [--guidance 1.0]
Bars (from demos): endpoint < 15 mm, |z_end_err| < 10 mm, commit_ratio ~ 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data.schema import NormStats
from phantom.data.windows import WindowSampler
from phantom.train import common as C
from phantom.train.builder import build_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--nfe", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lead-s", type=float, default=None,
                    help="anchor t0 this many seconds before the first close "
                         "(default: the chunk duration, so the chunk ENDS at "
                         "the close — no post-grasp lift inside the window)")
    ap.add_argument("--max-episodes", type=int, default=80)
    ap.add_argument("--no-ema", dest="ema", action="store_false", default=True,
                    help="GO scripts deploy with --ema; match that by default")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev, dt = ("cuda" if torch.cuda.is_available() else "cpu"), torch.bfloat16
    if dev == "cpu":
        dt = torch.float32
    hw = load_hardware(args.hardware)
    paths = load_paths()
    from phantom.config.model import PhantomModelConfig
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = PhantomModelConfig.from_dict(payload["configs"]["model"])
    pm = build_model(hw, paths, student=False, tiny=False, mc=mc, device=dev, dtype=dt)
    C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw, load_ema=args.ema, payload=payload)
    ns = payload["norm_stats"]
    a_mean = np.asarray(ns["mean"]["action"], dtype=np.float64)
    a_std = np.asarray(ns["std"]["action"], dtype=np.float64)
    norm = NormStats(mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
                     std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})

    data_root = Path(args.data)
    sampler = WindowSampler(hw, pm.bb, norm, student=False, seed=0)
    val_eps = C.manifest_split(data_root, "val")     # None = every episode
    ds = C.WindowDataset(data_root, sampler, episodes=val_eps, windows_per_episode=1,
                         resample=False, seed=0)
    pm.rf.eval()
    rows = []
    chunk_s = hw.control.chunk_horizon / hw.control.action_rate_hz
    lead = args.lead_s if args.lead_s is not None else chunk_s
    def make_batch(ep, t0):
        item = sampler.sample(ep, t0)
        batch = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).unsqueeze(0)
            elif isinstance(v, str):
                batch[k] = [v]
            elif isinstance(v, (int, float, np.floating)):
                batch[k] = torch.tensor([v])
            else:
                batch[k] = v
        batch = C.to_device(batch, dev, dt)
        # no privileged future contact package (deploy never has it)
        for k in list(batch):
            if k == "events" or k.startswith("cpk_"):
                batch[k] = torch.zeros_like(batch[k])
        return item, batch

    skipped = 0
    for wi in ds.index[: args.max_episodes]:
        tc = ds._close_time(wi.episode)
        if tc is None:
            skipped += 1
            continue
        t0 = tc - lead
        if not (wi.lo <= t0 <= wi.hi):
            skipped += 1           # chunk would not span the close — not a terminal window
            continue
        item, batch = make_batch(wi.episode, t0)
        gt = batch["action_chunk"][0].float().cpu().numpy().astype(np.float64) * a_std + a_mean
        preds = []
        with torch.no_grad():
            for s in range(args.seeds):
                pm.rf._gen = torch.Generator().manual_seed(1000 + s)
                # steady-state deploy path: every rig replan after the first
                # passes the TRUE previous package — reproduce it with a prior
                # window one chunk earlier (first-replan path otherwise)
                prev_cpk = None
                t_prev = t0 - chunk_s
                if t_prev >= wi.lo:
                    _, pb = make_batch(wi.episode, t_prev)
                    prev_cpk = pm.rf.sample(pb, nfe=args.nfe, guidance_scale=args.guidance).cpk
                p = pm.rf.sample(batch, nfe=args.nfe, guidance_scale=args.guidance,
                                 prev_cpk=prev_cpk)
                preds.append(p.actions_B_H_A[0].float().cpu().numpy().astype(np.float64) * a_std + a_mean)
        for pr in preds:
            cg, cp = np.cumsum(gt[:, :3], axis=0), np.cumsum(pr[:, :3], axis=0)
            end_err = float(np.linalg.norm(cp[-1] - cg[-1]) * 1000)
            z_err = float((cp[-1, 2] - cg[-1, 2]) * 1000)
            gt_desc = float(-cg[-1, 2]); pr_desc = float(-cp[-1, 2])
            commit = pr_desc / gt_desc if abs(gt_desc) > 2e-3 else float("nan")
            def first_close(a):
                h = np.nonzero(a[:, 6] > 0.45)[0]
                return int(h[0]) if len(h) else len(a)
            rows.append({"episode": wi.episode.name, "task": item.get("text", "?"),
                         "endpoint_err_mm": end_err, "z_end_err_mm": z_err,
                         "commit_ratio": commit,
                         "close_step_err": first_close(pr) - first_close(gt)})
    if not rows:
        print("no windows with a gripper close found"); return 1
    print(f"episodes skipped (no close / chunk cannot span the close): {skipped}")
    tasks = sorted({r["task"] for r in rows})
    def mean(key, sel):
        v = np.array([r[key] for r in sel if np.isfinite(r[key])])
        return float(v.mean()) if len(v) else float("nan")
    summary = {"nfe": args.nfe, "guidance": args.guidance, "n": len(rows),
               "endpoint_err_mm": mean("endpoint_err_mm", rows),
               "z_end_err_mm": mean("z_end_err_mm", rows),
               "commit_ratio": mean("commit_ratio", rows),
               "close_step_err": mean("close_step_err", rows),
               "per_task": {t: {k: mean(k, [r for r in rows if r["task"] == t])
                                for k in ("endpoint_err_mm", "z_end_err_mm", "commit_ratio", "close_step_err")}
                            for t in tasks}}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
