"""Offline open-loop eval: sample REAL action chunks on held-out val windows
and compare them to the demo actions.

val_action_v_mse is the flow-matching velocity loss — it never checked what
Euler sampling actually produces. This does: for each val window, run
rf.sample() exactly like deploy (encode_gen=False path) and measure the
sampled chunk against the ground-truth demo chunk.

Discriminates: [model/training is bad] vs [deploy distribution shift is bad].
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data.windows import WindowSampler
from phantom.train import common as C
from phantom.train.builder import build_model


def to_dev(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--nfe", type=int, default=5)
    ap.add_argument("--max-windows", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    hw = load_hardware(args.hardware)
    paths = load_paths()
    from phantom.config.model import PhantomModelConfig
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = PhantomModelConfig.from_dict(payload["configs"]["model"])
    print("ckpt model config:", {k: v for k, v in payload["configs"]["model"].items()
                                 if k in ("rope_time_mode", "cond_dropout_p", "action_t_max_of_two")})
    pm = build_model(hw, paths, student=False, tiny=False, mc=mc, device=dev, dtype=dt)
    C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw, load_ema=True, payload=payload)
    ns = payload["norm_stats"]
    a_mean = np.asarray(ns["mean"]["action"], dtype=np.float64)
    a_std = np.asarray(ns["std"]["action"], dtype=np.float64)

    data_root = Path(args.data)
    norm = C.load_norm_stats(data_root) if hasattr(C, "load_norm_stats") else None
    if norm is None:
        from phantom.data.schema import NormStats
        d = json.loads((data_root / "norm_stats.json").read_text())
        norm = NormStats(
            mean={k: np.asarray(v, dtype=np.float32) for k, v in d["mean"].items()},
            std={k: np.asarray(v, dtype=np.float32) for k, v in d["std"].items()})

    sampler = WindowSampler(hw, pm.bb, norm, student=False, seed=0)
    val_eps = C.manifest_split(data_root, "val")
    ds = C.WindowDataset(data_root, sampler, episodes=val_eps, resample=False, seed=0)
    stride = max(1, len(ds) // args.max_windows)
    idxs = list(range(0, len(ds), stride))[: args.max_windows]
    print(f"val windows: {len(ds)} total, evaluating {len(idxs)} (stride {stride}), nfe={args.nfe}")

    def denorm(a):  # (H, A) normalized -> physical
        return a * a_std + a_mean

    rows = []
    pm.rf.eval()
    for n, i in enumerate(idxs):
        item = ds[i]
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
        batch = to_dev(batch, dev, dt)

        gt = batch["action_chunk"][0].float().cpu().numpy().astype(np.float64)
        preds = []
        with torch.no_grad():
            for s in range(args.seeds):
                pm.rf._gen = torch.Generator().manual_seed(1000 + s)
                p = pm.rf.sample(batch, nfe=args.nfe)
                preds.append(p.actions_B_H_A[0].float().cpu().numpy().astype(np.float64))

        gt_d, pr_d = denorm(gt), [denorm(p) for p in preds]
        gp = np.linalg.norm(gt_d[:, :3], axis=1)          # |dpos| per step, m
        pp = np.linalg.norm(pr_d[0][:, :3], axis=1)
        # cosine of step direction where GT moves at all
        mask = gp > 1e-4
        cos = float(np.mean([
            np.dot(pr_d[0][j, :3], gt_d[j, :3]) / (pp[j] * gp[j] + 1e-12)
            for j in range(len(gp)) if mask[j]])) if mask.any() else float("nan")
        seed_var = float(np.mean(np.linalg.norm((pr_d[0] - pr_d[1])[:, :3], axis=1))) if args.seeds > 1 else 0.0
        rows.append({
            "idx": i,
            "task": batch.get("text", ["?"])[0],
            "gt_dpos_mean_mm": float(gp.mean() * 1000),
            "pred_dpos_mean_mm": float(pp.mean() * 1000),
            "mag_ratio": float(pp.mean() / (gp.mean() + 1e-9)),
            "dir_cosine": cos,
            "grip_gt_rng": [float(gt_d[:, 6].min()), float(gt_d[:, 6].max())],
            "grip_pred_rng": [float(pr_d[0][:, 6].min()), float(pr_d[0][:, 6].max())],
            "grip_mae": float(np.abs(pr_d[0][:, 6] - gt_d[:, 6]).mean()),
            "norm_mse_sampled": float(((preds[0] - gt) ** 2).mean()),
            "seed_var_dpos_mm": seed_var * 1000,
            "head_cos": float(np.dot(pr_d[0][:4, :3].ravel(), gt_d[:4, :3].ravel()) /
                              (np.linalg.norm(pr_d[0][:4, :3]) * np.linalg.norm(gt_d[:4, :3]) + 1e-12)),
            "tail_cos": float(np.dot(pr_d[0][12:, :3].ravel(), gt_d[12:, :3].ravel()) /
                              (np.linalg.norm(pr_d[0][12:, :3]) * np.linalg.norm(gt_d[12:, :3]) + 1e-12)),
        })
        if n % 8 == 0:
            r = rows[-1]
            print(f"[{n}/{len(idxs)}] {r['task']:<12} mag_ratio={r['mag_ratio']:.2f} "
                  f"cos={r['dir_cosine']:.2f} sampled_mse={r['norm_mse_sampled']:.3f}", flush=True)

    def agg(key):
        v = np.array([r[key] for r in rows if np.isfinite(r[key])])
        return {"mean": float(v.mean()), "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75))}

    summary = {
        "nfe": args.nfe, "n_windows": len(rows),
        "mag_ratio": agg("mag_ratio"),
        "dir_cosine": agg("dir_cosine"),
        "head_cos": agg("head_cos"), "tail_cos": agg("tail_cos"),
        "sampled_norm_mse": agg("norm_mse_sampled"),
        "flow_val_mse_reference": 0.066,
        "seed_var_dpos_mm": agg("seed_var_dpos_mm"),
        "grip_mae": agg("grip_mae"),
        "per_task": {},
    }
    for t in sorted({r["task"] for r in rows}):
        tr = [r for r in rows if r["task"] == t]
        summary["per_task"][t] = {
            "n": len(tr),
            "mag_ratio": float(np.mean([r["mag_ratio"] for r in tr])),
            "dir_cosine": float(np.nanmean([r["dir_cosine"] for r in tr])),
            "gt_mm": float(np.mean([r["gt_dpos_mean_mm"] for r in tr])),
            "pred_mm": float(np.mean([r["pred_dpos_mean_mm"] for r in tr])),
        }
    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
