"""Which inputs does the action head actually listen to?

Same-seed sampling on val windows with one input knocked out at a time;
report mean |delta action| vs baseline per knockout + best-of-4-seed
direction cosine (multimodality check: does SOME seed capture the GT mode?).
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


def collate1(item):
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
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--nfe", type=int, default=5)
    ap.add_argument("--max-windows", type=int, default=24)
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
    a_std = np.asarray(ns["std"]["action"], dtype=np.float64)
    a_mean = np.asarray(ns["mean"]["action"], dtype=np.float64)

    data_root = Path(args.data)
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

    KNOCKOUTS = {
        "baseline": lambda b: b,
        "prev_chunk_zero": lambda b: {**b, "prev_chunk": torch.zeros_like(b["prev_chunk"])},
        "text_empty": lambda b: {**b, "text": [""] * len(b["text"])},
        "wrist_zero": lambda b: {**b, "wrist": torch.zeros_like(b["wrist"])},
        "tactile_zero": lambda b: {**b, **{k: torch.zeros_like(b[k])
                                           for k in ("gel", "fields", "contact_state") if k in b}},
        "video_gray": lambda b: {**b, "video": torch.zeros_like(b["video"]),
                                 "gel": b.get("gel")},
    }

    def sample_actions(batch, seed):
        pm.rf._gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            p = pm.rf.sample(batch, nfe=args.nfe)
        return p.actions_B_H_A[0].float().cpu().numpy().astype(np.float64)

    deltas = {k: [] for k in KNOCKOUTS if k != "baseline"}
    best_cos, first_cos = [], []
    pm.rf.eval()
    for n, i in enumerate(idxs):
        raw = collate1(ds[i])
        gt = raw["action_chunk"][0].numpy().astype(np.float64) * a_std + a_mean
        base_batch = to_dev(raw, dev, dt)
        base = sample_actions(base_batch, 1000)
        base_d = base * a_std + a_mean

        # knockouts, same seed -> pure input-sensitivity
        for k, fn in KNOCKOUTS.items():
            if k == "baseline":
                continue
            kb = fn(dict(base_batch))
            kb = {kk: vv for kk, vv in kb.items() if vv is not None}
            a = sample_actions(kb, 1000)
            dpos = np.linalg.norm(((a - base) * a_std)[:, :3], axis=1)
            deltas[k].append(float(dpos.mean() * 1000))       # mm

        # multimodality: best-of-4 seeds direction cosine vs GT
        cs = []
        for s in range(4):
            a = sample_actions(base_batch, 2000 + s) * a_std + a_mean
            num = np.dot(a[:, :3].ravel(), gt[:, :3].ravel())
            den = np.linalg.norm(a[:, :3]) * np.linalg.norm(gt[:, :3]) + 1e-12
            cs.append(float(num / den))
        best_cos.append(max(cs))
        first_cos.append(cs[0])
        if n % 6 == 0:
            print(f"[{n}/{len(idxs)}] " +
                  " ".join(f"{k}={deltas[k][-1]:.1f}mm" for k in deltas), flush=True)

    summary = {
        "nfe": args.nfe, "n_windows": len(idxs),
        "knockout_mean_dpos_shift_mm": {k: float(np.mean(v)) for k, v in deltas.items()},
        "dir_cosine_first_seed": float(np.mean(first_cos)),
        "dir_cosine_best_of_4": float(np.mean(best_cos)),
    }
    print("\n==== ABLATION SUMMARY ====")
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
