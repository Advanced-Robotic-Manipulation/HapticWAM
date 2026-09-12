#!/usr/bin/env python
"""Offline endpoint error for a fine-tuned pi0.5 checkpoint, on a LeRobot split.

The number this prints is `endpoint_err_mm`, defined exactly as
`tools/terminal_eval.py` defines it for our own students:

    endpoint_err_mm = || cumsum(pred[:H, :3])[-1] - cumsum(gt[:H, :3])[-1] || * 1000

i.e. how far apart the predicted and demonstrated TCP end up after integrating
H = 16 pose deltas (1.6 s at 10 Hz), which is the horizon `ChunkExecutor`
actually plays. Rotation and the gripper column are reported alongside but the
headline is the translation endpoint, so a pi0.5 row can sit next to a student
row in the paper without a footnote.

Windows are drawn the way terminal_eval draws them: anchored in the last 1.5 s
before the episode's first gripper close, because that is where the rig's
failure mode lives (closing 35-80 mm above the grasp). `--anchor uniform`
spreads them over the whole episode instead, which is the easier metric and
should be quoted as such.

`zero_endpoint_err_mm` is the same metric for a policy that predicts NO motion.
It is the scale reference: a model that has learnt nothing scores that, so a
fine-tune must be far below it before any of this means anything.

The inference path is the deploy path -- `phantom/inference/lerobot_policy.py`
runs exactly this chain, so a number here and a number on the rig disagree only
because of the rig:

    ds[i] -> preprocessor -> predict_action_chunk(num_steps=nfe)
          -> postprocessor (once per chunk step) -> numpy deltas

Usage
-----
    python tools/pi05_offline_eval.py \
        --ckpt ~/lerobot/runs/pi05_phantom_expert_v1/checkpoints/last/pretrained_model \
        --data ~/lerobot/data/phantom_pi05/val \
        [--horizon 16] [--nfe 10] [--anchor close] [--max-windows 200] \
        [--out pi05_offline_eval.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

CLOSE_APERTURE = 0.45  # terminal_eval's absolute close rule


def percentiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": round(float(x.mean()), 3),
        "median": round(float(np.median(x)), 3),
        "p90": round(float(np.percentile(x, 90)), 3),
        "max": round(float(x.max()), 3),
    }


def close_index(grip: np.ndarray) -> int | None:
    """First step whose gripper COMMAND crosses the close threshold."""
    hit = np.nonzero(grip > CLOSE_APERTURE)[0]
    return int(hit[0]) if hit.size else None


def pick_windows(ds, horizon: int, anchor: str, per_episode: int,
                 max_windows: int, rng: np.random.Generator) -> list[int]:
    """Global frame indices to evaluate at."""
    froms = ds.meta.episodes["dataset_from_index"]
    tos = ds.meta.episodes["dataset_to_index"]
    starts: list[int] = []
    for ei in range(ds.meta.total_episodes):
        lo, hi = int(froms[ei]), int(tos[ei])
        n = hi - lo
        if n <= horizon + 1:
            continue
        if anchor == "close":
            # the episode's own gripper COMMAND column, read straight off the
            # single-step action of each frame (action[..., 6])
            grip = np.array([float(ds.hf_dataset[i]["action"][6]) for i in range(lo, hi)])
            ci = close_index(grip)
            if ci is None:
                continue
            # windows whose chunk still spans the close, ending in the 1.5 s
            # before it -- terminal_eval's window rule
            w_hi = min(ci, n - horizon - 1)
            w_lo = max(0, min(ci - 15, w_hi))
            if w_hi <= w_lo:
                continue
            cand = np.arange(w_lo, w_hi + 1)
        else:
            cand = np.arange(0, n - horizon - 1)
        if cand.size == 0:
            continue
        take = cand if cand.size <= per_episode else rng.choice(cand, per_episode, replace=False)
        starts.extend(lo + int(s) for s in np.sort(np.atleast_1d(take)))
    if len(starts) > max_windows:
        starts = sorted(rng.choice(np.array(starts), max_windows, replace=False).tolist())
    return starts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="a checkpoint's pretrained_model/ dir")
    p.add_argument("--data", type=Path, required=True, help="the val split root")
    p.add_argument("--horizon", type=int, default=16,
                   help="steps integrated; ChunkExecutor plays 16")
    p.add_argument("--nfe", type=int, default=None,
                   help="flow-matching steps; default = the checkpoint's own")
    p.add_argument("--anchor", choices=("close", "uniform"), default="close")
    p.add_argument("--per-episode", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    ckpt = args.ckpt.expanduser()
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = args.device
    chunk = int(getattr(cfg, "chunk_size", 50))
    if args.horizon > chunk:
        raise SystemExit(f"--horizon {args.horizon} > checkpoint chunk_size {chunk}")

    root = args.data.expanduser()
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    ds = LeRobotDataset(repo_id=info.get("repo_id") or "local/ds", root=root,
                        delta_timestamps={ACTION: [i / fps for i in range(chunk)]})

    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.to(args.device)
    policy.eval()
    dev = {"device": args.device}
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": dev},
        postprocessor_overrides={"device_processor": dev})

    img_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
    print(f"checkpoint {ckpt}")
    print(f"  chunk_size {chunk}  horizon {args.horizon}  fps {fps}  cameras {img_keys}")
    print(f"dataset {root}  {ds.meta.total_episodes} episodes  {ds.meta.total_frames} frames")

    rng = np.random.default_rng(args.seed)
    starts = pick_windows(ds, args.horizon, args.anchor, args.per_episode,
                          args.max_windows, rng)
    print(f"windows {len(starts)}  anchor={args.anchor}")
    if not starts:
        print("no evaluable window found")
        return 1

    H = args.horizon
    rows: list[dict] = []
    t_start = time.perf_counter()
    for w, i in enumerate(starts):
        sample = ds[i]
        if bool(sample["action_is_pad"][:H].any()):
            continue
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v] if k == "task" else v)
                 for k, v in sample.items()}
        gt = sample[ACTION][:H].numpy().astype(np.float64)
        with torch.no_grad():
            obs = pre(batch)
            if hasattr(policy, "_queues"):
                # policies with an observation history (Diffusion Policy, ACT with
                # n_obs_steps > 1) stack their queues inside predict_action_chunk;
                # a fresh window is one observation, so prime the queues the way
                # lerobot's select_action does (repeat the first observation).
                from lerobot.policies.utils import populate_queues
                policy.reset()
                img_keys = list(getattr(getattr(policy, "config", None), "image_features", []) or [])
                if "observation.images" in policy._queues and img_keys:
                    # Diffusion Policy stacks its camera features under one key in
                    # select_action before queueing; mirror it here
                    obs = dict(obs)
                    obs["observation.images"] = torch.stack([obs[k] for k in img_keys], dim=-4)
                policy._queues = populate_queues(policy._queues, obs, exclude_keys=[ACTION])
                # the dataset window carries the ground-truth ACTION; predict_action_chunk
                # stacks every batch key that has a queue, and the action queue is empty
                obs = {k: v for k, v in obs.items() if k != ACTION}
            out = policy.predict_action_chunk(obs, num_steps=args.nfe) \
                if args.nfe is not None else policy.predict_action_chunk(obs)
            # the deploy adapter unnormalises ONCE PER CHUNK STEP; mirror it
            out = torch.stack([post(out[:, k, :]) for k in range(out.shape[1])], dim=1)
        pred = out[0, :H].float().cpu().numpy().astype(np.float64)

        cg, cp = np.cumsum(gt[:, :3], axis=0), np.cumsum(pred[:, :3], axis=0)
        rg, rp = np.cumsum(gt[:, 3:6], axis=0), np.cumsum(pred[:, 3:6], axis=0)
        gi, pi_ = close_index(gt[:, 6]), close_index(pred[:, 6])
        rows.append({
            "frame": int(i),
            "endpoint_err_mm": float(np.linalg.norm(cp[-1] - cg[-1]) * 1000.0),
            "z_end_err_mm": float((cp[-1, 2] - cg[-1, 2]) * 1000.0),
            "rot_end_err_deg": float(np.degrees(np.linalg.norm(rp[-1] - rg[-1]))),
            "grip_mae": float(np.abs(pred[:, 6] - gt[:, 6]).mean()),
            "close_step_err": float(pi_ - gi) if (gi is not None and pi_ is not None)
                              else float("nan"),
            # a policy that proposes NO motion -- the scale reference
            "zero_endpoint_err_mm": float(np.linalg.norm(cg[-1]) * 1000.0),
        })
        if (w + 1) % 25 == 0:
            print(f"  {w + 1}/{len(starts)} windows "
                  f"({(time.perf_counter() - t_start) / (w + 1):.2f} s/window)", flush=True)

    if not rows:
        print("every window was padded; nothing evaluated")
        return 1

    summary = {
        "checkpoint": str(ckpt),
        "dataset": str(root),
        "anchor": args.anchor,
        "horizon": H,
        "nfe": args.nfe if args.nfe is not None else getattr(cfg, "num_inference_steps", None),
        "n_windows": len(rows),
        "s_per_window": round((time.perf_counter() - t_start) / len(rows), 3),
        "metrics": {k: percentiles(np.array([r[k] for r in rows]))
                    for k in ("endpoint_err_mm", "z_end_err_mm", "rot_end_err_deg",
                              "grip_mae", "close_step_err", "zero_endpoint_err_mm")},
    }
    print("=" * 72)
    for name, m in summary["metrics"].items():
        if m.get("n"):
            print(f"{name:24s} mean {m['mean']:9.3f}  median {m['median']:9.3f}  "
                  f"p90 {m['p90']:9.3f}  max {m['max']:9.3f}  n={m['n']}")
    e = summary["metrics"]["endpoint_err_mm"]["mean"]
    z = summary["metrics"]["zero_endpoint_err_mm"]["mean"]
    print(f"\nendpoint {e:.1f} mm vs {z:.1f} mm for a no-motion policy "
          f"({e / z:.2f}x)" if z else "")
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
