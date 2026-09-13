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

Matched windows (`--windows`)
-----------------------------
`--anchor close` draws its OWN windows, so a baseline row and a PHANTOM row are
two different samples of the same val split and can only be compared in
aggregate. `--windows eval5/v6_teacher.json` instead scores exactly the
(episode, t0) windows that `tools/terminal_eval.py` scored, 4 seeds each, and
writes `episode` / `t0` / `seed` into every row -- the identity
`phantom/eval/stats.py offline` joins on, so the two files pair window by
window. `--episodes-root data/phantom-episodes` makes the t0 -> frame map exact
instead of modelling the tick grid (see `frame_in_episode`).

    python tools/pi05_offline_eval.py --ckpt <ckpt> --data <val> --nfe 10 \
        --windows ~/data2/phantom-v6/eval5/v6_teacher.json \
        --episodes-root ~/data2/lerobot/phantom_repo/data/phantom-episodes \
        --out baseline_pi05_20k_matched.json
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


def load_terminal_windows(path: Path) -> tuple[list[dict], list[int]]:
    """A terminal_eval json -> its unique (episode, t0) windows + its seed list.

    terminal_eval writes one row per (window, seed); 124 episodes x 4 seeds is
    496 rows over 124 distinct windows. The windows keep the file's order.
    """
    d = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    src = d.get("rows", d) if isinstance(d, dict) else d
    seeds = sorted({int(r["seed"]) for r in src if r.get("seed") is not None})
    wins, seen = [], set()
    for r in src:
        ep, t0 = r.get("episode"), r.get("t0")
        if ep is None or t0 is None:
            continue
        key = (str(ep), round(float(t0), 6))
        if key not in seen:
            seen.add(key)
            wins.append({"episode": str(ep), "t0": float(t0)})
    return wins, (seeds or [0])


def frame_in_episode(meta_row: dict, t0: float, fps: int,
                     act_ts: np.ndarray | None = None) -> int:
    """LeRobot frame index inside the episode for terminal_eval's anchor `t0`.

    terminal_eval's GT chunk is the action rows at the first recorded tick
    >= t0 + j/fps (j = 1..H, `WindowSampler._action_grid(..., future=True)`).
    `tools/export_lerobot.py` gives LeRobot frame k the action row
    `act_index_lo + k + 1`, so the two chunks start on the SAME action row when

        k = (first action row at/after t0 + 1/fps) - act_index_lo - 1.

    With `act_ts` (the episode's real action timestamps, t_master seconds) that
    is exact. Without it the tick grid is modelled as
    `first_frame_t + k * action_dt_mean_s`, both recorded per episode by the
    export in `phantom_episodes.json`; the two agree unless the recording
    jittered by more than half a tick inside the episode.
    """
    lo = int(meta_row["act_index_lo"])
    # 1 us of slack: t0 + 1/fps is a float sum, and a tick that should land ON
    # it must not be pushed a whole frame later by the last bit of the mantissa
    t_first = t0 + 1.0 / fps - 1e-6
    if act_ts is not None:
        j = int(np.searchsorted(np.asarray(act_ts, dtype=np.float64), t_first, side="left"))
        return j - lo - 1
    dt = float(meta_row.get("action_dt_mean_s") or 0.0) or (1.0 / fps)
    return int(np.ceil((t_first - float(meta_row["first_frame_t"])) / dt - 1e-9)) - 1


def map_terminal_windows(wins: list[dict], episodes: list[dict], *, fps: int,
                         horizon: int, act_ts: dict | None = None) \
        -> tuple[list[dict], list[dict]]:
    """(mapped, skipped). `episodes` is the export's phantom_episodes.json.

    A window is skipped, never guessed, when its episode is not in this export
    or when the mapped chunk would run off either end of the episode.
    """
    by_name = {e["episode"]: e for e in episodes}
    mapped, skipped = [], []
    for w in wins:
        m = by_name.get(w["episode"])
        if m is None:
            skipped.append({**w, "reason": "episode not in this LeRobot export"})
            continue
        k = frame_in_episode(m, w["t0"], fps, (act_ts or {}).get(w["episode"]))
        n = int(m["n_frames"])
        if k < 0:
            skipped.append({**w, "reason": f"t0 before the exported frames (k={k})"})
            continue
        if k + horizon > n:
            skipped.append({**w, "reason": f"chunk past the episode end (k={k}, n_frames={n})"})
            continue
        mapped.append({**w, "episode_index": int(m["episode_index"]), "frame_in_ep": k})
    return mapped, skipped


def read_action_times(episodes_root: Path, episodes: list[dict]) -> dict:
    """episode name -> its recorded action timestamps, straight off the episode
    store's `actions.zarr/ts` (zarr only, no phantom import: this runs inside
    the LeRobot venv). Episodes whose directory is absent are simply left out,
    and those windows fall back to the mean-tick model."""
    import zarr
    out = {}
    for e in episodes:
        d = Path(episodes_root).expanduser() / e["path"] / "actions.zarr"
        if not d.is_dir():
            continue
        try:
            out[e["episode"]] = np.asarray(zarr.open_group(str(d), mode="r")["ts"][:],
                                           dtype=np.float64)
        except Exception as exc:      # a half-written episode must not kill the eval
            print(f"  act_ts unreadable for {e['episode']}: {exc}")
    return out


def base_row(i: int, gt: np.ndarray, pred: np.ndarray) -> dict:
    """The row both modes share -- key order frozen, the old mode's json is
    byte-identical to what it was before --windows existed."""
    cg, cp = np.cumsum(gt[:, :3], axis=0), np.cumsum(pred[:, :3], axis=0)
    rg, rp = np.cumsum(gt[:, 3:6], axis=0), np.cumsum(pred[:, 3:6], axis=0)
    gi, pi_ = close_index(gt[:, 6]), close_index(pred[:, 6])
    return {
        "frame": int(i),
        "steps_scored": int(pred.shape[0]),
        "endpoint_err_mm": float(np.linalg.norm(cp[-1] - cg[-1]) * 1000.0),
        "z_end_err_mm": float((cp[-1, 2] - cg[-1, 2]) * 1000.0),
        "rot_end_err_deg": float(np.degrees(np.linalg.norm(rp[-1] - rg[-1]))),
        "grip_mae": float(np.abs(pred[:, 6] - gt[:, 6]).mean()),
        "close_step_err": float(pi_ - gi) if (gi is not None and pi_ is not None)
                          else float("nan"),
        # a policy that proposes NO motion -- the scale reference
        "zero_endpoint_err_mm": float(np.linalg.norm(cg[-1]) * 1000.0),
    }


def head_metrics(gt: np.ndarray, pred: np.ndarray, head_steps: int = 8) -> dict:
    """terminal_eval's head_* block: the same endpoint over the first
    `head_steps` actions (the part of the chunk the rig usually executes
    before it replans)."""
    cg, cp = np.cumsum(gt[:, :3], axis=0), np.cumsum(pred[:, :3], axis=0)
    h = max(1, min(int(head_steps), len(cg)))
    return {
        "head_endpoint_err_mm": float(np.linalg.norm(cp[h - 1] - cg[h - 1]) * 1000.0),
        "head_z_err_mm": float((cp[h - 1, 2] - cg[h - 1, 2]) * 1000.0),
        "head_steps": int(h),
    }


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
    p.add_argument("--windows", type=Path, default=None,
                   help="a tools/terminal_eval.py json: score EXACTLY its "
                        "(episode, t0) windows, so the rows join pairwise with "
                        "the PHANTOM rows in phantom/eval/stats.py offline")
    p.add_argument("--episodes-root", type=Path, default=None,
                   help="with --windows: data/phantom-episodes, for an exact "
                        "t0 -> frame map read off the recorded action "
                        "timestamps instead of the per-episode mean tick")
    p.add_argument("--act-ts", type=Path, default=None,
                   help="with --windows: {episode: [action timestamps]} json "
                        "written by --dump-act-ts on a box that HAS the raw "
                        "episodes -- same exact map, no episode store needed")
    p.add_argument("--dump-act-ts", type=Path, default=None,
                   help="write --episodes-root's action timestamps to this json "
                        "and exit (feeds --act-ts on another box)")
    p.add_argument("--seeds", type=int, default=None,
                   help="with --windows: how many noise seeds per window "
                        "(default: the seeds the windows file itself used)")
    p.add_argument("--head-steps", type=int, default=8,
                   help="with --windows: head_* horizon, terminal_eval's 8")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.dump_act_ts:
        if not args.episodes_root:
            raise SystemExit("--dump-act-ts needs --episodes-root")
        eps = json.loads((args.data.expanduser() / "phantom_episodes.json")
                         .read_text(encoding="utf-8"))
        ts = read_action_times(args.episodes_root, eps)
        args.dump_act_ts.write_text(json.dumps(
            {k: [float(x) for x in v] for k, v in ts.items()}))
        print(f"wrote {args.dump_act_ts}: {len(ts)}/{len(eps)} episodes")
        return 0

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
    cfg_ = getattr(policy, "config", None)
    if cfg_ is not None and hasattr(cfg_, "n_action_steps") and hasattr(cfg_, "horizon") \
            and hasattr(cfg_, "n_obs_steps") and cfg_.n_action_steps < args.horizon:
        # score as many steps as the policy can execute from one chunk (DP: horizon - n_obs_steps + 1)
        cfg_.n_action_steps = min(args.horizon, cfg_.horizon - cfg_.n_obs_steps + 1)
        print(f"n_action_steps raised to {cfg_.n_action_steps} for scoring")
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
    froms = ds.meta.episodes["dataset_from_index"]
    win_meta: list[dict | None] = []
    skipped_windows: list[dict] = []
    if args.windows:
        wins, file_seeds = load_terminal_windows(args.windows)
        seeds = list(range(args.seeds)) if args.seeds else file_seeds
        episodes = json.loads((root / "phantom_episodes.json").read_text(encoding="utf-8"))
        if args.act_ts:
            act_ts = {k: np.asarray(v, dtype=np.float64) for k, v in
                      json.loads(args.act_ts.read_text(encoding="utf-8")).items()}
        elif args.episodes_root:
            act_ts = read_action_times(args.episodes_root, episodes)
        else:
            act_ts = None
        mapped, skipped_windows = map_terminal_windows(
            wins, episodes, fps=fps, horizon=args.horizon, act_ts=act_ts)
        starts = [int(froms[w["episode_index"]]) + w["frame_in_ep"] for w in mapped]
        win_meta = mapped
        print(f"windows {len(starts)} of {len(wins)} from {args.windows} "
              f"(skipped {len(skipped_windows)}; seeds {seeds}; "
              f"mapping={'exact act_ts' if act_ts else 'per-episode mean tick'})")
        for sk in skipped_windows[:10]:
            print(f"  skipped {sk['episode']} t0={sk['t0']:.4f}: {sk['reason']}")
    else:
        seeds = [None]
        starts = pick_windows(ds, args.horizon, args.anchor, args.per_episode,
                              args.max_windows, rng)
        win_meta = [None] * len(starts)
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
            if win_meta[w] is not None:
                skipped_windows.append({**{k: v for k, v in win_meta[w].items()
                                           if k in ("episode", "t0")},
                                        "reason": "LeRobot padded the chunk"})
            continue
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v] if k == "task" else v)
                 for k, v in sample.items()}
        gt_full = sample[ACTION][:H].numpy().astype(np.float64)
        for seed in seeds:
            gt = gt_full
            if seed is not None:
                torch.manual_seed(int(seed))
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
            if pred.shape[0] < H:
                # a policy whose executable chunk is shorter than the horizon (Diffusion
                # Policy: horizon 16 with n_obs_steps 2 -> 15 executable steps) is scored
                # over its own length; the row records it so the table can say so
                gt = gt_full[: pred.shape[0]]

            row = base_row(i, gt, pred)
            if win_meta[w] is not None:
                # the identity phantom/eval/stats.py joins on: (episode, t0, seed)
                row = {"episode": win_meta[w]["episode"], "t0": win_meta[w]["t0"],
                       "seed": int(seed), "episode_index": win_meta[w]["episode_index"],
                       "frame_in_ep": win_meta[w]["frame_in_ep"], **row,
                       **head_metrics(gt, pred, args.head_steps)}
            rows.append(row)
        if (w + 1) % 25 == 0:
            print(f"  {w + 1}/{len(starts)} windows "
                  f"({(time.perf_counter() - t_start) / (w + 1):.2f} s/window)", flush=True)

    if not rows:
        print("every window was padded; nothing evaluated")
        return 1

    keys = ["endpoint_err_mm", "z_end_err_mm", "rot_end_err_deg",
            "grip_mae", "close_step_err", "zero_endpoint_err_mm"]
    summary = {
        "checkpoint": str(ckpt),
        "dataset": str(root),
        "anchor": args.anchor,
        "horizon": H,
        "nfe": args.nfe if args.nfe is not None else getattr(cfg, "num_inference_steps", None),
        "n_windows": len(rows),
        "s_per_window": round((time.perf_counter() - t_start) / len(rows), 3),
        "metrics": {k: percentiles(np.array([r[k] for r in rows])) for k in keys},
    }
    if args.windows:
        summary["metrics"]["head_endpoint_err_mm"] = percentiles(
            np.array([r["head_endpoint_err_mm"] for r in rows]))
        summary.update({
            "windows_file": str(args.windows),
            "window_mapping": "exact act_ts" if act_ts else "mean tick",
            "seeds": [int(x) for x in seeds],
            "n_scored_windows": len({(r["episode"], round(r["t0"], 4)) for r in rows}),
            "n_episodes": len({r["episode"] for r in rows}),
            "n_windows_requested": len(wins),
            "n_windows_skipped": len(skipped_windows),
            "steps_scored": sorted({r["steps_scored"] for r in rows}),
            "skipped_windows": skipped_windows,
        })
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
