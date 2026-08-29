"""Replay recorded rig episodes through the DEPLOY stack itself.

Unlike tools/replay_rig.py (which samples via policy._batch_from_obs + rf.sample
with K tiled seeds), this builds the policy exactly as run_deploy.build_policy
does (EMA, LoRA fold, dtype, flags) and drives `PhantomPolicy.replan()` replan by
replan, chaining prev_plan the way PlannerLoop does. One full pass per seed. If
this reproduces the rig where replay_rig does not, the difference is inside the
deploy policy object; if it does not, the difference is in the recorded INPUTS
(e.g. the JPEG re-encode of the scene frame) — E0 discriminator, 2026-08-29.

    python tools/replay_deploy_path.py --ckpt <pt> --hardware configs/hardware.nuc.yaml \
        --episodes <ep_dir>... [--nfe 5] [--seeds 4] [--persistent-noise] [--out json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phantom.config.hardware import load_hardware          # noqa: E402
from phantom.config.paths import load_paths                # noqa: E402
from phantom.scripts.run_deploy import build_policy        # noqa: E402
from tools.replay_rig import RigEpisode, chunk_metrics, summarize   # noqa: E402

log = logging.getLogger("replay_deploy_path")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--nfe", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--persistent-noise", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="skip run_deploy's LoRA fold")
    ap.add_argument("--deploy-rng", action="store_true",
                    help="reproduce the rig's ACTUAL noise: keep the sampler generator at its "
                         "constructor seed, run the warm-up replan exactly as run_deploy does, "
                         "then replay episode-0 traces (seeds forced to 1). A match to the mm "
                         "proves the rig ran one fixed noise draw (2026-08-29)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.deploy_rng:
        args.seeds = 1
    hw = load_hardware(args.hardware)
    paths = load_paths()
    dargs = SimpleNamespace(system="teacher", ckpt=args.ckpt, device="cuda", tiny=False,
                            ema=True, nfe=args.nfe, guidance=args.guidance,
                            persistent_noise=args.persistent_noise, k_seeds=1,
                            parity_fixes=False, drop_video=False, compile=False,
                            flex=False, fp8=False, task="", text="")
    if args.no_merge:
        import phantom.backbone.loader as bl
        bl.merge_lora = lambda net: 0
    policy = build_policy(dargs, hw, paths)
    teacher = True
    out = []
    for ep_dir in args.episodes:
        ep = RigEpisode(Path(ep_dir), hw)
        policy.task_text = ep.meta.text or ep.meta.task
        per_seed = []          # per_seed[j] = list of chunks per accepted replan
        for j in range(args.seeds):
            if args.deploy_rng:
                # exactly run_deploy's sequence: fresh generator at seed 0
                # (rf.py constructor), one warm-up replan on fake_obs, reset
                policy.rf._gen = torch.Generator().manual_seed(0)
                from phantom.scripts.bench_inference import fake_obs
                policy.reset_episode()
                with torch.no_grad():
                    policy.replan(fake_obs(hw, teacher=True), None, np.zeros(6))
            else:
                policy.rf._gen = torch.Generator().manual_seed(1000 + j)
            policy.reset_episode()
            prev_plan, prev_fields, chunks = None, None, []
            for i, r in enumerate(ep.trace):
                t = ep.t_master(i)
                snap, prev_fields = ep.snapshot(t, prev_fields, teacher=teacher)
                if not r.get("accepted", True):
                    continue
                tcp = snap.ur_state[2 * hw.arm.dof:2 * hw.arm.dof + 6]
                with torch.no_grad():
                    plan = policy.replan(snap, prev_plan, tcp)
                prev_plan = plan
                chunks.append(np.asarray(plan.actions, dtype=np.float64))
            per_seed.append(chunks)
        n = min(len(c) for c in per_seed)
        acc = [r for r in ep.trace if r.get("accepted", True)]
        rows = []
        for k in range(n):
            row = summarize([chunk_metrics(per_seed[j][k]) for j in range(args.seeds)],
                            chunk_metrics(np.asarray(acc[k]["actions"], dtype=np.float64)))
            row.update(episode=ep.path.name, replan=k)
            rows.append(row)
            log.info("%s replan %2d: head_dz %7.1f +-%5.1f mm (trace %7.1f, in-spread %s) "
                     "grip_max %.2f", ep.path.name, k, row["head_dz"], row["head_dz_std"],
                     row["trace_head_dz"], row["trace_in_spread"], row["grip_max"])
        out.append({"episode": ep.path.name, "rows": rows})
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
