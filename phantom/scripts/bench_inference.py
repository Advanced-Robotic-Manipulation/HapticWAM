"""[Linux/5090] Offline inference-latency benchmark for the deploy replan path.

Times PhantomPolicy.replan() end to end on synthetic observations (no rig, no
sensors) with a CUDA-event breakdown of the two GPU consumers: the Wan VAE
encodes and the per-NFE-step DiT forwards. This is the harness for judging
every deploy-latency lever (nfe, --drop-video, --compile, batching changes)
against the smoke-run baseline of ~1.05 s/replan (logs_deploy_smoke.log).

    python -m phantom.scripts.bench_inference \
        --ckpt runs/teacher_v3_790eps/teacher_020000.pt [--nfe 5] \
        [--drop-video] [--compile] [--iters 10]
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.inference.policy import ObsSnapshot
from phantom.scripts.run_deploy import build_policy

log = logging.getLogger("bench_inference")


def fake_obs(hw, teacher: bool) -> ObsSnapshot:
    rng = np.random.default_rng(0)
    snap = ObsSnapshot(
        t=0.0,
        rgb=rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        wrist_window=rng.normal(size=(hw.wrist_ft.window_len, 6))
        .astype(np.float32),
        ur_state=rng.normal(size=(hw.ur_state_dim,)).astype(np.float32),
        reactive=0.1,
    )
    if teacher:
        fn = hw.n_fingers
        kd = hw.recording.keyframe_ds
        snap.gel = rng.integers(0, 255, (fn, 480, 640), dtype=np.uint8)
        snap.fields = rng.normal(size=(fn, kd.h, kd.w, hw.tactile.field_ch)) \
            .astype(np.float32)
        snap.contact_state = rng.normal(size=(fn, hw.contact_state_dim)) \
            .astype(np.float32)
    return snap


class CudaEventTimer:
    """Wraps a callable; accumulates GPU time via CUDA events per outer call."""

    def __init__(self, fn):
        self.fn = fn
        self.events: list[tuple] = []

    def __call__(self, *a, **k):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = self.fn(*a, **k)
        e.record()
        self.events.append((s, e))
        return out

    def drain_ms(self) -> tuple[float, int]:
        torch.cuda.synchronize()
        total = sum(s.elapsed_time(e) for s, e in self.events)
        n = len(self.events)
        self.events.clear()
        return total, n


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--system", default="teacher")
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--drop-video", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--flex", action="store_true")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--dump-actions", default="",
                    help="save the final replan's denormalized actions (.npy) "
                         "for cross-config parity checks")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--k-seeds", type=int, default=1,
                    help="K-seed batched sampling (run_deploy --k-seeds): profile the "
                         "replan latency cost on the DEPLOY GPU before booking rig time")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="reseed the sampling noise before EVERY replan (parity across levers)")
    ap.add_argument("--persistent-noise", action="store_true", default=True)
    ap.add_argument("--out", default="", help="write the summary (medians, config) as JSON")
    ap.add_argument("--dump", default="", help="npz with per-iteration actions and p_evt (parity input)")
    ap.add_argument("--parity-against", default="",
                    help="npz from a previous --dump run with the same --seed: report max |delta|")
    args = ap.parse_args(argv)
    args.device = "cuda"
    args.tiny = False
    args.task = "bench"
    args.text = ""

    assert torch.cuda.is_available(), "this benchmark is GPU-only"
    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=True)

    do_compile, args.compile = args.compile, False   # compile here, with our mode
    policy = build_policy(args, hw, paths)
    rf = policy.rf

    if do_compile:
        from phantom.backbone.loader import compile_blocks
        compile_blocks(rf.net, mode=args.compile_mode)

    net_timer = CudaEventTimer(rf.net.forward)
    rf.net.forward = net_timer
    vae_timer = CudaEventTimer(rf.vae.encode)
    rf.vae.encode = vae_timer

    obs = fake_obs(hw, teacher=(args.system == "teacher"))
    tcp_pose = np.zeros(6, dtype=np.float64)

    prev_plan = None
    rows = []
    dumped_actions, dumped_p_evt = [], []
    for i in range(args.warmup + args.iters):
        if args.seed is not None:
            rf._gen.manual_seed(int(args.seed) + i)
            torch.manual_seed(int(args.seed) + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        plan = policy.replan(obs, prev_plan, tcp_pose)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        net_ms, net_n = net_timer.drain_ms()
        vae_ms, vae_n = vae_timer.drain_ms()
        prev_plan = plan
        if i >= args.warmup:
            rows.append((wall, net_ms, vae_ms))
            dumped_actions.append(np.asarray(plan.actions, dtype=np.float32))
            dumped_p_evt.append(np.asarray(getattr(plan, "p_evt", np.zeros(5)), dtype=np.float32))
        log.info("replan %d%s: wall=%.0f ms | net=%.0f ms (%d calls) | "
                 "vae=%.0f ms (%d calls) | other=%.0f ms",
                 i, " (warmup)" if i < args.warmup else "", wall * 1e3,
                 net_ms, net_n, vae_ms, vae_n, wall * 1e3 - net_ms - vae_ms)

    walls = [r[0] * 1e3 for r in rows]
    nets = [r[1] for r in rows]
    vaes = [r[2] for r in rows]
    nfe = policy.nfe
    log.info("=" * 72)
    log.info("config: nfe=%d drop_video=%s compile=%s iters=%d",
             nfe, policy.drop_video, do_compile, args.iters)
    log.info("replan wall:  median %.0f ms  (min %.0f / max %.0f)",
             statistics.median(walls), min(walls), max(walls))
    log.info("DiT forwards: median %.0f ms total -> %.1f ms/step at nfe=%d",
             statistics.median(nets), statistics.median(nets) / nfe, nfe)
    log.info("VAE encodes:  median %.0f ms", statistics.median(vaes))
    log.info("other (CPU/pack/norm): median %.0f ms",
             statistics.median([w - n - v for w, n, v in
                                zip(walls, nets, vaes)]))
    if args.dump_actions:
        np.save(args.dump_actions, prev_plan.actions)
        log.info("actions of the last replan saved to %s", args.dump_actions)
    summary = {"nfe": nfe, "k_seeds": policy.k_seeds, "drop_video": bool(policy.drop_video),
               "compile": bool(do_compile), "compile_mode": args.compile_mode if do_compile else None,
               "flex": bool(args.flex), "fp8": bool(args.fp8), "iters": args.iters, "seed": args.seed,
               "wall_ms_median": statistics.median(walls), "wall_ms_min": min(walls), "wall_ms_max": max(walls),
               "net_ms_median": statistics.median(nets), "vae_ms_median": statistics.median(vaes),
               "other_ms_median": statistics.median([w - n - v for w, n, v in zip(walls, nets, vaes)]),
               "ckpt": args.ckpt, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    if args.dump:
        np.savez(args.dump, actions=np.stack(dumped_actions), p_evt=np.stack(dumped_p_evt))
        log.info("per-iteration actions/p_evt saved to %s", args.dump)
    if args.parity_against:
        ref = np.load(args.parity_against)
        a, b = ref["actions"], np.stack(dumped_actions)
        n = min(len(a), len(b))
        da = float(np.abs(a[:n] - b[:n]).max()); dp = float(np.abs(ref["p_evt"][:n] - np.stack(dumped_p_evt)[:n]).max())
        nan = bool(np.isnan(b).any())
        summary.update(parity_against=args.parity_against, parity_max_abs_action_delta=da,
                       parity_max_abs_p_evt_delta=dp, parity_nan=nan)
        log.info("PARITY vs %s: max |delta action| = %.3g, max |delta p_evt| = %.3g, nan=%s",
                 args.parity_against, da, dp, nan)
    if args.out:
        import json
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        log.info("summary written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
