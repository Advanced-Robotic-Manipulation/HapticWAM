"""[Linux/5090] Deployment entry point: the §6e receding-horizon loop on the
real rig (or fully mocked for a dry run on any machine).

    python -m phantom.scripts.run_deploy --system teacher --ckpt <teacher.pt> \
        --task fragile_grasp [--episodes 3] [--max-replans 20]
    python -m phantom.scripts.run_deploy --system student --ckpt <student.pt> ...
    # dry run without hardware or checkpoint:
    python -m phantom.scripts.run_deploy --system teacher --tiny --task smoke
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data.schema import NormStats
from phantom.deploy.planner import SYSTEM_MODES
from phantom.deploy.runtime import DeploymentRuntime
from phantom.inference.policy import PhantomPolicy
from phantom.train import common as C
from phantom.train.builder import build_model

log = logging.getLogger("run_deploy")


def build_policy(args, hw, paths) -> PhantomPolicy:
    student = args.system != "teacher"
    dtype = torch.bfloat16 if (args.device == "cuda" and not args.tiny) else torch.float32
    # build with the checkpoint's OWN model config (rope time mode, acc
    # feedback mode, ...) — building code defaults silently changes the
    # deployed geometry vs what was trained (v4 audit 2026-08-14)
    mc = payload = None
    if args.ckpt:
        payload = torch.load(str(Path(args.ckpt)), map_location="cpu",
                             weights_only=False)
        saved_mc = payload.get("configs", {}).get("model")
        if isinstance(saved_mc, dict):
            from phantom.config.model import PhantomModelConfig
            mc = PhantomModelConfig.from_dict(saved_mc)
            log.info("model config from checkpoint: rope=%s cond_dropout=%.2f "
                     "acc=%s", mc.rope_time_mode, mc.cond_dropout_p,
                     mc.acc.self_anticipation)
    pm = build_model(hw, paths, student=student, tiny=args.tiny, mc=mc,
                     load_base=not args.tiny, device=args.device, dtype=dtype,
                     inference=True)
    norm = NormStats.identity()
    if args.ckpt:
        payload = C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw,
                                            load_ema=args.ema, payload=payload)
        if payload.get("norm_stats"):
            ns = payload["norm_stats"]
            norm = NormStats(
                mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
                std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})
    from phantom.backbone import loader as bl
    if args.ckpt and not args.tiny:
        # exact: fold LoRA deltas into the base weights (removes the adapter
        # matmuls from every hot linear)
        bl.merge_lora(pm.rf.net)
    if getattr(args, "fp8", False):
        # bench-only escape hatch: torchao 0.18 rowwise FP8 currently produces
        # NaN actions on this stack (measured 2026-08-13) — never expose it as
        # a deploy flag until the parity check in bench_inference passes
        bl.quantize_fp8(pm.rf.net)
    if getattr(args, "flex", False):
        bl.enable_flex_attention(pm.rf.net)
    if getattr(args, "compile", False):
        bl.compile_blocks(pm.rf.net, mode=getattr(args, "compile_mode", "default")
                          or "default")
    return PhantomPolicy(pm, norm, nfe=args.nfe, drop_video=args.drop_video,
                         task_text=(args.text or args.task),
                         persistent_noise=getattr(args, "persistent_noise", False))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEM_MODES, required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--task", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-replans", type=int, default=20)
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--drop-video", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the DiT blocks (adds ~1-2 min warmup)")
    ap.add_argument("--compile-mode", default="default",
                    help="torch.compile mode (e.g. reduce-overhead for cudagraphs)")
    ap.add_argument("--persistent-noise", action="store_true",
                    help="hold the sampling noise fixed within an episode — "
                         "fresh noise per replan re-rolls the plan direction "
                         "(rig: consec-replan cosine 0.16-0.35)")
    ap.add_argument("--flex", action="store_true",
                    help="FlexAttention self-attn (block-sparse structural mask; "
                         "action parity vs SDPA verified to ~6e-4)")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--no-home", dest="home", action="store_false", default=True,
                    help="skip the pre-episode arm homing to the task's demo "
                         "start pose (real drivers home by default)")
    ap.add_argument("--allow-ood-start", action="store_true",
                    help="start the episode even if the live TCP is beyond "
                         "--max-start-sigma from the task's demo start "
                         "distribution (postmortem: 2-6 sigma starts caused "
                         "place-phase behavior in 7/7 episodes)")
    ap.add_argument("--max-start-sigma", type=float, default=2.5,
                    help="refuse episode start beyond this many sigma from the "
                         "task's demo start distribution (default 2.5: the "
                         "postmortem episode-1 pose was 2.65 sigma and failed)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    out_root = Path(args.out) if args.out else \
        paths.episodes_root() / "deploy" / time.strftime("%Y%m%d")

    policy = build_policy(args, hw, paths)
    any_real = hw.mode.drivers == "real" or "real" in hw.mode.overrides.values()
    arm_real = hw.mode.resolve("arm") == "real"
    if getattr(args, "compile", False) or (any_real and not args.tiny):
        # Warmup on the first forward — OUTSIDE the episode. Compile ate 1-2
        # min inside the first episode; even uncompiled, the first replan ran
        # 1.66-1.86 s vs the 1.6 s chunk budget in EVERY postmortem episode
        # (CUDA context + autotune) — so warm up unconditionally on the rig.
        try:
            from phantom.scripts.bench_inference import fake_obs
            log.info("warmup replan...")
            policy.replan(fake_obs(hw, teacher=(args.system == "teacher")),
                          None, np.zeros(6))
            policy.reset_episode()
        except Exception:
            if any_real:
                # a failed warmup on real hardware means either a broken model
                # (better to die here than mid-episode) or a cold first replan
                # that blows the 1.6 s chunk budget — fail closed
                log.exception("warmup replan FAILED on a real-hardware session")
                return 3
            log.exception("warmup failed (mock session — continuing)")
    from phantom.deploy import start_pose as sp
    stats = sp.load_start_stats().get(args.task)
    if arm_real and stats is None:
        # a missing/misspelled task must not silently disable the OOD gate on
        # a real arm (postmortem: 7/7 episodes failed from OOD starts)
        if not args.allow_ood_start:
            log.error("no start-pose stats for task %r (configs/start_poses.yaml"
                      " has: %s). Fix the task name, or pass --allow-ood-start "
                      "to run ungated deliberately.",
                      args.task, sorted(sp.load_start_stats()))
            return 2
        log.warning("task %r has no start stats — homing and the OOD gate are "
                    "DISABLED (--allow-ood-start)", args.task)
    rng = np.random.default_rng()

    def _gate(rt) -> tuple[float, bool]:
        """(worst sigma incl. gripper, gripper settled)."""
        st = rt.rig.arm.get_state()
        gs = rt.rig.gripper.get_state()
        sig, table = sp.start_sigma_report(stats, st.tcp_pose, gs.position)
        print(f"live state vs {args.task} demo start distribution:\n{table}")
        return float(np.max(sig)), float(getattr(gs, "obj", 3.0)) == 3.0

    with DeploymentRuntime(hw, policy, mode=args.system, out_root=out_root) as rt:
        for i in range(args.episodes):
            if arm_real:
                ep_tag = f"episode {i + 1}/{args.episodes}"
                # -- stage 1: home the arm to the task's demo start ---------
                if args.home and stats is not None:
                    input(f"{ep_tag}: clear the arm's path. Enter to MOVE ARM "
                          f"to the {args.task} demo start pose (slow move, "
                          "E-stop in hand)...")
                    try:
                        sp.move_to_start(rt.rig.arm, rt.rig.gripper, hw, stats,
                                         rng=rng)
                    except Exception:
                        log.exception("homing move FAILED (protective stop / "
                                      "Local mode / no move_l?) — jog the arm "
                                      "to the pose below by hand; the gate "
                                      "re-checks before anything runs")
                # -- stage 2: gate loop + operator confirm ------------------
                if stats is not None:
                    while True:
                        worst, settled = _gate(rt)
                        ok = worst <= args.max_start_sigma and settled
                        if ok or args.allow_ood_start:
                            if not ok:
                                log.warning("OOD start ALLOWED by flag: "
                                            "%.1f sigma, settled=%s",
                                            worst, settled)
                            break
                        if not settled:
                            log.warning("gripper not settled (OBJ != 3) — "
                                        "waiting up to 10 s...")
                            if sp.wait_gripper_settled(rt.rig.gripper):
                                continue
                        log.error("START GATE: %.1f sigma from the demo start "
                                  "(max %.1f) or gripper unsettled. Fix it "
                                  "(re-run homing / jog / reactivate gripper), "
                                  "then Enter to re-check. Ctrl-C aborts.",
                                  worst, args.max_start_sigma)
                        input("re-check when ready...")
                    while True:
                        input(f"{ep_tag}: place the object, confirm the scene "
                              "is safe. Enter to START EPISODE...")
                        # re-gate: the arm may have been bumped/jogged while
                        # the operator set the scene
                        worst, settled = _gate(rt)
                        if (worst <= args.max_start_sigma and settled) \
                                or args.allow_ood_start:
                            break
                        log.error("state drifted while setting the scene "
                                  "(%.1f sigma, settled=%s) — fix it (jog / "
                                  "re-settle gripper), then Enter to re-check",
                                  worst, settled)
                else:
                    input(f"{ep_tag}: reset scene, Enter to start...")
                # RealSense auto-exposure needs seconds after the stream opens
                # to settle; the first plans otherwise condition on dark
                # frames. Field-verified 2026-08-11: unsettled AE made the
                # policy flail confidently and slam the table.
                log.info("camera AE settle...")
                time.sleep(5.0)
            res = rt.run_episode(task=args.task, text=args.text,
                                 max_replans=args.max_replans,
                                 policy_name=f"{args.system}")
            log.info("episode %d: %s (replans=%d stop=%s safety_events=%d)",
                     i, res.episode_path, res.n_replans, res.stopped_reason,
                     res.safety_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
