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
        # EMA is the DEPLOY artifact (run_eval builds every campaign policy
        # with ema=True): running the rig on raw last-step weights meant the
        # rig and the eval numbers came from different artifacts of the same
        # checkpoint. Default True here, --no-ema/--raw to opt out.
        want_ema = bool(getattr(args, "ema", True))
        has_ema = bool((payload or {}).get("ema"))
        log.info("checkpoint weights: %s (%s)",
                 "EMA" if (want_ema and has_ema) else "raw (last step)",
                 "--no-ema/--raw" if not want_ema
                 else "EMA requested" if has_ema
                 else "EMA requested but this checkpoint has none")
        payload = C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw,
                                            load_ema=want_ema, payload=payload)
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
                         persistent_noise=getattr(args, "persistent_noise", False),
                         guidance=getattr(args, "guidance", 1.0))


# Episode outcomes that mean the robot's RTDE CONTROL SCRIPT is (probably)
# dead: a UR protective stop kills it outright, and both executor_crash and
# motion_stall are what a rejected servoJ looks like from the outside. The
# socket stays up in all three cases, so nothing else notices — every later
# episode would home-fail, start on a hand-jogged OOD pose and record 1-2
# zero-motion replans (audit 2026-08-26). docs/deployment_runtime.md's
# protective-stop recovery is implemented here.
_CONTROL_DEAD_REASONS = ("protective_stop", "executor_crash", "motion_stall")
_RECONNECT_TRIES = 3
_SCRIPT_START_TIMEOUT_S = 5.0


def recover_control(arm, reason: str) -> bool:
    """Rebuild the RTDE control session between episodes. True if the arm is
    servo-able again.

    Blocks on the operator until the robot itself reports it may be controlled
    (protective stop cleared, mode RUNNING, safety NORMAL), rebuilds the
    control interface, and then VERIFIES the new control script is actually
    playing — isConnected() is only the socket, which stays up while the script
    is dead, so without this check the failure would resurface as a rejected
    servoJ in the middle of the next episode."""
    log.error("previous episode ended with %r — the RTDE control script must be "
              "assumed dead; rebuilding it before the next episode "
              "(deployment_runtime.md: protective-stop recovery)", reason)
    for attempt in range(1, _RECONNECT_TRIES + 1):
        ok, why = arm.is_ready_for_control()
        while not ok:
            log.error("robot is not ready for control: %s", why)
            input("clear the fault on the pendant (Enable robot), then Enter "
                  "to re-check. Ctrl-C aborts...")
            ok, why = arm.is_ready_for_control()
        try:
            arm.reconnect_control()
        except Exception:
            log.exception("reconnect_control FAILED (attempt %d/%d)",
                          attempt, _RECONNECT_TRIES)
            continue
        deadline = time.perf_counter() + _SCRIPT_START_TIMEOUT_S
        while time.perf_counter() < deadline:
            if arm.program_running():
                log.info("RTDE control rebuilt and the control script is "
                         "running — the arm is servo-able again")
                return True
            time.sleep(0.1)
        log.error("control script is NOT running %.0fs after reconnect "
                  "(attempt %d/%d)", _SCRIPT_START_TIMEOUT_S, attempt,
                  _RECONNECT_TRIES)
    return False


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEM_MODES, required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--task", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-replans", type=int, default=20)
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="observation-guidance weight (classifier-free; v4 "
                         "trained with cond-dropout for this). >1 sharpens "
                         "obs->action coupling at ~2x replan latency")
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
    # EMA weights are the deploy artifact and what run_eval loads for every
    # campaign policy — deploying raw last-step weights ran the rig on a
    # DIFFERENT artifact than the eval numbers came from (Codex review
    # 2026-08-27). `--ema` is kept (a no-op now) so the existing GO scripts
    # and docs/inference.md commands still parse.
    ema = ap.add_mutually_exclusive_group()
    ema.add_argument("--ema", dest="ema", action="store_true", default=True,
                     help="use the checkpoint's EMA weights (DEFAULT; kept for "
                          "backward compatibility with existing GO scripts)")
    ema.add_argument("--no-ema", "--raw", dest="ema", action="store_false",
                     help="deploy the raw last-step weights instead of EMA "
                          "(debugging escape hatch — NOT what eval measures)")
    ap.add_argument("--no-home", dest="home", action="store_false", default=True,
                    help="skip the pre-episode arm homing to the task's demo "
                         "start pose (real drivers home by default)")
    ap.add_argument("--allow-ood-start", action="store_true",
                    help="start the episode even if the live TCP is beyond "
                         "--max-start-sigma from the task's demo start "
                         "distribution (postmortem: 2-6 sigma starts caused "
                         "place-phase behavior in 7/7 episodes)")
    ap.add_argument("--seed", type=int, default=None,
                    help="reseed the policy's sampling generator per episode "
                         "(seed + episode index) — reproducible noise draws for "
                         "paired trials")
    ap.add_argument("--no-label-prompt", dest="label_prompt", action="store_false",
                    default=True, help="skip the post-episode success/notes prompt")
    ap.add_argument("--max-start-sigma", type=float, default=2.5,
                    help="refuse episode start beyond this many sigma from the "
                         "task's demo start distribution (default 2.5: the "
                         "postmortem episode-1 pose was 2.65 sigma and failed)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

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
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=Path(__file__).resolve().parents[2],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha = "nogit"
    ckpt_real = str(Path(args.ckpt).resolve()) if args.ckpt else ""
    cond_tags = [f"nfe{policy.nfe}", f"g{policy.guidance}",
                 "pnoise" if args.persistent_noise else "freshnoise",
                 f"ckpt:{Path(ckpt_real).name}", f"git:{sha}",
                 f"seed:{args.seed}" if args.seed is not None else "seed:none"]
    log.info("episode condition tags: %s", cond_tags)

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

    prev_reason: str | None = None
    with DeploymentRuntime(hw, policy, mode=args.system, out_root=out_root) as rt:
        for i in range(args.episodes):
            if arm_real:
                ep_tag = f"episode {i + 1}/{args.episodes}"
                # -- stage 0: restore control after a protective stop -------
                if prev_reason in _CONTROL_DEAD_REASONS:
                    if not recover_control(rt.rig.arm, prev_reason):
                        log.error("RTDE control could NOT be recovered — "
                                  "refusing to start %s. Everything after this "
                                  "would be a zero-motion episode on a "
                                  "hand-jogged start pose. Fix the robot and "
                                  "restart the process.", ep_tag)
                        return 4
                    prev_reason = None
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
            ep_tags = list(cond_tags)
            if args.seed is not None:
                policy.rf._gen = torch.Generator().manual_seed(args.seed + i)
                policy.rf.reset_episode_noise()
                ep_tags[-1] = f"seed:{args.seed + i}"     # the ACTUAL per-episode seed
            res = rt.run_episode(task=args.task, text=args.text,
                                 max_replans=args.max_replans,
                                 policy_name=f"{args.system}", tags=ep_tags)
            log.info("episode %d: %s (replans=%d stop=%s safety_events=%d)",
                     i, res.episode_path, res.n_replans, res.stopped_reason,
                     res.safety_events)
            prev_reason = res.stopped_reason
            if res.fatal_reason:
                # Not recoverable by recover_control(): either a sensor worker
                # process is gone (its ring has no writer) or a stale executor/
                # gripper thread still owns the device. Both need a fresh
                # process; another episode here would silently run on frozen
                # streams or race the old worker on the Robotiq socket. Label
                # this episode first — it is exactly the one worth diagnosing.
                log.error("DEPLOYMENT SESSION OVER (%s) — refusing to start "
                          "another episode in this process. Fix the rig and "
                          "relaunch run_deploy.", res.fatal_reason)
            if arm_real and args.label_prompt and res.episode_path:
                # success is otherwise hardcoded None on every deploy episode
                # (audit 2026-08-20): the session produced unlabeled anecdotes
                ans = input("outcome? [s]uccess / [f]ail / [c]ontaminated / "
                            "Enter=skip, then optional notes: ").strip()
                if ans:
                    code, _, note = ans.partition(" ")
                    c = code[:1].lower()
                    # 'contaminated' (hand in frame, bumped scene, sensor glitch)
                    # is NOT a failure demonstration: success=False would make
                    # is_failure_demo() treat it as a deliberate failure and
                    # train contact/event heads on it. Keep success=None and
                    # tag it so dataset listers can exclude it.
                    succ = {"s": True, "f": False}.get(c)
                    extra = " CONTAMINATED" if c == "c" else ""
                    rt.recorder.relabel(Path(res.episode_path), success=succ,
                                        notes=f"operator: {ans}{extra}")
                    if c == "c":
                        mp = Path(res.episode_path) / "meta.json"
                        import json as _json
                        m = _json.loads(mp.read_text())
                        m["tags"] = list(m.get("tags") or []) + ["contaminated"]
                        mp.write_text(_json.dumps(m, indent=1))
            if res.fatal_reason:
                # returning from inside the `with` runs DeploymentRuntime
                # .__exit__: session.stop() + disconnect_all()
                return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
