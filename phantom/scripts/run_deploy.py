"""[Linux/5090] Deployment entry point: the §6e receding-horizon loop on the
real rig (or fully mocked for a dry run on any machine).

    python -m phantom.scripts.run_deploy --system teacher --ckpt <teacher.pt> \
        --task fragile_grasp [--episodes 3] [--max-replans 20]
    python -m phantom.scripts.run_deploy --system student --ckpt <student.pt> ...
    # dry run without hardware or checkpoint:
    python -m phantom.scripts.run_deploy --system teacher --tiny --task smoke

Deploy levers from the 2026-08-28 review (all default OFF, all tagged into the
episode's condition tags so an A/B arm is reconstructable from the recording):

    # arm A - baseline, exactly what GO_ANY.sh runs today
    ... --task waffles
    # arm B - the four levers, one at a time or bundled
    ... --task waffles --parity-fixes --terminal-veto --k-seeds 4 --nfe 3 --compile

`--nfe 3` and `--compile` are the LATENCY levers (P4: latency == the replan
interval, so only steps 0-9 of each 16-step chunk ever run). `--k-seeds` pushes
the other way and must be profiled on the deploy GPU first — see its help.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import select
import sys
import time
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data.schema import NormStats
from phantom.deploy.planner import SYSTEM_MODES, WRIST_MASKED_MODES
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
        if getattr(args, "terminal_veto", False):
            # BEFORE the model is built: an ACC-less checkpoint makes the veto
            # a silent no-op, and finding that out after a 2-minute build (on
            # the rig, with the operator waiting) is worse than useless
            assert_acc_head(payload, str(args.ckpt))
    if args.system in WRIST_MASKED_MODES:
        # the sensor-free comparative arms (P10A): zero the wrist window
        # inside the model too, not only in the snapshot, so what the model
        # sees is exactly what a mask_wrist-trained arm saw. mask_wrist is an
        # input ablation applied ON TOP of trained weights, which is why
        # load_phantom_checkpoint's config check ignores it.
        from phantom.config.model import PhantomModelConfig
        mc = (dataclasses.replace(mc, mask_wrist=True) if mc is not None
              else PhantomModelConfig(student=student, mask_wrist=True))
        log.info("system %s: wrist F/T window MASKED (recorded, not read)",
                 args.system)
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
                         guidance=getattr(args, "guidance", 1.0),
                         parity_fixes=getattr(args, "parity_fixes", False),
                         k_seeds=getattr(args, "k_seeds", 1),
                         close_p=getattr(args, "veto_p_close", 0.5))


# Episode outcomes that mean the robot's RTDE CONTROL SCRIPT is (probably)
# dead: a UR protective stop kills it outright, and both executor_crash and
# motion_stall are what a rejected servoJ looks like from the outside. The
# socket stays up in all three cases, so nothing else notices — every later
# episode would home-fail, start on a hand-jogged OOD pose and record 1-2
# zero-motion replans (audit 2026-08-26). docs/deployment_runtime.md's
# protective-stop recovery is implemented here.
# `safety_stop` joins them (audit 2026-08-27): the STOP_EPISODE path calls
# URArm.stop(), whose stopL failure is SWALLOWED as a log line, and one of the
# conditions that raises it is `arm_stale` — a dead RTDE stream. Verifying the
# control script before the next episode costs one is_ready_for_control() call
# on a healthy robot and nothing else.
# `servo_stop_failed` is the executor telling us servoStop() failed against a
# script that is still playing: `URArm._servo_active` is stuck True, so the
# next homing move_l would be refused outright. reconnect_control() replaces
# the script and clears the guard (ur.py invariant), which is exactly the fix.
# `worker_died` is here for completeness of the predicate — a dead sensor
# worker is ALSO session-fatal, so main() returns 5 before it could ever reuse
# the arm; any other caller asking "is the control script suspect?" should
# still get True.
_CONTROL_DEAD_REASONS = ("protective_stop", "executor_crash", "motion_stall",
                         "servo_stop_failed", "safety_stop", "worker_died")
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


#: the count cap that stands in for a time budget when no wall clock is set.
#: `--max-replans` parses to None so `main` can tell "the operator chose 40"
#: from "nobody said anything" (revalidation 2026-08-31 #1).
DEFAULT_MAX_REPLANS = 40


def resolve_max_replans(args) -> int | None:
    """The replan cap the episode actually runs with.

    `PlannerLoop.run` checks the replan COUNT before the wall clock, so a
    default of 40 makes `--max-episode-s` dead at any latency below
    `budget / 40`: the documented arm-B line (`--nfe 1`, 172 ms replans,
    `--max-episode-s 35`) stopped at `replan_cap` after 7.2 s against 16-31 s
    demos. When a wall-clock budget is in force and the operator did not name a
    count, LIFT the count entirely and let the clock govern. An explicit
    `--max-replans` always wins — including alongside `--max-episode-s`."""
    if args.max_replans is not None:
        return int(args.max_replans)
    if float(args.max_episode_s) > 0:
        return None
    return DEFAULT_MAX_REPLANS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEM_MODES, required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--task", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-replans", type=int, default=None,
                    help=f"episode cap in replans (~0.9 s each; default "
                         f"{DEFAULT_MAX_REPLANS}). Demos take 16-31 s and a "
                         f"rollout that retries needs room: 20 cut every 08-28 "
                         f"retry short. LEFT UNSET the count cap is LIFTED "
                         f"whenever --max-episode-s > 0, so the wall clock is "
                         f"the only budget; pass it explicitly to bind it")
    ap.add_argument("--max-episode-s", type=float, default=35.0,
                    help="episode WALL-CLOCK cap (s). The replan cap is not a "
                         "time budget: at --nfe 1 (172 ms replans) 40 replans "
                         "is a ~7 s episode against 16-31 s demos, and "
                         "PlannerLoop checks the COUNT first — so unless "
                         "--max-replans is passed explicitly this budget "
                         "replaces it. 0 or negative disables it")
    ap.add_argument("--nfe", type=int, default=None,
                    help="Euler steps per replan (default: the checkpoint's mc.nfe, 5). "
                         "LATENCY LEVER (review P4): the loop is compute-bound and "
                         "latency == replan interval == 0.97 s on 08-28, so only steps "
                         "0-9 of every 16-step chunk ever execute. --nfe 3 is the "
                         "first thing to try for L <= 0.5 s; --nfe 1 emits "
                         "x0 = eps - v(eps, t=1) = E[x0|obs] (deterministic, no "
                         "persistent-noise velocity offset, ~5x cheaper). Both are "
                         "unused by GO_ANY.sh — benchmark them with "
                         "phantom.scripts.bench_inference before a session")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="observation-guidance weight (classifier-free; v4 "
                         "trained with cond-dropout for this). >1 sharpens "
                         "obs->action coupling at ~2x replan latency")
    ap.add_argument("--drop-video", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the DiT blocks (adds ~1-2 min warmup, taken "
                         "OUTSIDE the episode by the unconditional warmup replan). "
                         "LATENCY LEVER (review P4), also unused by GO_ANY.sh: pair it "
                         "with --nfe 3 and, on a card with the kernels, --flex. "
                         "--compile-mode reduce-overhead adds cudagraphs. Status: the "
                         "wiring is intact (backbone/loader.compile_blocks wraps each "
                         "DiT block with dynamic=False; bench_inference exposes the "
                         "same path), but NO test covers it and it has never been "
                         "timed on the rig NUC — measure it with bench_inference "
                         "--compile before trusting it")
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
    ap.add_argument("--max-play-steps", type=int, default=10,
                    help="cap chunk playback at this many action steps and "
                         "hold until the next plan (steps beyond HEAD_STEPS=9 "
                         "are never validated by a replan; all four 09-01 "
                         "whips began in that tail). 0 = uncapped.")
    ap.add_argument("--lift-complete-z", type=float, default=0.0,
                    help="success auto-stop: tactile-confirmed grasp held "
                         "above this TCP z (m) ends the episode cleanly, "
                         "gripper held. 0 (default since 09-04) disables: "
                         "waffles demos carry the object ~8 s past the lift "
                         "apex, so the 0.32 m rule cut every good grasp "
                         "before transport; the operator ends episodes with "
                         "Enter. Pass 0.32 to restore the 09-01 behaviour.")
    ap.add_argument("--lift-complete-fz", type=float, default=2.5,
                    help="per-pad |fz| (N) that counts as a confirmed grasp "
                         "for the success auto-stop")
    ap.add_argument("--lift-complete-hold", type=float, default=0.4,
                    help="seconds the grasp+height condition must hold")
    ap.add_argument("--policy-server", default=None, metavar="ADDR",
                    help="attach to a resident policy server instead of "
                         "loading the model here ('auto' = try 127.0.0.1:7777 "
                         "and fall back to a local load; 'host:port' = "
                         "require it). See phantom.scripts.policy_server.")
    ap.add_argument("--home-joints", action="store_true",
                    help="before the slow moveL homing, moveJ to the task's demo JOINT "
                         "configuration (start_poses.yaml q_mean). Fixes a wrapped wrist / "
                         "flipped IK branch that the TCP gate cannot see. Path must be clear.")
    ap.add_argument("--z-floor", type=float, default=None,
                    help="no-go floor for the commanded TCP z (m). Default: the task's demo "
                         "tcp_z_min - --z-floor-margin from start_poses.yaml")
    ap.add_argument("--z-floor-margin", type=float, default=0.01)
    ap.add_argument("--hitbox-margin", type=float, default=0.03,
                    help="STOP hitbox = task demo TCP envelope (start_poses.yaml tcp_min/max) "
                         "+/- this margin (m); a commanded target outside ends the episode")
    ap.add_argument("--no-hitbox", action="store_true")
    ap.add_argument("--max-tcp-speed", type=float, default=None,
                    help="lower the executor's commanded TCP speed cap (m/s) below "
                         "hardware.yaml arm.limits.tcp_speed_m_s for this run")
    ap.add_argument("--no-z-floor", action="store_true",
                    help="run with only the hardware.yaml workspace box (not on a real arm "
                         "unless you mean it)")
    ap.add_argument("--max-start-sigma", type=float, default=2.5,
                    help="refuse episode start beyond this many sigma from the "
                         "task's demo start distribution (default 2.5: the "
                         "postmortem episode-1 pose was 2.65 sigma and failed)")
    # ---- deploy levers from the 2026-08-28 review. ALL DEFAULT OFF: the rig
    # A/B can only attribute an effect if each one is switched independently,
    # and every one of them writes its state into the episode condition tags.
    ap.add_argument("--parity-fixes", action="store_true",
                    help="P2: build prev_chunk from the MEASURED arm history + the "
                         "gripper commands actually sent (what WindowSampler feeds in "
                         "training) instead of the policy's own last proposal; take the "
                         "contact-state dt from the tactile ring timestamps; compute "
                         "`reactive` from the two consecutive fields_ds frames rather "
                         "than the last two replans; and summarise the ACC prev_cpk at "
                         "step round(latency/latent_dt) so it lines up with training's "
                         "two-pass anchor")
    ap.add_argument("--terminal-veto", action="store_true",
                    help="P3: scripted terminal commitment guard. Masks a commanded "
                         "gripper close unless p_contact > --veto-p-close or the TCP is "
                         "within 15 mm of the task z floor, and on a phantom grasp "
                         "(p_evt[none] > --veto-p-none one replan after a close) opens "
                         "to the demo start aperture and forbids any lift so the policy "
                         "re-descends. Logged per replan under `terminal_veto`")
    ap.add_argument("--veto-p-close", type=float, default=0.5,
                    help="theta_close on p_contact = 1 - p_evt[none] (default 0.5; "
                         "calibrate on the 08-20 gate trace). Also the threshold the "
                         "K-seed head-descent rejection uses")
    ap.add_argument("--veto-p-none", type=float, default=0.9)
    ap.add_argument("--veto-z-margin", type=float, default=None,
                    help="height (m) of the terminal veto's 'a demo would close "
                         "here' band above the task's demo tcp_z_min. Default: "
                         "sized so the band top is the task's demo close p95 + "
                         "15 mm (eval/grasp_label.Z_MAX_MM)")
    ap.add_argument("--veto-max-retries", type=int, default=3,
                    help="open/re-descend cycles allowed per episode before the episode "
                         "ends with reason `veto_retry_cap`")
    ap.add_argument("--k-seeds", type=int, default=1,
                    help="P6: sample K chunks per replan in ONE batched denoise and "
                         "select (reject a head descent < 50%% of the K-max while "
                         "p_contact is low, then take the chunk nearest the previous "
                         "accepted plan). PROFILE FIRST: cost scales ~linearly in K on "
                         "a 2B DiT, which fights the P4 latency target")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def episode_seed(base_seed, episode_index: int) -> int:
    """The sampling-noise seed for episode i: base + i when --seed was given
    (reproducible A/B), otherwise a fresh random draw — never the generator's
    constructor default, which made every rig session sample the same noise."""
    if base_seed is not None:
        return int(base_seed) + int(episode_index)
    import os
    return int.from_bytes(os.urandom(4), "little")


def resolve_z_floor(args, stats) -> float | None:
    """The z no-go floor (m) for this run, or None when disabled/unknown."""
    if getattr(args, "no_z_floor", False):
        return None
    if getattr(args, "z_floor", None) is not None:
        return float(args.z_floor)
    zmin = getattr(stats, "tcp_z_min", None) if stats is not None else None
    if zmin is None:
        return None
    return float(zmin) - float(getattr(args, "z_floor_margin", 0.01))


def veto_z_margin(args, stats) -> float:
    """Height of the close-mask's "a demo would close here" band above the
    task's demo `tcp_z_min`, in metres.

    Default: the band top IS the task's demo close ceiling, `Z_MAX_MM` =
    demo z_close p95 + 15 mm (phantom/eval/grasp_label.py) — the same table
    tools/label_grasps.py judges the session by. Measured from the already
    LOWERED safety floor with a hard-coded 15 mm the hatch topped out at
    57-81 mm, i.e. 30-90 mm below every demo close and ~50 mm below the
    model's own predicted close height (110.8 mm, E9), so it never fired and
    every close hung on an uncalibrated `p_contact` (VALIDATION_0830 P0 #5)."""
    if getattr(args, "veto_z_margin", None) is not None:
        return float(args.veto_z_margin)
    from phantom.eval.grasp_label import Z_MAX_MM
    z_max = Z_MAX_MM.get(str(args.task).removesuffix("_fail"))
    z_ref = getattr(stats, "tcp_z_min", None) if stats is not None else None
    if z_max is None or z_ref is None:
        log.warning("no demo close ceiling for task %r (Z_MAX_MM=%s, "
                    "tcp_z_min=%s) — the veto's at-floor band falls back to "
                    "15 mm", args.task, z_max, z_ref)
        return 0.015
    return max(0.015, float(z_max) / 1000.0 - float(z_ref))


def build_veto(args, stats, z_floor: float | None):
    """TerminalVeto for this run, or None when --terminal-veto is absent.

    The close-mask's "already low enough to close" band is measured from the
    task's demo `tcp_z_min` (NOT from the lowered safety floor the executor
    clamps to) and reaches the demo close p95; the open aperture is the task's
    demo START aperture (start_poses.yaml gripper_mean) — the aperture the
    demos approach with, and therefore the in-distribution thing to reopen
    to."""
    if not getattr(args, "terminal_veto", False):
        return None
    from phantom.deploy.planner import TerminalVeto
    from phantom.train.common import CLOSE_ABS_POS, CLOSE_ABS_RISE
    open_ap = float(getattr(stats, "gripper_mean", 0.0) or 0.0)
    if stats is None:
        log.warning("--terminal-veto with no start-pose stats for %r: the "
                    "recovery rule will reopen to 0.0 (fully open) and the "
                    "close-mask has no z band", args.task)
    z_ref = getattr(stats, "tcp_z_min", None) if stats is not None else None
    return TerminalVeto(
        p_close=float(args.veto_p_close), p_none=float(args.veto_p_none),
        max_retries=int(args.veto_max_retries), z_floor=z_floor,
        z_ref=None if z_ref is None else float(z_ref),
        z_margin=veto_z_margin(args, stats),
        open_aperture=open_ap, close_pos=CLOSE_ABS_POS, close_rise=CLOSE_ABS_RISE)


def envelope_conflict(hw) -> str | None:
    """The floor-is-a-CLAMP fix holds only while the STOP hitbox's lower z edge
    sits at or below the clamped workspace floor.

    `--hitbox-margin 5 --z-floor-margin 10` silently restores the 2026-08-28
    bug: the executor clamps a deep target ONTO the floor and the hitbox then
    calls that same target an exit, so every deep descent ends the episode
    (`STOP ['workspace_clamp','hitbox_exit']`). Both knobs are advertised in
    docs/rig_session_v5.md, and nothing asserted the invariant
    (validation_0830/safety-final.md §5)."""
    hb = hw.safety.hitbox_m
    if hb is None:
        return None
    floor = float(hw.safety.workspace_m.z[0])
    if float(hb.z[0]) <= floor + 1e-9:
        return None
    return (f"UNSAFE ENVELOPE: the STOP hitbox floor ({hb.z[0] * 1000:.0f} mm) "
            f"is ABOVE the z clamp floor ({floor * 1000:.0f} mm), so every "
            f"target the clamp pins on the floor is a hitbox_exit and ends the "
            f"episode. Raise --hitbox-margin above --z-floor-margin (or lower "
            f"--z-floor-margin).")


def has_acc_head(payload: dict) -> bool:
    """True when the checkpoint carries ACC-head weights."""
    for key in ("phantom_modules", "ema", "model", "state_dict"):
        sd = (payload or {}).get(key)
        if isinstance(sd, dict) and any("phantom_acc" in str(k) for k in sd):
            return True
    return False


def assert_acc_head(payload: dict, ckpt: str) -> None:
    """The terminal veto reads `p_evt`. With no ACC head the model returns
    `p_evt = zeros(5)` -> `p_none = 0` -> `p_contact = 1`, so the close mask
    and the K-seed rejection silently become no-ops that log `close_allowed`:
    an arm labelled `veto:on` that measures nothing (validation §9)."""
    if not has_acc_head(payload):
        raise RuntimeError(
            f"--terminal-veto with a checkpoint that has NO ACC head ({ckpt}): "
            "p_evt would be zeros(5), p_contact 1.0, and every close would be "
            "allowed while the trace logged `close_allowed`. Refusing — deploy "
            "an ACC checkpoint or drop --terminal-veto.")



def make_lift_complete(rt, z_m: float, fz_n: float, hold_s: float):
    """Success auto-stop (rig 2026-09-01): both tactile pads loaded AND the
    TCP above the lift height, sustained -> the episode ends cleanly holding
    the object. Demos end right after the lift; the "carry" beyond it is
    out-of-distribution and twice ran the arm into the full-extension
    singularity. Returns None (disabled) when thresholds are non-positive.

    |fz| is BASELINE-CORRECTED per episode: the left pad carried a 0.7-2.2 N
    zero offset that drifted through the 09-01 session (right 0.01-0.05 N),
    so a raw absolute threshold partly measures bias. The first call (arm
    still at the start pose, pads untouched) captures the baseline.
    Thresholds from the 22-episode evaluation: 2.5 N / 0.4 s / z 0.32 m fires
    in all 4 real lifted grasps, 1.8-2.7 s ahead of every whip it can see,
    and in 0 of the other 18 episodes."""
    if z_m <= 0 or fz_n <= 0:
        return None
    state = {"since": None, "base": None}

    def check() -> bool:
        try:
            _, arm = rt.session.rings["arm"].latest(1)
            z = float(np.asarray(arm["tcp_pose"][0]).reshape(-1)[2])
            raw = []
            for side in ("left", "right"):
                _, tac = rt.session.rings[f"tactile_{side}"].latest(1)
                raw.append(float(np.asarray(tac["wrench"][0]).reshape(-1)[2]))
        except Exception:
            return False
        if state["base"] is None:
            state["base"] = raw                # first replan: pads untouched
            return False
        fzs = [abs(r - b) for r, b in zip(raw, state["base"])]
        good = z > z_m and all(f > fz_n for f in fzs)
        now = time.perf_counter()
        if good:
            if state["since"] is None:
                state["since"] = now
            return (now - state["since"]) >= hold_s
        state["since"] = None
        return False

    return check


def persist_stop(recorder, res) -> None:
    """Write WHY the episode ended next to its data (09-04): stop.json with
    the reason, every safety event (kind/value/ticks/time) and the arm state at
    the stop, plus `stop:<reason>` / `evt:<kind>` tags in meta.json. Until now
    the reason only went to the terminal, so a stop could not be diagnosed
    from disk. Best effort — never fails the session."""
    if not res.episode_path:
        return
    try:
        import json as _json
        ep = Path(res.episode_path)
        doc = {"stopped_reason": res.stopped_reason, "n_replans": res.n_replans,
               "safety_events": res.events, "stop_state": res.stop_state,
               "fatal_reason": res.fatal_reason}
        (ep / "stop.json").write_text(_json.dumps(doc, indent=1), encoding="utf-8")
        tags = [f"stop:{res.stopped_reason}"]
        tags += ["evt:" + str(e["kind"]) for e in res.events]
        recorder.relabel(ep, tags=tags)
    except Exception:
        log.exception("could not persist the stop reason for %s", res.episode_path)


def make_operator_stop():
    """Episode-scoped stop button: pressing Enter DURING the episode ends it
    cleanly (`planner.run` breaks with stop_reason "operator_stop"; the
    gripper is not touched, so a held object is not dropped). Any Enter
    presses still buffered from the start prompts are drained on arming so a
    double-tap cannot instantly end the episode. Polling stdin is a zero-
    timeout select in the planner's own loop — no thread, no race with the
    outcome prompt that follows the episode."""
    if not sys.stdin.isatty():        # piped/test runs: no stop button
        return None
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()          # drain buffered lines
    fired = {"v": False}
    def check() -> bool:
        if fired["v"]:
            return True
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            fired["v"] = True
            log.info("operator stop requested — ending the episode cleanly")
            return True
        return False
    return check

VERDICT_PROMPT = ("outcome? [s]uccess / [f]ail / [c]ontaminated "
                  "(append d for DAMAGE, e.g. 'fd') / Enter=skip, "
                  "then optional notes "
                  # a let-go stop releases the fingers itself (executor._halt);
                  # this is the manual path if it could not reach the gripper
                  "[gripper still closed? run ./GRIPPER_OPEN.sh]: ")


def parse_verdict(ans: str) -> tuple[str, bool]:
    """Split an operator answer into (verdict code, damage flag).

    The code is the FIRST whitespace-separated token; everything after it is
    free-text notes. A single trailing 'd' on that token is the damage flag
    (`sd`, `fd`, `cd`) — the eval protocol reports damage as its own rate, so
    a destructive success has to stay distinguishable from a clean one, and a
    second y/N prompt is one more thing to answer per episode on a rig where
    the operator's hands are on the E-stop.

    Deliberately lenient in the same way the original was: only the token's
    FIRST character selects the verdict ('success!' -> 's'), so a two-char
    token is the only thing read as a damage flag ('success' does not set it).
    Returns ('', False) for anything unrecognized — the caller must not
    promote the episode."""
    ans = (ans or "").strip()
    if not ans:
        return "", False
    code, _, _note = ans.partition(" ")
    c = code[:1].lower()
    if c not in ("s", "f", "c"):
        return "", False
    damage = len(code) == 2 and code[1].lower() == "d"
    return c, damage


def label_episode(recorder, ep_path: Path, ans: str) -> str:
    """Apply the operator's post-episode verdict. Returns the verdict code
    ('s'/'f'/'c'/'' for skipped).

    The episode arrives here already filed as status='aborted' + tag
    'unlabeled' by DeploymentRuntime.run_episode (P9), so this function's job
    is to PROMOTE it:
      s/f  -> success True/False, status='finalized', 'unlabeled' cleared;
      c    -> 'contaminated' (hand in frame, bumped scene, sensor glitch). NOT
              a failure demonstration: success=False would make
              is_failure_demo() treat it as a deliberate failure and train the
              contact/event heads on it. success stays None and the tag keeps
              it out of every training index (is_trainable_episode);
      ''   -> nothing at all: the episode STAYS aborted + unlabeled. An
              operator who hits Enter has not accepted the take.

    A trailing 'd' on the verdict token (`sd`/`fd`/`cd`) additionally records
    meta.damage=True + the training-inert `damaged` tag — the same field
    eval/trial_runner's "Damage/breakage?" prompt writes, so a campaign and a
    plain run produce comparable episodes.
    """
    ans = (ans or "").strip()
    c, damage = parse_verdict(ans)
    if not c:
        log.warning("no usable operator verdict for %s (%s) — left "
                    "status='aborted' + tag 'unlabeled', excluded from "
                    "training", ep_path.name,
                    "skipped" if not ans else f"unrecognized {ans!r}")
        return ""
    note = f"operator: {ans}" + (" DAMAGE" if damage else "")
    if c == "c":
        recorder.relabel(ep_path, success=None, status="aborted",
                         tags=["contaminated"], remove_tags=["unlabeled"],
                         damage=damage, notes=f"{note} CONTAMINATED")
        return c
    recorder.relabel(ep_path, success=(c == "s"), status="finalized",
                     remove_tags=["unlabeled"], damage=damage, notes=note)
    return c


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    explicit_max_replans = args.max_replans is not None
    args.max_replans = resolve_max_replans(args)
    if args.max_replans is None:
        log.info("replan cap LIFTED: --max-episode-s %.1f s is the episode "
                 "budget (pass --max-replans to bind a count as well)",
                 float(args.max_episode_s))
    elif explicit_max_replans and float(args.max_episode_s) > 0:
        log.info("episode budget: %d replans OR %.1f s, whichever comes first "
                 "(the count is checked first)", args.max_replans,
                 float(args.max_episode_s))

    hw = load_hardware(args.hardware)
    # The per-task safety overrides below are model_copy()s, so each one changes
    # hw.config_hash(). Episodes are stamped with the BASE hash (the config as
    # loaded from yaml — the one the demos were recorded under), otherwise every
    # rollout trips train_teacher's CONFIG DRIFT gate and the whole DAgger set
    # is unusable. The overrides are recorded separately, under
    # meta.deploy_overrides, so the envelope stays auditable.
    base_hw = hw
    deploy_overrides: dict = {}
    from phantom.deploy import start_pose as sp
    stats = sp.load_start_stats().get(args.task)
    hb_lo, hb_hi = getattr(stats, "tcp_min", None), getattr(stats, "tcp_max", None)
    hitbox_on = not args.no_hitbox and hb_lo is not None and hb_hi is not None
    if hitbox_on:
        from phantom.deploy.safety import apply_hitbox
        hw = apply_hitbox(hw, hb_lo, hb_hi, args.hitbox_margin)
        hb = hw.safety.hitbox_m
        deploy_overrides["hitbox_m"] = {"x": list(hb.x), "y": list(hb.y),
                                        "z": list(hb.z),
                                        "margin_m": float(args.hitbox_margin)}
        log.info("STOP hitbox (demo envelope +/- %.0f mm): x %s y %s z %s mm", args.hitbox_margin * 1000,
                 [round(v * 1000) for v in hb.x], [round(v * 1000) for v in hb.y], [round(v * 1000) for v in hb.z])
    elif hw.mode.resolve("arm") == "real":
        log.warning("NO task hitbox (%s) — a lost policy can wander anywhere inside the workspace box",
                    "--no-hitbox" if args.no_hitbox else "no tcp_min/tcp_max in start_poses.yaml")
    # per-task no-go floor: the executor clamps every commanded target to the
    # workspace box, so raising its z lower bound to (demo z_min - margin)
    # stops a runaway descent where the demos never went (rig 2026-08-28:
    # whiteboard rollout 40 mm below any demo, waffles pack slammed)
    z_floor = resolve_z_floor(args, stats)
    if z_floor is not None:
        from phantom.deploy.safety import apply_z_floor
        hw = apply_z_floor(hw, z_floor)
        deploy_overrides["z_floor_m"] = float(hw.safety.workspace_m.z[0])
        log.info("z no-go floor: commanded TCP z clamped to >= %.0f mm (workspace z now %s m)",
                 hw.safety.workspace_m.z[0] * 1000, hw.safety.workspace_m.z)
    elif hw.mode.resolve("arm") == "real":
        log.warning("NO task z floor (%s) — only the hardware.yaml workspace box (z >= %.0f mm) "
                    "protects the table", "--no-z-floor" if args.no_z_floor else "no tcp_z_min in start_poses.yaml",
                    hw.safety.workspace_m.z[0] * 1000)
    conflict = envelope_conflict(hw)
    if conflict is not None:
        log.error("%s", conflict)
        return 2
    if args.max_tcp_speed is not None:
        from phantom.deploy.safety import apply_tcp_speed_limit
        hw = apply_tcp_speed_limit(hw, args.max_tcp_speed)
        deploy_overrides["tcp_speed_m_s"] = float(hw.arm.limits.tcp_speed_m_s)
    log.info("executor TCP speed cap: %.2f m/s", hw.arm.limits.tcp_speed_m_s)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    out_root = Path(args.out) if args.out else \
        paths.episodes_root() / "deploy" / time.strftime("%Y%m%d")

    policy = None
    if args.policy_server:
        from phantom.inference.remote import DEFAULT_PORT, RemotePolicy
        addr = args.policy_server
        host, port = (("127.0.0.1", DEFAULT_PORT) if addr == "auto"
                      else (addr.rsplit(":", 1)[0], int(addr.rsplit(":", 1)[1])))
        cfg = dict(nfe=args.nfe, guidance=getattr(args, "guidance", 1.0),
                   k_seeds=getattr(args, "k_seeds", 1),
                   parity_fixes=getattr(args, "parity_fixes", False),
                   persistent_noise=getattr(args, "persistent_noise", False),
                   task_text=(args.text or args.task),
                   drop_video=args.drop_video,
                   close_p=getattr(args, "veto_p_close", 0.5))
        try:
            policy = RemotePolicy((host, port), cfg)
            log.info("using policy server at %s:%d (ckpt %s, warm=%s) — "
                     "no local model load", host, port,
                     policy.info.get("ckpt"), policy.info.get("warmed"))
            if args.ckpt and str(args.ckpt) not in str(policy.info.get("ckpt")) \
                    and str(policy.info.get("ckpt")) not in str(args.ckpt):
                log.warning("policy server holds %s but --ckpt asked for %s — "
                            "the SERVER's model runs. Restart the server to "
                            "switch models.", policy.info.get("ckpt"), args.ckpt)
        except Exception:
            if addr != "auto":
                log.exception("policy server %s required but unreachable", addr)
                return 3
            log.info("no policy server on %s:%d — loading locally", host, port)
    if policy is None:
        policy = build_policy(args, hw, paths)
    remote = policy.__class__.__name__ == "RemotePolicy"
    any_real = hw.mode.drivers == "real" or "real" in hw.mode.overrides.values()
    arm_real = hw.mode.resolve("arm") == "real"
    if not remote and (getattr(args, "compile", False) or (any_real and not args.tiny)):
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
    if remote:
        # provenance must name the model that actually ran — the server's
        ckpt_real = str(policy.info.get("ckpt", ckpt_real))
    cond_tags = [f"nfe{policy.nfe}", f"g{policy.guidance}",
                 "pnoise" if args.persistent_noise else "freshnoise",
                 f"ckpt:{Path(ckpt_real).name}", f"git:{sha}",
                 f"seed:{args.seed}" if args.seed is not None else "seed:none",
                 # safety envelope provenance (rig 2026-08-28)
                 # honest under --no-z-floor: the raw workspace bound is NOT
                 # a task floor, and the two used to be indistinguishable
                 (f"zfloor:{round(hw.safety.workspace_m.z[0] * 1000)}mm"
                  if z_floor is not None else "zfloor:none"),
                 f"hitbox:{round(args.hitbox_margin * 1000)}mm" if hitbox_on else "hitbox:none",
                 f"vmax:{hw.arm.limits.tcp_speed_m_s:.2f}",
                 # deploy levers (review 2026-08-28) — ALWAYS tagged, on or off,
                 # so an A/B arm can never be reconstructed from memory alone
                 f"parity:{'on' if args.parity_fixes else 'off'}",
                 (f"veto:pc{args.veto_p_close:.2f}/pn{args.veto_p_none:.2f}/"
                  f"r{args.veto_max_retries}" if args.terminal_veto else "veto:off"),
                 f"kseeds:{max(1, args.k_seeds)}"]
    veto = build_veto(args, stats, z_floor)
    if veto is not None:
        log.info("terminal veto ON: p_close=%.2f p_none=%.2f max_retries=%d "
                 "z_floor=%s open_aperture=%.2f", veto.p_close, veto.p_none,
                 veto.max_retries,
                 "none" if veto.z_floor is None else f"{veto.z_floor * 1000:.0f}mm",
                 veto.open_aperture)
    log.info("episode condition tags: %s", cond_tags)

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

    def _gate(rt) -> tuple[float, bool]:
        """(worst sigma incl. gripper, gripper settled)."""
        st = rt.rig.arm.get_state()
        gs = rt.rig.gripper.get_state()
        sig, table = sp.start_sigma_report(stats, st.tcp_pose, gs.position,
                                           q=getattr(st, "q", None))
        print(f"live state vs {args.task} demo start distribution:\n{table}")
        return float(np.max(sig)), float(getattr(gs, "obj", 3.0)) == 3.0

    prev_reason: str | None = None
    open_aperture = float(getattr(stats, "gripper_mean", 0.0) or 0.0)
    with DeploymentRuntime(hw, policy, mode=args.system, out_root=out_root,
                           parity_fixes=args.parity_fixes, veto=veto,
                           base_hw=base_hw, open_aperture=open_aperture,
                           deploy_overrides=deploy_overrides,
                           max_play_steps=(args.max_play_steps or None)) as rt:
        for i in range(args.episodes):
            # ONE seed per episode, drawn before anything random happens: the
            # sampler noise AND the homing jitter come from it. The jitter used
            # to run on its own unseeded default_rng (+-27 mm per axis on
            # waffles — the same order as the effect the A/B measures), so
            # `--seed`'s "reproducible ... for paired trials" was false.
            ep_seed = episode_seed(args.seed, i)
            rng = np.random.default_rng(ep_seed)
            start_tag = None
            if arm_real:
                ep_tag = f"episode {i + 1}/{args.episodes}"
                # -- stage 0: restore control after a protective stop -------
                recovered_from = (prev_reason
                                  if prev_reason in _CONTROL_DEAD_REASONS else None)
                if recovered_from is not None:
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
                        homed = sp.move_to_start(
                            rt.rig.arm, rt.rig.gripper, hw, stats,
                            home_joints=args.home_joints, rng=rng)
                        # provenance: the REALISED start, not just the seed —
                        # the pairing of an A/B cell is only readable with it
                        if homed is not None:
                            tcp_t, grip_t = homed
                            start_tag = ("start:" + ",".join(
                                f"{v * 1000:.0f}" for v in np.asarray(tcp_t)[:3])
                                + f"mm/g{float(grip_t):.2f}")
                    except Exception:
                        if recovered_from is not None:
                            # We JUST rebuilt the control script and verified it
                            # is playing, and move_l still failed: the arm is
                            # not controllable in a way this process can fix
                            # (sticky servo guard, Local mode, IK reject).
                            # Continuing here is what turned one bad episode
                            # into a whole campaign of hand-jogged OOD starts —
                            # refuse instead of logging (audit 2026-08-27).
                            log.exception(
                                "homing move FAILED right after RTDE control "
                                "was rebuilt for %r — the arm is not "
                                "controllable from this process. Refusing %s. "
                                "Check the pendant (Local mode?) and restart "
                                "run_deploy.", recovered_from, ep_tag)
                            return 4
                        log.exception("homing move FAILED (protective stop / "
                                      "Local mode / no move_l?) — jog the arm "
                                      "to the pose below by hand; the gate "
                                      "re-checks before anything runs")
                # -- stage 2: gate loop + operator confirm ------------------
                if stats is not None:
                    auto_homes = 0
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
                        if args.home and auto_homes < 2:
                            # self-correct instead of asking the operator to
                            # jog + restart: a JOINT-space home fixes exactly
                            # what the joint gate measures (incl. a wrapped
                            # wrist — it unwinds). Slow move; countdown so the
                            # operator can e-stop if the path is not clear.
                            auto_homes += 1
                            log.warning("START GATE: %.1f sigma — AUTO-HOMING "
                                        "(slow joint-space move) in 3 s. "
                                        "E-stop if the path is not clear.",
                                        worst)
                            time.sleep(3.0)
                            try:
                                sp.move_to_start(rt.rig.arm, rt.rig.gripper,
                                                 hw, stats, home_joints=True,
                                                 rng=rng)
                                continue
                            except Exception:
                                log.exception("auto-home FAILED — manual fix")
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
                        if args.home:
                            log.warning("bumped off the start (%.1f sigma) — "
                                        "AUTO-HOMING in 3 s. E-stop if the "
                                        "path is not clear.", worst)
                            time.sleep(3.0)
                            try:
                                sp.move_to_start(rt.rig.arm, rt.rig.gripper,
                                                 hw, stats, home_joints=True,
                                                 rng=rng)
                            except Exception:
                                log.exception("auto-home FAILED — fix by hand")
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
            # Seed the SAMPLING noise per episode and record it. Without this
            # the rectified-flow generator stays at its constructor seed
            # (rf.py: manual_seed(0)), so every deploy process replayed the
            # SAME noise sequence: the warm-up replan consumed a fixed number
            # of draws and episode 0 of every rig session since 08-14 sampled
            # one identical persistent-noise tensor — a 6th-percentile "slow"
            # draw (16-seed replay, 2026-08-29: the rig's chunk == the slowest
            # of 16 seeds at every replan of every episode-0 trace).
            if hasattr(policy, "remote_reset"):
                policy.remote_reset(ep_seed)
            elif hasattr(policy.rf, "reset_episode_noise"):
                policy.rf._gen = torch.Generator().manual_seed(ep_seed)
                policy.rf.reset_episode_noise()
            else:                       # test stubs without a sampler
                log.warning("policy has no sampling generator to seed")
            ep_tags = [t for t in ep_tags if not t.startswith("seed:")] + [f"seed:{ep_seed}"]
            if start_tag is not None:
                ep_tags.append(start_tag)
            budget = float(args.max_episode_s)
            op_stop = make_operator_stop()
            if op_stop is not None:
                print(">> press Enter at any time to END the episode cleanly "
                      "(motion stops, gripper stays as-is)")
            lift_done = make_lift_complete(rt, args.lift_complete_z,
                                           args.lift_complete_fz,
                                           args.lift_complete_hold)
            res = rt.run_episode(task=args.task, text=args.text,
                                 max_replans=args.max_replans,
                                 max_episode_s=budget if budget > 0 else None,
                                 policy_name=f"{args.system}", tags=ep_tags,
                                 stop_check=op_stop,
                                 success_check=lift_done)
            log.info("episode %d: %s (replans=%d stop=%s safety_events=%d)",
                     i, res.episode_path, res.n_replans, res.stopped_reason,
                     res.safety_events)
            for ev in res.events:
                log.info("  safety event: %s value=%.3f x%d at t=%.2f",
                         ev["kind"], ev["value"], ev["count"], ev["t"])
            persist_stop(rt.recorder, res)
            if (res.stopped_reason in ("safety_stop", "wrist_extension")
                    and lift_done is not None and lift_done()):
                # the 500 Hz wrist-extension stop can beat the per-replan
                # lift detector on a good grasp (they sit ~20 mm of wd
                # apart) — do not let a SUCCESS get logged as an abort
                log.info("NOTE: the safety stop fired DURING a confirmed "
                         "hold (pads loaded, lifted) — label this episode "
                         "a SUCCESS; the object should still be in the "
                         "gripper.")
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
                # (audit 2026-08-20): the session produced unlabeled anecdotes.
                # run_episode has already filed this episode as
                # status='aborted' + tag 'unlabeled' (P9): a verdict PROMOTES
                # it back to 'finalized', no verdict leaves it excluded.
                # drain Enter presses buffered during the episode (a second
                # operator-stop tap otherwise answers this prompt with an
                # empty line and the label is silently discarded —
                # verification 09-01)
                if sys.stdin.isatty():
                    while select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                ans = input(VERDICT_PROMPT).strip()
                label_episode(rt.recorder, Path(res.episode_path), ans)
            elif res.episode_path:
                log.warning("episode %s got no operator verdict (%s) — left "
                            "status='aborted' + tag 'unlabeled', excluded from "
                            "training", Path(res.episode_path).name,
                            "no --label-prompt" if not args.label_prompt
                            else "no arm")
            if res.fatal_reason:
                # returning from inside the `with` runs DeploymentRuntime
                # .__exit__: session.stop() + disconnect_all()
                return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
