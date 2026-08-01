"""Teleop episode collection (pipeline.md §7: 750 episodes / 5 tasks + 20-30
deliberate-failure episodes per fragile task).

Hotkeys (keyboard device; also active alongside the spacemouse):
    SPACE  start / stop+save episode        g  mark success and stop
    b      mark failure and stop            f  tag deliberate-failure (then subtype)
    ESC    abort episode (discard)          0  re-zero wrist F/T
    ~      quit after current episode

Usage:
    python -m phantom.scripts.record_episodes --task fragile_grasp \
        [--teleop keyboard|spacemouse|echo] [--operator NAME] [--hardware ...] [--out ...]

The echo device (lab exoskeleton) is joint-space: the loop drives servo_j and
derives the recorded Δ-EE actions from measured TCP poses, additionally
logging raw joint targets to the actions_qtarget stream.

Runs identically against mocks (mode.drivers: mock) — rehearse the SOP anywhere.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data.derived import pose_delta
from phantom.data.schema import STREAM_ACTIONS_QTARGET, EpisodeMeta
from phantom.deploy.safety import SafetyAction, SafetyMonitor
from phantom.drivers.factory import make_rig
from phantom.recording.recorder import EpisodeRecorder
from phantom.recording.workers import SensorSession, make_gripper_poller
from phantom.timesync.clock import IdentityClock, MasterClock

log = logging.getLogger("record")

FAILURE_SUBTYPES = ("over_squeeze", "induced_slip", "near_limit")


def make_motion_device(hw, teleop: str, kbd):
    """Teleop device by name (shared by the CLI and the web-panel session)."""
    if teleop == "spacemouse":
        from phantom.teleop.spacemouse import SpaceMouseTeleop
        return SpaceMouseTeleop()
    if teleop == "echo":
        if hw.teleop is None or hw.teleop.echo is None:
            raise RuntimeError("echo teleop requires a teleop.echo section in "
                               "configs/hardware.yaml")
        from phantom.teleop.echo import EchoTeleop
        return EchoTeleop(hw.teleop.echo)
    if teleop == "none":
        from phantom.teleop.null import NullTeleop
        return NullTeleop()
    if kbd is None:
        raise RuntimeError("keyboard teleop needs an interactive terminal — "
                           "pick echo/spacemouse/none in the panel")
    return kbd


def run_collection(hw, cfg, out_root: Path, *, panel=None, enable_viz: bool = False,
                   use_keyboard: bool | None = None) -> None:
    """Bring the rig up and run the collection loop until quit.

    cfg needs .task/.text/.operator/.teleop (+ optional .target_episodes);
    the web panel (viz/session.py SessionRunner) calls this in a thread with
    `panel` injected — the CLI path calls it directly."""
    out_root.mkdir(parents=True, exist_ok=True)
    use_keyboard = sys.stdin.isatty() if use_keyboard is None else use_keyboard
    kbd = None
    if use_keyboard:
        from phantom.teleop.keyboard import KeyboardTeleop
        kbd = KeyboardTeleop()
    motion = make_motion_device(hw, cfg.teleop, kbd)

    rig = make_rig(hw, control=True)
    # tactile sensors are opened by the SensorSession worker processes below;
    # the parent must not also open them (a real DM-Tac is single-open)
    rig.worker_owned_tactile = True
    with rig:
        clock = (IdentityClock() if hw.mode.resolve("arm") == "mock"
                 else MasterClock.calibrate(rig.arm))
        session = SensorSession.start(hw, rig, session_id=str(int(time.time()) % 10_000_000))
        # this script sends gripper.move() straight from its teleop loop (no
        # GripperPilot), so a standalone poller thread for get_state() is
        # safe here -- unlike collect.py's session.py, there's no second
        # thread racing it for RobotiqGripper's socket lock.
        gripper_poller = make_gripper_poller(rig.gripper, hw, session.rings["gripper"])
        gripper_poller.start()
        recorder = EpisodeRecorder(session, clock, out_root)
        # same force-safety circuit as deployment (wrist wrench, fingertip
        # force/indentation e-stop, staleness, workspace) — teleop is when
        # the pads are most at risk
        safety = SafetyMonitor(hw, session.rings)
        viz = None
        if enable_viz:
            try:
                from phantom.viz.rerun_logger import RerunLogger
                viz = RerunLogger(hw, session.rings)
                viz.start()
                if panel is not None:
                    panel.state.update(rerun_url=viz.viewer_url)
            except RuntimeError as e:
                log.warning("rerun disabled: %s", e)
        if panel is not None:
            panel.state.update(task=cfg.task, operator=cfg.operator,
                               teleop=cfg.teleop, out_dir=str(out_root),
                               target_episodes=getattr(cfg, "target_episodes", 0))
        if kbd is not None:
            kbd.start()
        if motion is not kbd:
            motion.start()
        try:
            _teleop_loop(hw, rig, session, recorder, kbd, motion, safety, cfg,
                         panel=panel, viz=viz)
        finally:
            recorder.stop(abort=True)
            if kbd is not None:
                kbd.stop()
            if motion is not kbd:
                motion.stop()
            if viz is not None:
                viz.stop()
            gripper_poller.stop()
            session.stop()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--text", default="", help="language instruction (default: task name)")
    ap.add_argument("--teleop", choices=("keyboard", "spacemouse", "echo", "none"),
                    default="keyboard")
    ap.add_argument("--operator", default="")
    ap.add_argument("--target-episodes", type=int, default=0,
                    help="progress target shown in the panel (0 = open-ended)")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--panel", action="store_true",
                    help="web control panel + embedded rerun live view")
    ap.add_argument("--panel-host", default="127.0.0.1",
                    help="0.0.0.0 to reach the panel from a rig tablet (trusted LAN)")
    ap.add_argument("--panel-port", type=int, default=8788)
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()
    out_root = Path(args.out) if args.out else \
        paths.episodes_root() / time.strftime("%Y%m%d") / args.task

    panel = None
    if args.panel:
        from phantom.viz.panel import PanelServer, PanelState
        state = PanelState(task=args.task, operator=args.operator,
                           teleop=args.teleop, phase="running")
        panel = PanelServer(state, host=args.panel_host, port=args.panel_port)
        panel.start()
        print(f"[record] control panel: {panel.url}")
    try:
        run_collection(hw, args, out_root, panel=panel, enable_viz=args.panel)
    finally:
        if panel is not None:
            panel.stop()
    return 0


def _teleop_loop(hw, rig, session, recorder, kbd, motion, safety, args,
                 *, panel=None, viz=None) -> None:
    def note(text: str, level: str = "INFO") -> None:
        if viz is not None:
            viz.log_event(text, level=level)
    period = 1.0 / hw.control.action_rate_hz
    dt_servo = 1.0 / hw.control.executor_rate_hz
    recording = False
    ep_count = 0
    tags: list[str] = []
    # joint-space leader (Echo) state: previous measured TCP for Δ-EE
    # derivation; the smooth servo streamer is created on the first q_target
    prev_tcp: np.ndarray | None = None
    streamer = None
    safety_hold = False   # motion suspended until safety.recovered()
    last_grip_sent: float | None = None
    ep_name = ""
    tick_hz = 0.0
    grip_deadband = (hw.teleop.echo.gripper_deadband
                     if hw.teleop is not None and hw.teleop.echo is not None else 0.0)
    print(f"[record] task={args.task}  SPACE=start/stop g=success b=fail f=tag "
          f"ESC=abort 0=zeroFT ~=quit")
    while True:
        t_tick = time.perf_counter()
        cmd = motion.poll()
        buttons = dict(cmd.buttons)
        if kbd is not None and motion is not kbd:
            buttons.update(kbd.poll().buttons)
        if panel is not None:
            buttons.update(panel.pop_buttons())

        if buttons.get("quit"):
            if recording:
                recorder.stop(abort=True)
            if streamer is not None:
                streamer.stop()
            print("[record] quit")
            return
        if buttons.get("zero_ft"):
            rig.arm.zero_ft()
        if buttons.get("failure_tag") and recording:
            sub = FAILURE_SUBTYPES[len([t for t in tags if t in FAILURE_SUBTYPES])
                                   % len(FAILURE_SUBTYPES)]
            tags += ["deliberate_failure", sub]
            print(f"[record] tagged deliberate_failure/{sub}")
        if buttons.get("abort") and recording:
            recorder.stop(abort=True)
            recording = False
            print("[record] episode ABORTED")
            note("episode aborted", "WARN")
            if panel is not None:
                panel.state.add_episode(name=ep_name, outcome="discarded", tags=tags)
        end_success = buttons.get("success")
        end_fail = buttons.get("fail")
        toggle = buttons.get("start_stop")
        if recording and (toggle or end_success or end_fail):
            path = recorder.stop(success=bool(end_success) if (end_success or end_fail)
                                 else None)
            recording = False
            print(f"[record] saved {path}")
            note(f"episode saved: {path.name}")
            if panel is not None:
                panel.state.add_episode(
                    name=path.name,
                    outcome="success" if end_success else "fail" if end_fail else "saved",
                    tags=tags)
        elif not recording and toggle:
            if not session.all_alive():
                print("[record] ERROR: a stream worker died; restart the session")
                return
            tags = []
            meta = EpisodeMeta(task=args.task, text=args.text or args.task,
                               tags=tags, operator=args.operator, policy="teleop")
            if hw.wrist_ft.bias_on_episode_start:
                rig.arm.zero_ft()
            path = recorder.start(meta, f"ep_{args.task}_{int(time.time())}_{ep_count:03d}")
            ep_count += 1
            recording = True
            prev_tcp = None   # restart Δ-EE derivation at the episode boundary
            print(f"[record] RECORDING -> {path.name}")
            note(f"RECORDING {path.name}")
            if panel is not None:
                panel.state.update(t_ep_start=time.time())
        ep_name = path.name if recording else ""

        # motion: stream the teleop command to the arm + gripper, log the action
        state = rig.arm.get_state()

        # --- safety circuit: the SAME monitor as deployment, every tick ----
        if safety_hold:
            if safety.recovered():
                safety_hold = False
                print("[record] safety: forces back below limits — teleop resumed")
                note("teleop resumed after safety hold")
            else:
                wait = period - (time.perf_counter() - t_tick)
                if wait > 0:
                    time.sleep(wait)
                continue   # no commands while held
        target = (state.tcp_pose + np.concatenate([cmd.dpose[:3], cmd.dpose[3:]])
                  if cmd.q_target is None else None)
        verdict = safety.check(t_tick, target if target is not None else state.tcp_pose)
        if verdict.action == SafetyAction.PROTECTIVE_STOP:
            if recording:
                recorder.stop(abort=True)
                recording = False
            if streamer is not None:
                streamer.park()
            safety_hold = True
            print("[record] PROTECTIVE STOP — clear it on the pendant; teleop resumes "
                  "when forces are back below limits")
            note("PROTECTIVE STOP", "ERROR")
            if panel is not None:
                panel.state.update(safety_hold=True, recording=False,
                                   last_safety="protective_stop")
            continue
        if verdict.action == SafetyAction.STOP_EPISODE:
            kinds = ",".join(e.kind for e in verdict.events)
            if recording:
                recorder.stop(success=False, notes=f"safety_stop:{kinds}")
                recording = False
            if streamer is not None:
                streamer.park()
            safety_hold = True
            print(f"[record] SAFETY STOP ({kinds}) — episode saved as failure; "
                  "release the load to resume")
            note(f"SAFETY STOP: {kinds}", "ERROR")
            if panel is not None:
                panel.state.update(safety_hold=True, recording=False,
                                   last_safety=kinds)
                if ep_name:
                    panel.state.add_episode(name=ep_name, outcome="safety_stop",
                                            tags=[kinds])
            continue

        if cmd.q_target is None:
            # Δ-EE devices (keyboard / spacemouse): command IS the action
            action = np.concatenate([cmd.dpose, [cmd.gripper]]).astype(np.float32)
            if recording:
                recorder.record_action(time.perf_counter(), action)
            if verdict.action == SafetyAction.CLAMP:
                target = safety.clamp_target(target)
            rig.arm.servo_l(target, dt_servo, hw.arm.servoj.lookahead_time_s,
                            hw.arm.servoj.gain)
        else:
            # joint-space leader (Echo): motion is owned by the high-rate
            # JointServoStreamer (control_rate_hz servo_j through the
            # accel-limited tracker — no 10 Hz step-and-hold, no engage
            # jump, workspace hold at control rate). This loop only feeds
            # it the filtered target and handles recording.
            if streamer is None:
                from phantom.teleop.streamer import JointServoStreamer
                streamer = JointServoStreamer(hw, rig.arm)
                streamer.start()
                print(f"[record] joint streamer up @ {streamer.rate_hz:.0f} Hz")
            streamer.set_target(cmd.q_target)
            # recorded Δ-EE action = measured TCP delta between consecutive
            # ticks (one tick late by construction; actions and states carry
            # independent timestamps, windows sample by nearest ts)
            if recording:
                if prev_tcp is not None:
                    action = np.concatenate(
                        [pose_delta(prev_tcp, state.tcp_pose), [cmd.gripper]]
                    ).astype(np.float32)
                    recorder.record_action(time.perf_counter(), action)
                    recorder.record_action(time.perf_counter(),
                                           np.asarray(cmd.q_target, dtype=np.float32),
                                           stream=STREAM_ACTIONS_QTARGET)
                prev_tcp = state.tcp_pose.copy()
        # gripper: deadband so the URCap socket only sees meaningful changes
        if last_grip_sent is None or abs(cmd.gripper - last_grip_sent) > grip_deadband:
            rig.gripper.move(cmd.gripper, hw.gripper.default_speed,
                             hw.gripper.default_force)
            last_grip_sent = cmd.gripper

        if panel is not None:
            dt_tick = time.perf_counter() - t_tick
            tick_hz = 0.9 * tick_hz + 0.1 / max(dt_tick, period) if tick_hz else 1.0 / period
            panel.state.update(
                recording=recording, episode=ep_name, ep_count=ep_count,
                safety_hold=safety_hold, tick_hz=tick_hz,
                wrench_N=float(np.linalg.norm(state.ft[:3])),
                gripper=float(cmd.gripper))

        wait = period - (time.perf_counter() - t_tick)
        if wait > 0:
            time.sleep(wait)


if __name__ == "__main__":
    sys.exit(main())
