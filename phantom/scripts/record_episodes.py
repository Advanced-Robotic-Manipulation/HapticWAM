"""Teleop episode collection (pipeline.md §7: 750 episodes / 5 tasks + 20-30
deliberate-failure episodes per fragile task).

Hotkeys (keyboard device; also active alongside the spacemouse):
    SPACE  start / stop+save episode        g  mark success and stop
    b      mark failure and stop            f  tag deliberate-failure (then subtype)
    ESC    abort episode (discard)          0  re-zero wrist F/T
    ~      quit after current episode

Usage:
    python -m phantom.scripts.record_episodes --task fragile_grasp \
        [--teleop keyboard|spacemouse] [--operator NAME] [--hardware ...] [--out ...]

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
from phantom.data.schema import EpisodeMeta
from phantom.drivers.factory import make_rig
from phantom.recording.recorder import EpisodeRecorder
from phantom.recording.workers import SensorSession
from phantom.timesync.clock import IdentityClock, MasterClock

log = logging.getLogger("record")

FAILURE_SUBTYPES = ("over_squeeze", "induced_slip", "near_limit")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--text", default="", help="language instruction (default: task name)")
    ap.add_argument("--teleop", choices=("keyboard", "spacemouse"), default="keyboard")
    ap.add_argument("--operator", default="")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()
    out_root = Path(args.out) if args.out else \
        paths.episodes_root() / time.strftime("%Y%m%d") / args.task
    out_root.mkdir(parents=True, exist_ok=True)

    from phantom.teleop.keyboard import KeyboardTeleop
    kbd = KeyboardTeleop()
    if args.teleop == "spacemouse":
        from phantom.teleop.spacemouse import SpaceMouseTeleop
        motion = SpaceMouseTeleop()
    else:
        motion = kbd

    rig = make_rig(hw, control=True)
    with rig:
        clock = (IdentityClock() if hw.mode.resolve("arm") == "mock"
                 else MasterClock.calibrate(rig.arm))
        session = SensorSession.start(hw, rig, session_id=str(int(time.time()) % 10_000_000))
        recorder = EpisodeRecorder(session, clock, out_root)
        kbd.start()
        if motion is not kbd:
            motion.start()
        try:
            _teleop_loop(hw, rig, session, recorder, kbd, motion, args)
        finally:
            recorder.stop(abort=True)
            kbd.stop()
            if motion is not kbd:
                motion.stop()
            session.stop()
    return 0


def _teleop_loop(hw, rig, session, recorder, kbd, motion, args) -> None:
    period = 1.0 / hw.control.action_rate_hz
    dt_servo = 1.0 / hw.control.executor_rate_hz
    recording = False
    ep_count = 0
    tags: list[str] = []
    print(f"[record] task={args.task}  SPACE=start/stop g=success b=fail f=tag "
          f"ESC=abort 0=zeroFT ~=quit")
    while True:
        t_tick = time.perf_counter()
        cmd = motion.poll()
        buttons = dict(cmd.buttons)
        if motion is not kbd:
            buttons.update(kbd.poll().buttons)

        if buttons.get("quit"):
            if recording:
                recorder.stop(abort=True)
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
        end_success = buttons.get("success")
        end_fail = buttons.get("fail")
        toggle = buttons.get("start_stop")
        if recording and (toggle or end_success or end_fail):
            path = recorder.stop(success=bool(end_success) if (end_success or end_fail)
                                 else None)
            recording = False
            print(f"[record] saved {path}")
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
            print(f"[record] RECORDING -> {path.name}")

        # motion: stream the teleop delta to the arm + gripper, log the action
        action = np.concatenate([cmd.dpose, [cmd.gripper]]).astype(np.float32)
        if recording:
            recorder.record_action(time.perf_counter(), action)
        state = rig.arm.get_state()
        target = state.tcp_pose + np.concatenate([cmd.dpose[:3], cmd.dpose[3:]])
        if not hw.safety.workspace_m.contains(target[:3]):
            target[:3] = np.clip(target[:3],
                                 [hw.safety.workspace_m.x[0], hw.safety.workspace_m.y[0],
                                  hw.safety.workspace_m.z[0]],
                                 [hw.safety.workspace_m.x[1], hw.safety.workspace_m.y[1],
                                  hw.safety.workspace_m.z[1]])
        rig.arm.servo_l(target, dt_servo, hw.arm.servoj.lookahead_time_s,
                        hw.arm.servoj.gain)
        rig.gripper.move(cmd.gripper, hw.gripper.default_speed, hw.gripper.default_force)

        wait = period - (time.perf_counter() - t_tick)
        if wait > 0:
            time.sleep(wait)


if __name__ == "__main__":
    sys.exit(main())
