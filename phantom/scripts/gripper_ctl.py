"""Gripper control from the shell — open / close / reset the Robotiq OUTSIDE a
deploy session.

Rig 2026-08-28: a policy closed the gripper pad-on-pad, the episode ended and
the operator had no way to open it again ("it slams into itself and you can't
open it after"). The URCap socket accepts a fresh connection at any time, so:

    python -m phantom.scripts.gripper_ctl open   --hardware configs/hardware.nuc.yaml
    python -m phantom.scripts.gripper_ctl close  --hardware configs/hardware.nuc.yaml [--pos 0.9]
    python -m phantom.scripts.gripper_ctl reset  --hardware configs/hardware.nuc.yaml
    python -m phantom.scripts.gripper_ctl status --hardware configs/hardware.nuc.yaml

`reset` = deactivate (ACT 0) -> activate (ACT 1, the gripper runs its
calibration stroke: keep fingers clear) -> open. Use it after an e-stop /
power cycle when `open` reports the gripper as not activated.

Run it when no deploy process is driving the gripper (between runs); the
force/position ceilings are the same pad-safe clamps the deploy path uses
(base.Gripper.clamp_*).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from phantom.config.hardware import load_hardware

log = logging.getLogger("gripper_ctl")


def make_gripper(hw):
    if hw.mode.resolve("gripper") == "mock":
        from phantom.drivers.mock.robotiq import MockGripper
        return MockGripper(hw.gripper)
    from phantom.drivers.real.robotiq import RobotiqGripper
    return RobotiqGripper(hw.gripper, hw.arm.ip)


def wait_settled(gripper, timeout_s: float = 6.0):
    """Poll until OBJ != 0 (moving) or timeout; returns the last state."""
    deadline = time.perf_counter() + timeout_s
    st = gripper.get_state()
    while time.perf_counter() < deadline and st.moving:
        time.sleep(0.1)
        st = gripper.get_state()
    return st


def describe(st) -> str:
    obj = int(getattr(st, "obj", 3))
    what = {0: "moving", 1: "stopped on contact while OPENING",
            2: "stopped on contact while CLOSING (holding something)",
            3: "at requested position (nothing between the pads)"}.get(obj, str(obj))
    return f"position {st.position:.2f} (0 open .. 1 closed)  OBJ={obj}: {what}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("open", "close", "reset", "status"))
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--pos", type=float, default=None,
                    help="target for close (default: pad-safe ceiling) / open (default 0)")
    ap.add_argument("--speed", type=float, default=None, help="0..1 (default: gripper.default_speed)")
    return ap


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    hw = load_hardware(args.hardware)
    g = make_gripper(hw)
    speed = hw.gripper.default_speed if args.speed is None else args.speed
    force = hw.gripper.default_force          # clamped to the pad ceiling inside move()
    g.connect()
    try:
        if args.cmd == "status":
            print(describe(g.get_state()))
            return 0
        if args.cmd == "reset":
            log.info("deactivating (ACT 0) ...")
            if hasattr(g, "_set"):
                g._set(act=0)
                time.sleep(0.5)
            log.info("activating (ACT 1) — calibration stroke, keep fingers clear ...")
        # activate() is idempotent on an activated gripper (STA already 3) and
        # is exactly what an un-activated one (after e-stop / power cycle) needs
        g.activate()
        if args.cmd == "close":
            target = 1.0 if args.pos is None else args.pos      # clamp_position applies the pad ceiling
        else:
            target = 0.0 if args.pos is None else args.pos
        log.info("%s -> %.2f (speed %.2f, force %.2f)", args.cmd, target, speed, force)
        g.move(target, speed, force)
        st = wait_settled(g)
        print(describe(st))
        return 0 if not st.moving else 1
    finally:
        g.disconnect()


if __name__ == "__main__":
    sys.exit(main())
