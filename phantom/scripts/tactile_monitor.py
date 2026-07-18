"""Minimal CLI tactile monitor — bench-day / data-collection sanity tool.

One status line per sensor per refresh: resultant |F| and Fz, peak |depth|,
contact area, effective read rate. No GUI dependencies (phantom is GUI-less
by design); the DearPyGui viewers from the Denmark rig remain available as
reference in incoming/DM-Tac-SDK/scripts/.

Usage:
    python -m phantom.scripts.tactile_monitor [--hardware ...] [--sensor NAME]
        [--refresh-hz 5]

    r + Enter   re-zero the no-contact reference on all monitored sensors
                (keep the pads untouched!)
    q + Enter / Ctrl-C   quit

Runs identically against mocks (mode.drivers/overrides: tactile: mock).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

from phantom.config.hardware import load_hardware
from phantom.drivers.base import TactileSensor


def _make_sensors(hw, only: str | None) -> dict[str, TactileSensor]:
    entries = [s for s in hw.tactile.sensors if only is None or s.name == only]
    if not entries:
        raise SystemExit(f"no tactile sensor named {only!r} in the hardware config")
    sensors: dict[str, TactileSensor] = {}
    if hw.mode.resolve("tactile") == "mock":
        from phantom.drivers.mock.dmtac import MockTactileSensor
        from phantom.drivers.mock.scenario import ContactScenario
        scenario = ContactScenario(seed=0)
        for i, s in enumerate(entries):
            sensors[s.name] = MockTactileSensor(hw.tactile, s, scenario, finger_index=i)
    else:
        from phantom.drivers.real.dmtac import DmTacSensor
        for s in entries:
            sensors[s.name] = DmTacSensor(hw.tactile, s)
    return sensors


def _stdin_commands(out: list[str], stop: threading.Event) -> None:
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            return
        out.append(line.strip().lower())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--sensor", default=None, help="monitor a single sensor by name")
    ap.add_argument("--refresh-hz", type=float, default=5.0)
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    sensors = _make_sensors(hw, args.sensor)
    for name, s in sensors.items():
        print(f"[monitor] connecting {name} (dev_id={s.sensor.dev_id}) ...")
        s.connect()
    print("[monitor] r+Enter = reset reference, q+Enter / Ctrl-C = quit")

    cmds: list[str] = []
    stop = threading.Event()
    threading.Thread(target=_stdin_commands, args=(cmds, stop),
                     daemon=True, name="monitor-stdin").start()

    last_t = {name: time.perf_counter() for name in sensors}
    rate = dict.fromkeys(sensors, 0.0)
    interval = 1.0 / max(args.refresh_hz, 0.1)
    next_print = 0.0
    try:
        while True:
            while cmds:
                c = cmds.pop(0)
                if c == "q":
                    return 0
                if c == "r":
                    print("[monitor] resetting reference — keep pads untouched")
                    for s in sensors.values():
                        s.reset_reference()
            lines = []
            for name, s in sensors.items():
                f = s.read()   # blocks until the next sample at native rate
                now = time.perf_counter()
                dt = now - last_t[name]
                last_t[name] = now
                rate[name] = 0.9 * rate[name] + 0.1 * (1.0 / max(dt, 1e-6)) \
                    if rate[name] else 1.0 / max(dt, 1e-6)
                f_mag = float(np.linalg.norm(f.wrench[:3]))
                lines.append(
                    f"{name}: |F|={f_mag:6.2f}  Fz={f.wrench[2]:+6.2f}  "
                    f"peak|depth|={float(np.abs(f.depth).max()):6.3f}  "
                    f"area={f.contact_area_mm2:7.1f}mm2  {rate[name]:5.1f}Hz")
            if time.perf_counter() >= next_print:
                print("  |  ".join(lines))
                next_print = time.perf_counter() + interval
    except KeyboardInterrupt:
        return 0
    finally:
        stop.set()
        for s in sensors.values():
            try:
                s.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
