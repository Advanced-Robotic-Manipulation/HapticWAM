"""The operator's entry point: web-first data collection.

    python -m phantom.scripts.panel [--host 0.0.0.0] [--port 8788] [--hardware ...]

Starts ONLY the web panel (no robot motion yet) and prints its URL. The
operator opens it in a browser, fills the session wizard (task, operator,
teleop device, episode target) and presses "Запустить сессию" — the backend
brings up the rig, sensors, safety circuit, recorder and rerun view, and the
whole collection runs from the browser. Ctrl-C here shuts everything down.

`--host 0.0.0.0` exposes the panel to the rig LAN (tablet at the robot);
the buttons move a robot — trusted networks only.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from phantom.config.hardware import load_hardware
from phantom.viz.panel import PanelServer, PanelState
from phantom.viz.session import SessionRunner, setup_payload

log = logging.getLogger("panel")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    state = PanelState(phase="setup")
    panel = PanelServer(state, host=args.host, port=args.port)
    runner = SessionRunner(hw, panel)
    panel.runner = runner
    panel.setup_provider = lambda: setup_payload(hw)
    panel.start()
    print(f"[panel] открой в браузере:  {panel.url}")
    print("[panel] Ctrl-C — остановить всё")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[panel] останавливаю...")
        if runner.is_running():
            runner.stop_session()
            deadline = time.time() + 15.0
            while runner.is_running() and time.time() < deadline:
                time.sleep(0.2)
        panel.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
