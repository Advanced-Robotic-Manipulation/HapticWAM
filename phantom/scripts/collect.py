"""The operator's entry point for Echo teleop + data collection.

    python -m phantom.scripts.collect [--config configs/data_collect.yaml]
                                      [--host 0.0.0.0] [--port 8899]

Starts the web control panel; everything else (sessions, episodes, the
DM-Tac safeguard, offload to the external drive, and playback/verification
of already-recorded episodes) is driven from the browser.
See docs/data_collect_app.md for the full write-up — why this exists (the
device-rate teleop fix), the safeguard/resume flow, collection modes, and
the configs/data_collect.yaml fields.

`--host 0.0.0.0` exposes the panel to the rig LAN (tablet at the robot); the
buttons move a robot and open/close the gripper — trusted networks only.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="phantom.scripts.collect")
    ap.add_argument("--config", default=None,
                    help="data_collect yaml (default configs/data_collect.yaml)")
    ap.add_argument("--host", default=None,
                    help="override panel.host (0.0.0.0 for a rig tablet)")
    ap.add_argument("--port", type=int, default=None, help="override panel.port")
    args = ap.parse_args(argv)

    from phantom.data_collect.config import load_collect
    from phantom.data_collect.panel import (CollectPanel, CollectPanelState,
                                            CollectRunner)
    from phantom.data_collect.playback import PlaybackController
    from phantom.data_collect.session import CollectApp

    cc, hw = load_collect(args.config)

    state = CollectPanelState(
        mode=cc.default_mode,
        viewer_mode=cc.rerun.viewer,
        safeguard_enabled=cc.safeguard.enabled,
        safeguard_force_limit_n=cc.safeguard.force_limit_n,
        safeguard_depth_limit=cc.safeguard.depth_limit)

    def setup_provider() -> dict:
        drive = Path(cc.storage.external_drive)
        try:
            free_gb = round(shutil.disk_usage(drive).free / 1e9, 1) \
                if drive.exists() else 0.0
        except OSError:
            free_gb = 0.0
        return {
            "rig_name": hw.meta.rig_name,
            "mode": hw.mode.drivers,
            "arm": f"{hw.arm.model} @ {hw.arm.ip}",
            "sensors": [f"{s.name}:{s.dev_id}" for s in hw.tactile.sensors],
            "camera_serial": hw.cameras.scene.serial or "(first device)",
            "drive": str(drive),
            "drive_present": drive.exists(),
            "drive_free_gb": free_gb,
            "staging_root": cc.storage.staging_root,
            "default_mode": cc.default_mode,
        }

    # runner/panel/app wiring (panel needs the runner; the runner needs the
    # app; the app needs the panel — construct panel first, attach after)
    panel = CollectPanel(state,
                         host=args.host or cc.panel.host,
                         port=args.port or cc.panel.port,
                         setup_provider=setup_provider)
    app = CollectApp(cc, hw, panel)
    runner = CollectRunner(app, panel)
    panel.runner = runner
    panel.safeguard_provider = lambda: app.safeguard
    # episode playback/verify shares the runner's start gate so "session vs
    # playback" exclusivity cannot race between two HTTP threads
    playback = PlaybackController(cc, hw, panel, session_busy=runner.busy,
                                  gate_lock=runner.gate_lock)
    panel.playback = playback
    runner.playback = playback

    panel.start()
    print(f"[collect] control panel: {panel.url}")
    print(f"[collect] rig: {hw.meta.rig_name}  drivers: {hw.mode.drivers}  "
          f"drive: {cc.storage.external_drive}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        # graceful: wait for the session AND the offload to finish — killing
        # daemon threads mid-teardown/mid-offload risks the episode data.
        # A second Ctrl-C force-exits for the truly-stuck case.
        print("[collect] shutting down — waiting for session/offload "
              "(Ctrl-C again to force)")
        try:
            if runner.is_running():
                runner.stop_session()
            while runner.busy():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("[collect] FORCED exit — episodes in staging may need a "
                  "manual 'Offload now' next start")
    finally:
        playback.stop()
        panel.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
