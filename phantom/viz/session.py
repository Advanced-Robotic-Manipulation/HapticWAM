"""SessionRunner: drives a full collection session from the web panel.

The panel is the primary entry point (scripts/panel.py): the operator fills
the wizard, POST /api/session/start lands here, and run_collection (the same
code path as the record_episodes CLI) is executed in a worker thread with the
panel injected. Phase transitions and errors surface back through PanelState
so the UI can render them.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
import types
from pathlib import Path

from phantom.config.hardware import HardwareConfig
from phantom.config.paths import load_paths

log = logging.getLogger(__name__)

_TASK_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def setup_payload(hw: HardwareConfig) -> dict:
    """Rig summary for the panel's setup view."""
    try:
        paths = load_paths()
        out_root = str(paths.episodes_root())
    except Exception:
        out_root = "runs/episodes"
    try:
        free_gb = round(shutil.disk_usage(Path(out_root).anchor or "/").free / 1e9, 1)
    except OSError:
        free_gb = 0.0
    try:
        import rerun  # noqa: F401  ([viz] extra)
        viz_ok = True
    except ImportError:
        viz_ok = False
    return {
        "rig_name": hw.meta.rig_name,
        "mode": hw.mode.drivers,
        "overrides": dict(hw.mode.overrides),
        "bench_verified": hw.meta.bench_verified,
        "arm": f"{hw.arm.model} @ {hw.arm.ip}",
        "sensors": [f"{s.name}:{s.dev_id}" for s in hw.tactile.sensors],
        "echo_configured": hw.teleop is not None and hw.teleop.echo is not None,
        "viz_available": viz_ok,
        "out_root": out_root,
        "disk_free_gb": free_gb,
    }


class SessionRunner:
    """Bridges POST /api/session/{start,stop} to run_collection in a thread."""

    def __init__(self, hw: HardwareConfig, panel):
        self.hw = hw
        self.panel = panel
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def start_session(self, cfg: dict) -> str | None:
        """Validate + launch; returns a user-readable error or None."""
        if self._thread is not None and self._thread.is_alive():
            return "сессия уже запущена"
        task = str(cfg.get("task", "")).strip()
        if not _TASK_RE.match(task):
            return "имя задачи: латиница/цифры/_- (1–64 символа)"
        teleop = cfg.get("teleop", "echo")
        if teleop not in ("echo", "spacemouse", "keyboard", "none"):
            return f"неизвестное устройство {teleop!r}"
        if teleop == "echo" and (self.hw.teleop is None or self.hw.teleop.echo is None):
            return "Echo не настроен в configs/hardware.yaml (teleop.echo)"
        run_cfg = types.SimpleNamespace(
            task=task,
            text=str(cfg.get("text", "")).strip(),
            operator=str(cfg.get("operator", "")).strip(),
            teleop=teleop,
            target_episodes=max(0, int(cfg.get("target_episodes", 0) or 0)),
        )
        try:
            paths = load_paths()
            out_root = paths.episodes_root() / time.strftime("%Y%m%d") / task
        except Exception:
            out_root = Path("runs/episodes") / time.strftime("%Y%m%d") / task
        self.panel.state.reset_session(
            phase="starting", task=task, text=run_cfg.text,
            operator=run_cfg.operator, teleop=teleop,
            target_episodes=run_cfg.target_episodes, out_dir=str(out_root))
        self._thread = threading.Thread(target=self._run, args=(run_cfg, out_root),
                                        daemon=True, name="session-runner")
        self._thread.start()
        return None

    def stop_session(self) -> None:
        """Graceful: quit lands in the loop like the '~' hotkey."""
        self.panel.state.update(phase="stopping")
        self.panel.push_button("quit")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _run(self, run_cfg, out_root: Path) -> None:
        from phantom.scripts.record_episodes import run_collection
        try:
            self.panel.state.update(phase="running")
            run_collection(self.hw, run_cfg, out_root, panel=self.panel,
                           enable_viz=True, use_keyboard=False)
            self.panel.state.update(phase="setup")
        except Exception as e:
            log.exception("session failed")
            self.panel.state.update(phase="setup", error=str(e))
        finally:
            self.panel.pop_buttons()   # drop stale commands from the dead session
