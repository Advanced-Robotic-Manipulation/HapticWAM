"""Exclusive rig lease — one robot-owning process per arm per host.

Rig 2026-09-04: three Ctrl-Z'd `run_deploy` processes each held an RTDE
control connection ("RTDE input registers are already in use") and the next
launch fought them for the arm (issue #8). The RTDE controller refuses a
second control script, but only AFTER the new process has started talking to
the robot, and a suspended owner never lets go.

The lease is an `flock` on a per-arm file: acquired before any control call
(`URArm.connect(control=True)`), released when the process exits or the file
descriptor is closed — a suspended (SIGSTOP / Ctrl-Z) owner keeps it, and the
error names that owner so the operator can `fg` or kill it instead of
guessing. A process that cannot get the lease makes NO robot call.

Scope: one host. A second host talking to the same arm is still stopped by
the controller's own register check; the lease exists so a process on this
host reports the owner without touching the robot.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path


class RigBusy(RuntimeError):
    """Another process on this host owns the arm. No robot call was made."""


def lease_path(key: str) -> Path:
    d = Path(os.environ.get("PHANTOM_RIG_LOCK_DIR", "/tmp"))
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in key)
    return d / f"phantom-rig-{safe}.lock"


def _proc_state(pid: int) -> str:
    """'T (stopped)' for a Ctrl-Z'd owner, '' if unknown (macOS / gone)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split(")")[-1].split()
        return {"T": "T (stopped — fg it or kill it)", "R": "R (running)",
                "S": "S (sleeping)", "Z": "Z (zombie)"}.get(fields[0], fields[0])
    except Exception:
        try:
            os.kill(pid, 0)
            return "alive"
        except ProcessLookupError:
            return "gone (stale lease file — safe to remove)"
        except Exception:
            return ""


def acquire(key: str):
    """Return an open file object holding the lease (keep it referenced for
    as long as the arm is in use). Raises RigBusy naming the owner."""
    path = lease_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.seek(0)
        try:
            owner = json.loads(f.read() or "{}")
        except Exception:
            owner = {}
        f.close()
        pid = owner.get("pid")
        since = owner.get("since")
        if pid == os.getpid():
            raise RigBusy(
                f"rig '{key}' is already owned by THIS process (pid {pid}): a "
                "second control-mode URArm in one process — disconnect the "
                "first one. Nothing was sent to the robot.")
        raise RigBusy(
            f"rig '{key}' is owned by pid {pid} "
            f"({_proc_state(int(pid)) if pid else 'unknown'}), "
            f"cmd: {owner.get('cmd', '?')}, since "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since)) if since else '?'}"
            f" — lease file {path}. Nothing was sent to the robot. "
            "Bring that process to the foreground and end it (or kill it) "
            "before launching.")
    f.seek(0)
    f.truncate()
    f.write(json.dumps({"pid": os.getpid(), "since": time.time(),
                        "cmd": " ".join(sys.argv)[:300]}))
    f.flush()
    return f


def owner(key: str) -> dict | None:
    """Read-only: who holds the lease (None if free). Never blocks, never
    touches the robot — for preflight prints."""
    path = lease_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "a+") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return None
            except OSError:
                f.seek(0)
                d = json.loads(f.read() or "{}")
                if d.get("pid"):
                    d["state"] = _proc_state(int(d["pid"]))
                return d
    except Exception:
        return None
