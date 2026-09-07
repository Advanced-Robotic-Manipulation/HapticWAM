"""Recorded orchestration for the six-call post-study diagnostic on compute3.

This script owns only the dedicated inference server it creates. It never
connects to robot hardware or changes a primary-study process or source file.
"""

import fcntl
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
LIVE = Path("/home/physicalai/phantom-icra-2027/phantom")
SOURCE = BASE / "source_teacher_v2"
STUDY = BASE / "runs/teacher_robustness_v2_delivery"
OUT = BASE / "runs/teacher_rgb_transfer_v1"
HELPER = BASE / "review_tools/tools/sim/diagnose_rgb_transfer.py"
PRIMARY_PID = 1379886
PORT = 7798


def write_status(status, **extra):
    target = OUT / "orchestration/status.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(
        json.dumps(dict(status=status, time_unix_s=time.time(), **extra), indent=2)
        + "\n"
    )
    temp.replace(target)


def main():
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    (OUT / "orchestration").mkdir(parents=True, exist_ok=True)
    with (OUT / "orchestration/owner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "execution").exists() or (OUT / "server").exists():
            raise RuntimeError(
                "Diagnostic output already exists; preserve it and do not retry silently"
            )
        server = None
        calls_complete = False
        try:
            write_status("waiting_for_primary_study")
            while not (STUDY / "study_complete.json").exists():
                cmdline = Path(f"/proc/{PRIMARY_PID}/cmdline")
                try:
                    current_command = cmdline.read_bytes()
                except FileNotFoundError:
                    current_command = b""
                if str(STUDY / "run_study.py").encode() not in current_command:
                    if (STUDY / "study_complete.json").exists():
                        break
                    raise RuntimeError(
                        "Owned primary study exited without its completion marker; diagnostic not launched"
                    )
                time.sleep(5)
            sys.path.insert(0, str(SOURCE))
            spec = importlib.util.spec_from_file_location("rgb_diagnostic", HELPER)
            diagnostic = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(diagnostic)
            completed = diagnostic.validate_study_complete(STUDY)
            if (OUT / "execution").exists() or (OUT / "server").exists():
                raise RuntimeError(
                    "Diagnostic output already exists; preserve it and do not retry silently"
                )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", PORT))
            server_dir = OUT / "server"
            server_dir.mkdir()
            command = [
                str(LIVE / ".venv/bin/python"),
                str(SOURCE / "tools/sim/policy_server.py"),
                "--repo",
                str(LIVE),
                "--ckpt",
                str(LIVE / "runs/teacher_v5_ftA/teacher_001500.pt"),
                "--expected-sha256",
                diagnostic.CHECKPOINT_SHA,
                "--hardware",
                str(BASE / "runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml"),
                "--out",
                str(server_dir),
                "--port",
                str(PORT),
                "--system",
                "teacher",
                "--nfe",
                "1",
                "--k-seeds",
                "4",
                "--guidance",
                "1",
                "--task-text",
                "waffles",
                "--parity-fixes",
                "--persistent-noise",
            ]
            (server_dir / "command.json").write_text(
                json.dumps(command, indent=2) + "\n"
            )
            with (server_dir / "server.log").open("w") as log:
                server = subprocess.Popen(
                    command,
                    cwd=LIVE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                write_status(
                    "loading_dedicated_server",
                    owned_server_pid=server.pid,
                    primary_completion=completed,
                )
                ready = server_dir / "ready.json"
                deadline = time.monotonic() + 600
                while not ready.exists():
                    if server.poll() is not None:
                        raise RuntimeError(
                            f"Dedicated server exited {server.returncode}"
                        )
                    if time.monotonic() > deadline:
                        raise TimeoutError("Dedicated server warmup exceeded 600s")
                    time.sleep(0.5)
                command = [
                    str(LIVE / ".venv/bin/python"),
                    str(HELPER),
                    "--observation",
                    str(
                        BASE
                        / "runs/teacher_robustness_v2/screen/rollouts/fta1500_nfe1_k4__start_1787395928__seed903101/observations/0000.npz"
                    ),
                    "--initial-state",
                    str(
                        SOURCE
                        / "configs/sim/initial_states/waffles_aug22_1787395928_000.json"
                    ),
                    "--episode",
                    "/home/physicalai/phantom-icra-2027/data/full/tasks/waffles/ep_waffles_1787395928_000",
                    "--out",
                    str(OUT / "execution"),
                    "--execute",
                    "--completed-study",
                    str(STUDY),
                    "--server-ready",
                    str(ready),
                    "--owned-server-pid",
                    str(server.pid),
                ]
                (OUT / "orchestration/client_command.json").write_text(
                    json.dumps(command, indent=2) + "\n"
                )
                write_status(
                    "executing_six_first_plan_calls", owned_server_pid=server.pid
                )
                env = dict(
                    os.environ,
                    PYTHONPATH=str(SOURCE),
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                with (OUT / "orchestration/client.log").open("w") as client_log:
                    subprocess.run(
                        command,
                        cwd=LIVE,
                        env=env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        timeout=180,
                        check=True,
                    )
                calls_complete = True
                write_status(
                    "calls_complete_cleanup_pending",
                    owned_server_pid=server.pid,
                    calls=6,
                )
        except BaseException as exc:
            write_status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if server is not None and server.poll() is None:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(server.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    server.wait(timeout=10)
            if calls_complete:
                write_status(
                    "complete",
                    owned_server_pid=server.pid,
                    calls=6,
                    owned_server_exited=True,
                )


if __name__ == "__main__":
    main()
