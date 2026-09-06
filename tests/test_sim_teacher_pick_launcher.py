"""Managed teacher runs must preserve process ownership and input identity."""

import json
import signal
import socket
import subprocess
import sys

import pytest

from tools.sim import run_teacher_pick as launcher


@pytest.fixture
def args(tmp_path, monkeypatch):
    source, live, runtime, episode, isaac = [
        tmp_path / name
        for name in ("source", "live", "debug/runtime", "episode", "isaac")
    ]
    files = {
        source / "tools/sim/policy_server.py": "# inference only\n",
        source / "tools/sim/run_waffles.py": "# sim\n",
        source / "tools/sim/launch_waffles.sh": "#!/bin/sh\nexit 0\n",
        runtime / "waffles_teacher_post_hand.json": "{}",
        runtime / "nfe1_k4.json": json.dumps(launcher.SETTINGS),
        runtime / "hardware_campaign.yaml": "meta: {}\n",
        runtime / "veto_fd4a032.json": json.dumps(
            {
                "implementation": "fd4a032",
                "config": {
                    "z_ref": 0.0415,
                    "z_floor": 0.0315,
                    "z_margin": 0.0615,
                    "open_aperture": 0.232,
                },
            }
        ),
        runtime.parent / "sept4_policy_initial_state.json": "{}",
        runtime.parent / "sept4_no_contact_sensor_baseline.npz": "baseline fixture",
        live / "runs/teacher_v5_ftA/teacher_001500.pt": "checkpoint fixture",
        isaac / "setup_python_env.sh": "# setup\n",
        episode / "replay.npz": "replay fixture",
        episode / "manifest.json": "{}",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for path in (live / ".venv/bin/python", isaac / "kit/python/bin/python3"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(sys.executable)
    (source / "tools/sim/launch_waffles.sh").chmod(0o755)
    monkeypatch.setattr(
        launcher,
        "TEACHER_SHA256",
        launcher.sha256(live / "runs/teacher_v5_ftA/teacher_001500.pt"),
    )
    monkeypatch.setattr(launcher, "capture_versions", lambda _: {"fixture": True})
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return launcher.parser().parse_args(
        [
            "--source",
            str(source),
            "--live-repo",
            str(live),
            "--runtime",
            str(runtime),
            "--episode",
            str(episode),
            "--output",
            str(tmp_path / "managed"),
            "--isaac-root",
            str(isaac),
            "--port",
            str(port),
        ]
    )


class Process:
    def __init__(self, pid, *, wait_error=None):
        self.pid = pid
        self.returncode = None
        self.wait_error = wait_error

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.wait_error:
            raise self.wait_error
        self.returncode = 0
        return self.returncode


def ready(args, plan, process):
    return {
        "pid": process.pid,
        "status": "ready",
        "warmed": True,
        "checkpoint_sha256": launcher.TEACHER_SHA256,
        "effective": launcher.SETTINGS,
        "hardware_sha256": plan["inputs"]["hardware"]["sha256"],
        "system": "teacher",
        "weights": "EMA",
        "port": args.port,
    }


def test_dry_run_preserves_venv_path_and_starts_no_subprocess(
    args, monkeypatch, capsys
):
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("dry run spawned process"),
    )
    argv = ["launcher"]
    for name in (
        "source",
        "live_repo",
        "runtime",
        "episode",
        "output",
        "isaac_root",
        "port",
    ):
        argv += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    monkeypatch.setattr(sys, "argv", argv + ["--dry-run"])
    launcher.main()
    result = json.loads(capsys.readouterr().out)
    assert result["commands"]["server"][0] == str(args.live_repo / ".venv/bin/python")
    assert result["coverage"] == "manifold_patch"
    assert result["checkpoint_weights"] == "EMA"
    assert "--raw" not in result["commands"]["server"]
    assert not args.output.exists()


def test_occupied_port_fails_without_creating_output_or_signaling_owner(
    args, monkeypatch
):
    monkeypatch.setattr(
        launcher.os, "killpg", lambda *_: pytest.fail("signaled port owner")
    )
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", 0))
        foreign.listen()
        args.port = foreign.getsockname()[1]
        with pytest.raises(RuntimeError, match="occupied"):
            launcher.make_plan(args)
    assert not args.output.exists()


@pytest.mark.parametrize("port", [7777, 7778, 7783])
def test_rig_ports_are_reserved(args, port):
    args.port = port
    with pytest.raises(ValueError, match="reserved"):
        launcher.make_plan(args)


def test_existing_output_and_changed_checkpoint_are_rejected(args):
    args.output.mkdir()
    with pytest.raises(FileExistsError):
        launcher.make_plan(args)
    args.output.rmdir()
    (args.live_repo / "runs/teacher_v5_ftA/teacher_001500.pt").write_text(
        "different weights"
    )
    with pytest.raises(ValueError, match="step1500"):
        launcher.make_plan(args)


def test_startup_failure_cleans_only_newly_owned_server(args, monkeypatch):
    plan = launcher.make_plan(args)
    server = Process(12345)
    created, stopped = [], []

    def spawn(command, **kwargs):
        created.append(command)
        assert kwargs["start_new_session"]
        return server

    def fail_ready(*_):
        raise RuntimeError("warmup failed")

    def stop(process):
        if process:
            stopped.append(process)
            process.returncode = -15

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(launcher, "wait_ready", fail_ready)
    monkeypatch.setattr(launcher, "stop_owned", stop)
    with pytest.raises(RuntimeError, match="warmup failed"):
        launcher.run_managed(args, plan)
    assert len(created) == 1 and stopped == [server]
    ledger = json.loads((args.output / "managed.json").read_text())
    assert ledger["status"] == "failed"
    assert ledger["cleanup"]["owned_server_exit_code"] == -15
    assert not ledger["cleanup"]["foreign_processes_signaled"]


def test_rollout_timeout_cleans_both_owned_processes(args, monkeypatch):
    plan = launcher.make_plan(args)
    server = Process(12345)
    child = Process(12346, wait_error=subprocess.TimeoutExpired("Isaac", 1))
    processes = iter((server, child))
    stopped = []
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_a, **_k: next(processes))
    monkeypatch.setattr(launcher, "wait_ready", lambda *_: ready(args, plan, server))

    def stop(process):
        if process:
            stopped.append(process)
            process.returncode = -15

    monkeypatch.setattr(launcher, "stop_owned", stop)
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.run_managed(args, plan)
    assert stopped == [child, server]
    assert json.loads((args.output / "managed.json").read_text())["status"] == "failed"


def test_owned_ready_marker_with_wrong_settings_never_starts_isaac(args, monkeypatch):
    plan = launcher.make_plan(args)
    server = Process(12345)
    calls = []
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda command, **_: calls.append(command) or server,
    )
    marker = ready(args, plan, server)
    marker["effective"] = {**launcher.SETTINGS, "k_seeds": 1}
    monkeypatch.setattr(launcher, "wait_ready", lambda *_: marker)
    monkeypatch.setattr(
        launcher,
        "stop_owned",
        lambda process: setattr(process, "returncode", -15) if process else None,
    )
    with pytest.raises(RuntimeError, match="settings or hardware"):
        launcher.run_managed(args, plan)
    assert len(calls) == 1
    assert server.returncode == -15


def test_foreign_ready_marker_is_rejected(args):
    server_dir = args.output / "server"
    server_dir.mkdir(parents=True)
    (server_dir / "ready.json").write_text(
        json.dumps(
            {
                "pid": 99999,
                "checkpoint_sha256": launcher.TEACHER_SHA256,
                "status": "ready",
                "warmed": True,
            }
        )
    )
    with pytest.raises(RuntimeError, match="owned warmed teacher"):
        launcher.wait_ready(Process(12345), server_dir, 1)


def test_stop_owned_signals_only_live_handle_group(monkeypatch):
    sent = []
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    completed = Process(12345)
    completed.returncode = 0
    launcher.stop_owned(completed)
    active = Process(12346)
    launcher.stop_owned(active)
    assert sent == [(12346, signal.SIGTERM)]
