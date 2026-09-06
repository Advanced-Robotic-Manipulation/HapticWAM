#!/usr/bin/env python3
"""Run the measured-start teacher pick diagnostic with an owned policy server.

Run on the Isaac host. --runtime names teacher_debug_v1/runtime; its parent
contains sept4_policy_initial_state.json and the static tactile baseline.
--dry-run prints the exact commands and checks inputs without starting GPU jobs.
An exit-zero rollout means the harness completed, not that the object was picked.
The optional contact-coverage approximation and both veto versions are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

TEACHER_SHA256 = "67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e"
SETTINGS = {
    "nfe": 1,
    "guidance": 1.0,
    "k_seeds": 4,
    "parity_fixes": True,
    "persistent_noise": True,
    "task_text": "waffles",
    "drop_video": False,
    "close_p": 0.5,
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "live-repo", "runtime", "episode", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--coverage", choices=("point", "manifold_patch"), default="manifold_patch"
    )
    p.add_argument("--veto-version", choices=("fd4a032", "live"), default="fd4a032")
    p.add_argument(
        "--scene-config",
        type=Path,
        help="Explicit alternate scene, e.g. the 1ms integration diagnostic",
    )
    p.add_argument("--initial-state", type=Path)
    p.add_argument("--tactile-baseline", type=Path)
    p.add_argument("--placement-release-config", type=Path)
    p.add_argument("--record-packet-support", action="store_true")
    p.add_argument(
        "--robot-usd",
        type=Path,
        help="Reuse a known articulation; omit to import the pinned source URDF",
    )
    p.add_argument(
        "--isaac-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ISAAC_SIM_ROOT", "/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0"
            )
        ),
    )
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--server-timeout", type=float, default=600.0)
    p.add_argument("--rollout-timeout", type=float, default=1800.0)
    p.add_argument("--dry-run", action="store_true")
    return p


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_port(port):
    if not 1024 <= port <= 65535 or 7777 <= port <= 7783:
        raise ValueError("Choose a dedicated port outside reserved rig ports 7777–7783")
    # Read-only ownership check. The server also binds before allocating a model
    # so a racing owner makes server startup fail, never replace that process.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"Port {port} is occupied; no process was signaled"
            ) from error


def make_plan(args):
    for name in ("source", "live_repo", "runtime", "episode", "output", "isaac_root"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output must be new: {args.output}")
    if args.episode.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be outside the prepared source episode")
    if (
        not 0 < args.duration < float("inf")
        or not 0 < args.server_timeout < float("inf")
        or not 0 < args.rollout_timeout < float("inf")
    ):
        raise ValueError("Duration and timeouts must be finite and positive")
    check_port(args.port)
    debug = args.runtime.parent
    paths = {
        "scene": args.scene_config or args.runtime / "waffles_teacher_post_hand.json",
        "inference": args.runtime / "nfe1_k4.json",
        "hardware": args.runtime / "hardware_campaign.yaml",
        "veto": args.runtime / f"veto_{args.veto_version}.json",
        "initial_state": args.initial_state
        or debug / "sept4_policy_initial_state.json",
        "tactile_baseline": args.tactile_baseline
        or debug / "sept4_no_contact_sensor_baseline.npz",
        "checkpoint": args.live_repo / "runs/teacher_v5_ftA/teacher_001500.pt",
        # Do not resolve this symlink: resolving .venv/bin/python bypasses venv discovery.
        "server_python": args.live_repo / ".venv/bin/python",
        "server_script": args.source / "tools/sim/policy_server.py",
        "isaac_launcher": args.source / "tools/sim/launch_waffles.sh",
        "isaac_runner": args.source / "tools/sim/run_waffles.py",
        "isaac_python": args.isaac_root / "kit/python/bin/python3",
        "isaac_setup": args.isaac_root / "setup_python_env.sh",
        "episode_replay": args.episode / "replay.npz",
        "episode_manifest": args.episode / "manifest.json",
    }
    if args.robot_usd:
        paths["robot_usd"] = args.robot_usd
    if args.placement_release_config:
        paths["placement_release"] = args.placement_release_config
    paths = {name: path.expanduser().absolute() for name, path in paths.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    for name in ("server_python", "isaac_python", "isaac_launcher"):
        if not os.access(paths[name], os.X_OK):
            raise ValueError(f"Not executable: {paths[name]}")
    settings = json.loads(paths["inference"].read_text())
    if settings != SETTINGS:
        raise ValueError(
            f"Expected explicit native NFE1/K4 settings {SETTINGS}; got {settings}"
        )
    veto = json.loads(paths["veto"].read_text())
    if (
        veto.get("implementation") != args.veto_version
        or not {"z_ref", "z_floor", "z_margin", "open_aperture"}
        <= veto.get("config", {}).keys()
    ):
        raise ValueError(
            "Veto file must match selected version and explicitly specify task geometry/aperture"
        )
    inputs = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
        if name not in ("server_python", "isaac_python")
    }
    if inputs["checkpoint"]["sha256"] != TEACHER_SHA256:
        raise ValueError("Checkpoint is not the audited ftA teacher step1500 artifact")
    server = [
        str(paths["server_python"]),
        str(paths["server_script"]),
        "--repo",
        str(args.live_repo),
        "--ckpt",
        str(paths["checkpoint"]),
        "--expected-sha256",
        TEACHER_SHA256,
        "--system",
        "teacher",
        "--device",
        "cuda",
        "--port",
        str(args.port),
        "--out",
        str(args.output / "server"),
        "--hardware",
        str(paths["hardware"]),
        "--nfe",
        "1",
        "--k-seeds",
        "4",
        "--guidance",
        "1.0",
        "--task-text",
        "waffles",
        "--parity-fixes",
        "--persistent-noise",
    ]
    rollout = [
        str(paths["isaac_launcher"]),
        "--mode",
        "policy",
        "--episode",
        str(args.episode),
        "--output",
        str(args.output / "rollout"),
        "--config",
        str(paths["scene"]),
        "--duration",
        str(args.duration),
        "--seed",
        str(args.seed),
        "--policy-server",
        f"127.0.0.1:{args.port}",
        "--policy-mode",
        "teacher",
        "--policy-config",
        str(paths["inference"]),
        "--hardware-config",
        str(paths["hardware"]),
        "--ignore-episode-overrides",
        "--max-play-steps",
        "10",
        "--save-policy-observations",
        "--policy-initial-state",
        str(paths["initial_state"]),
        "--terminal-veto-config",
        str(paths["veto"]),
        "--tactile",
        "measured_baseline_proxy",
        "--tactile-baseline",
        str(paths["tactile_baseline"]),
        "--gel-contact-coverage",
        args.coverage,
        "--wrist",
        "contact_proxy",
        "--skip-stage-export",
    ]
    if args.robot_usd:
        rollout += ["--robot-usd", str(paths["robot_usd"])]
    if args.placement_release_config:
        rollout += ["--placement-release-config", str(paths["placement_release"])]
    if args.record_packet_support:
        rollout += ["--record-packet-support"]
    return {
        "schema_version": 1,
        "status": "planned",
        "seed": args.seed,
        "coverage": args.coverage,
        "veto_version": args.veto_version,
        "checkpoint_weights": "EMA",
        "settings": SETTINGS,
        "environment": {"ISAAC_SIM_ROOT": str(args.isaac_root)},
        "commands": {"server": server, "rollout": rollout},
        "command_display": {
            "server": shlex.join(server),
            "rollout": shlex.join(rollout),
        },
        "inputs": inputs,
        "cwd": {"server": str(args.live_repo), "rollout": str(args.source)},
        "output": str(args.output),
        "hardware_launcher_used": False,
        "robot_import": "explicit cached articulation"
        if args.robot_usd
        else "pinned source URDF imported by harness",
        "interpretation": "Diagnostic only. Runtime completion is not a claim of acquisition, retention or task success; inspect recorded physics and policy traces.",
    }


def wait_ready(process, directory, timeout):
    deadline = time.monotonic() + timeout
    next_message = time.monotonic()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Owned policy server exited {process.returncode}; see {directory / 'server.log'}"
            )
        marker = directory / "ready.json"
        if marker.exists():
            ready = json.loads(marker.read_text())
            if (
                ready.get("pid") != process.pid
                or ready.get("checkpoint_sha256") != TEACHER_SHA256
                or ready.get("status") != "ready"
                or not ready.get("warmed")
            ):
                raise RuntimeError(
                    "Ready marker does not identify the owned warmed teacher"
                )
            return ready
        if time.monotonic() >= next_message:
            print(f"Waiting for owned teacher server PID {process.pid}", flush=True)
            next_message = time.monotonic() + 30
        time.sleep(0.25)
    raise TimeoutError(f"Owned teacher server did not become ready within {timeout}s")


def stop_owned(process, timeout=20):
    """Only called with this launcher's Popen handles; never look up foreign PIDs."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=timeout)
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def capture_versions(args):
    result = {"controller_python": sys.version, "platform": sys.platform}
    for name, path in (("source", args.source), ("live_repo", args.live_repo)):
        result[name] = {}
        for key, git_args in (
            ("head", ["rev-parse", "HEAD"]),
            ("status", ["status", "--porcelain=v1", "--untracked-files=normal"]),
        ):
            proc = subprocess.run(
                ["git", "-C", str(path), *git_args],
                capture_output=True,
                text=True,
                check=False,
            )
            result[name][key] = proc.stdout.strip() if proc.returncode == 0 else None
    return result


def run_managed(args, plan):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "server").mkdir()
    record = dict(plan)
    record.update(
        status="starting", started_unix_s=time.time(), versions=capture_versions(args)
    )
    record["source_files_sha256"] = {
        str(path.relative_to(args.source)): sha256(path)
        for directory in (
            args.source / "phantom/sim",
            args.source / "tools/sim",
            args.source / "assets/sim",
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    ledger = args.output / "managed.json"
    write_json(ledger, record)
    server = child = None
    old_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        check_port(args.port)
        with (args.output / "server/server.log").open("w") as server_log:
            server = subprocess.Popen(
                plan["commands"]["server"],
                cwd=args.live_repo,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["owned_server_pid"] = server.pid
            write_json(ledger, record)
            record["server_ready"] = wait_ready(
                server, args.output / "server", args.server_timeout
            )
            ready = record["server_ready"]
            if (
                ready.get("effective") != SETTINGS
                or ready.get("hardware_sha256") != plan["inputs"]["hardware"]["sha256"]
                or ready.get("system") != "teacher"
                or ready.get("weights") != "EMA"
                or ready.get("port") != args.port
            ):
                raise RuntimeError(
                    "Owned server settings or hardware differ from the reviewed plan"
                )
            record["status"] = "running"
            write_json(ledger, record)
            with (args.output / "isaac.log").open("w") as log:
                child = subprocess.Popen(
                    plan["commands"]["rollout"],
                    cwd=args.source,
                    env={**os.environ, **plan["environment"]},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                record["owned_isaac_pid"] = child.pid
                write_json(ledger, record)
                record["rollout_exit_code"] = child.wait(timeout=args.rollout_timeout)
            if record["rollout_exit_code"] != 0:
                raise RuntimeError(
                    f"Isaac exited {record['rollout_exit_code']}; see {args.output / 'isaac.log'}"
                )
            if (
                not (args.output / "rollout/run.json").is_file()
                or not (args.output / "rollout/sim_trace.npz").is_file()
            ):
                raise RuntimeError(
                    "Isaac returned zero without completed run/trace artifacts"
                )
            record["status"] = "runtime_completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        for name, process in (("isaac", child), ("server", server)):
            try:
                stop_owned(process)
            except (OSError, subprocess.TimeoutExpired) as error:
                cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
        signal.signal(signal.SIGTERM, old_term)
        record["finished_unix_s"] = time.time()
        record["cleanup"] = {
            "owned_server_exit_code": server.poll() if server else None,
            "owned_isaac_exit_code": child.poll() if child else None,
            "errors": cleanup_errors,
            "foreign_processes_signaled": False,
        }
        if cleanup_errors:
            record["status"] = "cleanup_failed"
        write_json(ledger, record)
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
    return record


def main():
    args = parser().parse_args()
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    result = run_managed(args, plan)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output"],
                "cleanup": result["cleanup"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
