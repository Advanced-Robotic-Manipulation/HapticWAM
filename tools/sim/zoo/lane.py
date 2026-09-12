#!/usr/bin/env python3
"""One execution lane of a sim zoo study on compute3: owned servers, Isaac trials, scoring.

Runs its lane's trials in study order, grouped by (model, recipe): one owned
policy server per group (PHANTOM ``tools/sim/policy_server.py`` or the LeRobot
``phantom.scripts.lerobot_server`` for pi0.5), one Isaac process per trial via
the frozen runtime's ``launch_waffles.sh``, then ``score_trial.py``. Every trial
directory is preserved; a failed trial is recorded, never retried.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ISAAC = "/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0"
# the pi0.5 adapter (phantom.scripts.lerobot_server) lives on main / the box checkout, not in the
# frozen v10-based sim runtime; the server is only an RPC peer, so it runs from the box checkout
# exactly as serve_bg.sh does. Nothing there is modified.
BOX_REPO = "/home/physicalai/phantom-icra-2027/phantom"
PHANTOM_PY = "/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python"
PI05_PY = "/home/physicalai/phantom-icra-2027/pi05venv/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def port_free(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def gpu_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


class Server:
    def __init__(self, study, model, recipe, port, directory):
        self.study, self.model, self.recipe, self.port, self.dir = study, model, recipe, port, Path(directory)
        self.proc = None

    def command(self):
        m, r, runtime = self.model, self.recipe, self.study["runtime"]
        if m["kind"] == "lerobot":
            return [PI05_PY, "-m", "phantom.scripts.lerobot_server", "--ckpt", m["checkpoint"], "--port", str(self.port),
                    "--hardware", self.study["inputs"] + "/hardware_input.yaml", "--policy-type", "pi05",
                    "--device", "cuda", "--action-space", "delta", "--image-size", "224", "--task", r["task_text"]]
        cmd = [PHANTOM_PY, runtime + "/tools/sim/policy_server.py", "--repo", runtime, "--ckpt", m["checkpoint"],
               "--expected-sha256", m["sha256"], "--system", m["system"], "--port", str(self.port),
               "--out", str(self.dir), "--hardware", self.study["inputs"] + "/hardware_input.yaml",
               "--nfe", str(r["nfe"]), "--guidance", str(r["guidance"]), "--k-seeds", str(r["k_seeds"]),
               "--task-text", r["task_text"],
               "--parity-fixes" if r["parity_fixes"] else "--no-parity-fixes",
               "--persistent-noise" if r["persistent_noise"] else "--no-persistent-noise"]
        if r.get("action_time_origin") == "observation":
            cmd += ["--action-time-origin", "observation"]
        # inference-latency levers under test (policy_server --compile / --flex); recorded in the
        # server's info and therefore in every trial's policy_info.json
        cmd += list(r.get("server_extra_args", []))
        if not r.get("use_ema", True):
            cmd.append("--raw")
        return cmd

    def start(self, timeout_s=900):
        self.dir.mkdir(parents=True, exist_ok=True)
        if not port_free(self.port):
            raise RuntimeError(f"port {self.port} busy")
        root = BOX_REPO if self.model["kind"] == "lerobot" else self.study["runtime"]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=root)
        log = (self.dir / "server_console.log").open("a")
        self.proc = subprocess.Popen(self.command(), cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        write(self.dir / "owned_pid.json", {"pid": self.proc.pid, "command": self.command(), "started_unix_s": time.time()})
        deadline = time.monotonic() + timeout_s
        ready_file = self.dir / "ready.json"
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early (rc={self.proc.returncode}); see {self.dir}/server_console.log")
            if self.model["kind"] == "lerobot":
                text = (self.dir / "server_console.log").read_text(errors="replace")
                if "\nREADY " in text or text.startswith("READY "):
                    break
                if "Traceback" in text or "CUDA out of memory" in text:
                    raise RuntimeError("lerobot server failed; see server_console.log")
            elif ready_file.exists():
                break
            if time.monotonic() > deadline:
                raise RuntimeError("server startup timed out")
            time.sleep(2)
        write(self.dir / "ready_snapshot.json", read(ready_file) if ready_file.exists() else {"kind": "lerobot", "port": self.port})

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=15)
        write(self.dir / "server_exit.json", {"returncode": self.proc.returncode if self.proc else None, "unix_s": time.time()})


def trial_command(study, trial, model, recipe, port, out_dir):
    inputs, fixed = study["inputs"], study["fixed"]
    cmd = [study["runtime"] + "/tools/sim/launch_waffles.sh", "--mode", "policy",
           "--episode", study["prepared_episode"], "--output", str(out_dir), "--config", inputs + "/scene.json",
           "--duration", str(fixed["horizon_s"]), "--seed", str(trial["seed"]),
           "--policy-server", f"127.0.0.1:{port}", "--policy-mode", model["policy_mode"],
           "--policy-config", f"{inputs}/policy_config__{trial['recipe']}.json",
           "--hardware-config", inputs + "/hardware_input.yaml", "--ignore-episode-overrides",
           "--max-play-steps", str(recipe.get("max_play_steps", fixed["max_play_steps"])),
           "--grip-play-steps", str(recipe.get("grip_play_steps", fixed["grip_play_steps"])),
           "--observation-delay-s", "0.0", "--inference-delay-add-s", "0.0",
           "--tactile", "measured_baseline_proxy", "--tactile-baseline", study["tactile_baseline"],
           "--wrist", "gripper_contact_proxy", "--gel-contact-coverage", "manifold_patch_v2",
           "--skip-stage-export", "--policy-initial-state", f"{inputs}/initial_states/{trial['start']}.json",
           "--policy-delivery-clock", "rpc_wall",
           "--placement-release-config", inputs + "/" + recipe.get("placement_release_config", "placement_release.json"),
           *(["--gripper-max-close-cmd", str(recipe["gripper_max_close_cmd"])] if recipe.get("gripper_max_close_cmd") is not None else []),
           "--boundary-projection-config", inputs + "/" + recipe.get("boundary_projection_config", "boundary_projection.json"),
           "--placement-controller-profile", "minimal_v5",
           "--servo-reach-limiter", "--servo-constraint-hold-s", str(fixed["servo_constraint_hold_s"]),
           "--experimental-adaptive-policy", "--save-policy-observations", "--record-gel-contacts",
           "--record-packet-support", "--record-robot-environment-contacts",
           "--no-progress-stop-s", str(fixed["no_progress_stop_s"])]
    if model["kind"] == "lerobot":
        # no ACC head: neither the terminal veto nor the minimal_v5 port (both
        # consume p_evt) can run; the rig's pi0.5 preset makes the same choice
        cmd = [c for c in cmd if c not in ("--placement-controller-profile", "minimal_v5")]
    elif recipe.get("terminal_veto", True):
        cmd += ["--terminal-veto-config", inputs + "/terminal_veto.json"]
    if fixed.get("robot_usd"):
        cmd += ["--robot-usd", fixed["robot_usd"]]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rollout-timeout", type=float, default=2400)
    parser.add_argument("--only", nargs="*", help="restrict to these trial ids")
    args = parser.parse_args()
    study = read(args.study)
    raw = Path(study["raw"])
    lane_dir = raw / f"lane{args.lane}"
    lane_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = lane_dir / "ledger.json"
    ledger = read(ledger_path) if ledger_path.exists() else {"lane": args.lane, "port": args.port, "trials": {}}
    trials = [t for t in study["trials"] if t["lane"] == args.lane and (not args.only or t["id"] in args.only)]
    groups = []
    for t in trials:
        key = (t["model"], t["recipe"])
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(t)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for (model_id, recipe_id), members in groups:
        model, recipe = study["models"][model_id], study["recipes"][recipe_id]
        server = Server(study, model, recipe, args.port, lane_dir / f"server__{model_id}__{recipe_id}")
        try:
            server.start()
        except Exception as error:  # noqa: BLE001 - preserved, never retried
            for t in members:
                ledger["trials"][t["id"]] = {"status": "server_failed", "error": repr(error)}
            write(ledger_path, ledger)
            server.stop()
            continue
        try:
            for t in members:
                out_dir = raw / "rollouts" / t["id"]
                if out_dir.exists():
                    ledger["trials"][t["id"]] = ledger["trials"].get(t["id"], {"status": "preexisting_preserved"})
                    write(ledger_path, ledger)
                    continue
                out_dir.mkdir(parents=True)
                cmd = trial_command(study, t, model, recipe, args.port, out_dir)
                status = {"status": "running", "command": cmd, "started_unix_s": time.time(),
                          "gpu_pids_at_start": gpu_pids(), "server_pid": server.proc.pid}
                write(out_dir / "run_status.json", status)
                ledger["trials"][t["id"]] = status
                write(ledger_path, ledger)
                with (out_dir / "isaac.log").open("w") as log:
                    child = subprocess.Popen(cmd, cwd=study["runtime"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    try:
                        code = child.wait(timeout=args.rollout_timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        code = "timeout"
                status.update(status="finished" if code == 0 else "runtime_error", exit_code=code, finished_unix_s=time.time())
                write(out_dir / "run_status.json", status)
                reference = (study.get("latency_reference_p95_s") or {}).get(recipe_id)
                score = subprocess.run([PHANTOM_PY, study["runtime"] + "/tools/sim/zoo/score_trial.py", "--trial", str(out_dir),
                                        "--thresholds", study["inputs"] + "/thresholds.json", "--runtime", study["runtime"]]
                                       + (["--latency-reference-p95", str(reference)] if reference else []),
                                       env=dict(env, CUDA_VISIBLE_DEVICES="", PYTHONPATH=study["runtime"]),
                                       capture_output=True, text=True)
                status["score_stdout"] = score.stdout[-600:]
                status["score_rc"] = score.returncode
                if score.returncode:
                    status["score_stderr"] = score.stderr[-1500:]
                # regenerable per-trial USD copy (33 MB) is dropped to keep RAM for trials
                shutil.rmtree(out_dir / "robot_asset", ignore_errors=True)
                result_path = out_dir / "trial_result.json"
                if result_path.exists():
                    r = read(result_path)
                    status.update(stage=r.get("stage"), stage_name=r.get("stage_name"), stop_reason=r.get("stop_reason"),
                                  score_status=r.get("status"), latency_confounded=(r.get("latency") or {}).get("latency_confounded"))
                write(out_dir / "run_status.json", status)
                ledger["trials"][t["id"]] = status
                write(ledger_path, ledger)
                print(json.dumps({k: status.get(k) for k in ("status", "stage_name", "stop_reason", "latency_confounded")}) + " " + t["id"], flush=True)
        finally:
            server.stop()
    ledger["completed_unix_s"] = time.time()
    write(ledger_path, ledger)
    print("lane done", flush=True)


if __name__ == "__main__":
    main()
