#!/usr/bin/env python3
"""One-shot v5 screen -> reserved confirmation transition; root launches this.

Default is a CPU-only binding check. --execute authorizes the already-declared
confirmation, after the exact owned screen exits and all24 cases pass auditing.
No resume, trial replacement, process killing, or automatic arm extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
ROOT = BASE / "runs/teacher_success_anchor_v5"
DRIVER = BASE / "source_teacher_anchor_driver_v5"
REVIEW = BASE / "review_teacher_anchor_v5"
PYTHON = "/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python"
PROTOCOL = DRIVER / "tests/fixtures/reference/teacher_success_anchor_v5/protocol.json"
SCREEN = DRIVER / "configs/sim/teacher_success_anchor_v5_screen.json"
PID = 2306707
BOOT_ID = "9b1fae31-5718-4674-bf3b-f3b413a89cc8"
START_TICKS = "102173374"
PINS = {
    PROTOCOL: "f67fe79a5267c6d253d7d90f67f2a367c8899f030959c8f42fa714d072ab7ac2",
    SCREEN: "3392e9f703ef008f2a16d6c133870192ddb1642526ef7adb3702b2e8a6c9b9c5",
    ROOT
    / "screen_launcher.json": "0e50d01e8224ef4f41d5a696895a2f7072f8a22934801e52adce12c85382e578",
    ROOT
    / "external_driver_manifest.json": "fc21a40ce032f0d8a0c5ae9246b6013733c88af25fb0258654ab1906b78448a2",
    ROOT
    / "screen/frozen_inputs.json": "020716a74584897ddc798a039c055384943828d45705165b6e80009e6be8ad0a",
    REVIEW
    / "review_manifest.json": "25e342f304ab0a3b81ffcb5ca8cd24dd4657cf1560aef2e0e379429dd8e3d45f",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def process_identity(pid, proc=Path("/proc")):
    directory = proc / str(pid)
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    except FileNotFoundError:
        return None
    return {
        "pid": pid,
        "start_ticks": fields[19],
        "state": fields[0],
        "argv": [a.decode() for a in argv] if argv != [b""] else [],
    }


def screen_running(expected_argv, proc=Path("/proc")):
    if (proc / "sys/kernel/random/boot_id").read_text().strip() != BOOT_ID:
        raise RuntimeError("Machine rebooted since captured screen identity")
    current = process_identity(PID, proc)
    if current is None:
        return False
    if current["start_ticks"] != START_TICKS:
        raise RuntimeError("Screen PID was reused; refusing ambiguous handoff")
    if current["state"] == "Z":
        return False
    if current["argv"] != expected_argv:
        raise RuntimeError("Screen process argv changed")
    return True


def verify_static():
    checks = []

    def check(path, expected):
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"Frozen hash mismatch: {path}")
        checks.append({"path": str(path), "sha256": actual})

    for path, expected in PINS.items():
        check(path, expected)
    design = read(SCREEN)
    rows = design["runtime_contract"]["source_and_input_hashes"]
    if len(rows) != 120:
        raise RuntimeError("Expected exactly120 admitted runtime/input bindings")
    for row in rows:
        check(row["path"], row["sha256"])
    for row in design["policies"]:
        check(row["checkpoint"], row["checkpoint_sha256"])
    manifest = read(ROOT / "external_driver_manifest.json")
    if manifest["source"] != str(DRIVER) or len(manifest["files"]) != 343:
        raise RuntimeError("External driver manifest identity/count changed")
    for name, expected in manifest["files"].items():
        check(DRIVER / name, expected)
    manifest = read(REVIEW / "review_manifest.json")
    for name, expected in manifest["source_sha256"].items():
        check(REVIEW / name, expected)
    for name, target in manifest["read_only_import_links"].items():
        if (REVIEW / name).resolve() != Path(target):
            raise RuntimeError(f"Review import link changed: {name}")
    frozen = read(ROOT / "screen/frozen_inputs.json")
    for field, root_field in (
        ("source_sha256", "source_root"),
        ("external_controller_source_sha256", "external_controller_source_root"),
        ("live_core_sha256", "live_repository"),
        ("episode_sha256", "prepared_episode"),
    ):
        for name, expected in frozen[field].items():
            check(Path(frozen[root_field]) / name, expected)
    return {
        "status": "passed",
        "checks": checks,
        "runtime_input_count": 120,
        "driver_file_count": 343,
        "review_module_count": len(manifest["source_sha256"]),
    }


def trial_keys(design):
    return {
        f"{p['id']}__{c['id']}__seed{s}"
        for p in design["policies"]
        for c in design["conditions"]
        for s in design["sampling_seeds"]
    }


def require_complete(directory, design, design_sha, controller_pid):
    expected = trial_keys(design)
    progress = read(directory / "progress.json")
    if (
        len(expected) != 24
        or progress.get("status") != "all_trials_completed"
        or progress.get("campaign_sha256") != design_sha
        or progress.get("controller_pid") != controller_pid
        or set(progress.get("trials", {})) != expected
    ):
        raise RuntimeError("Incomplete/unmatched controller ledger")
    folders = {p.name for p in (directory / "rollouts").iterdir() if p.is_dir()}
    if folders != expected:
        raise RuntimeError("Unexpected or missing raw rollout directories")
    for key in sorted(expected):
        status = read(directory / "rollouts" / key / "run_status.json")
        ledger = progress["trials"][key]
        if any(
            row.get("status") != "completed"
            or row.get("exit_code") != 0
            or row.get("reused", False)
            for row in (status, ledger)
        ):
            raise RuntimeError(f"Failed, reused or incomplete attempt: {key}")
        score = read(directory / "analysis/trials" / (key + ".json"))
        if (
            score.get("status") != "scored"
            or score.get("metrics", {}).get("valid_for_scoring") is not True
            or score["metrics"].get("thresholds") != design["thresholds"]
            or score.get("case", {}).get("campaign_sha256") != design_sha
        ):
            raise RuntimeError(f"Invalid or unbound authoritative score: {key}")
    return {"status": "passed", "completed_valid_trials": len(expected)}


def require_quiescent(directory):
    for p in (directory / "servers").glob("*/server.json"):
        data = read(p)
        if (
            data.get("dedicated_simulation_server") is not True
            or data.get("port") != 7799
        ):
            raise RuntimeError(f"Unexpected owned-server identity: {p}")
        current = process_identity(int(data["pid"]))
        if current is not None and current["state"] != "Z":
            raise RuntimeError(f"Archived owned server PID still exists: {data['pid']}")
    with socket.socket() as sock:
        # A closed prior server may leave TCP TIME_WAIT sockets; a live listener
        # still prevents this bind. No connection, listen or model RPC is made.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 7799))


def require_selection(path, campaign_sha, stage):
    selected = read(path)
    if (
        selected.get("status") != "complete_valid_matched_stage"
        or selected.get("expected_trials") != 24
        or selected.get("available_trials") != 24
        or selected.get("missing_trial_keys")
        or selected.get("invalid_trial_keys")
        or selected.get("campaign_sha256") != campaign_sha
        or selected.get("protocol_sha256") != PINS[PROTOCOL]
        or selected.get("stage") != stage
    ):
        raise RuntimeError("Selection is incomplete, invalid or hash-unbound")
    return selected


def confirmation_command(original, campaign, output):
    command = list(original)
    for flag, value in (("--campaign", campaign), ("--output", output)):
        if command.count(flag) != 1:
            raise RuntimeError(f"Ambiguous original launch argument: {flag}")
        command[command.index(flag) + 1] = str(value)
    if command.count("--execute") != 1:
        raise RuntimeError("Expected exact original executed driver command")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Wait and execute the reserved24 only"
    )
    args = parser.parse_args()
    audit = verify_static()
    original = read(ROOT / "screen_launcher.json")["command"]
    running = screen_running(original)
    if not args.execute:
        print(
            json.dumps(
                {
                    "static": audit["status"],
                    "screen_running": running,
                    "runtime_inputs": 120,
                    "driver_files": 343,
                    "actions": "none; omit --execute for checks only",
                }
            ),
            flush=True,
        )
        return 0
    work = ROOT / "transition"
    work.mkdir(exist_ok=False)
    ledger = {
        "schema_version": 1,
        "wrapper_pid": os.getpid(),
        "wrapper_sha256": sha(__file__),
        "screen_identity": {
            "pid": PID,
            "boot_id": BOOT_ID,
            "start_ticks": START_TICKS,
            "argv": original,
        },
        "events": [],
        "status": "waiting_screen",
    }
    deadline = time.monotonic() + 6 * 3600
    child = None

    def event(stage, **values):
        row = {
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **values,
        }
        ledger["events"].append(row)
        ledger["status"] = stage
        temporary = work / "ledger.tmp"
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(work / "ledger.json")
        print(json.dumps(row), flush=True)

    def check_deadline():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Six-hour transition deadline exceeded; no processes killed"
            )

    def run(label, command):
        nonlocal child
        check_deadline()
        event(label, command=command)
        with (work / f"{label}.log").open("x") as log:
            child = subprocess.Popen(
                command,
                cwd=DRIVER,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            event(label + "_started", child_pid=child.pid)
            while child.poll() is None:
                check_deadline()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    event(label + "_running", child_pid=child.pid)
            code, pid = child.returncode, child.pid
        child = None
        event(label + "_exited", exit_code=code, child_pid=pid)
        if code != 0:
            raise RuntimeError(f"{label} failed with exit{code}; no retry")
        return pid

    def interrupted(signum, _frame):
        raise InterruptedError(
            f"Wrapper interrupted by signal{signum}; no child killed"
        )

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        write_new(work / "admission_before_wait.json", audit)
        event("waiting_screen")
        while screen_running(original):
            check_deadline()
            progress = read(ROOT / "screen/progress.json")
            if progress.get("status") in ("error", "interrupted"):
                raise RuntimeError("Screen reports infrastructure/controller failure")
            event(
                "waiting_screen",
                completed=sum(
                    r.get("status") == "completed"
                    for r in progress.get("trials", {}).values()
                ),
            )
            time.sleep(30)
        design = read(SCREEN)
        event(
            "screen_complete",
            **require_complete(ROOT / "screen", design, PINS[SCREEN], PID),
        )
        require_quiescent(ROOT / "screen")
        for p in (
            ROOT / "screen/selection",
            ROOT / "confirmation_campaign.json",
            ROOT / "confirmation",
            ROOT / "confirmation_plan",
        ):
            if p.exists():
                raise FileExistsError(
                    f"Preserve existing selection/attempt; refusing overwrite: {p}"
                )
        write_new(work / "admission_after_screen.json", verify_static())
        helper = [PYTHON, str(REVIEW / "tools/sim/teacher_anchor_compare.py")]
        common = ["--protocol", str(PROTOCOL)]
        screen_path = ROOT / "screen/campaign_snapshot.json"
        if sha(screen_path) != PINS[SCREEN]:
            raise RuntimeError("Screen snapshot changed")
        score_path = ROOT / "screen/selection/selection.json"
        run(
            "score_screen",
            helper
            + [
                "score",
                *common,
                "--campaign",
                str(screen_path),
                "--runs",
                str(ROOT / "screen/rollouts"),
                "--out",
                str(score_path.parent),
                "--stage",
                "screen",
            ],
        )
        screen_selection = require_selection(score_path, PINS[SCREEN], "screen")
        derived = ROOT / "confirmation_campaign.json"
        run(
            "derive_confirmation",
            helper
            + [
                "confirm",
                *common,
                "--campaign",
                str(screen_path),
                "--selection",
                str(score_path),
                "--out",
                str(derived),
            ],
        )
        derived_sha = sha(derived)
        confirm_design = read(derived)
        command = confirmation_command(original, derived, ROOT / "confirmation")
        plan_command = confirmation_command(
            original, derived, ROOT / "confirmation_plan"
        )
        plan_command.remove("--execute")
        run("plan_confirmation", plan_command)
        plan = read(ROOT / "confirmation_plan/plan.json")
        planned = {
            (b["policy_id"], c, s)
            for b in plan["checkpoint_blocks"]
            for c, s in b["trials"]
        }
        expected = {
            (p["id"], c["id"], s)
            for p in confirm_design["policies"]
            for c in confirm_design["conditions"]
            for s in confirm_design["sampling_seeds"]
        }
        if (
            plan.get("planned_trials") != 24
            or planned != expected
            or sum(len(b["trials"]) for b in plan["checkpoint_blocks"]) != 24
            or plan.get("campaign_sha256") != derived_sha
            or plan.get("executing") is not False
            or confirm_design["sampling_seeds"] != list(range(904501, 904513))
            or [p["id"] for p in confirm_design["policies"]]
            != screen_selection["selected_ids"]
        ):
            raise RuntimeError("Confirmation plan differs from reserved matched grid")
        write_new(work / "admission_before_confirmation.json", verify_static())
        require_quiescent(ROOT / "screen")
        if (ROOT / "confirmation").exists() or sha(derived) != derived_sha:
            raise RuntimeError(
                "Confirmation output appeared or design changed before launch"
            )
        confirm_pid = run("execute_confirmation", command)
        event(
            "confirmation_complete",
            **require_complete(
                ROOT / "confirmation", confirm_design, derived_sha, confirm_pid
            ),
        )
        require_quiescent(ROOT / "confirmation")
        write_new(work / "admission_after_confirmation.json", verify_static())
        result_path = ROOT / "confirmation/selection/selection.json"
        if result_path.parent.exists():
            raise FileExistsError(
                "Refusing to overwrite existing confirmation selection"
            )
        run(
            "score_confirmation",
            helper
            + [
                "score",
                *common,
                "--campaign",
                str(derived),
                "--runs",
                str(ROOT / "confirmation/rollouts"),
                "--out",
                str(result_path.parent),
                "--stage",
                "confirmation",
            ],
        )
        result = require_selection(result_path, derived_sha, "confirmation")
        event(
            "study_complete",
            screen_trials=24,
            confirmation_trials=24,
            selected_ids=screen_selection["selected_ids"],
            clear_simulator_winner=result["clear_simulator_winner"],
            conclusion=result["conclusion"],
            arm_extension_executed=False,
        )
        write_new(work / "final_status.json", ledger)
        return 0
    except BaseException as error:
        event(
            "aborted",
            error=f"{type(error).__name__}: {error}",
            outstanding_child_pid=child.pid
            if child is not None and child.poll() is None
            else None,
            processes_killed=False,
            retry_permitted=False,
        )
        write_new(work / "final_status.json", ledger)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
