#!/usr/bin/env python3
"""One-shot development diagnostic after completed v5 confirmation; no hardware.

Default verifies file bindings only. --execute waits for the already-running
v5 study, then launches exactly the two frozen limiter cells once. It never
kills a process, retries a trial, or changes a campaign.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
V5 = BASE / "runs/teacher_success_anchor_v5"
ROOT = BASE / "runs/teacher_success_anchor_v6"
DRIVER = BASE / "source_teacher_anchor_driver_v6"
SOURCE = BASE / "source_teacher_anchor_limiter_v6"
PYTHON = "/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python"
CAMPAIGN = DRIVER / "configs/sim/teacher_success_anchor_v6_diagnostic.json"
PINS = {
    ROOT
    / "external_driver_manifest.json": "9c23de29bca1efb10ff44648f3bde33d044c1b2115b2984e8c51605a43c03777",
    ROOT
    / "diagnostic_source_preflight.json": "ce9b0a58157a8973b50fd7adec1709d2ba14e6ef969c586be11ab3a61778f9d1",
    CAMPAIGN: "55a5ad7852b1f08b6cce6a6d7f136454cac69504aed11ad1b441c36945aac9b6",
    ROOT
    / "diagnostic_preflight.json": "7a68986781a851bcd794d400b581eac838daeb1715c97273d78f544a86ff9412",
    ROOT
    / "source_audit/source_manifest.json": "df54897c11d4e1b6774c76e07382f67b0e9e31c0724d43a3deff50b2b13fc9b6",
    V5
    / "transition_to_confirmation.py": "95c316468c4f0f8221a529d91eea2823f73f0bed65fc04a2b9ed392b7c4be9bd",
    V5
    / "confirmation_campaign.json": "fe62f45fcaf2e71e4df890ff7303bf70a2b49e35fc2a745f6f7492a5f200b855",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify():
    checked = 0

    def check(path, expected):
        nonlocal checked
        if sha(path) != expected:
            raise RuntimeError(f"Frozen file changed: {path}")
        checked += 1

    for path, digest in PINS.items():
        check(path, digest)
    manifest = read(ROOT / "external_driver_manifest.json")
    if manifest["source"] != str(DRIVER) or len(manifest["files"]) != 344:
        raise RuntimeError("Unexpected external driver identity")
    for name, digest in manifest["files"].items():
        check(DRIVER / name, digest)
    frozen = read(ROOT / "diagnostic_source_preflight.json")
    for hashes, root in (
        ("source_sha256", "source_root"),
        ("external_controller_source_sha256", "external_controller_source_root"),
        ("live_core_sha256", "live_repository"),
        ("episode_sha256", "prepared_episode"),
    ):
        for name, digest in frozen[hashes].items():
            check(Path(frozen[root]) / name, digest)
    for item in frozen["adapter_profile_inputs"].values():
        check(item["path"], item["sha256"])
    for name in ("hardware", "robot_usd"):
        check(
            frozen["hardware_config" if name == "hardware" else name],
            frozen[name + "_sha256"],
        )
    design = read(CAMPAIGN)
    if design["sampling_seeds"] != [904301, 904302] or len(design["policies"]) != 1:
        raise RuntimeError("Unexpected development trial grid")
    for policy in design["policies"]:
        path = Path(policy["checkpoint"])
        if not path.is_absolute():
            path = Path(frozen["live_repository"]) / path
        check(path, policy["checkpoint_sha256"])
    if SOURCE != Path(frozen["source_root"]):
        raise RuntimeError("Unexpected runtime source")
    return {"status": "passed", "file_checks": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    initial = verify()
    if not args.execute:
        print(json.dumps(initial))
        return
    work = ROOT / "handoff"
    work.mkdir(exist_ok=False)
    write_new(work / "initial_admission.json", initial)
    child = None
    ledger = {"pid": os.getpid(), "source_sha256": sha(__file__), "events": []}

    def event(status, **details):
        row = {
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            **details,
        }
        ledger["status"] = status
        ledger["events"].append(row)
        temporary = work / "ledger.tmp"
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(work / "ledger.json")
        print(json.dumps(row), flush=True)

    try:
        event("waiting_v5_confirmation")
        deadline = time.monotonic() + 4 * 3600
        final = V5 / "transition/final_status.json"
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("v5 confirmation did not finish in four hours")
            try:
                completed = read(final)
            except (FileNotFoundError, json.JSONDecodeError):
                # The upstream final file is written once, without rename.
                # A visible but unfinished JSON is not a completed handoff.
                completed = None
            if completed is not None:
                break
            time.sleep(30)
        if completed.get("status") != "study_complete":
            raise RuntimeError("v5 transition did not finish successfully")
        spec = importlib.util.spec_from_file_location(
            "completed_v5_transition", V5 / "transition_to_confirmation.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.require_selection(
            V5 / "confirmation/selection/selection.json",
            PINS[V5 / "confirmation_campaign.json"],
            "confirmation",
        )
        module.require_quiescent(V5 / "confirmation")
        write_new(work / "admission_before_launch.json", verify())
        if (ROOT / "diagnostic").exists():
            raise FileExistsError("Refusing existing diagnostic output")
        command = [
            PYTHON,
            str(DRIVER / "tools/sim/run_policy_campaign.py"),
            "--campaign",
            str(CAMPAIGN),
            "--source",
            str(SOURCE),
            "--live-repo",
            "/home/physicalai/phantom-icra-2027/phantom",
            "--server-python",
            PYTHON,
            "--evidence",
            str(BASE / "evidence"),
            "--hardware-config",
            str(ROOT / "hardware_limiter.yaml"),
            "--robot-usd",
            str(
                BASE / "runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda"
            ),
            "--output",
            str(ROOT / "diagnostic"),
            "--port",
            "7799",
            "--execute",
        ]
        with (work / "execute.log").open("x") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            event(
                "diagnostic_started",
                child_pid=child.pid,
                command=command,
                completed_v5_selection_sha256=sha(
                    V5 / "confirmation/selection/selection.json"
                ),
            )
            code = child.wait()
        if code != 0:
            raise RuntimeError(f"Diagnostic driver exited {code}; outputs preserved")
        event(
            "diagnostic_driver_complete",
            child_pid=child.pid,
            exit_code=code,
            further_studies_launched=False,
        )
        write_new(work / "final_status.json", ledger)
    except BaseException as error:
        event(
            "aborted",
            error=f"{type(error).__name__}: {error}",
            outstanding_child_pid=child.pid if child and child.poll() is None else None,
            processes_killed=False,
            retry_permitted=False,
        )
        write_new(work / "final_status.json", ledger)
        raise


if __name__ == "__main__":
    main()
