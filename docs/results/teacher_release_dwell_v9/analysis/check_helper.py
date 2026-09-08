#!/usr/bin/env python3
"""CPU guards for V9 audit discovery/design; no recorded trial is analyzed."""

import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def rejected(call, label):
    try:
        call()
    except ValueError:
        return label
    raise AssertionError("Unexpected audit acceptance: " + label)


root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v9_audit_checked", root / "audit_v9.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
campaigns = root.parent / "launchplan/campaigns"
designs = [
    json.loads((campaigns / name).read_text())
    for name in ("control_dwell200.json", "treatment_dwell100.json")
]
assert helper.design_diff(designs)["treatment_s"] == 0.1
passed = ["actual frozen dwell-only designs accepted"]
wrong = copy.deepcopy(designs)
wrong[1]["adapter_profile"]["placement_release"]["open_command_max"] = 0.5
passed.append(
    rejected(lambda: helper.design_diff(wrong), "changed opening threshold rejected")
)
wrong_physics = copy.deepcopy(designs)
wrong_physics[1]["nominal_scene"]["waffle"]["mass"] += 0.001
passed.append(
    rejected(
        lambda: helper.design_diff(wrong_physics), "changed physical mass rejected"
    )
)
with TemporaryDirectory(prefix="v9_audit_guard_") as name:
    folder = Path(name)
    (folder / "launch_plan.json").write_text(json.dumps({"plans": []}))
    (folder / "complete.json").write_text(
        json.dumps({"status": "running", "trials": 3})
    )
    passed.append(
        rejected(
            lambda: helper.discover(folder),
            "incomplete campaign rejected before raw reads",
        )
    )
    (folder / "complete.json").write_text(
        json.dumps({"status": "four_valid_trials_completed", "trials": 4})
    )
    passed.append(
        rejected(lambda: helper.discover(folder), "wrong block denominator rejected")
    )
print(json.dumps({"status": "passed", "checks": passed}, indent=2))
