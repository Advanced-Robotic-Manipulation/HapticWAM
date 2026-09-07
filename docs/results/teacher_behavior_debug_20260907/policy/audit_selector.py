#!/usr/bin/env python3
"""Exercise the saved native selector without loading Torch/models or hardware."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXPECTED_POLICY_SHA = "e4d4c0797896f72bddb0d3c7383a1ebe6c8aeca02848b4456675c4bf7c35f435"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_POLICY_SHA:
        raise ValueError(
            "Must audit the native policy source identified by saved server metadata"
        )
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PhantomPolicy"
    )
    select = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_select_seed"
    )
    method_lines = [select.lineno, select.end_lineno]
    # Execute only this NumPy selector; never import the module's model stack.
    namespace = {"np": np, "Plan": object}
    exec(
        compile(ast.Module(body=[select], type_ignores=[]), str(args.source), "exec"),
        namespace,
    )
    select = namespace["_select_seed"]
    config = SimpleNamespace(HEAD_STEPS=9, HEAD_DZ_KEEP=0.5, close_p=0.5)
    a = np.zeros((4, 16, 7))
    a[:, :, 2] = np.array([-0.001, -0.003, -0.0005, 0.001])[:, None]
    picked, descent = select(config, a, 0.1, None, 0.0, 10.0)
    assert picked == 1 and descent["k_rejected"] == 3
    previous = SimpleNamespace(
        actions=np.zeros((16, 7)), action_times=np.arange(16) / 10
    )
    previous.actions[-1, 0] = 0.01
    b = np.zeros((2, 16, 7))
    b[1, :, 0] = 0.01
    expired_pick, expired = select(config, b, 0.9, previous, 5.0, 10.0)
    assert expired_pick == 1
    c = np.zeros((2, 16, 7))
    c[0, :, 0], c[1, :, 3] = 0.02, 0.03
    previous.actions[:] = 0
    mixed_pick, mixed = select(config, c, 0.9, previous, 0.0, 10.0)
    assert mixed_pick == 0
    print(
        json.dumps(
            {
                "policy_source": str(args.source),
                "policy_sha256": digest,
                "selected_method_lines": [
                    *method_lines,
                ],
                "cpu_only_checks": {
                    "low_contact_descent_prefilter": {
                        "passed": True,
                        "diagnostics": descent,
                    },
                    "expired_reference_still_selects_last_row_match": {
                        "passed": True,
                        "last_reference_time_s": 1.5,
                        "new_grid_start_s": 5.0,
                        "picked": expired_pick,
                        "diagnostics": expired,
                        "interpretation": "Reproduces a source behavior on synthetic values; cohort occurrence checked separately, not asserted.",
                    },
                    "metres_and_radians_have_unweighted_numerical_l2": {
                        "passed": True,
                        "picked": mixed_pick,
                        "diagnostics": mixed,
                        "candidate0_increment": "20mm translation, zero rotation per step",
                        "candidate1_increment": "zero translation, .03rad rotation per step",
                        "interpretation": "Documents implicit unit weighting; does not establish which physical trajectory is preferable.",
                    },
                },
                "unidentifiable_from_saved_cohort": "All nonselected candidate XYZ/rotation/gripper arrays and candidate feasibility scores are absent; only selected actions and per-K nine-step descent are saved.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
