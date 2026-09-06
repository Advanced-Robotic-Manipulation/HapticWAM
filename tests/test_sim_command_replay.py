import json

import numpy as np
import pytest

from phantom.sim.command_replay import RecordedDriveCommands


def trace(tmp_path, rows):
    path = tmp_path / "execution.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    return path


def row(t, q, fingers):
    return {
        "t": t,
        "target_q": [q] * 6,
        "target_finger_q": [fingers] * 2,
        "status": "drive_submitted",
    }


def test_causal_commands_keep_native_times_at_different_physics_rates(tmp_path):
    commands = RecordedDriveCommands(
        trace(tmp_path, [row(0.004, 1, 0.03), row(0.012, 2, 0.02)]),
        finger_limit_m=0.035,
    )
    assert commands.at(0) is None
    assert commands.at(0.003) is None
    for t in [0.004, 0.005, 0.006, 0.008, 0.011]:
        q, fingers = commands.at(t)
        np.testing.assert_array_equal(q, [1] * 6)
        np.testing.assert_array_equal(fingers, [0.03] * 2)
    for t in [0.012, 0.013, 5.0]:
        np.testing.assert_array_equal(commands.at(t)[0], [2] * 6)
    commands.at(0.012)[0][:] = 99
    np.testing.assert_array_equal(commands.at(0.012)[0], [2] * 6)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [row(0.004, 1, 0.03), row(0.004, 2, 0.02)],
        [row(0.004, float("nan"), 0.03)],
        [row(0.004, 1, 0.04)],
        [{**row(0.004, 1, 0.03), "status": "proposal"}],
    ],
)
def test_invalid_or_unsubmitted_commands_cannot_be_mechanics_truth(tmp_path, rows):
    with pytest.raises(ValueError):
        RecordedDriveCommands(trace(tmp_path, rows), finger_limit_m=0.035)
