"""Independent v1 timing/mechanics overlays: causality and immutable base guards."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_sim_teacher_anchor_finish import adapter, observe, plan
from test_sim_teacher_anchor_finish import (
    trees as trees,  # noqa: PLC0414 - pytest fixture re-export
)

from phantom.sim.command_replay import RecordedDriveCommands
from tools.sim.anchor_inference_timing import RecordedInferenceLatencies
from tools.sim.prepare_teacher_anchor_diagnostic import (
    BASE_RUNNER_SHA,
    RUNNER,
    prepare,
)

FIXTURES = Path(__file__).parent / "fixtures/teacher_anchor_v1"


def schedule(tmp_path, rows=None):
    p = tmp_path / "planner_trace.json"
    p.write_text(
        json.dumps(
            rows
            if rows is not None
            else [
                {"replan_id": 0, "t": 0.26, "latency_s": 0.016, "status": "active"},
                {"replan_id": 1, "t": 0.276, "latency_s": 0.040, "status": "active"},
            ]
        )
    )
    return RecordedInferenceLatencies(p)


def test_recorded_delay_controls_frozen_adapter_without_modifying_native_proposal(
    trees, monkeypatch, tmp_path
):
    base, _, _ = trees
    ad = adapter(monkeypatch, base)
    delays = schedule(tmp_path)
    observe(ad, 0.26)
    raw = plan(0.26, 0.4)
    raw.latency_s = 0.95
    raw_times = raw.action_times.copy()
    ad.policy = SimpleNamespace(replan=lambda *_: raw)
    result = ad.replan(t=0.26, latency_s=delays.next(0, 0.26))
    assert raw.latency_s == 0.95
    np.testing.assert_array_equal(raw.action_times, raw_times)
    assert result.latency_s == 0.016
    assert result.diag["sim_native_inference_latency_s"] == 0.95
    assert result.diag["sim_effective_inference_latency_s"] == 0.016
    assert ad._pending[0] == pytest.approx(0.276)
    np.testing.assert_allclose(
        result.action_times,
        0.276 + np.arange(len(result.actions)) / ad.hw.control.action_rate_hz,
    )
    assert delays.next(1, 0.284) == 0.040
    assert delays.used[-1]["request_time_error_s"] == pytest.approx(0.008)
    with pytest.raises(RuntimeError, match="exhausted"):
        delays.next(2, 0.324)
    with pytest.raises(ValueError, match="exactly once"):
        delays.next(1, 0.324)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"replan_id": 1, "t": 0, "latency_s": 1}],
        [{"replan_id": 0, "t": 0, "latency_s": -1}],
        [{"replan_id": 0, "t": 0, "latency_s": float("nan")}],
        [{"replan_id": 0, "t": 0, "status": "inference_error"}],
        [{"replan_id": 0, "t": 0, "latency_s": 1, "status": "inference_started"}],
    ],
)
def test_invalid_or_incomplete_timing_is_rejected(tmp_path, rows):
    with pytest.raises(ValueError):
        schedule(tmp_path, rows)


@pytest.fixture
def runner_base(tmp_path):
    base = tmp_path / "base"
    path = base / RUNNER
    path.parent.mkdir(parents=True)
    path.write_text((FIXTURES / "run_waffles.py.txt").read_text())
    return base


@pytest.mark.parametrize(
    "variant,module,option",
    [
        (
            "latency_schedule",
            "tools/sim/anchor_inference_timing.py",
            "--anchor-latency-trace",
        ),
        (
            "accepted_commands",
            "phantom/sim/command_replay.py",
            "--anchor-command-trace",
        ),
    ],
)
def test_independent_overlay_has_exact_scope_and_preserves_initialization(
    runner_base, tmp_path, variant, module, option
):
    output = tmp_path / "output"
    before = (runner_base / RUNNER).read_text()
    manifest = prepare(runner_base, output, variant)
    after = (output / RUNNER).read_text()
    assert hashlib.sha256(before.encode()).hexdigest() == BASE_RUNNER_SHA
    assert (runner_base / RUNNER).read_text() == before
    assert set(manifest["output_sha256"]) == {RUNNER, module}
    assert option in after
    assert "finish_after_release" not in after
    # Independent physics initialization and step/render loop remain literal.
    start, end = "    qfull = np.zeros", "    writer = cv2.VideoWriter"
    assert (
        before[before.index(start) : before.index(end)]
        == after[after.index(start) : after.index(end)]
    )
    assert before.count("world.step(render=False)") == after.count(
        "world.step(render=False)"
    )
    assert before.count("world.render()") == after.count("world.render()")
    if variant == "accepted_commands":
        tree = ast.parse(after)
        guarded = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and ast.unparse(n.test)
            == "args.mode == 'policy' and recorded_commands is None"
        ]
        assert len(guarded) == 1
        branch = ast.get_source_segment(after, guarded[0])
        assert "policy = RemoteSimulationPolicy(" in branch
        assert "adapter = SimulationPolicyAdapter(" in branch
        # A real reader instance disables the entire policy connection block.
        check = compile(ast.Expression(guarded[0].test), "guard", "eval")
        assert not eval(
            check,
            {"args": SimpleNamespace(mode="policy"), "recorded_commands": object()},
        )
    with pytest.raises(FileExistsError):
        prepare(runner_base, output, variant)
    with pytest.raises(ValueError, match="protected"):
        prepare(runner_base, runner_base / "bad", variant)
    (runner_base / RUNNER).write_text(before + "\n")
    with pytest.raises(ValueError, match="source hash"):
        prepare(runner_base, tmp_path / "bad", variant)


def test_recorded_submission_loop_holds_same_targets_on_native_tick_grid(
    runner_base, tmp_path
):
    output = tmp_path / "output"
    prepare(runner_base, output, "accepted_commands")
    source = (output / RUNNER).read_text()
    tree = ast.parse(source)
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "recorded_commands is not None"
    )
    body = compile(
        ast.Module(body=branch.body, type_ignores=[]), "recorded-loop", "exec"
    )
    path = tmp_path / "commands.jsonl"
    rows = [
        {
            "t": 0.004,
            "status": "drive_submitted",
            "target_q": [0.4] * 6,
            "target_finger_q": [0.02, 0.02],
        },
        {
            "t": 0.012,
            "status": "drive_submitted",
            "target_q": [0.6] * 6,
            "target_finger_q": [0.01, 0.01],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    state = {
        "recorded_commands": RecordedDriveCommands(path, finger_limit_m=0.035),
        "ids": np.arange(6),
        "fingers": np.array([6, 7]),
        "desired": np.zeros(8),
    }
    for t in (0, 0.004, 0.008, 0.012, 0.016):
        state["t"] = t
        exec(body, state)  # noqa: S102 - hash-pinned generated source, no trace code
        expected = (
            np.zeros(8)
            if t < 0.004
            else np.r_[[0.4] * 6, 0.02, 0.02]
            if t < 0.012
            else np.r_[[0.6] * 6, 0.01, 0.01]
        )
        np.testing.assert_array_equal(state["desired"], expected)
