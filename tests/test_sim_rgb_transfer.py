import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sim import diagnose_rgb_transfer as diagnostic
from tools.sim.diagnose_rgb_transfer import (
    OBS_FIELDS,
    SEEDS,
    array_sha,
    compare_proposals,
    paired_calls,
    proposal,
    replace_rgb,
    same_clock_arm_indices,
    select_camera_index,
    validate_owned_server,
    validate_study_complete,
)


def observation():
    arrays = {name: np.zeros(1, dtype=np.float32) for name in OBS_FIELDS}
    arrays.update(
        t=np.asarray(0.26),
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        ur_state=np.arange(26, dtype=np.float32),
    )
    return arrays


def test_nearest_real_frame_is_causal_to_request_not_arbitrarily_future():
    assert select_camera_index([0.014, 0.077, 0.144, 0.210, 0.278], 0.2, 0.26) == 3
    assert select_camera_index([0.014, 0.077, 0.144, 0.210, 0.278], 0.2, 0.19) == 2
    with pytest.raises(ValueError, match="No causal"):
        select_camera_index([0.1, 0.2], 0.1, 0.05)


def test_arm_pose_join_uses_exact_selected_timestamps_with_unequal_stream_lengths():
    assert same_clock_arm_indices([0.0, 0.1, 0.2], [0.0, 0.1, 0.2, 0.3], 0.15) == (1, 1)
    with pytest.raises(ValueError, match="exact timestamps"):
        same_clock_arm_indices([0.0, 0.1, 0.2], [0.0, 0.1001, 0.2], 0.15)


def test_rgb_swap_preserves_every_non_rgb_tensor_dtype_shape_and_value():
    original = observation()
    swapped = replace_rgb(original, np.ones((4, 5, 3), dtype=np.uint8))
    assert not np.array_equal(swapped["rgb"], original["rgb"])
    assert all(
        array_sha(original[k]) == array_sha(swapped[k]) for k in original if k != "rgb"
    )
    swapped["ur_state"][0] = 999
    assert original["ur_state"][0] == 0
    with pytest.raises(ValueError, match="shape and dtype"):
        replace_rgb(original, np.ones((4, 5, 3), dtype=np.float32))


class StubPolicy:
    def __init__(self):
        self.events = []

    def remote_reset(self, seed):
        self.events.append(("reset", seed))
        self.seed = seed

    def replan(self, obs, prev, tcp):
        assert prev is None
        self.events.append(("replan", self.seed))
        actions = np.zeros((16, 7))
        actions[:, 0] = obs.rgb.mean() * 0.001 + (self.seed % 2) * 0.0001
        actions[:, 6] = 0.25
        return SimpleNamespace(
            t_created=float(obs.t),
            t0_pose=tcp,
            actions=actions,
            action_times=np.arange(16) * 0.1 + 0.8,
            latency_s=0.8,
            sigma=np.zeros(3),
            gate=0.0,
            p_evt=np.zeros(5),
            diag={"k_pick": 0},
        )


def test_exact_six_requests_with_seed_reset_and_repeat_controls_no_cpk():
    original = observation()
    swapped = replace_rgb(original, np.ones((4, 5, 3), dtype=np.uint8))
    policy = StubPolicy()
    rows, comparisons = paired_calls(policy, original, swapped)
    assert len(rows) == 6
    assert policy.events == [
        (kind, seed) for seed in SEEDS for _ in range(3) for kind in ("reset", "replan")
    ]
    for row in comparisons:
        assert row["rendered_repeat_control"]["endpoint_10step_xyz_difference_m"] == 0
        assert row["rgb_swap"]["endpoint_10step_xyz_difference_m"] == pytest.approx(
            0.01
        )


def test_non_rgb_counterfactual_change_refused_before_any_inference():
    original = observation()
    swapped = replace_rgb(original, np.ones((4, 5, 3), dtype=np.uint8))
    swapped["ur_state"][0] += 1
    policy = StubPolicy()
    with pytest.raises(ValueError, match="Non-RGB"):
        paired_calls(policy, original, swapped)
    assert policy.events == []


def test_native_delta_endpoint_uses_cumulative_steps_without_extra_dt():
    actions = np.zeros((16, 7))
    actions[:, 0] = 0.001
    p = SimpleNamespace(
        t_created=0,
        t0_pose=np.zeros(6),
        actions=actions,
        action_times=np.arange(16) * 0.1,
        latency_s=0.0,
        sigma=np.zeros(3),
        gate=0.0,
        p_evt=np.zeros(5),
        diag={},
    )
    result = proposal(p)
    assert result["endpoint_10step_pose"][0] == pytest.approx(0.01)
    assert result["endpoint_16step_pose"][0] == pytest.approx(0.016)
    assert compare_proposals(result, result)["xyz_step_rmse_m"] == 0


def test_study_guard_requires_both_complete_stages_and_exited_processes(
    tmp_path, monkeypatch
):
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(diagnostic, "PROC_ROOT", proc)
    root = tmp_path / "study"
    for stage, count in (("screen", 32), ("confirmation", 24)):
        folder = root / stage
        (folder / "selection").mkdir(parents=True)
        (folder / "selection/selection.json").write_text(
            json.dumps(
                {
                    "status": "complete_valid_matched_stage",
                    "stage": stage,
                    "expected_trials": count,
                    "available_trials": count,
                    "missing_trial_keys": [],
                    "invalid_trial_keys": [],
                }
            )
        )
        (folder / "progress.json").write_text(
            json.dumps(
                {
                    "status": "all_trials_completed",
                    "controller_pid": 4321,
                }
            )
        )
    assert set(validate_study_complete(root)) == {"screen", "confirmation"}
    (proc / "4321").mkdir()
    with pytest.raises(ValueError, match="controller must have exited"):
        validate_study_complete(root)
    (proc / "4321").rmdir()
    path = root / "confirmation/selection/selection.json"
    changed = json.loads(path.read_text())
    changed["available_trials"] = 23
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Both corrected study stages"):
        validate_study_complete(root)


def test_server_guard_pins_owned_pid_recipe_and_native_source(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    (proc / "5678").mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "PROC_ROOT", proc)
    ready_path = tmp_path / "server/ready.json"
    ready_path.parent.mkdir()
    (proc / "5678/cmdline").write_bytes(
        b"python\0/review/tools/sim/policy_server.py\0--out\0"
        + str(ready_path.parent).encode()
        + b"\0"
    )
    original = {
        "normalizers_sha256": "norm",
        "backbone_sha256": "base",
        "text_cache_sha256": "text",
        "hardware_sha256": "hardware",
        "inference_source_sha256": {"policy.py": "source"},
    }
    ready = {
        **original,
        "dedicated_simulation_server": True,
        "status": "ready",
        "pid": 5678,
        "port": 7801,
        "system": "teacher",
        "weights": "EMA",
        "checkpoint_sha256": diagnostic.CHECKPOINT_SHA,
        "effective": dict(diagnostic.SETTINGS),
    }
    ready_path.write_text(json.dumps(ready))
    assert validate_owned_server(ready_path, 5678, original) == ready
    changed = {**ready, "inference_source_sha256": {"policy.py": "changed"}}
    ready_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="inference_source_sha256"):
        validate_owned_server(ready_path, 5678, original)
    changed = {**ready, "port": 7777}
    ready_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="non-rig port"):
        validate_owned_server(ready_path, 5678, original)
    changed = {**ready, "effective": {**diagnostic.SETTINGS, "k_seeds": 1}}
    ready_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="inference settings"):
        validate_owned_server(ready_path, 5678, original)
