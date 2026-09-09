import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.sim.analyze_policy_campaign import (
    condition_scene,
    load_design,
    load_trials,
    matched_comparison,
    outcome_summary,
    paired_cluster_interval,
    scene_mismatches,
)


def record(policy, condition, seed, success, valid=True):
    return {
        "policy_id": policy,
        "condition_id": condition,
        "sampling_seed": seed,
        "status": "scored" if valid else "invalid",
        "metrics": {"valid_for_scoring": valid, "outcomes": {"full_task": success}},
    }


def test_paired_interval_preserves_seed_dependence_within_each_condition():
    # Both methods win one seed per condition: every paired condition effect is
    # zero. Resampling individual seed outcomes would invent scenario variance.
    a = [[1, 0], [1, 0], [1, 0]]
    b = [[0, 1], [0, 1], [0, 1]]
    result = paired_cluster_interval(a, b, replicates=500, seed=4)
    assert result["difference_a_minus_b"] == 0
    assert result["interval"] == [0, 0]
    assert result["degenerate_interval"] is True
    assert "does not establish equivalence" in result["interpretation"]


def test_paired_interval_is_reproducible_and_effect_sign_is_explicit():
    a, b = [[1, 1], [1, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]
    first = paired_cluster_interval(a, b, replicates=1000, seed=42)
    second = paired_cluster_interval(a, b, replicates=1000, seed=42)
    reverse = paired_cluster_interval(b, a, replicates=1000, seed=42)
    assert first == second
    assert first["difference_a_minus_b"] == 0.5
    assert reverse["difference_a_minus_b"] == -0.5
    np.testing.assert_allclose(reverse["interval"], -np.array(first["interval"])[::-1])


def test_pairing_joins_ids_and_requires_both_seeds_in_a_condition():
    design = {
        "conditions": [{"id": "z"}, {"id": "a"}],
        "sampling_seeds": [7, 9],
        "analysis": {
            "confidence_level": 0.95,
            "bootstrap_replicates": 100,
            "bootstrap_seed": 1,
        },
    }
    records = {}
    for policy in ["b", "a"]:
        for condition in ["a", "z"]:
            for seed in [9, 7]:
                records[policy, condition, seed] = record(
                    policy, condition, seed, policy == "a"
                )
    del records["b", "a", 9]
    result = matched_comparison(records, design, "a", "b")
    assert result["included_condition_ids"] == ["z"]
    assert result["excluded_condition_ids"] == ["a"]
    assert result["matched_rollouts_per_policy"] == 2
    assert result["status"] == "partial_matched_suite"
    assert result["estimate"]["difference_a_minus_b"] == 1


def test_invalid_observed_success_and_missing_trials_remain_unresolved():
    records = [
        record("a", "x", 1, True),
        record("a", "x", 2, False),
        record("a", "y", 1, True, valid=False),
    ]
    result = outcome_summary(records, 4, "full_task")
    assert result["observed_true"] == 1
    assert result["rate_among_valid_only"] == 0.5
    assert result["missing_or_invalid"] == 2
    assert result["scheduled_denominator_rate_bounds"] == [0.25, 0.75]


def test_frozen_design_is_balanced_and_single_family_perturbations_are_effective():
    path = Path(__file__).resolve().parents[1] / "configs/sim/policy_campaign.json"
    design, _ = load_design(path)
    assert len(design["conditions"]) == 20
    assert design["planned_counts"]["total"] == 160
    assert design["policies"][2]["architecture"] == "student"
    nominal = design["nominal_scene"]
    friction = next(c for c in design["conditions"] if c["id"] == "packet_friction_085")
    scene = condition_scene(design, friction)
    assert scene["waffle"]["static_friction"] == pytest.approx(0.65 * 0.85)
    assert scene["gripper"]["pad_friction"] == nominal["gripper"]["pad_friction"]
    assert nominal["waffle"]["static_friction"] == 0.65  # No mutation.
    camera = next(c for c in design["conditions"] if c["family"] == "camera")
    transformed = condition_scene(design, camera)
    np.testing.assert_allclose(
        np.array(transformed["camera"]["world_from_cv"])[:3, 3],
        np.array(nominal["camera"]["world_from_cv"])[:3, 3],
    )
    assert transformed["waffle"] == nominal["waffle"]


def test_scene_audit_detects_unapplied_perturbation_but_allows_trace_rate_change():
    design, _ = load_design(
        Path(__file__).resolve().parents[1] / "configs/sim/policy_campaign.json"
    )
    condition = next(
        c for c in design["conditions"] if c["family"] == "object_placement"
    )
    expected = condition_scene(design, condition)
    assert "waffle.center" in scene_mismatches(expected, design["nominal_scene"])
    actual = copy.deepcopy(expected)
    actual["physics"]["render_hz"] = 10
    assert scene_mismatches(expected, actual) == []
    actual["physics"]["dt"] = 0.008
    assert scene_mismatches(expected, actual) == ["physics.dt"]


def test_unfrozen_design_is_rejected_before_scoring(tmp_path):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps({"status": "proposed"}))
    with pytest.raises(ValueError, match="frozen"):
        load_design(path)


def test_manifest_hash_and_duplicate_keys_cannot_silently_choose_a_trial(tmp_path):
    design = {
        "policies": [{"id": "a"}],
        "conditions": [{"id": "nominal"}],
        "sampling_seeds": [1, 2],
    }
    out = tmp_path / "analysis"
    first = tmp_path / "runs/first"
    first.mkdir(parents=True)
    manifest = {
        "campaign_sha256": "wrong",
        "policy_id": "a",
        "condition_id": "nominal",
        "sampling_seed": 1,
    }
    (first / "case.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_trials(tmp_path / "runs", design, "correct", out)
    manifest["campaign_sha256"] = "correct"
    (first / "case.json").write_text(json.dumps(manifest))
    second = tmp_path / "runs/second"
    second.mkdir()
    (second / "case.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate trial key"):
        load_trials(tmp_path / "runs", design, "correct", out)


def audited_runtime():
    from tools.sim.analyze_policy_campaign import runtime_audit
    from tools.sim.run_policy_campaign import inference_config

    design, _ = load_design(
        Path(__file__).resolve().parents[1] / "configs/sim/policy_campaign.json"
    )
    policy, condition = design["policies"][1], design["conditions"][0]
    info = {
        "effective": inference_config(design),
        "ckpt_sha": policy["checkpoint_sha256"],
        "mode": policy["policy_mode"],
        "hardware_effective": copy.deepcopy(
            design["runtime_hardware"]["effective_model"]
        ),
        "observation_delay_s": 0.0,
        "inference_delay_add_s": 0.0,
        "max_play_steps": 10,
        "planner_stall_watchdog": True,
        "terminal_veto": False,
        "phantom_recovery": False,
        "tactile_model": "contact_proxy",
        "wrist_model": "contact_proxy",
    }
    server = {
        "effective": inference_config(design),
        "checkpoint_sha256": policy["checkpoint_sha256"],
        "system": policy["architecture"],
        "weights": "EMA",
        "hardware_sha256": design["runtime_hardware"]["sha256"],
        "hardware_config_hash": design["runtime_hardware"]["config_hash"],
    }
    return runtime_audit, design, policy, condition, info, server


def test_recipe_variant_does_not_inherit_other_candidates_runtime():
    from tools.sim.analyze_policy_campaign import policy_settings
    from tools.sim.run_policy_campaign import inference_config

    audit, design, policy, condition, info, server = audited_runtime()
    original = copy.deepcopy(design["inference_settings"])
    variant = {
        **policy,
        "id": "same_weights_other_recipe",
        "inference_settings": {"nfe": 1, "k_seeds": 4},
    }
    reasons = audit(
        design, variant, condition, info, server, {"duration_s": 30}, [0, 30], None
    )
    assert "policy_effective_nfe_differs_from_campaign" in reasons
    assert "server_effective_k_seeds_differs_from_campaign" in reasons
    info["effective"] = inference_config(design, variant)
    server["effective"] = inference_config(design, variant)
    assert (
        audit(
            design, variant, condition, info, server, {"duration_s": 30}, [0, 30], None
        )
        == []
    )
    assert design["inference_settings"] == original
    assert policy_settings(design, policy) == original


def test_condition_start_and_fixed_delivery_must_match_measured_inputs():
    audit, design, policy, condition, info, server = audited_runtime()
    condition = {
        **condition,
        "initial_state": {"path": "/measured/start.json", "sha256": "measured-start-b"},
    }
    design["delivery_latency_s"] = 1.0
    reasons = audit(
        design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
    )
    assert "effective_initial_state_hash_differs_from_campaign" in reasons
    assert "effective_delivery_latency_override_differs_from_campaign" in reasons
    info["policy_initial_state_provenance"] = {"sha256": "measured-start-b"}
    info["policy_latency_override_s"] = 1.0
    assert (
        audit(
            design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
        )
        == []
    )


def test_delivery_feedback_profile_rejects_missing_or_historical_veto_state():
    audit, design, policy, condition, info, server = audited_runtime()
    design["adapter_profile"] = {"terminal_veto_feedback_source": "current_delivery"}
    reason = "effective_terminal_veto_feedback_source_differs_from_campaign"
    for reported in (None, "request_snapshot_historical"):
        info["terminal_veto_feedback_source"] = reported
        assert reason in audit(
            design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
        )
    info["terminal_veto_feedback_source"] = "current_delivery"
    assert (
        audit(
            design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
        )
        == []
    )


def test_gripper_wrist_profile_is_explicit_and_audited():
    audit, design, policy, condition, info, server = audited_runtime()
    design["adapter_profile"] = {"wrist_model": "gripper_contact_proxy"}
    assert "effective_wrist_model_differs_from_campaign" in audit(
        design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
    )
    info["wrist_model"] = "gripper_contact_proxy"
    assert (
        audit(
            design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
        )
        == []
    )


def test_campaign_commands_apply_variant_and_start_without_moving_packet(tmp_path):
    from argparse import Namespace

    from tools.sim.run_policy_campaign import server_command, simulation_command

    _, design, policy, condition, _, _ = audited_runtime()
    policy = {
        **policy,
        "id": "recipe_b",
        "inference_settings": {"nfe": 1, "k_seeds": 4, "max_play": 6},
    }
    condition = {
        **condition,
        "initial_state": {"path": "/measured/start-b.json", "sha256": "b"},
    }
    design["adapter_profile"] = {
        "initial_state": {"path": "/measured/start-a.json", "sha256": "a"}
    }
    design["delivery_latency_s"] = 1.0
    args = Namespace(
        source=tmp_path,
        live_repo=tmp_path,
        server_python=tmp_path / "python",
        port=7799,
        hardware_config=tmp_path / "hardware.yaml",
        evidence=tmp_path / "evidence",
        output=tmp_path / "output",
    )
    server = server_command(args, design, policy, tmp_path / "server")
    command = simulation_command(
        args, design, policy, condition, 13, tmp_path / "trial", None
    )
    assert server[server.index("--nfe") + 1] == "1"
    assert server[server.index("--k-seeds") + 1] == "4"
    assert (
        command[command.index("--policy-initial-state") + 1] == "/measured/start-b.json"
    )
    assert command[command.index("--policy-latency") + 1] == "1.0"
    assert command[command.index("--max-play-steps") + 1] == "6"
    assert command[command.index("--policy-config") + 1].endswith(
        "inference_recipe_b.json"
    )
    assert (
        condition_scene(design, condition)["waffle"]
        == design["nominal_scene"]["waffle"]
    )


def test_unsupported_recipe_override_cannot_silently_change_model_inputs():
    from tools.sim.analyze_policy_campaign import policy_settings

    _, design, policy, _, _, _ = audited_runtime()
    policy["inference_settings"] = {"mask_wrist": True}
    with pytest.raises(ValueError, match="Unsupported"):
        policy_settings(design, policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nfe", 1),
        ("guidance", 2),
        ("k_seeds", 4),
        ("parity_fixes", False),
        ("persistent_noise", False),
        ("task_text", "other"),
        ("drop_video", True),
        ("close_p", 0.8),
    ],
)
def test_effective_recipe_changes_are_invalid_even_with_correct_checkpoint(
    field, value
):
    audit, design, policy, condition, info, server = audited_runtime()
    assert (
        audit(
            design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
        )
        == []
    )
    info["effective"][field] = value
    assert f"policy_effective_{field}_differs_from_campaign" in audit(
        design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
    )


def test_full_hardware_ema_and_server_recipe_are_verified():
    audit, design, policy, condition, info, server = audited_runtime()
    info["hardware_effective"]["safety"]["grip_latch_fz_n"] += 1
    server["weights"] = "raw"
    server["effective"]["k_seeds"] = 4
    reasons = audit(
        design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None
    )
    assert "effective_hardware_differs_from_frozen_model" in reasons
    assert "effective_ema_setting_differs_from_campaign" in reasons
    assert "server_effective_k_seeds_differs_from_campaign" in reasons


def test_raw_trace_truncation_cannot_be_hidden_by_duration_header_or_stop():
    audit, design, policy, condition, info, server = audited_runtime()
    for stop in (None, "safety_stop"):
        reasons = audit(
            design, policy, condition, info, server, {"duration_s": 30}, [0, 10], stop
        )
        assert "raw_trace_end_disagrees_with_reported_duration" in reasons
    assert (
        audit(
            design,
            policy,
            condition,
            info,
            server,
            {"duration_s": 10},
            [0, 10],
            "safety_stop",
        )
        == []
    )
    reasons = audit(
        design, policy, condition, info, server, {"duration_s": 10}, [0, 10], None
    )
    assert "short_horizon_without_recorded_termination" in reasons
    reasons = audit(
        design,
        policy,
        condition,
        info,
        server,
        {"duration_s": 10},
        [0, 10],
        "lift_complete",
    )
    assert "early_success_termination_forbidden" in reasons


def test_campaign_blocks_cover_exact_grid_with_nominal_first():
    from tools.sim.run_policy_campaign import blocks

    design, _ = load_design(
        Path(__file__).resolve().parents[1] / "configs/sim/policy_campaign.json"
    )
    schedule = blocks(design)
    assert len(schedule) == 7
    assert all(
        [c for c, _ in b["trials"]] == ["nominal", "nominal"] for b in schedule[:3]
    )
    keys = [(b["policy_id"], c, s) for b in schedule for c, s in b["trials"]]
    assert len(set(keys)) == len(keys) == 160
    assert {p: sum(k[0] == p for k in keys) for p in {k[0] for k in keys}} == {
        p["id"]: 40 for p in design["policies"]
    }


def test_controller_dry_run_preserves_venv_executable_symlink(
    tmp_path, monkeypatch, capsys
):
    import sys

    from tools.sim.run_policy_campaign import main

    repo = Path(__file__).resolve().parents[1]
    venv = tmp_path / "live/.venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_policy_campaign.py",
            "--campaign",
            str(repo / "configs/sim/policy_campaign.json"),
            "--live-repo",
            str(tmp_path / "live"),
            "--evidence",
            str(tmp_path / "evidence"),
            "--hardware-config",
            str(tmp_path / "not-read.yaml"),
            "--output",
            str(output),
        ],
    )
    main()
    plan = json.loads((output / "plan.json").read_text())
    assert plan["server_python"] == str(venv / "python")
    assert plan["executing"] is False
    assert plan["planned_trials"] == 160
    capsys.readouterr()


def test_owned_cleanup_tolerates_process_exit_race(monkeypatch):
    from tools.sim.run_policy_campaign import stop_owned

    class Exiting:
        pid = 23456
        waited = False

        def poll(self):
            return None

        def wait(self, timeout):
            self.waited = True

    def already_exited(*_):
        raise ProcessLookupError

    monkeypatch.setattr("os.killpg", already_exited)
    process = Exiting()
    stop_owned(process)
    assert process.waited


def test_missing_execution_and_planner_logs_are_unresolved_not_empty(tmp_path):
    design = {
        "policies": [{"id": "a"}],
        "conditions": [{"id": "nominal"}],
        "sampling_seeds": [1, 2],
    }
    folder = tmp_path / "runs/trial"
    folder.mkdir(parents=True)
    (folder / "case.json").write_text(
        json.dumps(
            {
                "campaign_sha256": "frozen",
                "policy_id": "a",
                "condition_id": "nominal",
                "sampling_seed": 1,
            }
        )
    )
    for name in (
        "run.json",
        "sim_trace.npz",
        "effective_config.json",
        "policy_info.json",
        "server_ready.json",
    ):
        (folder / name).write_text("{}")
    records = load_trials(tmp_path / "runs", design, "frozen", tmp_path / "analysis")
    result = records["a", "nominal", 1]
    assert result["status"] == "missing_inputs"
    assert result["missing_inputs"] == ["execution_trace.jsonl", "planner_trace.json"]


def test_frozen_source_inventory_covers_watchdog_assets_and_episode(
    tmp_path, monkeypatch
):
    from argparse import Namespace

    from tools.sim.run_policy_campaign import source_manifest

    for name in (
        "phantom/sim/deployment_filters.py",
        "tools/sim/run_waffles.py",
        "tools/sim/launch_waffles.sh",
        "assets/sim/waffles/packet_top.png",
        "configs/sim/waffles.json",
    ):
        path = tmp_path / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    ignored = tmp_path / "source/assets/sim/__pycache__/cache.pyc"
    ignored.parent.mkdir()
    ignored.write_bytes(b"cache")
    monkeypatch.setattr("tools.sim.run_policy_campaign.fingerprint", lambda p: str(p))
    args = Namespace(
        source=tmp_path / "source",
        live_repo=tmp_path / "live",
        evidence=tmp_path / "evidence",
        hardware_config=tmp_path / "hardware.yaml",
        robot_usd=None,
    )
    result = source_manifest(
        args, "design", {"prepared_episode": "evidence/fit/episode"}
    )
    assert "phantom/sim/deployment_filters.py" in result["source_sha256"]
    assert "phantom/deploy/safety.py" in result["source_sha256"]
    assert "phantom/config/hardware.py" in result["live_core_sha256"]
    assert "assets/sim/waffles/packet_top.png" in result["source_sha256"]
    assert all("__pycache__" not in p for p in result["source_sha256"])
    assert set(result["episode_sha256"]) == {"replay.npz", "manifest.json"}
    assert (
        "tools/sim/run_policy_campaign.py"
        in result["external_controller_source_sha256"]
    )
    assert (
        "tools/sim/analyze_policy_campaign.py"
        in result["external_controller_source_sha256"]
    )
    assert (
        "phantom/sim/policy_metrics.py" in result["external_controller_source_sha256"]
    )


def test_teacher_only_analysis_does_not_invent_policy_comparison():
    from tools.sim.analyze_policy_campaign import analyze

    design, sha = load_design(
        Path(__file__).resolve().parents[1] / "configs/sim/policy_campaign.json"
    )
    design["policies"] = design["policies"][:1]
    design["analysis"].update(
        primary_comparison=None, exploratory_comparisons=[], secondary_comparisons=[]
    )
    result = analyze(design, sha, {})
    assert set(result["policies"]) == {"teacher"}
    assert result["paired_comparisons"] == []
    assert result["policies"]["teacher"]["missing"] == 40


def test_campaign_passes_frozen_teacher_sensor_and_release_profile(tmp_path):
    from argparse import Namespace

    from tools.sim.run_policy_campaign import simulation_command

    design, _ = load_design(
        Path(__file__).resolve().parents[1] / "configs/sim/policy_campaign.json"
    )
    design["adapter_profile"] = {
        "tactile_model": "measured_baseline_proxy",
        "wrist_model": "gripper_contact_proxy",
        "gel_contact_coverage": "manifold_patch",
        "initial_state": {"path": "/fixtures/initial.json"},
        "tactile_baseline": {"path": "/fixtures/baseline.npz"},
        "terminal_veto": {"implementation": "fd4a032"},
        "placement_release": {"explicit": True},
        "record_packet_support": True,
        "save_policy_observations": True,
    }
    args = Namespace(
        source=tmp_path / "source",
        evidence=tmp_path / "evidence",
        output=tmp_path / "out",
        port=7796,
        hardware_config=tmp_path / "hardware.yaml",
    )
    cmd = simulation_command(
        args,
        design,
        design["policies"][0],
        design["conditions"][0],
        4242,
        tmp_path / "trial",
        None,
    )
    assert cmd[cmd.index("--tactile") + 1] == "measured_baseline_proxy"
    assert cmd[cmd.index("--wrist") + 1] == "gripper_contact_proxy"
    assert cmd[cmd.index("--policy-initial-state") + 1] == "/fixtures/initial.json"
    assert cmd[cmd.index("--placement-release-config") + 1] == str(
        args.output / "runtime/placement_release.json"
    )
    assert "--record-packet-support" in cmd and "--save-policy-observations" in cmd


def optional_native_runtime():
    audit, design, policy, condition, info, server = audited_runtime()
    design["adapter_profile"] = {
        "placement_controller_profile": "minimal_v5", "grip_play_steps": 10,
    }
    info.update(
        placement_controller_profile={
            "id": "minimal_v5", "veto_implementation": "fd4a032",
            "veto_feedback_source": "request_snapshot_historical",
        },
        grip_play_steps=10, effective_grip_play_steps=10,
    )
    return audit, design, policy, condition, info, server


def test_explicit_native_controller_metadata_and_split_cap_pass_audit():
    audit, design, policy, condition, info, server = optional_native_runtime()
    assert audit(design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None) == []


@pytest.mark.parametrize("field,value,reason", [
    ("placement_controller_profile", None, "effective_placement_controller_profile_differs_from_campaign"),
    ("placement_controller_profile", {"id": "other"}, "effective_placement_controller_profile_differs_from_campaign"),
    ("placement_controller_profile", {
        "id": "minimal_v5", "veto_implementation": "fd4a032",
        "veto_feedback_source": "current_delivery",
    }, "effective_placement_controller_semantics_differ_from_campaign"),
    ("grip_play_steps", 16, "effective_grip_play_steps_differs_from_campaign"),
    ("effective_grip_play_steps", 16, "effective_combined_gripper_playback_cap_differs_from_campaign"),
])
def test_new_declared_native_settings_cannot_silently_use_historical_behavior(field, value, reason):
    audit, design, policy, condition, info, server = optional_native_runtime()
    info[field] = value
    assert reason in audit(design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None)


@pytest.mark.parametrize("grip_cap,effective", [(None, 10), (4, 4), (16, 10)])
def test_declared_effective_grip_cap_accounts_for_pose_cap(grip_cap, effective):
    audit, design, policy, condition, info, server = optional_native_runtime()
    design["adapter_profile"]["grip_play_steps"] = grip_cap
    info.update(grip_play_steps=grip_cap, effective_grip_play_steps=effective)
    assert audit(design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None) == []


def test_undeclared_optional_fields_preserve_historical_audit():
    audit, design, policy, condition, info, server = audited_runtime()
    info.update(grip_play_steps=10, effective_grip_play_steps=10,
                placement_controller_profile={"id": "not_a_historical_declaration"})
    assert audit(design, policy, condition, info, server, {"duration_s": 30}, [0, 30], None) == []
