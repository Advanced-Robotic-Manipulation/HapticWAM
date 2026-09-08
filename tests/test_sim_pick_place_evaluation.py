"""Task acceptance requires sustained physical evidence, including failed trials."""

import numpy as np

from tools.sim.evaluate_pick_place import evaluate_arrays, image_alignment


def complete_trial():
    t = np.arange(0, 10.001, 0.05)
    height = np.interp(
        t, [0, 2.25, 3, 7, 7.5, 10], [0.025, 0.025, 0.225, 0.225, 0.025, 0.025]
    )
    position = np.c_[np.zeros((len(t), 2)), height]
    contact = (t >= 2) & (t < 7)
    trace = {
        "t": t,
        "physics_t": t.copy(),
        "q": np.zeros((len(t), 6)),
        "target_q": np.zeros((len(t), 6)),
        "tcp": np.c_[position + [0, 0, 0.1], np.zeros((len(t), 3))],
        "waffle_position": position,
        "waffle_orientation_wxyz": np.tile([1, 0, 0, 0], (len(t), 1)),
        "gripper": np.c_[np.where(contact, 0.7, 0.2)],
        "pad_packet_normal_force": np.repeat(contact[:, None] * 2.0, 2, axis=1),
    }
    run = {
        "episode": "complete",
        "mode": "dynamics",
        "object_dynamics": {
            "rigid_body_dynamic": True,
            "kinematic": False,
            "attachments": [],
            "pose_writes_after_initialization": 0,
        },
    }
    cfg = {
        "waffle": {"size": [0.2, 0.02, 0.03]},
        "bin": {"center": [0, 0, 0], "size": [0.5, 0.5, 0.2], "wall": 0.01},
    }
    real = {"t": t, "q": np.zeros((len(t), 6))}
    timeline = {
        "episode": "complete",
        "visual_adjudication": {"complete_pick_place": True},
        "telemetry_events": {
            "dual_pad_contact_above_2_sensor_units_s": 2.0,
            "lift_20mm_above_min_s": 2.35,
            "release_closure_drop_0p15_s": 7.0,
        },
    }
    return trace, run, cfg, real, timeline


def test_complete_physical_carry_and_settle_can_pass_with_images_unverified():
    result = evaluate_arrays(*complete_trial())
    assert result["physical_verdict"] == "pass_within_declared_tolerances"
    assert result["image_proxy_verdict"] == "unverified"
    assert result["metrics"]["release"]["packet_filtered_both_pads_unloaded_s"] == 7
    assert result["metrics"]["final_bin"]["contained_fraction"] == 1


def test_one_contact_frame_and_unfiltered_table_force_cannot_certify_carry():
    trial = complete_trial()
    trace = trial[0]
    trace["pad_packet_normal_force"][:] = 0
    trace["pad_packet_normal_force"][40] = 100
    trace["pad_contact_force"] = np.full((len(trace["t"]), 2, 3), 100.0)
    result = evaluate_arrays(*trial)
    assert result["gates"]["timed_bilateral_packet_contact"]["pass"] is False
    assert result["gates"]["sustained_bilateral_carry_contact"]["pass"] is False
    del trace["pad_packet_normal_force"]
    result = evaluate_arrays(*trial)
    assert result["gates"]["sustained_bilateral_carry_contact"]["pass"] is None


def test_commands_without_actual_tracking_and_hidden_attachment_fail():
    trial = complete_trial()
    trial[0]["q"][:] = 0.3
    trial[1]["object_dynamics"]["attachments"] = ["fixed_joint"]
    result = evaluate_arrays(*trial)
    assert result["physical_verdict"] == "fail"
    assert result["gates"]["free_dynamic_object_provenance"]["pass"] is False
    assert result["gates"]["actual_drive_tracking"]["pass"] is False


def test_center_inside_bin_does_not_imply_oriented_packet_containment():
    trial = complete_trial()
    trial[0]["waffle_position"][trial[0]["t"] >= 7.5, 0] = 0.21
    result = evaluate_arrays(*trial)
    assert result["gates"]["final_oriented_bin_containment"]["pass"] is False


def test_no_post_release_window_cannot_certify_settling():
    trial = complete_trial()
    selection = trial[0]["t"] <= 7.3
    for key, value in trial[0].items():
        trial[0][key] = value[selection]
    result = evaluate_arrays(*trial)
    assert result["gates"]["recording_time_coverage"]["pass"] is False
    assert result["gates"]["post_release_settling"]["pass"] is None


def test_image_proxy_excludes_explicit_occlusion_and_reports_coverage():
    trace = complete_trial()[0]
    camera = {"world_from_cv": np.eye(4), "fx": 100, "fy": 100, "cx": 32, "cy": 24}
    cfg = {"camera": camera}
    pose = {
        "camera": camera,
        "conditional_initial_packet_center_m": [0, 0, 0.025],
        "assumed_packet_size_m": [0.2, 0.02, 0.03],
        "conditional_initial_packet_yaw_rad": 0,
    }
    tracks = [
        {"camera_t_s": stamp, "x_px": 32, "y_px": 24, "visibility": "visible"}
        for stamp in trace["t"]
    ]
    tracks[80].update(x_px=10000, visibility="partial_occlusion")
    result = image_alignment(trace, cfg, tracks, pose)
    assert result["rmse_px"] == 0
    assert result["coverage_fraction"] == 1
    assert result["excluded_nonvisible_frames"] == 1


def test_measured_box_uses_opening_and_independent_floor_for_containment():
    trial = complete_trial()
    trial[2]['bin'] = {
        'geometry_model': 'rectangular_envelope',
        'center': [0, 0, 0],
        'outer_size': [.400, .300, .190],
        'opening_size': [.360, .260],
        'floor_thickness': .004,
    }
    result = evaluate_arrays(*trial)
    # Bottom at10mm is inside above a4mm floor. Treating the20mm
    # outer/opening separation as floor thickness would reject this state.
    assert result['gates']['final_oriented_bin_containment']['pass'] is True
    trial[0]['waffle_position'][trial[0]['t'] >= 7.5, 0] = .09
    result = evaluate_arrays(*trial)
    # Right edge190mm fits outer envelope200mm but exceeds opening180mm.
    assert result['gates']['final_oriented_bin_containment']['pass'] is False
