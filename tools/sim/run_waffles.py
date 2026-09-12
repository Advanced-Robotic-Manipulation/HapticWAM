#!/usr/bin/env python3
"""Isaac Sim 6.0 waffles reconstruction, replay and closed-loop harness.

Run through launch_waffles.sh. No real device driver is constructed. Source
episodes are opened only by prepare_waffles.py, which writes separate exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/sim/waffles.json")
    p.add_argument(
        "--episode",
        type=Path,
        required=True,
        help="Prepared episode folder containing replay.npz",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=["replay", "dynamics", "policy", "contact_probe", "command_replay"],
        default="replay",
    )
    p.add_argument("--duration", type=float, default=None)
    p.add_argument(
        "--command-trace",
        type=Path,
        help="Recorded drive-submission JSONL for command_replay mechanics diagnostics",
    )
    p.add_argument(
        "--command-trace-manifest", type=Path,
        help="Required native command replay declaration: trace/config hashes, named joint ordering and radian drive semantics",
    )
    p.add_argument(
        "--command-finger-hold-at", type=float,
        help="Counterfactual native command_replay only: from this recorded command time retain the preceding submitted finger drives; keep every arm target unchanged",
    )
    p.add_argument(
        "--render-hz",
        type=float,
        help="Override media/trace sampling rate for tuning runs",
    )
    p.add_argument(
        "--skip-stage-export",
        action="store_true",
        help="Skip USD packaging during parameter tuning",
    )
    p.add_argument("--gui", action="store_true")
    p.add_argument(
        "--robot-usd", type=Path, help="Reuse a previously imported robot USD"
    )
    p.add_argument("--policy-server", default="127.0.0.1:7777")
    p.add_argument(
        "--experimental-adaptive-policy", action="store_true",
        help="Teacher-only exploratory native W2L inference; requires measured-baseline tactile inputs and complete observation/contact auditing",
    )
    p.add_argument(
        "--save-policy-observations",
        action="store_true",
        help="Save exact preprocessed-input snapshots before each policy request for diagnostics",
    )
    p.add_argument(
        "--policy-config", type=Path, help="Explicit remote inference settings JSON"
    )
    p.add_argument(
        "--terminal-veto-config",
        type=Path,
        help="Optional explicit terminal-veto JSON: implementation and config; no inferred task defaults",
    )
    p.add_argument(
        "--policy-initial-state",
        type=Path,
        help="Measured single-state JSON with q, gripper and wrist_ft; policy initialization only",
    )
    p.add_argument(
        "--placement-release-config",
        type=Path,
        help="Explicit opt-in measured-TCP gate allowing policy-commanded release from the grip latch",
    )
    p.add_argument(
        "--boundary-projection-config", type=Path, default=None,
        help="Explicit default-off bounded upper-Y projection treatment JSON",
    )
    p.add_argument(
        "--placement-controller-profile", choices=["minimal_v5"], default=None,
        help="Explicit shared native historical veto/request feedback plus release/FINISH profile",
    )
    p.add_argument(
        "--record-packet-support",
        action="store_true",
        help="Record independent packet-to-bin and packet-to-robot normal forces for strict placement scoring",
    )
    p.add_argument(
        "--record-robot-environment-contacts",
        action="store_true",
        help="Diagnostic only: save every robot body's explicit table/mat/bin/packet contacts; no controller or safety feedback",
    )
    p.add_argument(
        "--hardware-config", type=Path, default=REPO / "configs/hardware.nuc.yaml"
    )
    p.add_argument("--ignore-episode-overrides", action="store_true")
    p.add_argument("--max-play-steps", type=int, default=10)
    p.add_argument(
        "--grip-play-steps", type=int, default=None,
        help="Separate gripper chunk cap, also bounded by max-play-steps; default preserves legacy playback",
    )
    p.add_argument(
        "--servo-reach-limiter",
        action="store_true",
        help="Separate policy diagnostic: existing native .40 rad elbow / 1 rad/s commanded-joint limiter; measured safety thresholds stay unchanged",
    )
    p.add_argument("--observation-delay-s", type=float, default=0.0)
    p.add_argument(
        "--no-progress-stop-s", type=float, default=None,
        help="Policy mode: end the trial (reason no_progress_timeout) when the packet has not been "
             "lifted 3 cm by this simulated time; the independent scorer still scores the trial",
    )
    p.add_argument(
        "--servo-constraint-hold-s", type=float, default=None,
        help="Opt-in verified stationary servo hold deadline; requires --servo-reach-limiter",
    )
    p.add_argument("--inference-delay-add-s", type=float, default=0.0)
    p.add_argument(
        "--policy-mode",
        choices=["student", "vision_only", "teacher"],
        default="teacher",
    )
    p.add_argument(
        "--tactile",
        choices=["contact_proxy", "measured_baseline_proxy", "zero_ablation"],
        default="contact_proxy",
    )
    p.add_argument(
        "--tactile-baseline",
        type=Path,
        help="Single measured no-contact NPZ for measured_baseline_proxy",
    )
    p.add_argument(
        "--gel-contact-coverage",
        choices=["point", "manifold_patch", "manifold_patch_v2"],
        default="point",
        help="Explicit contact-area estimate for the tactile proxy; physics colliders and force limits are unchanged",
    )
    p.add_argument(
        "--record-gel-contacts",
        action="store_true",
        help="Record read-only contact geometry/separations during measured replay as well as policy runs",
    )
    p.add_argument(
        "--wrist",
        choices=["contact_proxy", "gripper_contact_proxy", "zero_ablation"],
        default="contact_proxy",
        help="Uncalibrated pad-only or full gripper external-normal-contact wrench plus recorded bias, or zero ablation",
    )
    p.add_argument(
        "--policy-latency",
        type=float,
        default=None,
        help="Override measured inference latency on sim clock",
    )
    p.add_argument(
        "--policy-delivery-clock",
        choices=["native", "rpc_wall"],
        default="native",
        help="Native inference duration (default), or measured full client policy call for delivery only; native action/CPK clocks stay unchanged",
    )
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument(
        "--probe-start-time",
        type=float,
        default=6.0,
        help="Recorded arm pose used to initialize the synthetic contact probe",
    )
    p.add_argument(
        "--push-at",
        type=float,
        default=None,
        help="Apply a packet velocity impulse at this sim time",
    )
    p.add_argument("--push-velocity", type=float, nargs=3, default=[0.15, 0, 0])
    p.add_argument("--friction-scale", type=float, default=1)
    p.add_argument("--object-offset", type=float, nargs=2, default=[0, 0])
    p.add_argument("--save-stage-only", action="store_true")
    args = p.parse_args()
    if args.servo_constraint_hold_s is not None:
        if (not args.servo_reach_limiter or args.mode != "policy"
                or not np.isfinite(args.servo_constraint_hold_s)
                or args.servo_constraint_hold_s <= 0):
            p.error("servo-constraint-hold-s must be finite, positive and used with policy servo-reach-limiter")
    if args.policy_delivery_clock == "rpc_wall":
        if args.policy_latency is not None:
            p.error("rpc_wall cannot be combined with --policy-latency")
        if args.mode != "policy":
            p.error("rpc_wall requires --mode policy")
    return args


def validate_adaptive_policy_experiment(args, cfg):
    """Require an explicit, audited exploratory teacher contract before Kit starts.

    The measured-motion gates support an experiment, not calibrated teacher
    inputs. This opt-in preserves that distinction and the mechanical guards.
    """
    from phantom.sim.gripper_articulation import is_adaptive

    opted_in = bool(getattr(args, "experimental_adaptive_policy", False))
    adaptive = is_adaptive(cfg)
    if opted_in and (not adaptive or args.mode != "policy"):
        raise ValueError("experimental-adaptive-policy requires native W2L policy mode")
    if not adaptive or args.mode != "policy":
        return None
    if not opted_in:
        raise ValueError(
            "Native adaptive W2L policy execution is uncalibrated; an audited "
            "teacher experiment requires --experimental-adaptive-policy."
        )
    if args.policy_mode not in ("teacher", "student", "vision_only"):
        raise ValueError("The experimental adaptive policy contract covers teacher, student and vision_only modes")
    if args.tactile != "measured_baseline_proxy" or args.wrist != "gripper_contact_proxy":
        raise ValueError("Adaptive teacher experiment requires measured_baseline_proxy and gripper_contact_proxy")
    if args.gel_contact_coverage != "manifold_patch_v2":
        raise ValueError("Adaptive teacher experiment requires the replay-tested manifold_patch_v2 mapping")
    if (not args.save_policy_observations or not args.record_gel_contacts
            or not args.record_packet_support or not args.record_robot_environment_contacts):
        raise ValueError(
            "Adaptive teacher experiment requires policy observations, gel contacts, "
            "packet-support and robot-environment contact records"
        )
    for name in ("tactile_baseline", "policy_config"):
        path = getattr(args, name, None)
        if path is None or not Path(path).is_file():
            raise ValueError(f"Adaptive teacher experiment requires an existing {name} file")
    from phantom.sim.remote_policy import CONFIGURABLE

    policy_bytes = Path(args.policy_config).read_bytes()
    try:
        settings = json.loads(policy_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Adaptive teacher policy_config must contain valid JSON") from error
    if not isinstance(settings, dict):
        raise ValueError("Adaptive teacher policy_config must be a JSON object")
    missing = set(CONFIGURABLE) - settings.keys()
    unknown = settings.keys() - set(CONFIGURABLE)
    if missing or unknown:
        raise ValueError(
            "Adaptive teacher policy_config must explicitly specify the supported "
            f"inference settings; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for name in ("nfe", "k_seeds"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"Adaptive teacher policy_config {name} must be a positive integer")
    for name in ("parity_fixes", "persistent_noise", "drop_video"):
        if type(settings[name]) is not bool:
            raise ValueError(f"Adaptive teacher policy_config {name} must be a boolean")
    for name in ("guidance", "close_p"):
        value = settings[name]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Adaptive teacher policy_config {name} must be finite and nonnegative")
    if settings["close_p"] > 1:
        raise ValueError("Adaptive teacher policy_config close_p must be in [0, 1]")
    if not isinstance(settings["task_text"], str) or not settings["task_text"].strip():
        raise ValueError("Adaptive teacher policy_config task_text must be a nonempty string")
    return {
        "status": "exploratory_uncalibrated_teacher_inference",
        "mechanics_prerequisite": "Native W2L at <=1ms; instantaneous joint/loop guards remain enabled",
        "tactile_mapping": "Measured no-contact baseline plus active-gel normal-contact proxy; optical, shear and force transfer unvalidated",
        "tactile_baseline_sha256": hashlib.sha256(Path(args.tactile_baseline).read_bytes()).hexdigest(),
        "policy_config_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "declared_policy_settings": settings,
        "hardware_transfer_qualified": False,
        "success_source": "Independent packet state and bin support, not controller FINISH",
    }


def measured_gripper_status(closure, target, contact_force_n):
    """Approximate Robotiq OBJ from physical closure, target and pad contact.

    The prismatic-jaw proxy cannot reproduce the hardware firmware's contact
    detector, so this convention is explicitly reported in run.json.
    """
    if abs(target - closure) <= 2 / 255:
        return 3
    if contact_force_n > 0.1:
        return 2 if target > closure else 1
    return 0


def measured_tcp_twist(previous, current, dt):
    """RTDE-style linear/world angular velocity; rotvec derivative is different."""
    from scipy.spatial.transform import Rotation

    if previous is None:
        return np.zeros(6)
    angular = (
        Rotation.from_rotvec(current[3:]) * Rotation.from_rotvec(previous[3:]).inv()
    ).as_rotvec() / dt
    return np.r_[(current[:3] - previous[:3]) / dt, angular]


def report_servo_limiter_execution(adapter, t, selection, gripper_command, rejects,
                                   *, hold_budget=None, qref=None, limits=None, dt=None,
                                   telemetry=None, submit_targets=None):
    """Report the accepted joint target, or a bounded ordinary controller stop.

    CPU-testable feedback boundary. A stall preserves grip and enters the same
    stopped-tick handling/observation tail as other controller stops.
    """
    from phantom.sim.kinematics import forward_pose
    if selection.reason.startswith("boundary_projection_"):
        adapter.request_stop(selection.reason)
        adapter.report_execution(t, accepted=False, reason=selection.reason, controller_stop=True)
        if telemetry is not None:
            telemetry.update(boundary_stop=selection.reason)
        return None, rejects

    def report_verified(sent_q, *, held=False):
        achieved = forward_pose(sent_q)
        boundary = getattr(adapter.safety, "boundary_projection", None)
        if boundary is not None:
            from phantom.deploy.boundary_projection import BoundaryProjectionStop
            try:
                boundary.verify_final(
                    t, achieved, sent_q,
                    verified=bool(selection.all_ik_valid and selection.all_ik_on_branch),
                    dt=dt, previous_pose=forward_pose(qref),
                )
            except BoundaryProjectionStop as error:
                # No proposed grip was submitted on this failed final check;
                # do not generate a new ACK or original-policy opening event.
                adapter.request_stop(error.reason)
                adapter.report_execution(t, accepted=False, reason=error.reason, controller_stop=True)
                if telemetry is not None:
                    telemetry.update(boundary_stop=error.reason)
                return None
            if submit_targets is None:
                raise ValueError("enabled boundary guard requires actual drive submission before ACK")
            # The callback must apply both verified joint and unchanged grip
            # targets, and raise if submission fails. No ACK exists yet.
            if submit_targets(sent_q, gripper_command) is False:
                raise RuntimeError("verified simulator drive submission rejected")
            boundary.note_submission(t, achieved, sent_q, mechanism="isaac_apply_action")
        adapter.report_execution(t, accepted=True, tcp_pose=achieved,
                                 gripper_command=gripper_command, held=held)
        if boundary is not None:
            boundary.acknowledge(t, achieved)
        return achieved

    if hold_budget is not None:
        from phantom.drivers.servo_hold import verified_constraint_hold

        held = verified_constraint_hold(selection, qref, dt, limits)
        if selection.accepted or held:
            sent_q = qref if held else selection.q
            state = hold_budget.check(t, sent_q, held=held)
            if telemetry is not None:
                telemetry.update(state, held=held)
            if state["timed_out"]:
                adapter.request_stop("servo_constraint_hold_timeout")
                boundary = getattr(adapter.safety, "boundary_projection", None)
                adapter.report_execution(t, accepted=False,
                                         gripper_command=gripper_command if boundary is None else None,
                                         reason="servo_constraint_hold_timeout", controller_stop=True)
                if boundary is not None and telemetry is not None:
                    telemetry["drive_not_submitted"] = True
                return None, rejects
            achieved = report_verified(sent_q, held=held)
            return achieved, rejects if held or achieved is None else 0

    if selection.accepted:
        achieved = report_verified(selection.q)
        return achieved, rejects if achieved is None else 0
    rejects += 1
    if rejects >= 25:
        adapter.request_stop("servo_limiter_stall")
    boundary = getattr(adapter.safety, "boundary_projection", None)
    if boundary is not None:
        if submit_targets is None:
            raise ValueError("enabled boundary guard requires actual drive submission before gripper ACK")
        if submit_targets(qref, gripper_command) is False:
            raise RuntimeError("simulator gripper drive submission rejected")
    adapter.report_execution(
        t, accepted=False, gripper_command=gripper_command, reason=selection.reason
    )
    return None, rejects


def apply_episode_overrides(hw, overrides):
    """Replay the recorded effective safety envelope without hardware imports."""
    from phantom.config.hardware import WorkspaceBox
    from phantom.deploy.safety import apply_tcp_speed_limit, apply_z_floor

    unknown = set(overrides) - {"hitbox_m", "z_floor_m", "tcp_speed_m_s"}
    if unknown:
        raise ValueError(
            f"Unsupported recorded deployment overrides: {sorted(unknown)}"
        )
    if "hitbox_m" in overrides:
        recorded = overrides["hitbox_m"]
        box = (
            None
            if recorded is None
            else WorkspaceBox(**{axis: recorded[axis] for axis in "xyz"})
        )
        hw = hw.model_copy(
            update={"safety": hw.safety.model_copy(update={"hitbox_m": box})}
        )
    if "z_floor_m" in overrides:
        hw = apply_z_floor(hw, float(overrides["z_floor_m"]))
    if "tcp_speed_m_s" in overrides:
        hw = apply_tcp_speed_limit(hw, float(overrides["tcp_speed_m_s"]))
    return hw


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported policy trace value: {type(value).__name__}")


class PolicyAudit:
    """Durable inference attempts and drive submissions, independent of Isaac."""

    def __init__(self, output):
        self.output = Path(output)
        self.plans = []
        self.active_replan_id = None
        self.execution = (self.output / "execution_trace.jsonl").open("w", buffering=1)
        try:
            self._save_plans()
        except BaseException:
            self.execution.close()
            raise

    def _save_plans(self):
        destination = self.output / "planner_trace.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.plans, indent=2, default=_json_value) + "\n"
        )
        temporary.replace(destination)

    def begin_replan(self, t):
        index = len(self.plans)
        self.plans.append(
            {"replan_id": index, "t": float(t), "status": "inference_started"}
        )
        self._save_plans()
        return index

    def finish_replan(self, index, plan, wall_time_s):
        self.plans[index].update(
            {
                "status": "pending_activation",
                "t_created": plan.t_created,
                "latency_s": plan.latency_s,
                "inference_wall_time_s": wall_time_s,
                "actions": plan.actions,
                "action_times": plan.action_times,
                "sigma": plan.sigma,
                "gate": plan.gate,
                "p_evt": plan.p_evt,
                "t0_pose": plan.t0_pose,
                "diagnostics": plan.diag,
            }
        )
        self._save_plans()

    def observed(self, snapshot):
        """Persist exactly the numpy inputs supplied to the inference policy."""
        folder = self.output / "observations"
        folder.mkdir(exist_ok=True)
        index = len(self.plans) - 1
        values = {
            name: np.asarray(getattr(snapshot, name))
            for name in (
                "t",
                "rgb",
                "wrist_window",
                "ur_state",
                "gel",
                "fields",
                "contact_state",
                "reactive",
                "prev_chunk",
            )
            if getattr(snapshot, name) is not None
        }
        np.savez_compressed(folder / f"{index:04d}.npz", **values)
        (folder / f"{index:04d}.json").write_text(
            json.dumps(
                {
                    "replan_id": index,
                    "sensor_times": snapshot.sensor_times,
                    "present_fields": sorted(values),
                    "scope": "Exact raw observation passed to native PhantomPolicy preprocessing",
                },
                indent=2,
            )
            + "\n"
        )

    def delivered(self, plan, snapshot, t, activated):
        """Keep the delivered transform separate from the raw policy proposal."""
        with (self.output / "delivered_plans.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "captured_snapshot_t": snapshot.t,
                        "delivery_t": t,
                        "activated": activated,
                        "actions": plan.actions,
                        "diagnostics": plan.diag,
                    },
                    default=_json_value,
                )
                + "\n"
            )

    def fail_replan(self, index, error):
        self.plans[index].update(
            {
                "status": "inference_error",
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        if getattr(error, "infrastructure_invalid", False):
            self.plans[index]["error_category"] = "instrumentation_or_input_invalid"
        self._save_plans()

    def executed(self, row):
        # A successful articulation.apply_action call confirms submission of a
        # setpoint. It does not imply the measured robot reached the setpoint.
        if row.get("diagnostics", {}).get("plan_activated"):
            pending = [
                i
                for i, plan in enumerate(self.plans)
                if plan["status"] == "pending_activation"
                and float(plan["action_times"][0]) <= row["t"] + 1e-9
            ]
            if pending:
                if self.active_replan_id is not None:
                    self.plans[self.active_replan_id]["status"] = "superseded"
                self.active_replan_id = pending[-1]
                self.plans[self.active_replan_id].update(
                    status="active", activated_at=row["t"]
                )
                self._save_plans()
        self.execution.write(
            json.dumps(
                {
                    **row,
                    "status": row.get("status", "drive_submitted"),
                    "active_replan_id": self.active_replan_id,
                },
                default=_json_value,
            )
            + "\n"
        )

    def close(self):
        try:
            for row in self.plans:
                if row["status"] == "pending_activation":
                    row["status"] = "not_activated_before_end"
            self._save_plans()
        finally:
            self.execution.close()


def load_command_replay(args, cfg):
    """Validate target truth before Kit starts, preserving the legacy format."""
    from phantom.sim.command_replay import NativeFingerHoldReplay, NativeRecordedDriveCommands, RecordedDriveCommands
    from phantom.sim.gripper_articulation import is_adaptive, is_articulated

    if not args.command_trace or not args.policy_initial_state:
        raise ValueError("command_replay requires an explicit command trace and measured initial state")
    manifest = getattr(args, "command_trace_manifest", None)
    hold_at = getattr(args, "command_finger_hold_at", None)
    if hold_at is not None and getattr(args, "mode", None) != "command_replay":
        raise ValueError("--command-finger-hold-at requires command_replay mode")
    if is_adaptive(cfg):
        if manifest is None:
            raise ValueError("Native command_replay requires --command-trace-manifest with named radian drive references")
        replay = NativeRecordedDriveCommands(args.command_trace, manifest_path=manifest, cfg=cfg)
        return NativeFingerHoldReplay(replay, hold_at_s=hold_at) if hold_at is not None else replay
    if hold_at is not None:
        raise ValueError("--command-finger-hold-at requires native adaptive command replay")
    if is_articulated(cfg):
        raise ValueError("Legacy command traces contain prismatic jaw metres; coupled W2L command replay remains unsupported")
    if manifest is not None:
        raise ValueError("A native command manifest cannot describe legacy prismatic replay")
    return RecordedDriveCommands(args.command_trace, finger_limit_m=cfg["gripper"]["stroke"] / 2)


def main():
    args = arguments()
    if args.command_finger_hold_at is not None and args.mode != "command_replay":
        raise ValueError("--command-finger-hold-at requires command_replay mode")
    if args.servo_reach_limiter and args.mode != "policy":
        raise ValueError("--servo-reach-limiter is only supported in policy mode")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    from phantom.sim.gripper_articulation import is_adaptive

    if is_adaptive(cfg):
        from phantom.sim.gripper_adaptive import validate_physics_timestep
        validate_physics_timestep(cfg)
    experiment = validate_adaptive_policy_experiment(args, cfg)
    if experiment is not None:
        (args.output / "adaptive_policy_experiment.json").write_text(
            json.dumps(experiment, indent=2) + "\n"
        )

    if args.mode == "command_replay":
        load_command_replay(args, cfg)
    if args.render_hz is not None:
        if args.render_hz <= 0:
            raise ValueError("render-hz must be positive")
        cfg["physics"]["render_hz"] = args.render_hz
    if args.friction_scale <= 0:
        raise ValueError("friction-scale must be positive")
    cfg["waffle"]["static_friction"] *= args.friction_scale
    cfg["waffle"]["dynamic_friction"] *= args.friction_scale
    cfg["waffle"]["center"][0] += args.object_offset[0]
    cfg["waffle"]["center"][1] += args.object_offset[1]
    data = np.load(args.episode / "replay.npz", allow_pickle=False)
    duration = (
        float(data["t"][-1])
        if args.duration is None
        else (
            args.duration
            if args.mode in ("policy", "command_replay")
            else min(args.duration, float(data["t"][-1]))
        )
    )
    if duration <= 0:
        raise ValueError("duration must be positive")
    for delay in (args.observation_delay_s, args.inference_delay_add_s):
        if not np.isfinite(delay) or delay < 0:
            raise ValueError(
                "observation/inference delays must be finite and nonnegative"
            )
    (args.output / "effective_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": not args.gui,
            "width": 640,
            "height": 480,
            "renderer": "RaytracedLighting",
            "anti_aliasing": 3,
            "multi_gpu": False,
            "sync_loads": True,
        }
    )
    exit_code = 0
    try:
        run(app, args, cfg, data, duration)
    except BaseException:  # noqa: BLE001 - preserve failures before Kit exits the process
        import traceback

        failure = traceback.format_exc()
        print(failure, flush=True)
        (args.output / "FAILED.txt").write_text(failure)
        exit_code = 1
    finally:
        app.close(exit_code=exit_code)


def run(app, args, cfg, data, duration):
    import cv2
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pxr import PhysxSchema, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation

    from phantom.sim.kinematics import (
        JOINT_NAMES,
        forward_pose,
        inverse_kinematics,
    )
    from phantom.sim.scene import build_scene, import_robot
    from phantom.sim.gripper_articulation import (
        is_articulated, is_adaptive, finger_joint_names, joint_targets, drive_targets,
        closure_from_joint_positions, finger_drive_type,
    )

    articulated_gripper = is_articulated(cfg)
    adaptive_gripper = is_adaptive(cfg)
    dt = cfg["physics"]["dt"]
    fps = cfg["physics"]["render_hz"]
    robot_usd = (
        str(args.robot_usd.resolve())
        if args.robot_usd
        else import_robot(REPO, args.output, cfg)
    )
    world = World(stage_units_in_meters=1, physics_dt=dt, rendering_dt=1 / fps)
    stage = omni.usd.get_context().get_stage()
    paths = build_scene(stage, REPO, args.output, cfg, robot_usd)
    from phantom.sim.gripper_visual import build as build_gripper_visual

    gripper_visual = None
    housing = stage.GetPrimAtPath(paths["gripper_housing_path"])
    if not articulated_gripper:
        gripper_visual = build_gripper_visual(
            stage, paths["tool_path"], REPO / "assets/sim/robotiq",
            yaw_rad=-np.pi / 2 + cfg["gripper"].get("yaw", 0),
            pad_touch_command=cfg["gripper"]["pad_touch_command"],
        )
        for child in housing.GetChildren():
            if child.GetName().lower() in ("visual", "visuals") or (
                child.IsA(UsdGeom.Gprim) and not child.HasAPI(UsdPhysics.CollisionAPI)
            ):
                UsdGeom.Imageable(child).MakeInvisible()
        for path in paths["pad_paths"]:
            UsdGeom.Imageable(stage.GetPrimAtPath(path + "/Linkage")).MakeInvisible()
    roots = [
        str(p.GetPath())
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one robot articulation; found {roots}")
    physics_articulation = PhysxSchema.PhysxArticulationAPI.Apply(
        stage.GetPrimAtPath(roots[0])
    )
    physics_articulation.CreateSolverPositionIterationCountAttr(
        cfg["physics"]["solver_position_iterations"]
    )
    physics_articulation.CreateSolverVelocityIterationCountAttr(
        cfg["physics"]["solver_velocity_iterations"]
    )
    physics_articulation.CreateEnabledSelfCollisionsAttr(True)
    # SingleArticulation can address the root prim even when API sits on its fixed joint.
    root = roots[0]
    if stage.GetPrimAtPath(root).IsA(UsdPhysics.Joint):
        root = paths["robot_path"]
    robot = world.scene.add(SingleArticulation(prim_path=root, name="ur3"))
    packet = world.scene.add(
        SingleRigidPrim(prim_path=paths["waffle_path"], name="waffle")
    )
    tool_body = SingleRigidPrim(
        prim_path=paths["tool_path"], name="tool_feedback", reset_xform_properties=False
    )
    pads = [
        RigidPrim(
            prim_paths_expr=p,
            name=f"pad_{i}",
            track_contact_forces=True,
            contact_filter_prim_paths_expr=[paths["waffle_path"]],
            max_contact_count=64,
            reset_xform_properties=False,
        )
        for i, p in enumerate(paths["pad_paths"])
    ]
    support_views = None
    if args.record_packet_support:
        from tools.sim.object_support import PacketSupportViews

        support_views = PacketSupportViews(
            paths["waffle_path"],
            [
                f"/World/Bin/{name}"
                for name in ("Bottom", "Left", "Right", "Front", "Back")
            ],
            [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and str(prim.GetPath()).startswith(paths["robot_path"] + "/")
            ],
            rigid_prim_cls=RigidPrim,
        )
    robot_environment_views = None
    if args.record_robot_environment_contacts:
        from tools.sim.robot_environment_contacts import RobotEnvironmentContactViews

        robot_environment_views = RobotEnvironmentContactViews(
            [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and str(prim.GetPath()).startswith(paths["robot_path"] + "/")
            ],
            rigid_prim_cls=RigidPrim,
        )
    gripper_wrist = None
    if args.wrist == "gripper_contact_proxy":
        from tools.sim.gripper_wrist import GripperContactWrist

        gripper_wrist = GripperContactWrist(
            str(housing.GetPath()), paths["pad_paths"], rigid_prim_cls=RigidPrim,
            additional_actor_paths=[
                p for p in paths["gripper_body_paths"]
                if p not in [str(housing.GetPath()), *paths["pad_paths"]]
            ] if articulated_gripper else (),
        )
    gel_views = None
    if args.record_gel_contacts or (
        args.mode == "policy" and args.tactile == "measured_baseline_proxy"
    ):
        from tools.sim.gel_contact import GelContactViews, GelSurfaceGeometry

        environment_paths = [
            paths["waffle_path"],
            "/World/Bench/Slab",
            "/World/Mat/Base",
            *[
                f"/World/Bin/{name}"
                for name in ("Bottom", "Left", "Right", "Front", "Back")
            ],
            *[
                str(prim.GetPath())
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and str(prim.GetPath()).startswith(paths["robot_path"])
                and str(prim.GetPath()) not in paths["pad_paths"]
            ],
        ]
        gel_views = GelContactViews(
            paths["pad_paths"],
            environment_paths,
            rigid_prim_cls=RigidPrim,
            coverage=args.gel_contact_coverage,
            geometry=GelSurfaceGeometry(**cfg["gripper"]["gel_geometry"])
            if articulated_gripper else None,
        )
    cam = cfg["camera"]
    camera = Camera(
        "/World/SceneCamera", frequency=fps, resolution=tuple(cam["resolution"])
    )
    if "world_from_cv" in cam:
        twc = np.asarray(cam["world_from_cv"])
        orient = Rotation.from_matrix(twc[:3, :3] @ np.diag([1, -1, -1])).as_quat()
        camera.set_world_pose(
            position=twc[:3, 3], orientation=orient[[3, 0, 1, 2]], camera_axes="usd"
        )
    else:
        pos = np.asarray(cam["position"])
        target = np.asarray(cam["target"])
        z = (pos - target) / np.linalg.norm(pos - target)
        x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        orient = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
        camera.set_world_pose(
            position=pos, orientation=orient[[3, 0, 1, 2]], camera_axes="usd"
        )
    from phantom.sim.camera import configure_camera_intrinsics, validate_camera_intrinsics

    configure_camera_intrinsics(camera, cam)
    camera.set_clipping_range(0.02, 10)
    camera.set_focus_distance(1.0)
    world.reset()
    for pad in pads:
        pad.initialize()
    if gel_views is not None:
        gel_views.initialize()
    if support_views is not None:
        support_views.initialize()
    if robot_environment_views is not None:
        robot_environment_views.initialize()
    if gripper_wrist is not None:
        gripper_wrist.initialize()
    tool_body.initialize()
    camera.initialize()
    camera_projection_report = validate_camera_intrinsics(camera, cam)
    (args.output / "camera_projection.json").write_text(
        json.dumps(camera_projection_report, indent=2) + "\n"
    )
    if args.gui:
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().set_active_camera("/World/SceneCamera")

    def inspect_gui():
        if args.gui:
            print(
                "Scene ready for inspection; close the Isaac window to exit.",
                flush=True,
            )
            while app.is_running():
                world.step(render=False)
                world.render()

    names = list(robot.dof_names)
    ids = np.array([names.index(n) for n in JOINT_NAMES])
    finger_names = finger_joint_names(cfg)
    fingers = np.array([names.index(n) for n in finger_names])
    print(
        "PHANTOM_SCENE",
        json.dumps(
            {**paths, "articulation_root": root, "dofs": names, "robot_usd": robot_usd}
        ),
        flush=True,
    )
    initial_row = (
        int(np.argmin(np.abs(data["t"] - args.probe_start_time)))
        if args.mode == "contact_probe"
        else 0
    )
    q0 = np.asarray(data["q"][initial_row])
    g0 = float(data["gripper"][initial_row, 0])
    initial_state = None
    if args.policy_initial_state:
        if args.mode not in ("policy", "command_replay"):
            raise ValueError(
                "--policy-initial-state requires policy or command_replay mode"
            )
        initial_state = json.loads(args.policy_initial_state.read_text())
        q0 = np.asarray(initial_state["q"], dtype=float)
        g0 = float(initial_state["gripper"])
        initial_wrist = np.asarray(initial_state["wrist_ft"], dtype=float)
        if (
            q0.shape != (6,)
            or initial_wrist.shape != (6,)
            or not np.isfinite(np.r_[q0, initial_wrist, g0]).all()
            or not 0 <= g0 <= 1
        ):
            raise ValueError(
                "Initial state requires finite q[6], wrist_ft[6], gripper in [0,1]"
            )
    if gripper_visual is not None:
        gripper_visual.update(g0)
    recorded_commands = None
    if args.mode == "command_replay":
        recorded_commands = load_command_replay(args, cfg)
        (args.output / "command_replay.json").write_text(
            json.dumps(recorded_commands.metadata, indent=2) + "\n"
        )

    def finger_target(closure):
        # Direct replay is a kinematic nominal pose only. During physics, the
        # spring rest references must not be replaced by prescribed follower
        # angles; contact determines those passive positions.
        return (joint_targets if args.mode == "replay" else drive_targets)(closure, cfg)

    def finger_closure(values):
        return closure_from_joint_positions(values, cfg)

    native_mechanics_monitor = {"enabled": adaptive_gripper, "checks": 0}
    from phantom.sim.native_failure_audit import (
        NativeFailureHistory, json_value, read_diagnostic, runtime_readback,
    )
    native_failure_history = NativeFailureHistory() if adaptive_gripper else None

    def check_native_mechanics(phase, t, drive_references):
        if not adaptive_gripper:
            return
        from phantom.sim.gripper_adaptive import mechanical_diagnostics

        values = np.asarray(robot.get_joint_positions()[fingers], float)
        velocity_readback = read_diagnostic(
            lambda: np.asarray(robot.get_joint_velocities()[fingers], float)
        )
        velocities = velocity_readback.get("value")
        native_failure_history.observe(
            phase, t, finger_names, values, velocities, drive_references[fingers],
            velocity_read_error=velocity_readback.get("error"),
        )
        try:
            diagnostic = mechanical_diagnostics(values)
        except ValueError as error:
            diagnostic = {"passed": False, "error": str(error)}
        native_mechanics_monitor["checks"] += 1
        native_mechanics_monitor["last_phase"] = phase
        native_mechanics_monitor["last_t_s"] = float(t)
        maxima = native_mechanics_monitor.setdefault("observed_maxima", {})
        for key in ("coupling_max_abs_rad", "joint_limit_violation_max_rad", "loop_closure_max_m"):
            if key in diagnostic:
                maxima[key] = max(maxima.get(key, 0.), diagnostic[key])
        if not diagnostic["passed"]:
            failure = {"phase": phase, "t_s": float(t), "diagnostic": diagnostic,
                       "finger_joint_names": list(finger_names),
                       "finger_q_rad": json_value(values),
                       "finger_qd_rad_s": json_value(velocities),
                       "finger_qd_readback": velocity_readback,
                       "desired_drive_references_rad": dict(zip(finger_names, json_value(drive_references[fingers]))),
                       "recent_physics_steps": native_failure_history.report(),
                       "runtime_readback": read_diagnostic(lambda: runtime_readback(
                           stage, roots[0], paths["joint_paths"], robot, finger_names, fingers
                       )),
                       "failure_step_contacts": {"t_s": float(t), "physics_dt_s": float(dt)}}
            # These getters inspect the already completed physics step. They
            # neither advance physics nor alter policy/safety observation state.
            for label, reader in (
                ("gel", gel_views),
                ("robot_environment", robot_environment_views),
                ("gripper_environment", gripper_wrist.reader if gripper_wrist is not None else None),
            ):
                failure["failure_step_contacts"][label] = (
                    read_diagnostic(lambda reader=reader: reader.get_all(dt))
                    if reader is not None else {"available": False, "reason": "not configured"}
                )
            (args.output / "native_mechanics_failure.json").write_text(
                json.dumps(failure, indent=2, allow_nan=False) + "\n"
            )
            raise RuntimeError(f"Native gripper mechanics failed during {phase} at t={t:.6f}s; see native_mechanics_failure.json")
        native_mechanics_monitor["last_diagnostic"] = diagnostic

    qfull = np.zeros(len(names))
    qfull[ids] = q0
    qfull[fingers] = joint_targets(g0, cfg)
    initial_drive = qfull.copy()
    initial_drive[fingers] = finger_target(g0)
    robot.set_joint_positions(qfull)
    robot.set_joint_velocities(np.zeros_like(qfull))
    robot.apply_action(ArticulationAction(joint_positions=initial_drive))
    for name, index in zip(JOINT_NAMES, ids):
        UsdPhysics.DriveAPI(
            stage.GetPrimAtPath(paths["joint_paths"][name]), "angular"
        ).GetTargetPositionAttr().Set(float(np.degrees(qfull[index])))
    for name, index in zip(finger_names, fingers):
        drive = UsdPhysics.DriveAPI(
            stage.GetPrimAtPath(paths["joint_paths"][name]), finger_drive_type(cfg)
        )
        target = drive.GetTargetPositionAttr()
        if target.IsValid():
            target.Set(float(np.degrees(initial_drive[index]) if articulated_gripper else initial_drive[index]))

    # World.reset() initializes physics at the imported articulation's default
    # pose before measured q0 is assigned. Clear any packet impulse from that
    # transient pose as part of initial conditions, before settling/t=0.
    yaw = float(cfg["waffle"]["yaw"])
    packet.set_world_pose(
        position=np.asarray(cfg["waffle"]["center"]),
        orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
    )
    packet.set_linear_velocity(np.zeros(3))
    packet.set_angular_velocity(np.zeros(3))

    def tcp_measured(q):
        # Independent physics link feedback detects import/frame errors that
        # recomputing DH from q alone would conceal.
        pos, quat = tool_body.get_world_pose()
        rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        return np.r_[np.asarray(pos) + rot.apply([0, 0, 0.18]), rot.as_rotvec()]

    def interp(key, t):
        native = f"native_{key}"
        aliases = {
            "q": "arm_q",
            "qd": "arm_qd",
            "tcp": "arm_tcp_pose",
            "gripper": "gripper",
        }
        native = f"native_{aliases[key]}"
        nt = native + "_t"
        values = data[native] if native in data else data[key]
        tt = data[nt] if nt in data else data["t"]
        # prepare_waffles exports native times relative to common t0.
        return np.array(
            [np.interp(t, tt, values[:, i]) for i in range(values.shape[1])]
        )

    # Warm up renderer with joint pose held; settle object under gravity.
    for _ in range(45):
        robot.set_joint_positions(qfull)
        robot.set_joint_velocities(np.zeros_like(qfull))
        world.step(render=False)
        world.render()
    robot_settling = []
    native_finger_settling = []
    if args.mode in ("policy", "command_replay") or (adaptive_gripper and args.mode == "dynamics"):
        # Pose assignment during renderer initialization is not a controller
        # equilibrium. Let the position drives settle before enabling safety
        # and starting the trial clock; never waive a velocity safety check.
        settling_physics_origin = world.current_time
        for settle_step in range(int(np.ceil(2.0 / dt))):
            robot.apply_action(ArticulationAction(joint_positions=initial_drive))
            world.step(render=False)
            settling_t = world.current_time - settling_physics_origin if adaptive_gripper else settle_step * dt
            robot_settling.append(
                np.r_[
                    settling_t,
                    robot.get_joint_positions()[ids],
                    robot.get_joint_velocities()[ids],
                ]
            )
            if adaptive_gripper:
                native_finger_settling.append(np.r_[settling_t, robot.get_joint_positions()[fingers]])
                try:
                    check_native_mechanics("initialization_settling", settling_t, initial_drive)
                except RuntimeError:
                    np.savez_compressed(args.output / "robot_settling.npz", samples=robot_settling,
                                        native_finger_samples=native_finger_settling)
                    raise
        np.savez_compressed(args.output / "robot_settling.npz", samples=robot_settling,
                            **({"native_finger_samples": native_finger_settling} if adaptive_gripper else {}))
        world.render()
    camera.get_rgba()
    settled_position = np.asarray(packet.get_world_pose()[0])
    next_progress_check_t = 0.0
    settling_error = float(np.linalg.norm(settled_position - cfg["waffle"]["center"]))
    initialization = {
        "adaptive_passive_joints": adaptive_gripper,
        "initial_finger_seed_rad": qfull[fingers].tolist() if adaptive_gripper else None,
        "initial_finger_drive_references_rad": initial_drive[fingers].tolist() if adaptive_gripper else None,
        "passive_settling_s": 2.0 if adaptive_gripper and args.mode in ("dynamics", "policy", "command_replay") else 0.0,
        "policy_initial_state": initial_state,
        "policy_initial_state_sha256": hashlib.sha256(
            args.policy_initial_state.read_bytes()
        ).hexdigest()
        if args.policy_initial_state
        else None,
        "configured_packet_center_m": cfg["waffle"]["center"],
        "settled_packet_center_m": settled_position.tolist(),
        "settling_displacement_m": settling_error,
        "packet_velocity_m_s": np.asarray(packet.get_linear_velocity()).tolist(),
        "packet_support_contacts": support_views.get_all(dt)
        if support_views is not None
        else None,
        "configured_robot_q": q0.tolist(),
        "settled_robot_q": robot.get_joint_positions()[ids].tolist(),
        "settled_robot_qd": robot.get_joint_velocities()[ids].tolist(),
        "policy_robot_settling_s": 2.0
        if args.mode in ("policy", "command_replay")
        else 0.0,
        "robot_initial_max_joint_error_rad": float(
            np.max(np.abs(robot.get_joint_positions()[ids] - q0))
        ),
        "robot_initial_max_joint_speed_rad_s": float(
            np.max(np.abs(robot.get_joint_velocities()[ids]))
        ),
        "method": "Assign measured initial robot pose, reset packet pose and velocities once, then settle before t=0; no packet state writes during driven replay.",
    }
    if robot_environment_views is not None:
        initialization["robot_environment_contacts"] = robot_environment_views.get_all(
            dt
        )
    (args.output / "initialization.json").write_text(
        json.dumps(initialization, indent=2) + "\n"
    )
    check_native_mechanics("after_initialization_settling", 0., initial_drive)
    if adaptive_gripper:
        (args.output / "native_runtime_readback.json").write_text(
            json.dumps(read_diagnostic(lambda: runtime_readback(
                stage, roots[0], paths["joint_paths"], robot, finger_names, fingers
            )), indent=2, allow_nan=False) + "\n"
        )
        initialization["native_mechanics"] = native_mechanics_monitor["last_diagnostic"]
        initialization["settled_finger_q_rad"] = robot.get_joint_positions()[fingers].tolist()
        (args.output / "initialization.json").write_text(
            json.dumps(initialization, indent=2) + "\n"
        )
    if not np.isfinite(settling_error) or settling_error > 0.02:
        raise RuntimeError(
            f"Packet moved {settling_error:.4f}m before replay; initial geometry/settling is invalid"
        )
    if args.mode in ("policy", "command_replay") and (
        initialization["robot_initial_max_joint_error_rad"] > 0.02
        or initialization["robot_initial_max_joint_speed_rad_s"] > 0.05
    ):
        raise RuntimeError(
            "Robot failed pre-trial equilibrium checks: joint error must be <=0.02rad and speed <=0.05rad/s; no policy inference attempted"
        )
    # Save the initial scene after physics has written the joint pose to USD.
    # The flattened USD and packaged USDZ can also be opened in the Isaac GUI.
    if not args.skip_stage_export or args.save_stage_only:
        stage.GetRootLayer().Export(str(args.output / "waffles_scene.usda"))
        stage.Flatten().Export(str(args.output / "waffles_scene.usd"))
        from pxr import UsdUtils

        if not UsdUtils.CreateNewUsdzPackage(
            str(args.output / "waffles_scene.usd"),
            str(args.output / "waffles_scene.usdz"),
        ):
            raise RuntimeError("Could not package standalone waffles_scene.usdz")
    if args.save_stage_only:
        print(
            "PHANTOM_SCENE_SAVED", str(args.output / "waffles_scene.usdz"), flush=True
        )
        inspect_gui()
        return
    writer = cv2.VideoWriter(
        str(args.output / "sim.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        tuple(cam["resolution"]),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create sim.mp4")
    adapter = policy = audit = probe = None
    boundary_audit = None
    emergency_authority = emergency_trace = None
    boundary_loop_completed = False
    if args.mode == "contact_probe":
        if articulated_gripper:
            from phantom.sim.gripper_probe import initialize as initialize_probe
        else:
            from phantom.sim.contact_probe import initialize as initialize_probe

        tool_position, tool_orientation = tool_body.get_world_pose()
        probe = initialize_probe(
            packet, robot, fingers, qfull, cfg, tool_position, tool_orientation
        )
        probe.metadata["initial_recorded_pose_time_s"] = float(data["t"][initial_row])
        (args.output / "contact_probe.json").write_text(
            json.dumps(probe.metadata, indent=2) + "\n"
        )
    pending_execution = None
    hw = None
    servo_reach_limits = None
    servo_limiter_rejects = 0
    servo_hold_budget = None
    policy_ready_t = None
    next_tactile_t = 0.0
    tactile_sample = None
    policy_tactile_trace = []
    policy_gel_trace = []
    policy_gel_normal_trace = []
    policy_gel_contact_diagnostics = []
    robot_environment_contact_trace = []
    measured_sensor_proxy = None
    next_control = 0.0
    wrist_bias = np.zeros(6)
    wrist_contact_file = None
    wrist_value = None
    wrist_sample_t = None
    next_wrist_t = 0.0
    stop_after_step = False
    terminal_until = None

    def measured_pad_forces():
        # Missing physics contact sensing invalidates the proxy, not an all-zero
        # no-contact measurement. Let failures reach FAILED.txt and cleanup.
        return [
            np.asarray(pad.get_net_contact_forces(dt=dt)).reshape(-1, 3).sum(axis=0)
            for pad in pads
        ]

    def wrist_proxy(tcp, forces):
        if args.wrist == "zero_ablation":
            return np.zeros(6)
        if gripper_wrist is not None:
            if wrist_value is None:
                raise RuntimeError("Gripper wrist input has not been sampled")
            return wrist_value.copy()
        value = wrist_bias.copy()
        for pad, force in zip(pads, forces):
            position = np.asarray(pad.get_world_poses()[0]).reshape(-1, 3)[0]
            value[:3] += force
            value[3:] += np.cross(position - tcp[:3], force)
        return value

    def sample_gripper_wrist(t, tcp):
        nonlocal wrist_value, wrist_sample_t, next_wrist_t
        wrist_value, record = gripper_wrist.sample(tcp, wrist_bias, dt)
        wrist_sample_t = float(t)
        wrist_contact_file.write(json.dumps({"t": t, **record}) + "\n")
        next_wrist_t = t + control_dt

    def tactile_proxy(t, forces):
        if measured_sensor_proxy is not None:
            contacts = gel_views.get_all(dt)
            frame = measured_sensor_proxy.synthesize(
                contacts["normal_force_n"], contacts["contact_uv"]
            )
            policy_gel_normal_trace.append(np.asarray(contacts["normal_force_n"]))
            policy_gel_contact_diagnostics.append({"t": t, **contacts})
            return frame.as_sensor_dict(
                t, [sensor.name for sensor in hw.tactile.sensors]
            )
        out = {}
        for i, sensor in enumerate(hw.tactile.sensors):
            force = forces[i] if args.tactile == "contact_proxy" else np.zeros(3)
            # Explicit low-dimensional contact proxy. The SDK's photometric and
            # depth units are unknown; these fields must not be called calibrated.
            h, w = hw.recording.field_ds.hw
            yy, xx = np.mgrid[-1 : 1 : complex(h), -1 : 1 : complex(w)]
            pressure = np.exp(-(xx * xx + yy * yy) / 0.18) * min(
                np.linalg.norm(force) / 15, 0.8
            )
            fields = np.zeros((h, w, 8), np.float32)
            fields[..., 2] = pressure
            fields[..., 7] = pressure
            kh, kw = hw.recording.keyframe_ds.hw
            key = cv2.resize(fields, (kw, kh))
            gh, gw, _ = hw.tactile.infer_img.hwc
            gel = (
                np.clip(cv2.resize(pressure, (gw, gh)) * 150 + 70, 0, 255).astype(
                    np.uint8
                )
                if args.tactile == "contact_proxy"
                else np.zeros((gh, gw), np.uint8)
            )
            out[sensor.name] = {
                "t": t,
                "fields_ds": fields,
                "keyframe": key,
                "infer_img": gel,
                "wrench": np.r_[0, 0, np.linalg.norm(force), 0, 0, 0],
                "area": float((pressure > 0.05).mean()),
            }
        return out

    trace = {
        k: []
        for k in [
            "t",
            "physics_t",
            "q",
            "qd",
            "tcp",
            "tcp_nominal_fk",
            "gripper",
            "waffle_position",
            "frame_t",
            "target_q",
            "pad_force",
            "pad_position",
            "pad_orientation_wxyz",
            "pad_force_local",
            "pad_packet_force",
            "pad_contact_uv",
            "pad_contact_count",
            "pad_packet_normal_force",
            "waffle_orientation_wxyz",
        ]
    }
    events = []
    next_frame = 0.0
    pushed = False
    previous_tcp = None
    previous_frame_t = None
    rgb = None
    desired = initial_drive.copy()
    nsteps = int(np.floor((duration + (0.5 if args.mode == "policy" else 0)) / dt))
    physics_origin = world.current_time
    start = time.monotonic()
    try:
        if gripper_wrist is not None:
            # Command replay needs the same recorded residual as policy mode.
            # Loading this configuration is read-only and imports no driver.
            from phantom.config.hardware import load_hardware

            wrist_hw = load_hardware(args.hardware_config, quiet=True)
            control_dt = 1 / wrist_hw.control.executor_rate_hz
            if dt > control_dt + 1e-9:
                raise ValueError("Physics dt exceeds wrist sample/control period")
            if "native_arm_ft" in data:
                ft_idx = int(np.argmin(np.abs(data["native_arm_ft_t"])))
                wrist_bias = np.asarray(data["native_arm_ft"][ft_idx], dtype=float)
            if initial_state is not None:
                wrist_bias = initial_wrist.copy()
            wrist_contact_file = (args.output / "wrist_contact_trace.jsonl").open("w")
        if args.mode == "policy":
            audit = PolicyAudit(args.output)
            from phantom.config.hardware import load_hardware
            from phantom.sim.policy_adapter import SimulationPolicyAdapter
            from phantom.sim.remote_policy import RemoteSimulationPolicy
            from tools.sim.deployment_filters import PlannerStallWatchdog

            stall_watchdog = PlannerStallWatchdog()

            hw = load_hardware(args.hardware_config, quiet=True)
            if args.tactile == "measured_baseline_proxy":
                from phantom.sim.tactile_proxy import MeasuredBaselineTactileProxy

                if not args.tactile_baseline:
                    raise ValueError(
                        "measured_baseline_proxy requires --tactile-baseline"
                    )
                measured_sensor_proxy = MeasuredBaselineTactileProxy(
                    args.tactile_baseline
                )
            manifest = json.loads((args.episode / "manifest.json").read_text())
            overrides = (
                {}
                if args.ignore_episode_overrides
                else manifest.get("meta", {}).get("deploy_overrides", {})
            )
            hw = apply_episode_overrides(hw, overrides)
            if args.servo_reach_limiter:
                from phantom.drivers.servo_limiter import ServoLimits

                # Historical opt-in native values. Keep the measured stop and
                # all other hardware fields unchanged; no task geometry enters.
                hw = hw.model_copy(
                    update={
                        "safety": hw.safety.model_copy(
                            update={
                                "elbow_min_rad": 0.40,
                                "servo_joint_speed_max_rad_s": 1.0,
                            }
                        )
                    }
                )
                servo_reach_limits = ServoLimits(0.40, 1.0)
                if args.servo_constraint_hold_s is not None:
                    from phantom.drivers.servo_hold import ConstraintHoldBudget

                    servo_hold_budget = ConstraintHoldBudget(args.servo_constraint_hold_s)
                    hw = hw.model_copy(update={"safety": hw.safety.model_copy(
                        update={"servo_constraint_hold_s": args.servo_constraint_hold_s}
                    )})
            control_dt = 1 / hw.control.executor_rate_hz
            if dt > control_dt + 1e-9:
                raise ValueError(
                    "physics dt must not exceed the policy executor period"
                )
            source_text = str(manifest.get("meta", {}).get("text", ""))
            host, port = args.policy_server.rsplit(":", 1)
            policy = RemoteSimulationPolicy(
                (host, int(port)),
                config=(
                    json.loads(args.policy_config.read_text())
                    if args.policy_config
                    else {"parity_fixes": True, "task_text": source_text}
                ),
            )
            expected_student = args.policy_mode == "student"
            reported_student = policy.info.get("student")
            if (
                reported_student is not None
                and bool(reported_student) != expected_student
            ):
                raise ValueError(
                    "Checkpoint architecture and requested observation mode disagree"
                )
            terminal_veto = None
            terminal_veto_spec = None
            if args.terminal_veto_config:
                from tools.sim.deployment_filters import TerminalVetoFilter

                terminal_veto_spec = json.loads(args.terminal_veto_config.read_text())
                required = {"z_ref", "z_floor", "z_margin", "open_aperture"}
                if not required <= terminal_veto_spec["config"].keys():
                    raise ValueError(
                        "Terminal veto requires explicit task height and open aperture settings"
                    )
                terminal_veto = TerminalVetoFilter(
                    hw,
                    terminal_veto_spec["config"],
                    implementation=terminal_veto_spec["implementation"],
                    controller_profile=args.placement_controller_profile,
                )
            release_spec = None
            release_config = None
            boundary_spec = None
            if args.boundary_projection_config:
                from phantom.deploy.boundary_projection import BoundaryProjectionConfig
                boundary_spec = BoundaryProjectionConfig.from_dict(
                    json.loads(args.boundary_projection_config.read_text())).to_dict()
                if servo_reach_limits is None or servo_hold_budget is None:
                    raise ValueError("boundary projection requires verified shared servo limiter and hold budget")
            if args.placement_release_config:
                from phantom.sim.release_controller import PlacementReleaseConfig

                release_spec = json.loads(args.placement_release_config.read_text())
                release_config = PlacementReleaseConfig.from_dict(release_spec)
            adapter = SimulationPolicyAdapter(
                hw,
                policy,
                mode=args.policy_mode,
                max_play_steps=args.max_play_steps,
                grip_play_steps=args.grip_play_steps,
                open_aperture=terminal_veto.veto.open_aperture
                if terminal_veto
                else min(g0, hw.gripper.max_close_cmd),
                plan_filter=terminal_veto,
                observation_callback=audit.observed
                if args.save_policy_observations
                else None,
                delivered_plan_callback=audit.delivered,
                release_config=release_config,
                boundary_config=boundary_spec,
                controller_profile=args.placement_controller_profile,
                policy_delivery_clock=args.policy_delivery_clock,
            )
            adapter.reset(seed=args.seed)
            if adapter.safety.boundary_projection is not None:
                from phantom.sim.boundary_audit import BoundaryPhysicsAudit
                boundary_audit = BoundaryPhysicsAudit(
                    adapter.safety.boundary_projection, dt,
                    hashlib.sha256(args.boundary_projection_config.read_bytes()).hexdigest(),
                )
            if "native_arm_ft" in data:
                ft_idx = int(np.argmin(np.abs(data["native_arm_ft_t"])))
                wrist_bias = np.asarray(data["native_arm_ft"][ft_idx], dtype=float)
            if initial_state is not None:
                wrist_bias = initial_wrist.copy()
            (args.output / "policy_info.json").write_text(
                json.dumps(
                    {
                        **policy.info,
                        "task_text_source": str(args.policy_config)
                        if args.policy_config
                        else "prepared episode manifest.meta.text",
                        "wrist_model": args.wrist,
                        "wrist_proxy_metadata": gripper_wrist.metadata()
                        if gripper_wrist is not None
                        else None,
                        "wrist_sampling_rate_hz": 1 / control_dt
                        if gripper_wrist is not None
                        else None,
                        "initial_recorded_wrist_bias": wrist_bias.tolist(),
                        "observation_warmup_s": hw.wrist_ft.window_s,
                        "control_rate_hz": hw.control.executor_rate_hz,
                        "deploy_overrides": overrides,
                        "tactile_model": args.tactile,
                        "tactile_baseline_provenance": {
                            "path": str(args.tactile_baseline),
                            "sha256": hashlib.sha256(
                                args.tactile_baseline.read_bytes()
                            ).hexdigest(),
                        }
                        if measured_sensor_proxy
                        else None,
                        "tactile_proxy_parameters": asdict(
                            measured_sensor_proxy.parameters
                        )
                        if measured_sensor_proxy
                        else None,
                        "gel_contact_coverage": args.gel_contact_coverage
                        if measured_sensor_proxy
                        else None,
                        "tactile_contact_mapping": "Gel-only normal contact, explicit inner face/active patch filtering. Tangential force and displacement unmodeled. Static measured baseline, physics-responsive deformation; transfer unvalidated."
                        if measured_sensor_proxy
                        else None,
                        "mode": args.policy_mode,
                        "hardware_config": str(args.hardware_config),
                        "hardware_effective": hw.model_dump(mode="json"),
                        "max_play_steps": args.max_play_steps,
                        "grip_play_steps": args.grip_play_steps,
                        "effective_grip_play_steps": adapter._grip_play_limit(),
                        "placement_controller_profile": adapter.controller_profile_metadata,
                        "wrench_baseline_rows": adapter.wrench_baseline_rows,
                        "pre_first_plan_control": "retain_existing_drive_targets_without_execution_feedback",
                        "planner_stall_watchdog": True,
                        **(
                            {
                                "servo_reach_limiter": {
                                    **asdict(servo_reach_limits),
                                    "algorithm": "shared_native_bisection_slide_v1",
                                    "measured_wrist_extension_stop_m": hw.safety.wrist_extension_stop_m,
                                    "consecutive_reject_limit": 25,
                                    "tracking_guarantee": False,
                                    **({"constraint_hold_s": args.servo_constraint_hold_s,
                                        "constraint_hold_progress_rad": 0.001}
                                       if servo_hold_budget is not None else {}),
                                }
                            }
                            if servo_reach_limits is not None
                            else {}
                        ),
                        "terminal_veto": terminal_veto_spec or False,
                        "terminal_veto_feedback_source": terminal_veto.feedback_source(
                            adapter
                        )
                        if terminal_veto
                        else None,
                        "placement_release": release_spec,
                        "boundary_projection": boundary_spec,
                        "boundary_projection_config_sha256": hashlib.sha256(
                            args.boundary_projection_config.read_bytes()).hexdigest()
                        if args.boundary_projection_config else None,
                        "record_packet_support": args.record_packet_support,
                        "save_policy_observations": args.save_policy_observations,
                        "policy_initial_state_provenance": {
                            "path": str(args.policy_initial_state),
                            "sha256": hashlib.sha256(
                                args.policy_initial_state.read_bytes()
                            ).hexdigest(),
                        }
                        if args.policy_initial_state
                        else None,
                        "policy_latency_override_s": args.policy_latency,
                        "policy_delivery_clock": args.policy_delivery_clock,
                        "policy_call_wall_measurement": "Complete synchronous policy.replan client call; excludes observation callbacks and runner audit bookkeeping"
                        if args.policy_delivery_clock == "rpc_wall"
                        else None,
                        "inference_delivery_clock": "explicit fixed latency plus response delay; native compute and RPC times logged separately"
                        if args.policy_latency is not None
                        else "measured complete client policy call plus configured response delay; native Plan.latency_s and action grid preserved"
                        if args.policy_delivery_clock == "rpc_wall"
                        else "native inference latency plus explicitly configured response delay; RPC overhead is separately logged",
                        "phantom_recovery": bool(
                            terminal_veto and terminal_veto.implementation == "live"
                        ),
                        "observation_delay_s": args.observation_delay_s,
                        "inference_delay_add_s": args.inference_delay_add_s,
                        "observation_delay_scope": "all model inputs, causal history; safety and executor retain current feedback",
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
        for step in range(nsteps + 1):
            t = step * dt
            submitted_this_tick = False
            check_native_mechanics("execution", t, desired)
            qactual = robot.get_joint_positions()[ids]
            qd = robot.get_joint_velocities()[ids]
            tcp = tcp_measured(qactual)
            measured_capture_t = t  # no world.step occurs before submission
            if (
                gripper_wrist is not None
                and (wrist_value is None or args.mode != "policy" or stop_after_step)
                and t + 1e-9 >= next_wrist_t
            ):
                sample_gripper_wrist(t, tcp)
            if recorded_commands is not None:
                targets = recorded_commands.at(t)
                if targets is not None:
                    desired[ids], desired[fingers] = targets
            elif args.mode in ("replay", "dynamics"):
                desired[ids] = interp("q", t)
                desired[fingers] = finger_target(interp("gripper", t)[0])
                if args.mode == "replay":
                    robot.set_joint_positions(desired)
                    robot.set_joint_velocities(np.zeros_like(desired))
            elif probe is not None:
                desired[ids], finger_gap = probe.targets(t, qactual)
                desired[fingers] = finger_gap
            elif not stop_after_step and rgb is not None and t + 1e-9 >= next_control:
                next_control = t + control_dt
                closure = finger_closure(robot.get_joint_positions()[fingers])
                twist = measured_tcp_twist(previous_tcp, tcp, dt)
                forces = measured_pad_forces()
                if gripper_wrist is not None:
                    sample_gripper_wrist(t, tcp)
                measured_wrist = wrist_proxy(tcp, forces)
                target_closure = finger_closure(desired[fingers])
                obj = measured_gripper_status(
                    closure, target_closure, max(np.linalg.norm(f) for f in forces)
                )
                if t + 1e-9 >= next_tactile_t:
                    tactile_sample = tactile_proxy(t, forces)
                    policy_gel_trace.append(
                        np.stack(
                            [
                                tactile_sample[sensor.name]["infer_img"]
                                for sensor in hw.tactile.sensors
                            ]
                        )
                    )
                    policy_tactile_trace.append(
                        (
                            t,
                            np.array(forces)
                            if args.tactile != "zero_ablation"
                            else np.zeros_like(forces),
                        )
                    )
                    while next_tactile_t <= t + 1e-9:
                        next_tactile_t += 1 / hw.tactile.rate_hz
                if policy_ready_t is None:
                    policy_ready_t = t + hw.wrist_ft.window_s + args.observation_delay_s
                adapter.observe(
                    t,
                    rgb=rgb,
                    camera_t=previous_frame_t,
                    q=qactual,
                    qd=qd,
                    tcp_pose=tcp,
                    tcp_speed=twist,
                    gripper_state=[closure, obj],
                    wrist_ft=measured_wrist,
                    tactile=tactile_sample,
                )
                if (
                    args.no_progress_stop_s is not None
                    and t + 1e-9 >= args.no_progress_stop_s
                    and t + 1e-9 >= next_progress_check_t
                    and adapter.stopped_reason is None
                    and not getattr(adapter, "completed_reason", None)
                ):
                    next_progress_check_t = t + 0.5
                    packet_lift_now = float(np.asarray(packet.get_world_pose()[0])[2] - settled_position[2])
                    if packet_lift_now < 0.03:
                        adapter.request_stop("no_progress_timeout")
                        events.append({"t": t, "event": "no_progress_timeout",
                                       "packet_lift_m": packet_lift_now,
                                       "deadline_s": args.no_progress_stop_s})
                if (
                    t + 1e-9 >= policy_ready_t
                    and not getattr(adapter, "completed_reason", None)
                    and adapter.ready_for_replan(t)
                ):
                    stall_check = stall_watchdog.check(t, tcp, adapter._last_cmd)
                    events.append({"event": "planner_stall_watchdog", **stall_check})
                    if stall_check["stop_reason"]:
                        adapter.request_stop(stall_check["stop_reason"])
                    else:
                        replan_id = audit.begin_replan(t)
                        replan_started = time.monotonic()
                        try:
                            proposed = adapter.replan(
                                t=t,
                                latency_s=args.policy_latency,
                                observation_delay_s=args.observation_delay_s,
                                inference_delay_add_s=args.inference_delay_add_s,
                            )
                        except BaseException as error:
                            audit.fail_replan(replan_id, error)
                            raise
                        audit.finish_replan(
                            replan_id, proposed, time.monotonic() - replan_started
                        )
                command = adapter.step(t)
                # No accepted plan: keep every initial motor and passive-spring
                # drive reference; measured closure is observation-only.
                if command is not None:
                    pending_execution = {
                        "t": t,
                        "camera_t": previous_frame_t,
                        "measured_q": np.array(qactual, copy=True),
                        "measured_qd": np.array(qd, copy=True),
                        "measured_tcp": np.array(tcp, copy=True),
                        "measured_gripper": [closure, obj],
                        "measured_wrist_ft": measured_wrist,
                        "wrist_capture_t": wrist_sample_t
                        if gripper_wrist is not None
                        else t,
                        "tactile_capture_t": policy_tactile_trace[-1][0],
                        "requested_tcp": command.tcp_pose,
                        "gripper_command": command.gripper,
                        "command_dt": command.dt,
                        "stopped": command.stopped,
                        "stop_reason": command.reason,
                        "diagnostics": command.diagnostics,
                    }
                    # The gripper worker executes independently of arm IK, including
                    # the safety layer's release command on a stopped episode.
                    if command.stopped:
                        from phantom.sim.emergency_stop import (
                            EmergencySubmissionFault, MeasuredStopAuthority,
                            persist_submission_fault, write_receipt,
                        )
                        emergency_trace = (args.output / "emergency_stop_trace.jsonl").open("x")
                        boundary = adapter.safety.boundary_projection
                        emergency_authority = MeasuredStopAuthority(
                            hw=hw, stop_t=t, stop_reason=command.reason,
                            safety_events=command.diagnostics.get("safety_events", []),
                            physics_dt=dt,
                            feedback_max_age_s=boundary.config.feedback_max_age_s
                            if boundary is not None else control_dt,
                            emit=lambda row: write_receipt(emergency_trace, row),
                        )
                        stopped_target = desired.copy()
                        stopped_target[fingers] = finger_target(command.gripper)
                        stopped_target[ids] = qactual
                        try:
                            emergency_receipt = emergency_authority.submit(
                                step=step, t=t, capture_t=measured_capture_t,
                                q=qactual, qd=qd, tcp=tcp, candidate=stopped_target,
                                arm_ids=ids, gripper_command=command.gripper,
                                apply=lambda target: robot.apply_action(
                                    ArticulationAction(joint_positions=target)),
                                wrist_capture_t=wrist_sample_t,
                                tactile_capture_t=policy_tactile_trace[-1][0]
                                if policy_tactile_trace else None,
                            )
                        except EmergencySubmissionFault as error:
                            persist_submission_fault(error=error, audit=audit,
                                row=pending_execution, pause=world.pause)
                            pending_execution = None
                            raise
                        # Commit accepted state only after the actual setter
                        # returns without rejection; retain the exact old q/grip.
                        desired[:] = stopped_target
                        submitted_this_tick = True
                        pending_execution.update(status="emergency_stop_submitted",
                                                 emergency_stop=emergency_receipt)
                        adapter.report_execution(
                            t, accepted=True, tcp_pose=tcp, gripper_command=command.gripper
                        )
                        pending_execution.update(
                            ik_success=None,
                            ik_reason="safety_hold",
                            accepted_tcp=np.array(tcp, copy=True),
                        )
                        events.append(
                            {"t": t, "event": "policy_stop", "reason": command.reason}
                        )
                        stop_after_step = True
                        # Continue observing a released free body after a safety
                        # stop so the fixed settling criterion can be evaluated.
                        # The common trial horizon still bounds the observation.
                        terminal_until = min(t + 2.0, duration)
                    elif servo_reach_limits is not None:
                        from phantom.drivers.servo_limiter import select_servo_step

                        def nominal_servo_ik(pose, seed):
                            result = inverse_kinematics(
                                pose, seed, max_joint_delta_rad=0.35
                            )
                            # Return a converged off-branch solution so the shared
                            # selector explicitly rejects it before any shortening.
                            return (
                                result.q.tolist()
                                if result.success or result.reason == "branch_guard"
                                else []
                            )

                        boundary = adapter.safety.boundary_projection
                        try:
                            selection = None if boundary is None else boundary.terminal_selection(
                                forward_pose(desired[ids]), desired[ids].tolist(), control_dt, servo_reach_limits)
                        except Exception as error:
                            from phantom.deploy.boundary_projection import BoundaryProjectionStop
                            from phantom.drivers.servo_limiter import ServoStep
                            if not isinstance(error, BoundaryProjectionStop):
                                raise
                            selection = ServoStep(None, None, error.reason, "stop", None, None, 0)
                        if selection is None:
                            selection = select_servo_step(
                                command.tcp_pose,
                                forward_pose(desired[ids]),
                                desired[ids].tolist(),
                                control_dt,
                                nominal_servo_ik,
                                servo_reach_limits,
                            )
                            if boundary is not None:
                                selection = boundary.refine_rate_selection(
                                    selection, forward_pose(desired[ids]), desired[ids].tolist(),
                                    control_dt, nominal_servo_ik, servo_reach_limits, forward_pose)
                        held_qref = desired[ids].copy()
                        pending_execution.update(
                            ik_success=selection.accepted,
                            ik_reason=selection.reason,
                            servo_reach_limiter={
                                "mode": selection.mode,
                                "violation": selection.violation,
                                "fraction": selection.fraction,
                                "ik_calls": selection.ik_calls,
                                "all_ik_valid": selection.all_ik_valid,
                                "all_ik_on_branch": selection.all_ik_on_branch,
                            },
                        )
                        hold_telemetry = {}

                        def submit_verified_targets(joints, grip):
                            nonlocal submitted_this_tick
                            submitted = desired.copy()
                            submitted[ids] = joints
                            submitted[fingers] = finger_target(grip)
                            ok = robot.apply_action(ArticulationAction(joint_positions=submitted))
                            if ok is False:
                                raise RuntimeError("verified simulator drive submission rejected")
                            desired[:] = submitted
                            submitted_this_tick = True

                        achieved, servo_limiter_rejects = report_servo_limiter_execution(
                            adapter, t, selection, command.gripper, servo_limiter_rejects,
                            hold_budget=servo_hold_budget, qref=held_qref,
                            limits=servo_reach_limits, dt=control_dt, telemetry=hold_telemetry,
                            submit_targets=submit_verified_targets if boundary is not None else None,
                        )
                        if hold_telemetry.get("boundary_stop") or hold_telemetry.get("drive_not_submitted"):
                            # Final IK rejection precedes both motor target
                            # submission and any new gripper acknowledgement.
                            pending_execution["gripper_command"] = adapter._last_grip
                            if hold_telemetry.get("boundary_stop"):
                                events.append({"t": t, "event": "controller_stop",
                                               "reason": hold_telemetry["boundary_stop"]})
                        else:
                            desired[fingers] = finger_target(command.gripper)
                        if achieved is not None and selection.accepted:
                            desired[ids] = selection.q
                        if hold_telemetry:
                            pending_execution["servo_reach_limiter"]["constraint_hold"] = hold_telemetry
                            if hold_telemetry.get("held") and achieved is not None:
                                pending_execution.update(ik_success=True, ik_reason="constraint_hold")
                        if hold_telemetry.get("timed_out"):
                            pending_execution.update(ik_success=None, ik_reason="servo_constraint_hold_timeout")
                            events.append({"t": t, "event": "controller_stop", "reason": "servo_constraint_hold_timeout"})
                        elif achieved is None:
                            events.append({"t": t, "event": "ik_rejected", "reason": selection.reason})
                        pending_execution["accepted_tcp"] = achieved
                        pending_execution["servo_reach_limiter"]["consecutive_rejects"] = (
                            servo_limiter_rejects
                        )
                    else:
                        desired[fingers] = finger_target(command.gripper)
                        ik = inverse_kinematics(
                            command.tcp_pose, desired[ids], max_joint_delta_rad=0.35
                        )
                        pending_execution.update(
                            ik_success=ik.success,
                            ik_reason=ik.reason,
                            ik_position_error_m=ik.position_error_m,
                            ik_rotation_error_rad=ik.rotation_error_rad,
                            ik_iterations=ik.iterations,
                        )
                        if ik.success:
                            delta = np.clip(
                                ik.q - desired[ids],
                                -hw.arm.limits.joint_speed_rad_s * control_dt,
                                hw.arm.limits.joint_speed_rad_s * control_dt,
                            )
                            desired[ids] += delta
                            adapter.report_execution(
                                t,
                                accepted=True,
                                tcp_pose=forward_pose(desired[ids]),
                                gripper_command=command.gripper,
                            )
                            pending_execution["accepted_tcp"] = forward_pose(desired[ids])
                        else:
                            adapter.report_execution(
                                t,
                                accepted=False,
                                gripper_command=command.gripper,
                                reason=ik.reason,
                            )
                            events.append(
                                {
                                    "t": t,
                                    "event": "ik_rejected",
                                    "reason": ik.reason,
                                    "position_error_m": ik.position_error_m,
                                    "rotation_error_rad": ik.rotation_error_rad,
                                }
                            )
                            pending_execution["accepted_tcp"] = None
                    pending_execution["target_q"] = desired[ids].copy()
                    pending_execution["target_finger_q"] = desired[fingers].copy()
            if boundary_audit is not None:
                # This is the same measured state read at loop entry; physics
                # has not advanced during control. Include the newly accepted
                # drive target before submission, and every subsequent hold.
                boundary_audit.note_selected(
                    adapter.safety.boundary_projection.last.get("selected_tcp_pose"),
                    kind=adapter.safety.boundary_projection.last.get("selected_target_kind", "continuation"),
                )
                boundary_audit.sample(
                    step, t, tcp, forward_pose(desired[ids]), qactual, qd,
                    phase="stop_hold" if stop_after_step else "finish_hold"
                    if adapter.completed_reason else "active",
                )
            if not submitted_this_tick:
                if emergency_authority is not None:
                    try:
                        emergency_authority.submit(
                            step=step, t=t, capture_t=measured_capture_t,
                            q=qactual, qd=qd, tcp=tcp, candidate=desired,
                            arm_ids=ids, gripper_command=adapter._last_grip,
                            apply=lambda target: robot.apply_action(
                                ArticulationAction(joint_positions=target)),
                            wrist_capture_t=wrist_sample_t,
                            tactile_capture_t=policy_tactile_trace[-1][0]
                            if policy_tactile_trace else None,
                        )
                    except EmergencySubmissionFault as error:
                        persist_submission_fault(error=error, audit=audit,
                            row={"t": t, "stopped": True,
                                 "stop_reason": adapter.stopped_reason,
                                 "measured_q": qactual, "measured_qd": qd,
                                 "measured_tcp": tcp}, pause=world.pause)
                        raise
                else:
                    robot.apply_action(ArticulationAction(joint_positions=desired))
            if pending_execution is not None:
                audit.executed(pending_execution)
                pending_execution = None
            if args.push_at is not None and t >= args.push_at and not pushed:
                packet.set_linear_velocity(np.array(args.push_velocity))
                events.append(
                    {
                        "t": t,
                        "event": "packet_velocity_impulse",
                        "velocity": args.push_velocity,
                    }
                )
                pushed = True
            last_tick = t + dt >= (
                terminal_until if terminal_until is not None else duration
            )
            if boundary_audit is not None:
                # Enabled audits include the final readback at the terminal
                # horizon, not just the step preceding it. Default path stays.
                last_tick = t + 1e-9 >= (terminal_until if terminal_until is not None else duration)
            render = t + 1e-9 >= next_frame or last_tick
            previous_tcp = tcp
            if render:
                if args.record_gel_contacts and args.mode != "policy":
                    policy_gel_contact_diagnostics.append(
                        {"t": t, **gel_views.get_all(dt)}
                    )
                if gripper_visual is not None:
                    gripper_visual.update(finger_closure(robot.get_joint_positions()[fingers]))
                world.render()
                rgba = camera.get_rgba()
                if rgba is None or rgba.size == 0:
                    raise RuntimeError("Camera returned no render")
                rgb = np.ascontiguousarray(rgba[:, :, :3]).astype(np.uint8)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not trace["frame_t"]:
                    cv2.imwrite(
                        str(args.output / "sim_first.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    )
                if len(trace["frame_t"]) % 60 == 0:
                    print(
                        f"PHANTOM_FRAME t={t:.3f} elapsed={time.monotonic() - start:.1f}",
                        flush=True,
                    )
                actual = robot.get_joint_positions()
                forces = []
                pad_positions = []
                pad_orientations = []
                local_forces = []
                packet_forces = []
                contact_uvs = []
                contact_counts = []
                packet_normal_forces = []
                for pad in pads:
                    forces.append(
                        np.asarray(pad.get_net_contact_forces(dt=dt))
                        .reshape(-1, 3)
                        .sum(axis=0)
                    )
                    pad_pos, pad_quat = pad.get_world_poses()
                    pad_positions.append(np.asarray(pad_pos).reshape(-1, 3)[0])
                    orientation = np.asarray(pad_quat).reshape(-1, 4)[0]
                    pad_orientations.append(orientation)
                    local_forces.append(
                        Rotation.from_quat(orientation[[1, 2, 3, 0]])
                        .inv()
                        .apply(forces[-1])
                    )
                    packet_forces.append(
                        np.asarray(pad.get_contact_force_matrix(dt=dt))
                        .reshape(-1, 3)
                        .sum(axis=0)
                    )
                    force_data, points, _, _, counts, starts = (
                        pad.get_contact_force_data(dt=dt)
                    )
                    contact_ids = [
                        j
                        for count, first in zip(
                            np.asarray(counts).ravel(), np.asarray(starts).ravel()
                        )
                        for j in range(int(first), int(first + count))
                    ]
                    contact_counts.append(len(contact_ids))
                    weights = np.abs(np.asarray(force_data).ravel()[contact_ids])
                    packet_normal_forces.append(float(weights.sum()))
                    if weights.sum() > 1e-8:
                        centroid = np.average(
                            np.asarray(points)[contact_ids], axis=0, weights=weights
                        )
                        local = (
                            Rotation.from_quat(orientation[[1, 2, 3, 0]])
                            .inv()
                            .apply(centroid - pad_positions[-1])
                        )
                        contact_uvs.append(
                            np.clip(
                                local[1:3]
                                / (np.asarray(cfg["gripper"]["pad_size"])[1:3] / 2),
                                -1,
                                1,
                            )
                        )
                    else:
                        contact_uvs.append([np.nan, np.nan])
                trace["t"].append(t)
                trace["frame_t"].append(t)
                trace["physics_t"].append(world.current_time - physics_origin)
                trace["q"].append(actual[ids].copy())
                trace["qd"].append(robot.get_joint_velocities()[ids].copy())
                trace["tcp"].append(tcp_measured(actual[ids]))
                trace["tcp_nominal_fk"].append(forward_pose(actual[ids]))
                trace["target_q"].append(desired[ids].copy())
                measured_closure = finger_closure(actual[fingers])
                target_closure = finger_closure(desired[fingers])
                trace.setdefault("finger_q", []).append(actual[fingers].copy())
                trace.setdefault("target_finger_q", []).append(desired[fingers].copy())
                status = measured_gripper_status(
                    measured_closure,
                    target_closure,
                    max((np.linalg.norm(f) for f in forces), default=0),
                )
                trace["gripper"].append([measured_closure, status])
                trace["waffle_position"].append(packet.get_world_pose()[0])
                trace["waffle_orientation_wxyz"].append(packet.get_world_pose()[1])
                trace["pad_force"].append(forces)
                trace["pad_position"].append(pad_positions)
                trace["pad_orientation_wxyz"].append(pad_orientations)
                trace["pad_force_local"].append(local_forces)
                trace["pad_packet_force"].append(packet_forces)
                trace["pad_contact_uv"].append(contact_uvs)
                trace["pad_contact_count"].append(contact_counts)
                trace["pad_packet_normal_force"].append(packet_normal_forces)
                if gripper_wrist is not None:
                    trace.setdefault("wrist_capture_t", []).append(
                        np.nan if wrist_sample_t is None else wrist_sample_t
                    )
                    trace.setdefault("wrist_ft", []).append(
                        wrist_bias.copy() if wrist_value is None else wrist_value.copy()
                    )
                if support_views is not None:
                    support = support_views.get_all(dt)
                    for key in ("packet_robot_normal_force", "packet_bin_normal_force"):
                        trace.setdefault(key, []).append(support[key])
                if robot_environment_views is not None:
                    robot_environment_contact_trace.append(
                        {"t": t, **robot_environment_views.get_all(dt)}
                    )
                for key in (
                    "q",
                    "qd",
                    "tcp",
                    "waffle_position",
                    "waffle_orientation_wxyz",
                    "pad_force",
                    "pad_position",
                    "pad_orientation_wxyz",
                    "pad_packet_force",
                    "pad_packet_normal_force",
                ):
                    if not np.isfinite(trace[key][-1]).all():
                        raise RuntimeError(f"Nonfinite simulated {key} at t={t:.6f}s")
                next_frame += 1 / fps
                previous_frame_t = t
            if last_tick:
                break
            if step < nsteps:
                world.step(render=False)
        boundary_loop_completed = True
        cv2.imwrite(
            str(args.output / "sim_last.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
    finally:
        if emergency_trace is not None:
            emergency_trace.close()
        writer.release()
        if boundary_audit is not None:
            report = boundary_audit.finalize(
                completed=boundary_loop_completed,
                expected_end_s=terminal_until if terminal_until is not None else duration,
                stopped_reason=adapter.stopped_reason, completed_reason=adapter.completed_reason,
                completed_at_s=adapter.completed_at_s,
            )
            (args.output / "boundary_physics_audit.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n")
        if adaptive_gripper:
            (args.output / "native_mechanics_monitor.json").write_text(
                json.dumps(native_mechanics_monitor, indent=2, allow_nan=False) + "\n"
            )
        if wrist_contact_file is not None:
            wrist_contact_file.close()
        # Preserve measured states even when a policy or transport call fails.
        # FAILED.txt and the planner audit retain the failure; partial traces
        # must never be interpreted as a completed rollout by the scorer.
        np.savez_compressed(
            args.output / "sim_trace.npz",
            **{k: np.asarray(v) for k, v in trace.items()},
        )
        if args.mode == "policy":
            np.savez_compressed(
                args.output / "policy_tactile.npz",
                t=np.asarray([row[0] for row in policy_tactile_trace]),
                pad_force=np.asarray([row[1] for row in policy_tactile_trace]),
                gel=np.asarray(policy_gel_trace, dtype=np.uint8),
                **(
                    {"gel_normal_force": np.asarray(policy_gel_normal_trace)}
                    if measured_sensor_proxy is not None
                    else {}
                ),
            )
        if measured_sensor_proxy is not None or args.record_gel_contacts:
            (args.output / "gel_contact_trace.json").write_text(
                json.dumps(
                    policy_gel_contact_diagnostics, default=_json_value, indent=2
                )
                + "\n"
            )
        if robot_environment_views is not None:
            (args.output / "robot_environment_contact_trace.json").write_text(
                json.dumps(robot_environment_contact_trace, indent=2) + "\n"
            )
        try:
            if audit is not None:
                audit.close()
        finally:
            if policy is not None:
                policy.close()
    report = {
        "mode": args.mode,
        "command_replay": recorded_commands.metadata
        if recorded_commands is not None
        else None,
        "episode": str(args.episode),
        "frames": len(trace["t"]),
        "duration_s": float(trace["t"][-1]) if trace["t"] else 0,
        "wall_time_s": time.monotonic() - start,
        "events": events,
        "robot_usd": robot_usd,
        "joint_state_source": "actual PhysX articulation joints and tool0 rigid body pose; independent nominal FK also logged",
        "native_mechanics_monitor": native_mechanics_monitor if adaptive_gripper else None,
        "adaptive_policy_experiment": (
            json.loads((args.output / "adaptive_policy_experiment.json").read_text())
            if adaptive_gripper and args.mode == "policy" else None
        ),
        "object_dynamics": {
            "rigid_body_dynamic": True,
            "kinematic": False,
            "attachments": [],
            "pose_writes_after_initialization": 0,
            "initialization": "configured table pose before settling; contact_probe alone subsequently uses its documented one-time synthetic starting pose before t=0",
        },
        "tactile_model": args.tactile if args.mode == "policy" else None,
        "wrist_model": args.wrist
        if args.mode == "policy" or gripper_wrist is not None
        else None,
        "wrist_proxy_metadata": gripper_wrist.metadata()
        if gripper_wrist is not None
        else None,
        "wrist_sampling_rate_hz": 1 / control_dt if gripper_wrist is not None else None,
        "initial_recorded_wrist_bias": wrist_bias.tolist()
        if gripper_wrist is not None
        else None,
        "gripper_OBJ_model": "estimated from physical closure/target/contact; unvalidated",
        "gripper_model": cfg["gripper"].get("model", "legacy_sliding_pad_proxy"),
        "gripper_articulation": paths["gripper_articulation"],
        "finger_joint_names": list(finger_names),
        "finger_joint_units": "radians" if articulated_gripper else "metres",
        "gripper_feedback_source": "measured master joint angle" if articulated_gripper else "mean of two measured sliders",
        "finger_target_semantics": (
            "Direct replay: nominal eight-joint kinematic pose, not measured passive angles"
            if adaptive_gripper and args.mode == "replay" else
            "Motor position and passive spring rest references; passive positions evolve under contact"
            if adaptive_gripper else "nominal coupled gripper targets"
        ),
        "gel_geometry": cfg["gripper"].get("gel_geometry") if articulated_gripper else None,
        "contact_trace_source": (
            "PhysX contacts between separate supplier wear-layer rigid bodies and packet; "
            "hard sensor housings/adapters/linkage are separate actors. Active optical-area "
            "contact is further filtered by gel_contact_trace.json. UV uses local pad Y/Z."
            if articulated_gripper else
            "PhysX contacts filtered between each pad rigid body (including backing/linkage colliders) and packet; UV is force-weighted world contact centroid transformed to pad Y/Z and normalized/clipped by pad dimensions; NaN indicates no contact force."
        ),
        "control_rate_hz": hw.control.executor_rate_hz if hw else None,
        "validation_status": "reconstruction prototype; dynamics and tactile transfer unvalidated",
        "policy_stop_reason": adapter.stopped_reason if adapter else None,
        "policy_completed_reason": getattr(adapter, "completed_reason", None),
        "policy_delivery_clock": args.policy_delivery_clock if adapter else None,
        "policy_completion_is_task_success": False,
        **(
            {
                "servo_reach_limiter": {
                    **asdict(servo_reach_limits),
                    "algorithm": "shared_native_bisection_slide_v1",
                    "measured_wrist_extension_stop_m": hw.safety.wrist_extension_stop_m,
                    "consecutive_reject_limit": 25,
                    "final_consecutive_rejects": servo_limiter_rejects,
                    "tracking_guarantee": False,
                    **({"constraint_hold_s": args.servo_constraint_hold_s,
                        "constraint_hold_progress_rad": 0.001}
                       if servo_hold_budget is not None else {}),
                }
            }
            if servo_reach_limits is not None
            else {}
        ),
        "post_stop_observation_s": 2.0 if adapter else None,
        "record_packet_support": args.record_packet_support,
        "record_gel_contacts": args.record_gel_contacts,
        "gel_contact_coverage": args.gel_contact_coverage
        if gel_views is not None
        else None,
        "packet_support_source": "Independent PhysX normal-contact sums from the free packet to all robot rigid bodies and the five bin colliders; not exposed to policy or release controller"
        if support_views is not None
        else None,
        "packet_support_filter_paths": {
            "packet": support_views.packet_path,
            "bin": support_views.bin_paths,
            "robot": support_views.robot_paths,
        }
        if support_views is not None
        else None,
        "camera_intrinsics_px": camera_projection_report["intrinsics_px"],
        "camera_projection": camera_projection_report,
    }
    if robot_environment_views is not None:
        report["robot_environment_contact_diagnostic"] = {
            "file": "robot_environment_contact_trace.json",
            "sampling": "scene frame times, same t as sim_trace; contact impulses from current physics step",
            "robot_paths": robot_environment_views.robot_paths,
            "environment_paths": robot_environment_views.environment_paths,
            "policy_or_safety_feedback": False,
            "self_collision_observed": False,
        }
    if probe is not None:
        positions = np.asarray(trace["waffle_position"])
        times = np.asarray(trace["t"])
        hold = positions[(times >= 3) & (times < 7)]
        report["contact_probe"] = {
            "kind": "synthetic diagnostic; not real-recording validation",
            "initial_packet_z_m": float(positions[0, 2]),
            "hold_lift_m": float(np.median(hold[:, 2]) - positions[0, 2])
            if len(hold)
            else None,
            "final_packet_z_m": float(positions[-1, 2]),
            "peak_pad_contact_force_N": float(
                np.linalg.norm(trace["pad_force"], axis=-1).max()
            ),
            "last_target_diagnostics": probe.diagnostics,
        }
    (args.output / "run.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )
        + "\n"
    )
    print(
        "PHANTOM_COMPLETE",
        json.dumps({"frames": report["frames"], "output": str(args.output)}),
        flush=True,
    )
    inspect_gui()


if __name__ == "__main__":
    main()
