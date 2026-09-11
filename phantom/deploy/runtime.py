"""Deployment runtime (pipeline.md §6e): wires rig + sensor session + recorder
+ policy + planner loop + chunk executor + safety for one episode/trial.

Architecture note: tactile sensors run in their own processes (SensorSession);
the planner runs in the caller's thread (GPU-bound, torch releases the GIL)
and the executor in a dedicated thread at executor_rate_hz. This is the
pragmatic single-host layout; the zmq 3-process split is a documented upgrade
path if executor jitter under load proves too high (docs/deployment_runtime.md).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.data.schema import EpisodeMeta
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.planner import (
    PlannerLoop,
    SnapshotBuilder,
    TerminalVeto,
    snapshot_stop_reason,
)
from phantom.deploy.safety import SafetyMonitor
from phantom.drivers.factory import make_rig
from phantom.inference.policy import PhantomPolicy
from phantom.recording.recorder import EpisodeRecorder
from phantom.recording.workers import SensorSession, new_session_id
from phantom.timesync.clock import IdentityClock, MasterClock

log = logging.getLogger(__name__)


# Floor on the ring warm-up budget, independent of any camera model.
_WARMUP_FLOOR_S = 20.0
_WARMUP_MARGIN_S = 5.0
_WARMUP_LOG_PERIOD_S = 2.0


def warmup_timeout_s() -> float:
    """How long to wait for the rings to fill before giving up.

    Derived from the scene camera driver, not guessed: RealSenseCamera tolerates
    _REBUILD_AFTER_TIMEOUTS blocking wait_for_frames() timeouts (5 s each, the
    pyrealsense2 default) before it rebuilds the pipeline, so a camera that
    heals itself can take ~17 s to deliver its FIRST frame. The old flat 10 s
    aborted at two thirds of that first rebuild — the episode died with
    "no camera frames yet" while the driver was still recovering normally
    (rig 2026-08-27)."""
    from phantom.drivers.real.realsense import first_frame_budget_s
    return max(_WARMUP_FLOOR_S, first_frame_budget_s() + _WARMUP_MARGIN_S)


def _wait_rings_warm(snapshots: SnapshotBuilder, timeout_s: float | None = None) -> None:
    """Block until every ring the snapshot reads has its first sample.

    The real rig streams long before an operator starts an episode, but the
    first episode right after connect — and any mock-driver dry run, where
    the session and the episode start together — races the workers' first
    frames."""
    if timeout_s is None:
        timeout_s = warmup_timeout_s()
    deadline = time.monotonic() + timeout_s
    next_log = time.monotonic()
    log.info("waiting up to %.0fs for the sensor rings to warm "
             "(one RealSense pipeline rebuild fits inside this budget)...",
             timeout_s)
    while True:
        try:
            snapshots.build()
            return
        except (AssertionError, IndexError) as e:
            now = time.monotonic()
            # say WHAT is missing: "no camera frames yet" vs "no gripper state
            # yet" point at completely different hardware, and the operator was
            # previously given only a bare traceback at the deadline
            waiting_on = str(e).splitlines()[0] if str(e) else type(e).__name__
            if now >= deadline:
                log.error("rings did not warm within %.0fs — still waiting on: "
                          "%s", timeout_s, waiting_on)
                raise
            if now >= next_log:
                log.info("ring warm-up (%.0fs left): waiting on %s",
                         deadline - now, waiting_on)
                next_log = now + _WARMUP_LOG_PERIOD_S
            time.sleep(0.05)


# Stop reasons that end the whole DEPLOYMENT SESSION, not just the episode:
# nothing in-process can put the rig back in a runnable state, so continuing
# would only produce corrupt episodes (a dead sensor worker cannot be
# respawned — SensorSession.start() wires the rings once).
_SESSION_FATAL_REASONS = ("worker_died",)


def fatal_reason(executor) -> str | None:
    """Why this PROCESS must not start another episode (None = it may).

    Distinct from EpisodeResult.stopped_reason, which run_deploy uses to decide
    whether to REBUILD the RTDE control script and carry on: these are the
    conditions rebuilding cannot fix."""
    if getattr(executor, "join_failed", False):
        # stop() could not join the servo/gripper thread: the arm's servo
        # session and the one Robotiq socket are still owned by a worker this
        # process can no longer address, so a second executor would race it.
        return "executor_join_timeout"
    if executor.stopped_reason in _SESSION_FATAL_REASONS:
        return executor.stopped_reason
    return None


@dataclass
class EpisodeResult:
    episode_path: Path | None
    stopped_reason: str | None
    n_replans: int
    safety_events: int
    trace_path: Path | None
    # non-None => this PROCESS must not start another episode (see
    # _SESSION_FATAL_REASONS and ChunkExecutor.join_failed). run_deploy exits.
    fatal_reason: str | None = None
    # per-condition safety events (kind/value/t/count) and the arm state at
    # the stop — persisted by run_deploy into stop.json + meta tags so a stop
    # can be diagnosed from disk instead of from terminal scrollback (09-04)
    events: list = field(default_factory=list)
    stop_state: dict = field(default_factory=dict)
    # Controller state is separate from a subsequent stop and object success.
    completed_reason: str | None = None
    completed_at_s: float | None = None


class DeploymentRuntime:
    def _arm_state_now(self) -> dict:
        """Latest arm sample as plain floats (tcp_pose, q, qd, wrist-centre
        distance) — best effort, never raises."""
        try:
            _, arm = self.session.rings["arm"].latest(1)
            q = np.asarray(arm["q"][0], dtype=float).reshape(-1)
            out = {"tcp_pose": np.asarray(arm["tcp_pose"][0], dtype=float).reshape(-1).tolist(),
                   "q_deg": np.degrees(q).tolist(),
                   "qd_max": float(np.max(np.abs(np.asarray(arm["qd"][0], dtype=float))))}
            dh = getattr(self.hw.safety, "ur_dh_a2_a3_d4_m", None)
            if dh is not None and q.size >= 3:
                a2, a3, d4 = dh
                out["wrist_dist_m"] = float(np.sqrt(a2 * a2 + a3 * a3 + 2 * a2 * a3 * np.cos(q[2]) + d4 * d4))
            arm_drv = getattr(getattr(self, "rig", None), "arm", None)
            live = {"hits": getattr(arm_drv, "_limiter_hits", None),
                    "holds": getattr(arm_drv, "_limiter_holds", None)}
            constraint_hold = getattr(arm_drv, "constraint_hold_last", None)
            if constraint_hold:
                live["constraint_hold"] = constraint_hold
            last = getattr(arm_drv, "limiter_last", None) or {}
            if last or live["hits"]:
                out["servo_limiter"] = last if last else live
            loss = getattr(arm_drv, "control_loss_last", None)
            if loss:
                out["control_loss"] = loss
            return out
        except Exception:
            return {}

    def __init__(self, hw: HardwareConfig, policy: PhantomPolicy, mode: str,
                 out_root: Path, *, parity_fixes: bool = False,
                 veto: TerminalVeto | None = None,
                 base_hw: HardwareConfig | None = None,
                 deploy_overrides: dict | None = None,
                 open_aperture: float = 0.0,
                 max_play_steps: int | None = None, release_config=None, boundary_config=None,
                 min_replan_s: float = 0.0,
                 grip_play_steps: int | None = None,
                 controller_profile: str | None = None):
        """`base_hw` is the config as LOADED FROM YAML, before run_deploy's
        per-task safety overrides (z floor / hitbox / TCP speed cap, applied
        with model_copy). Episodes are stamped with ITS config_hash so a
        rollout matches the demos recorded on the same rig — otherwise every
        override changes hw.config_hash() and train_teacher's CONFIG DRIFT
        gate (train_teacher.py:312) refuses the whole rollout set. The
        overrides themselves are recorded under meta.deploy_overrides, so the
        envelope an episode ran under is never lost. Default (None): stamp
        `hw` exactly as before."""
        self.hw = hw
        self.base_config_hash = (base_hw or hw).config_hash()
        self.deploy_overrides = dict(deploy_overrides or {})
        self.policy = policy
        self.mode = mode
        self.out_root = Path(out_root)
        # deploy levers (review 2026-08-28), both default OFF so the rig A/B
        # can attribute each one; run_deploy tags every episode with the state
        self.parity_fixes = bool(parity_fixes)
        self.veto = veto
        # the aperture a "let go" stop reopens to (task demo start aperture)
        self.open_aperture = float(open_aperture)
        self.max_play_steps = max_play_steps
        self.min_replan_s = float(min_replan_s or 0.0)
        self.grip_play_steps = grip_play_steps
        from phantom.deploy.release_controller import make_release_controller
        controller = make_release_controller(release_config, hw)
        self.release_config = None if controller is None else controller.config
        from phantom.deploy.boundary_projection import boundary_config as parse_boundary_config
        self.boundary_config = parse_boundary_config(boundary_config)
        if self.boundary_config is not None:
            from phantom.deploy.boundary_projection import make_boundary_projection
            make_boundary_projection(self.boundary_config, hw, {})  # Validate before opening any device.
            if hw.mode.resolve("arm") == "mock":
                raise ValueError("native boundary projection requires URArm; use the simulation adapter for physics")
        self.planner_class = PlannerLoop
        if controller_profile is not None:
            from phantom.deploy.minimal_v5 import planner_class, validate_profile
            profile_meta = validate_profile(controller_profile, release_config=self.release_config,
                                            veto=veto, mode=mode, hw=hw)
            self.deploy_overrides["placement_controller_profile"] = profile_meta
            self.deploy_overrides["placement_veto_feedback"] = profile_meta["veto_feedback_source"]
            self.planner_class = planner_class(controller_profile, PlannerLoop)
        self.rig = make_rig(hw, control=True)
        self.rig.worker_owned_tactile = True    # real DM-Tac is single-open
        self.session: SensorSession | None = None
        self.recorder: EpisodeRecorder | None = None

    def __enter__(self) -> "DeploymentRuntime":
        self.rig.connect_all()
        clock = (IdentityClock() if self.hw.mode.resolve("arm") == "mock"
                 else MasterClock.calibrate(self.rig.arm))
        self.session = SensorSession.start(self.hw, self.rig,
                                           session_id=new_session_id())
        self.recorder = EpisodeRecorder(self.session, clock, self.out_root)
        return self

    def __exit__(self, *exc) -> None:
        if self.session is not None:
            self.session.stop()
        self.rig.disconnect_all()

    # ------------------------------------------------------------------
    def run_episode(self, *, task: str, text: str = "", tags: list[str] | None = None,
                    max_replans: int = 20, max_episode_s: float | None = None,
                    policy_name: str = "", dagger_round: int = 0,
                    stop_check=None, success_check=None) -> EpisodeResult:
        assert self.session is not None and self.recorder is not None
        hw = self.hw
        # Fail closed BEFORE anything is recorded or the arm is driven: a
        # worker that died during the previous episode (or between them)
        # leaves its ring frozen, and every stream this episode would record
        # or condition on is then a stale copy of the last live sample.
        if not self.session.all_alive():
            log.error("a sensor/arm worker is DEAD before episode start — "
                      "refusing to run. The rings it feeds have no writer; "
                      "restart the process.")
            return EpisodeResult(episode_path=None, stopped_reason="worker_died",
                                 n_replans=0, safety_events=0, trace_path=None,
                                 fatal_reason="worker_died")
        meta = EpisodeMeta(task=task, text=text or task, tags=list(tags or []),
                           policy=policy_name or self.mode,
                           dagger_round=dagger_round,
                           config_hash=self.base_config_hash,
                           deploy_overrides=dict(self.deploy_overrides))
        # 1 s wall-clock resolution alone collided on fast retries (gate
        # reject -> relaunch inside the same second) and appended the second
        # episode onto the first zarr; a per-runtime sequence makes it unique
        self._ep_seq = getattr(self, "_ep_seq", -1) + 1
        ep_name = f"ep_{self.mode}_{task}_{int(time.time())}_{self._ep_seq:03d}"
        ep_path = self.recorder.start(meta, ep_name)
        if hasattr(self.policy, "reset_episode"):
            self.policy.reset_episode()

        safety = SafetyMonitor(hw, self.session.rings, boundary_config=self.boundary_config)
        executor = ChunkExecutor(hw, self.rig.arm, self.rig.gripper, safety,
                                 record_action=self.recorder.record_action,
                                 gripper_ring=self.session.rings["gripper"],
                                 open_aperture=self.open_aperture,
                                 max_play_steps=self.max_play_steps,
                                 grip_play_steps=self.grip_play_steps,
                                 release_config=self.release_config)
        snapshots = SnapshotBuilder(hw, self.session, self.mode,
                                    parity_fixes=self.parity_fixes,
                                    executor=executor,
                                    wrench_baseline_rows=int(
                                        getattr(self.policy, "wrench_baseline_rows", 0) or 0))
        trace: list = []
        # the policy's predicted contact package per replan (tactile
        # prediction error, scored offline against the recorded pads)
        from phantom.deploy.cpk_trace import CPK_TRACE_FILE, CpkTraceLog
        cpk_log = CpkTraceLog(hw)
        planner = self.planner_class(hw, self.policy, snapshots, executor, trace=trace,
                              session=self.session, veto=self.veto,
                              cpk_log=cpk_log,
                              **({"min_replan_s": self.min_replan_s} if self.min_replan_s > 0 else {}))

        # the executor thread owns + polls the gripper, so it must run BEFORE
        # ring warm-up — the snapshot hard-requires gripper state (no silent
        # zeros; that substitute collapsed the policy's action magnitude ~3x
        # on every rig episode before 2026-08-14)
        executor.start()
        saved = None
        trace_path = None
        try:
            try:
                _wait_rings_warm(snapshots)

                if hw.wrist_ft.bias_on_episode_start:
                    try:
                        self.rig.arm.zero_ft()
                    except Exception:
                        log.exception("zero_ft failed (continuing)")

                planner.run(max_replans=max_replans,
                            max_episode_s=max_episode_s,
                            stop_check=stop_check,
                            success_check=success_check)
            except AssertionError as e:
                # Last net for the snapshot's hard requirements (PlannerLoop.run
                # catches its own; this covers _wait_rings_warm timing out and
                # any future caller of build() in here). Escaping this frame
                # cost the operator the label prompt and left an empty episode
                # directory behind, with the failure surfacing as an rc-1
                # traceback from run_deploy (rig 2026-08-27).
                reason = snapshot_stop_reason(e)
                log.error("episode aborted before/around the planner loop (%s): "
                          "%s — ending it through the normal stop path so the "
                          "trace is saved and the episode can be labelled",
                          reason, e)
                executor.request_stop(reason)
        finally:
            planner.stop()
            executor.stop()
            try:
                saved = self.recorder.stop(success=None)
                if saved is not None:
                    # PROVISIONAL verdict (P9, review 2026-08-28): a deploy
                    # episode is finalized with success=None, which every
                    # lister reads as an ordinary full-weight demo — so a
                    # session where the operator hits Enter, runs with
                    # --no-label-prompt, or whose process dies before the
                    # prompt would train on-air closes at action_weight 1.0.
                    # File it as NOT training-ready right here, at the only
                    # point guaranteed to run; run_deploy's label prompt
                    # promotes it back to 'finalized' when a verdict arrives.
                    # Same guard the collect app got (data_collect/session.py
                    # _file_unlabeled) — deploy was missed.
                    try:
                        # + the stop reason (analyst 09-04 P3: every end
                        # condition of a 29-episode session had to be
                        # reconstructed from the streams)
                        reason = executor.stopped_reason or planner.stop_reason
                        self.recorder.relabel(saved, status="aborted",
                                              tags=["unlabeled",
                                                    f"stop:{reason or 'none'}"])
                    except Exception:
                        log.exception("could not file %s as unlabeled", saved)
            finally:
                # the trace must survive crashes — crashed episodes are
                # exactly the ones worth diagnosing (issue #2: the trace of
                # the 08-14 table-press episode was lost this way)
                saved_dir = saved if saved is not None else ep_path
                if saved_dir is not None and trace:
                    trace_path = saved_dir / "planner_trace.json"
                    trace_path.write_text(json.dumps(trace, indent=1),
                                          encoding="utf-8")
                if saved_dir is not None and len(cpk_log):
                    try:
                        cpk_log.save(saved_dir / CPK_TRACE_FILE)
                    except Exception:
                        log.exception("could not write %s", CPK_TRACE_FILE)
        return EpisodeResult(
            episode_path=saved,
            # the executor's reason wins; the planner's own caps (replan_cap /
            # episode_time_cap) are what used to read as stopped_reason None
            stopped_reason=executor.stopped_reason or planner.stop_reason,
            n_replans=len(trace), safety_events=len(safety.log_events),
            trace_path=trace_path, fatal_reason=fatal_reason(executor),
            completed_reason=getattr(executor, "completed_reason", None),
            completed_at_s=getattr(executor, "completed_at_s", None),
            events=[{"t": float(e.t), "kind": str(e.kind),
                     "value": float(e.value), "action": getattr(e.action, "value", str(e.action)),
                     "count": int(getattr(e, "count", 1))}
                    for e in safety.log_events],
            stop_state={**self._arm_state_now(),
                        "wrist_guard": safety.wrench_diagnostics(),
                        **({"boundary_projection": dict(safety.boundary_projection.last)}
                           if safety.boundary_projection is not None else {}),
                        **({"placement_release": executor.release_diagnostics()}
                           if getattr(executor, "release_controller", None) is not None else {}),
                        **({"at_halt": executor.halt_state}
                           if getattr(executor, "halt_state", None) else {}),
                        **({"crash": executor.crash_text}
                           if getattr(executor, "crash_text", None) else {})})
