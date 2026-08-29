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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phantom.config.hardware import HardwareConfig
from phantom.data.schema import EpisodeMeta
from phantom.deploy.executor import ChunkExecutor
from phantom.deploy.planner import (PlannerLoop, SnapshotBuilder, TerminalVeto,
                                    snapshot_stop_reason)
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


class DeploymentRuntime:
    def __init__(self, hw: HardwareConfig, policy: PhantomPolicy, mode: str,
                 out_root: Path, *, parity_fixes: bool = False,
                 veto: TerminalVeto | None = None,
                 base_hw: HardwareConfig | None = None,
                 deploy_overrides: dict | None = None):
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
                    max_replans: int = 20, policy_name: str = "",
                    dagger_round: int = 0) -> EpisodeResult:
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

        safety = SafetyMonitor(hw, self.session.rings)
        executor = ChunkExecutor(hw, self.rig.arm, self.rig.gripper, safety,
                                 record_action=self.recorder.record_action,
                                 gripper_ring=self.session.rings["gripper"])
        snapshots = SnapshotBuilder(hw, self.session, self.mode,
                                    parity_fixes=self.parity_fixes,
                                    executor=executor)
        trace: list = []
        planner = PlannerLoop(hw, self.policy, snapshots, executor, trace=trace,
                              session=self.session, veto=self.veto)

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

                planner.run(max_replans=max_replans)
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
                        self.recorder.relabel(saved, status="aborted",
                                              tags=["unlabeled"])
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
        return EpisodeResult(
            episode_path=saved, stopped_reason=executor.stopped_reason,
            n_replans=len(trace), safety_events=len(safety.log_events),
            trace_path=trace_path, fatal_reason=fatal_reason(executor))
