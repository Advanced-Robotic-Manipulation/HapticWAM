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
from phantom.deploy.planner import PlannerLoop, SnapshotBuilder
from phantom.deploy.safety import SafetyMonitor
from phantom.drivers.factory import make_rig
from phantom.inference.policy import PhantomPolicy
from phantom.recording.recorder import EpisodeRecorder
from phantom.recording.workers import SensorSession
from phantom.timesync.clock import IdentityClock, MasterClock

log = logging.getLogger(__name__)


def _wait_rings_warm(snapshots: SnapshotBuilder, timeout_s: float = 10.0) -> None:
    """Block until every ring the snapshot reads has its first sample.

    The real rig streams long before an operator starts an episode, but the
    first episode right after connect — and any mock-driver dry run, where
    the session and the episode start together — races the workers' first
    frames."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            snapshots.build()
            return
        except (AssertionError, IndexError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


@dataclass
class EpisodeResult:
    episode_path: Path | None
    stopped_reason: str | None
    n_replans: int
    safety_events: int
    trace_path: Path | None


class DeploymentRuntime:
    def __init__(self, hw: HardwareConfig, policy: PhantomPolicy, mode: str,
                 out_root: Path):
        self.hw = hw
        self.policy = policy
        self.mode = mode
        self.out_root = Path(out_root)
        self.rig = make_rig(hw, control=True)
        self.rig.worker_owned_tactile = True    # real DM-Tac is single-open
        self.session: SensorSession | None = None
        self.recorder: EpisodeRecorder | None = None

    def __enter__(self) -> "DeploymentRuntime":
        self.rig.connect_all()
        clock = (IdentityClock() if self.hw.mode.resolve("arm") == "mock"
                 else MasterClock.calibrate(self.rig.arm))
        self.session = SensorSession.start(self.hw, self.rig,
                                           session_id=str(int(time.time()) % 10_000_000))
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
        meta = EpisodeMeta(task=task, text=text or task, tags=list(tags or []),
                           policy=policy_name or self.mode,
                           dagger_round=dagger_round)
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
        snapshots = SnapshotBuilder(hw, self.session, self.mode)
        trace: list = []
        planner = PlannerLoop(hw, self.policy, snapshots, executor, trace=trace)

        # the executor thread owns + polls the gripper, so it must run BEFORE
        # ring warm-up — the snapshot hard-requires gripper state (no silent
        # zeros; that substitute collapsed the policy's action magnitude ~3x
        # on every rig episode before 2026-08-14)
        executor.start()
        saved = None
        trace_path = None
        try:
            _wait_rings_warm(snapshots)

            if hw.wrist_ft.bias_on_episode_start:
                try:
                    self.rig.arm.zero_ft()
                except Exception:
                    log.exception("zero_ft failed (continuing)")

            planner.run(max_replans=max_replans)
        finally:
            planner.stop()
            executor.stop()
            try:
                saved = self.recorder.stop(success=None)
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
            trace_path=trace_path)
