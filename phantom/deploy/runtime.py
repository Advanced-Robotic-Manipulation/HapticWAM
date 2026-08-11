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
        ep_name = f"ep_{self.mode}_{task}_{int(time.time())}"
        ep_path = self.recorder.start(meta, ep_name)

        safety = SafetyMonitor(hw, self.session.rings)
        executor = ChunkExecutor(hw, self.rig.arm, self.rig.gripper, safety,
                                 record_action=self.recorder.record_action)
        snapshots = SnapshotBuilder(hw, self.session, self.mode)
        _wait_rings_warm(snapshots)
        trace: list = []
        planner = PlannerLoop(hw, self.policy, snapshots, executor, trace=trace)

        if hw.wrist_ft.bias_on_episode_start:
            try:
                self.rig.arm.zero_ft()
            except Exception:
                log.exception("zero_ft failed (continuing)")

        executor.start()
        try:
            planner.run(max_replans=max_replans)
        finally:
            planner.stop()
            executor.stop()
            saved = self.recorder.stop(success=None)

        trace_path = None
        if saved is not None:
            trace_path = saved / "planner_trace.json"
            trace_path.write_text(json.dumps(trace, indent=1), encoding="utf-8")
        return EpisodeResult(
            episode_path=saved, stopped_reason=executor.stopped_reason,
            n_replans=len(trace), safety_events=len(safety.log_events),
            trace_path=trace_path)
