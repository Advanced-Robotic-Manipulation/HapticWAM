"""Student rollout collection for DAgger rounds: the deployment runtime in
student mode with the recorder always on (full-sensor streams recorded) —
~50 rollouts/task/round per pipeline.md §7."""

from __future__ import annotations

import logging
from pathlib import Path

from phantom.config.hardware import HardwareConfig
from phantom.deploy.runtime import DeploymentRuntime, EpisodeResult
from phantom.inference.policy import PhantomPolicy

log = logging.getLogger(__name__)


def collect_rollouts(hw: HardwareConfig, policy: PhantomPolicy, out_root: Path,
                     *, task: str, n_rollouts: int, dagger_round: int,
                     max_replans: int = 20) -> list[EpisodeResult]:
    results = []
    with DeploymentRuntime(hw, policy, mode="student", out_root=out_root) as rt:
        for i in range(n_rollouts):
            input(f"[dagger r{dagger_round}] reset scene for rollout "
                  f"{i + 1}/{n_rollouts} of task '{task}', then press Enter...")
            res = rt.run_episode(task=task, max_replans=max_replans,
                                 policy_name="student_hid",
                                 dagger_round=dagger_round)
            log.info("rollout %d: %s (%d replans, stop=%s)", i, res.episode_path,
                     res.n_replans, res.stopped_reason)
            results.append(res)
    return results
