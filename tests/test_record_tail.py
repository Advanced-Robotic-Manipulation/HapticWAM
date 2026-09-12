"""DeploymentRuntime keeps recording for `record_tail_s` after the arm stops."""
import time

import numpy as np

from phantom.deploy.runtime import DeploymentRuntime
from phantom_test_utils import make_small_hw
from test_p9_p10_fixes import _CrawlPolicy


def test_tail_runs_between_executor_stop_and_recorder_stop(tmp_path):
    hw = make_small_hw()
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="student", out_root=tmp_path,
                           record_tail_s=1.2) as rt:
        events = []
        real_tail, real_stop = rt._record_tail, rt.recorder.stop
        def tail(reason):
            t0 = time.monotonic(); real_tail(reason); events.append(("tail", time.monotonic() - t0))
        def stop(**kw):
            events.append(("recorder_stop", 0.0)); return real_stop(**kw)
        rt._record_tail, rt.recorder.stop = tail, stop
        res = rt.run_episode(task="whiteboard", max_replans=2, policy_name="student")
    assert res.episode_path is not None
    assert [e[0] for e in events] == ["tail", "recorder_stop"]   # tail first, then the recorder closes
    assert events[0][1] >= 1.1, events


def test_tail_is_skipped_when_a_worker_died(tmp_path):
    hw = make_small_hw()
    with DeploymentRuntime(hw, _CrawlPolicy(hw), mode="student", out_root=tmp_path,
                           record_tail_s=5.0) as rt:
        t0 = time.monotonic()
        rt._record_tail("worker_died")
        assert time.monotonic() - t0 < 0.5
        # a dead session ends the tail early as well
        rt.session.all_alive = lambda: False
        t0 = time.monotonic()
        rt._record_tail("finish")
        assert time.monotonic() - t0 < 0.5


def test_run_deploy_records_the_tail_in_overrides():
    import phantom.scripts.run_deploy as RD
    src = open(RD.__file__).read()
    assert 'deploy_overrides["record_tail_s"]' in src and "record_tail_s=float(getattr(args, \"record_tail_s\"" in src
