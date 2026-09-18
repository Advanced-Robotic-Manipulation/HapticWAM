"""Policy server / RemotePolicy protocol (rig 2026-09-01: model preloaded in
its own terminal; run_deploy attaches in seconds instead of a ~3 min load).

Round-trips run over a real localhost Listener/Client pair — the same wire
the rig uses — with a stub policy standing in for the model."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from phantom.inference.policy import ObsSnapshot, Plan
from phantom.inference.remote import (PHANTOM_AUTHKEY, PolicyServer,
                                      RemotePolicy, plan_from_wire,
                                      plan_to_wire)


class _FakeCpk:
    """Stands in for a ContactPackage — must never cross the wire."""

    def __init__(self, tag):
        self.tag = tag

    def __reduce__(self):  # make accidental pickling loud
        raise TypeError("ContactPackage crossed the wire")


class _StubRF:
    def __init__(self):
        self.resets = 0
        self._gen = None

    def reset_episode_noise(self):
        self.resets += 1


class _StubPolicy:
    def __init__(self):
        self.rf = _StubRF()
        self.nfe = 5
        self.guidance = 1.0
        self.k_seeds = 1
        self.parity_fixes = False
        self.persistent_noise = True
        self.task_text = ""
        self.drop_video = False
        self.close_p = 0.5
        self.action_time_origin = "inference_ready"  # real policy default
        self.seen_prev_cpk = []

    def replan(self, obs, prev_plan, tcp_pose):
        self.seen_prev_cpk.append(None if prev_plan is None else prev_plan.cpk)
        H, A = 4, 7
        return Plan(t_created=time.time(), t0_pose=np.asarray(tcp_pose, float),
                    actions=np.full((H, A), self.nfe, float),
                    action_times=np.arange(H) / 10.0, sigma=np.zeros(3),
                    gate=0.5, p_evt=np.zeros(5), cpk=_FakeCpk(len(self.seen_prev_cpk)),
                    latency_s=0.01)


def _snap():
    return ObsSnapshot(t=time.time(), rgb=np.zeros((4, 4, 3), np.uint8),
                       wrist_window=np.zeros((8, 6), np.float32),
                       ur_state=np.zeros(26, np.float32))


@pytest.fixture()
def server_client():
    pol = _StubPolicy()
    srv = PolicyServer(pol, ckpt="stub.pt")
    from multiprocessing.connection import Listener
    listener = Listener(("127.0.0.1", 0), authkey=PHANTOM_AUTHKEY)
    port = listener.address[1]

    def serve_one():
        with listener.accept() as conn:
            try:
                while True:
                    msg = conn.recv()
                    try:
                        conn.send(("ok", srv.handle(msg)))
                    except Exception:
                        import traceback
                        conn.send(("err", traceback.format_exc()))
            except EOFError:
                pass

    t = threading.Thread(target=serve_one, daemon=True)
    t.start()
    client = RemotePolicy(("127.0.0.1", port),
                          {"nfe": 1, "guidance": 2.0, "task_text": "waffles"})
    yield pol, client
    client.close()
    listener.close()


def test_configure_applies_and_mirrors_effective(server_client):
    pol, client = server_client
    assert pol.nfe == 1 and pol.guidance == 2.0 and pol.task_text == "waffles"
    # the client mirrors the EFFECTIVE values for provenance tags
    assert client.nfe == 1 and client.guidance == 2.0


def test_replan_roundtrip_keeps_cpk_server_side(server_client):
    pol, client = server_client
    p1 = client.replan(_snap(), None, np.zeros(6))
    assert p1.cpk is None and isinstance(p1._cpk_token, int)
    assert np.allclose(p1.actions, 1.0)          # built with configured nfe=1
    # planner-style mutation survives the wire back
    p1.t0_pose = p1.t0_pose + 0.123
    p2 = client.replan(_snap(), p1, np.ones(6))
    # the server matched the token back to the ORIGINAL package object
    assert isinstance(pol.seen_prev_cpk[1], _FakeCpk)
    assert pol.seen_prev_cpk[1].tag == 1
    assert p2._cpk_token != p1._cpk_token


def test_reset_episode_seeds_and_resets(server_client):
    pol, client = server_client
    client.remote_reset(1234)
    assert pol.rf.resets == 1 and pol.rf._gen is not None
    client.reset_episode()
    assert pol.rf.resets == 2


def test_wire_format_drops_cpk():
    plan = Plan(t_created=1.0, t0_pose=np.zeros(6), actions=np.zeros((2, 7)),
                action_times=np.zeros(2), sigma=np.zeros(3), gate=0.0,
                p_evt=np.zeros(5), cpk=_FakeCpk(0), latency_s=0.0)
    d = plan_to_wire(plan)          # would raise if it tried to pickle/copy cpk
    assert "cpk" not in d
    back = plan_from_wire(d, cpk=None)
    assert back.cpk is None and back.actions.shape == (2, 7)


@pytest.mark.requires_cosmos_repo
def test_real_tiny_policy_over_the_wire():
    """The riskiest plumbing: a REAL PhantomPolicy (tiny, CPU) served over the
    localhost protocol — two replans with the ContactPackage held server-side
    and the prev_plan token swapped back on the second call."""
    from multiprocessing.connection import Listener

    from phantom.config.model import AccConfig, PhantomModelConfig
    from phantom.config.paths import load_paths
    from phantom.data.schema import NormStats
    from phantom.inference.policy import PhantomPolicy
    from phantom.scripts.bench_inference import fake_obs
    from phantom.train.builder import build_model
    from phantom_test_utils import make_small_hw

    hw = make_small_hw()
    mc = PhantomModelConfig(acc=AccConfig(self_anticipation="two_pass"))
    pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False,
                     mc=mc)
    policy = PhantomPolicy(pm, NormStats.identity(), nfe=1, persistent_noise=True)
    srv = PolicyServer(policy, ckpt="tiny.pt")
    listener = Listener(("127.0.0.1", 0), authkey=PHANTOM_AUTHKEY)
    port = listener.address[1]

    def serve_one():
        with listener.accept() as conn:
            try:
                while True:
                    msg = conn.recv()
                    try:
                        conn.send(("ok", srv.handle(msg)))
                    except Exception:
                        import traceback
                        conn.send(("err", traceback.format_exc()))
            except EOFError:
                pass

    t = threading.Thread(target=serve_one, daemon=True)
    t.start()
    client = RemotePolicy(("127.0.0.1", port), {"nfe": 1, "k_seeds": 1})
    try:
        client.remote_reset(42)
        obs = fake_obs(hw, teacher=True)
        p1 = client.replan(obs, None, np.zeros(6))
        assert p1.cpk is None and p1.actions.ndim == 2
        assert np.isfinite(p1.actions).all()
        p2 = client.replan(obs, p1, np.zeros(6))
        assert np.isfinite(p2.actions).all()
        assert p2._cpk_token != p1._cpk_token
        # the second replan consumed the stored package as prev_cpk (the
        # store had it under p1's token)
        assert srv._store, "server-side package store is empty"
    finally:
        client.close()
        listener.close()
