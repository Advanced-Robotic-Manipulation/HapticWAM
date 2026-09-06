"""Wire compatibility with the resident policy service, without model imports."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from phantom.sim.policy_adapter import SimulationObservation
from phantom.sim.remote_policy import CONFIGURABLE, PLAN_FIELDS, RemoteSimulationPolicy


class Connection:
    def __init__(self):
        self.closed = False
        self.close_event = threading.Event()
        self.settings = {k: None for k in CONFIGURABLE}
        self.settings.update(task_text="recorded task", nfe=12, parity_fixes=True)
        self.requests = []
        self.available = True

    def send(self, message):
        self.requests.append(message)
        kind = message[0]
        if kind == "info":
            self.reply = ("ok", {"ckpt": "/runs/test.pt", "warmed": True})
        elif kind == "configure":
            self.settings.update(message[1])
            self.reply = ("ok", {"effective": self.settings.copy()})
        elif kind == "reset_episode":
            self.reply = ("ok", None)
        elif kind == "replan":
            obs, _, tcp = message[1:]
            values = {
                "t_created": obs.t,
                "t0_pose": tcp,
                "actions": np.zeros((16, 7)),
                "action_times": obs.t + np.arange(16) / 10,
                "sigma": np.zeros(3),
                "gate": 0.1,
                "p_evt": np.zeros(5),
                "latency_s": 0.05,
                "diag": {},
            }
            self.reply = ("ok", (values, 42))
        else:
            self.reply = ("err", "bad request")

    def poll(self, timeout):
        return self.available

    def recv(self):
        return self.reply

    def close(self):
        self.closed = True
        self.close_event.set()


def test_client_uses_builtin_wire_types_and_restores_shared_settings(monkeypatch):
    connection = Connection()
    monkeypatch.setattr("multiprocessing.connection.Client", lambda *a, **k: connection)
    with RemoteSimulationPolicy(
        config={"task_text": "waffle source task", "nfe": 4}
    ) as policy:
        assert policy.task_text == "waffle source task"
        assert policy.nfe == 4
        obs = SimulationObservation(
            0.2, np.zeros((4, 5, 3), np.uint8), np.zeros((16, 6)), np.zeros(26)
        )
        p = policy.replan(obs, None, np.zeros(6))
        sent = connection.requests[-1]
        assert type(sent[1]) is SimpleNamespace
        assert p._cpk_token == 42 and p.cpk is None
        policy.replan(obs, p, np.zeros(6))
        previous, token = connection.requests[-1][2]
        assert type(previous) is dict
        assert set(previous) == set(PLAN_FIELDS)
        assert token == 42
        policy.remote_reset(123)
        assert connection.requests[-1] == ("reset_episode", 123)
    assert connection.closed
    assert connection.settings["task_text"] == "recorded task"
    assert connection.settings["nfe"] == 12


def test_response_timeout_discards_connection(monkeypatch):
    connection = Connection()
    monkeypatch.setattr("multiprocessing.connection.Client", lambda *a, **k: connection)
    policy = RemoteSimulationPolicy(timeout_s=0.01)
    connection.available = False
    with pytest.raises(TimeoutError, match="connection closed"):
        policy.reset_episode()
    assert connection.closed
    with pytest.raises(ConnectionError, match="closed"):
        policy.reset_episode()
    policy.close()


def test_busy_server_handshake_times_out_and_late_connection_is_closed(monkeypatch):
    release = threading.Event()
    done = threading.Event()
    connection = Connection()

    def connect(*a, **k):
        release.wait(1)
        done.set()
        return connection

    monkeypatch.setattr("multiprocessing.connection.Client", connect)
    try:
        with pytest.raises(TimeoutError, match="another client"):
            RemoteSimulationPolicy(connect_timeout_s=0.01)
    finally:
        release.set()
    assert done.wait(1)
    assert connection.close_event.wait(1)


def test_unrecognized_override_is_refused_before_connect(monkeypatch):
    def unexpected(*a, **k):
        raise AssertionError("must not connect")

    monkeypatch.setattr("multiprocessing.connection.Client", unexpected)
    with pytest.raises(ValueError, match="unsupported"):
        RemoteSimulationPolicy(config={"hardware_ip": "example"})
