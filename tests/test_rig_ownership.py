"""Issue #8 (review 09-05): a busy policy server must be distinguishable
from an absent one over the REAL localhost protocol, a second client must
not be able to drive the policy, and one host gets one robot owner."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener

import pytest

from phantom.inference.remote import PHANTOM_AUTHKEY, PolicyServer, RemotePolicy
from tests.test_policy_server import _StubPolicy  # noqa: E402


@pytest.fixture
def live_server():
    srv = PolicyServer(_StubPolicy(), ckpt="stub.pt", ckpt_sha="abc123def456")
    listener = Listener(("127.0.0.1", 0), authkey=PHANTOM_AUTHKEY)
    port = listener.address[1]
    t = threading.Thread(target=srv.serve_forever,
                         kwargs={"port": port, "listener": listener}, daemon=True)
    t.start()
    time.sleep(0.05)
    yield srv, port
    try:
        listener.close()
    except Exception:
        pass


def test_info_answers_while_another_client_owns_the_policy(live_server):
    srv, port = live_server
    a = RemotePolicy(("127.0.0.1", port), {"nfe": 1})       # A attaches (configure = ownership)
    t0 = time.perf_counter()
    info = RemotePolicy.probe(("127.0.0.1", port))
    assert time.perf_counter() - t0 < 2.0, "probe must not wait for A to finish"
    assert info["busy"] is True and info["owner"] == 1 and info["ckpt_sha"] == "abc123def456"
    # a second CLIENT is refused with a clear Busy — never a local fallback
    with pytest.raises(RemotePolicy.Busy):
        RemotePolicy(("127.0.0.1", port), {"nfe": 1})
    a.close()
    time.sleep(0.1)
    assert RemotePolicy.probe(("127.0.0.1", port))["busy"] is False
    b = RemotePolicy(("127.0.0.1", port), {"nfe": 2})       # now B can own it
    assert b.info["busy"] is True and b.info["owner"] not in (None, 1)
    b.close()


def test_second_connection_cannot_drive_the_policy(live_server):
    srv, port = live_server
    a = RemotePolicy(("127.0.0.1", port), {"nfe": 1})
    from multiprocessing.connection import Client
    c = Client(("127.0.0.1", port), authkey=PHANTOM_AUTHKEY)
    c.send(("reset_episode", 1))
    status, payload = c.recv()
    assert status == "err" and "busy" in payload
    assert srv.policy.rf.resets == 0, "B's reset must not have reached the model"
    c.send(("info",))
    assert c.recv()[0] == "ok"
    c.close()
    a.remote_reset(3)
    assert srv.policy.rf.resets == 1
    a.close()


def test_probe_states_absent_vs_unreachable():
    import socket
    # ABSENT: nothing listens -> Absent (connection refused), fast
    s = socket.socket(); s.bind(("127.0.0.1", 0)); free = s.getsockname()[1]; s.close()
    with pytest.raises(RemotePolicy.Absent):
        RemotePolicy.probe(("127.0.0.1", free), timeout_s=2.0)
    # UNREACHABLE: a raw socket that accepts but never speaks -> Unreachable
    raw = socket.socket(); raw.bind(("127.0.0.1", 0)); raw.listen(1)
    port = raw.getsockname()[1]
    try:
        with pytest.raises(RemotePolicy.Unreachable):
            RemotePolicy.probe(("127.0.0.1", port), timeout_s=0.5)
    finally:
        raw.close()


def test_rig_lease_names_the_owner_and_makes_no_robot_call(tmp_path, monkeypatch):
    monkeypatch.setenv("PHANTOM_RIG_LOCK_DIR", str(tmp_path))
    from phantom.drivers.real import rig_lease
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "from phantom.drivers.real import rig_lease; import time; "
         "f = rig_lease.acquire('192.168.88.56'); print('HELD', flush=True); time.sleep(30)"],
        env={**os.environ, "PHANTOM_RIG_LOCK_DIR": str(tmp_path)},
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "HELD"
        with pytest.raises(rig_lease.RigBusy) as ei:
            rig_lease.acquire("192.168.88.56")
        msg = str(ei.value)
        assert f"pid {holder.pid}" in msg and "Nothing was sent to the robot" in msg
        who = rig_lease.owner("192.168.88.56")
        assert who and who["pid"] == holder.pid
    finally:
        holder.kill(); holder.wait()
    # released with the process: a new owner succeeds
    f = rig_lease.acquire("192.168.88.56")
    f.close()


def test_urarm_takes_the_lease_before_any_robot_call(tmp_path, monkeypatch):
    """The driver acquires the lease at the top of _connect_control: with
    the lease held elsewhere, neither the dashboard 'stop old script'
    preflight (a robot call) nor an RTDE control interface happens."""
    import types
    monkeypatch.setenv("PHANTOM_RIG_LOCK_DIR", str(tmp_path))
    from phantom.config.hardware import load_hardware
    from phantom.drivers.real import rig_lease
    from phantom.drivers.real.ur import URArm
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    held = rig_lease.acquire(str(hw.arm.ip))
    arm = URArm(hw)
    built, dashboard = [], []
    monkeypatch.setitem(sys.modules, "rtde_control",
                        types.SimpleNamespace(RTDEControlInterface=lambda *a, **k: built.append(1)))
    arm._preflight_stop_old_script = lambda: dashboard.append("STOP")
    arm._probe_construct = lambda: None
    with pytest.raises(rig_lease.RigBusy):
        arm._connect_control()
    assert built == [], "no RTDE control interface may be constructed without the lease"
    assert dashboard == [], "the dashboard preflight is a robot call — never without the lease"
    assert URArm._live_ctrl_count == 0, "the process-wide controller slot must not stay reserved"
    held.close()


def test_operator_stop_needs_a_letter():
    from phantom.scripts.run_deploy import operator_stop_requested
    assert not operator_stop_requested("\n")
    assert not operator_stop_requested("   \n")
    assert not operator_stop_requested("s\n")          # 's' is the SUCCESS label, not stop
    assert operator_stop_requested("x\n")
    assert operator_stop_requested("STOP\n")


def test_ownership_race_loser_gets_busy_not_a_fallback(live_server):
    """Verify 09-05 #4: two clients that both saw busy=False and both send
    configure — the loser must raise Busy (run_deploy aborts), never a
    generic error that 'auto' would turn into a local load."""
    from multiprocessing.connection import Client
    srv, port = live_server
    a = Client(("127.0.0.1", port), authkey=PHANTOM_AUTHKEY)
    b = Client(("127.0.0.1", port), authkey=PHANTOM_AUTHKEY)
    for c in (a, b):
        c.send(("info",)); assert c.recv()[1]["busy"] is False
    a.send(("configure", {"nfe": 1})); assert a.recv()[0] == "ok"
    # b now uses the RemotePolicy path: its configure must surface as Busy
    rp = RemotePolicy.__new__(RemotePolicy)
    rp._conn = b
    with pytest.raises(RemotePolicy.Busy):
        rp._call("configure", {"nfe": 2})
    a.close()


def test_in_process_handle_never_becomes_a_phantom_owner(live_server):
    srv, port = live_server
    srv.handle(("reset_episode", 1))                 # direct, single-process use
    assert RemotePolicy.probe(("127.0.0.1", port))["busy"] is False


def test_disk_preflight_aborts_when_nearly_full(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    from phantom.scripts import run_deploy as rd
    DU = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DU(1e12, 1e12 - 3e9, 3e9))
    assert rd.preflight_disk(tmp_path / "episodes" / "deploy") == 4
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DU(1e12, 0, 1e12))
    assert rd.preflight_disk(tmp_path / "episodes" / "deploy") == 0
