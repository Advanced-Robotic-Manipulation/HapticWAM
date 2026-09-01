"""Remote policy: run_deploy's model half served by a resident process.

Rig sessions restart `run_deploy` constantly (protective stops, scene resets,
different flags) and every restart paid the ~3 min model load + warmup with
the operator standing at the arm (rig 2026-09-01). The server
(`phantom.scripts.policy_server`) loads the checkpoint ONCE and keeps it warm
on the GPU; `RemotePolicy` is a drop-in for `PhantomPolicy` on the robot side
— `PlannerLoop` only ever calls `.replan(obs, prev_plan, tcp_pose)`.

Protocol (multiprocessing.connection, localhost, authkey PHANTOM_AUTHKEY):
    ("info",)                        -> ("ok", {"ckpt": ..., "warmed": bool})
    ("configure", {attr: value})     -> ("ok", info)      # nfe, guidance, ...
    ("reset_episode", seed:int|None) -> ("ok", None)
    ("replan", obs, plan_dict|None, tcp_pose) -> ("ok", (plan_dict, token))
    on error                         -> ("err", traceback_string)

The ContactPackage never crosses the wire: plans travel as dicts with
`cpk` replaced by an integer token into the server's package store, and a
prev_plan sent back swaps the token for the stored package. Everything else
in a Plan / ObsSnapshot is numpy and pickles cleanly.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from phantom.inference.policy import ObsSnapshot, Plan

log = logging.getLogger(__name__)

PHANTOM_AUTHKEY = b"phantom-policy-v1"
DEFAULT_PORT = 7777

# PhantomPolicy attributes a client may reconfigure per launch without a
# model reload (all consumed at replan time, none change tensor shapes).
CONFIGURABLE = ("nfe", "guidance", "k_seeds", "parity_fixes",
                "persistent_noise", "task_text", "drop_video", "close_p")


def plan_to_wire(plan: Plan) -> dict:
    # NOT dataclasses.asdict: that deep-copies every field INCLUDING the
    # ContactPackage (GPU tensors) before we could drop it
    return {f.name: getattr(plan, f.name)
            for f in dataclasses.fields(Plan) if f.name != "cpk"}


def plan_from_wire(d: dict, cpk) -> Plan:
    return Plan(cpk=cpk, **d)


class RemotePolicy:
    """Client half. Duck-types the PhantomPolicy surface PlannerLoop and
    run_deploy use: `.replan`, `.task_text`, plus `remote_reset(seed)` in
    place of the local `rf` seeding dance."""

    CONNECT_TIMEOUT_S = 6.0

    def __init__(self, address: tuple[str, int], config: dict | None = None):
        # multiprocessing Client has NO timeout: against a server that is
        # busy with another client the TCP connect succeeds (listen backlog)
        # and the authkey handshake then blocks FOREVER — '--policy-server
        # auto' would hang instead of falling back, and PICK.sh's probe with
        # it (verification 09-01). Connect in a worker thread and give up.
        import threading
        from multiprocessing.connection import Client
        box: dict = {}

        def _connect():
            try:
                box["conn"] = Client(address, authkey=PHANTOM_AUTHKEY)
            except Exception as e:      # noqa: BLE001 — reported below
                box["err"] = e

        t = threading.Thread(target=_connect, daemon=True)
        t.start()
        t.join(self.CONNECT_TIMEOUT_S)
        if "conn" not in box:
            if "err" in box:
                raise ConnectionError(f"policy server at {address}: {box['err']}")
            raise ConnectionError(
                f"policy server at {address} did not answer within "
                f"{self.CONNECT_TIMEOUT_S:.0f}s — most likely another "
                "run_deploy is still attached (it serves one client at a "
                "time). Ctrl-C the other one or restart SERVE.sh.")
        self._conn = box["conn"]
        self.info = self._call("info")
        cfg = {k: v for k, v in (config or {}).items()
               if k in CONFIGURABLE and v is not None}
        self.info = self._call("configure", cfg)
        # mirror the EFFECTIVE policy attrs (nfe=None falls back to the
        # checkpoint default server-side; tags must show the real value)
        for k, v in self.info.get("effective", {}).items():
            setattr(self, k, v)
        log.info("remote policy: %s", self.info)

    def _call(self, *msg):
        self._conn.send(msg)
        status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"policy server error on {msg[0]!r}:\n{payload}")
        return payload

    # -- PhantomPolicy surface -----------------------------------------
    def replan(self, obs: ObsSnapshot, prev_plan: Plan | None,
               tcp_pose: np.ndarray) -> Plan:
        prev = None
        if prev_plan is not None:
            prev = (plan_to_wire(prev_plan), getattr(prev_plan, "_cpk_token", None))
        plan_d, token = self._call("replan", obs, prev, np.asarray(tcp_pose))
        plan = plan_from_wire(plan_d, cpk=None)
        plan._cpk_token = token
        return plan

    def remote_reset(self, seed: int | None) -> None:
        self._call("reset_episode", seed)

    def reset_episode(self) -> None:
        """Runtime's per-episode hook — plain noise reset, seed already set
        by remote_reset (same double-reset sequence as the local policy)."""
        self._call("reset_episode", None)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class PolicyServer:
    """Server half: owns one loaded PhantomPolicy, serves one client at a
    time, keeps the last few contact packages addressable by token."""

    def __init__(self, policy, ckpt: str, keep_packages: int = 4):
        self.policy = policy
        self.ckpt = ckpt
        self.warmed = False
        self._store: dict[int, object] = {}
        self._next_token = 1
        self._keep = keep_packages

    # -- request handlers ----------------------------------------------
    def handle(self, msg: tuple):
        kind = msg[0]
        if kind == "info":
            return {"ckpt": self.ckpt, "warmed": self.warmed}
        if kind == "configure":
            for k, v in msg[1].items():
                if k in CONFIGURABLE:
                    setattr(self.policy, k, v)
            return {"ckpt": self.ckpt, "warmed": self.warmed,
                    "applied": sorted(msg[1].keys()),
                    "effective": {k: getattr(self.policy, k, None)
                                  for k in CONFIGURABLE}}
        if kind == "reset_episode":
            import torch
            seed = msg[1]
            if seed is not None and hasattr(self.policy.rf, "_gen"):
                self.policy.rf._gen = torch.Generator().manual_seed(int(seed))
            self.policy.rf.reset_episode_noise()
            self._store.clear()
            return None
        if kind == "replan":
            _, prev, tcp_pose = msg[1], msg[2], msg[3]
            obs = msg[1]
            prev_plan = None
            if prev is not None:
                prev_d, token = prev
                prev_plan = plan_from_wire(prev_d, cpk=self._store.get(token))
            plan = self.policy.replan(obs, prev_plan, np.asarray(tcp_pose))
            token = self._next_token
            self._next_token += 1
            if plan.cpk is not None:
                self._store[token] = plan.cpk
                for old in sorted(self._store)[:-self._keep]:
                    del self._store[old]
            return (plan_to_wire(plan), token)
        raise ValueError(f"unknown request {kind!r}")

    def warmup(self, hw, teacher: bool = True) -> None:
        """One synthetic replan so the first real episode pays no compile/
        alloc cost. Uses the same fake_obs run_deploy's warmup uses."""
        from phantom.scripts.bench_inference import fake_obs
        t0 = time.perf_counter()
        self.policy.replan(fake_obs(hw, teacher=teacher), None, np.zeros(6))
        self.policy.reset_episode()
        self.warmed = True
        log.info("warmup replan done in %.1f s", time.perf_counter() - t0)

    def serve_forever(self, port: int = DEFAULT_PORT) -> None:
        from multiprocessing.connection import Listener
        with Listener(("127.0.0.1", port), authkey=PHANTOM_AUTHKEY) as srv:
            log.info("policy server READY on 127.0.0.1:%d (ckpt %s)",
                     port, self.ckpt)
            print(f"READY ckpt={self.ckpt} port={port}", flush=True)
            while True:
                with srv.accept() as conn:
                    log.info("client connected")
                    try:
                        while True:
                            msg = conn.recv()
                            try:
                                conn.send(("ok", self.handle(msg)))
                            except Exception:
                                import traceback
                                conn.send(("err", traceback.format_exc()))
                    except EOFError:
                        log.info("client disconnected")
