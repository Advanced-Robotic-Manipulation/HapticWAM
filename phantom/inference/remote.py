"""Remote policy: run_deploy's model half served by a resident process.

Rig sessions restart `run_deploy` constantly (protective stops, scene resets,
different flags) and every restart paid the ~3 min model load + warmup with
the operator standing at the arm (rig 2026-09-01). The server
(`phantom.scripts.policy_server`) loads the checkpoint ONCE and keeps it warm
on the GPU; `RemotePolicy` is a drop-in for `PhantomPolicy` on the robot side
— `PlannerLoop` only ever calls `.replan(obs, prev_plan, tcp_pose)`.

Protocol (multiprocessing.connection, localhost, authkey PHANTOM_AUTHKEY):
    ("info",)                        -> ("ok", {"ckpt", "ckpt_sha", "warmed",
                                                "busy", "owner", "owner_since"})
                                        # answered for EVERY connection, even
                                        # while another client owns the server
    ("configure", {attr: value})     -> ("ok", info)      # nfe, guidance, ...
    ("reset_episode", seed:int|None) -> ("ok", None)
    ("replan", obs, plan_dict|None, tcp_pose) -> ("ok", (plan_dict, token))
    on error                         -> ("err", traceback_string)
    ("err", "busy: ...")             -> a non-info request from a second client
                                        while the first is still attached

Ownership (09-05, issue #8): the first connection that sends a
non-`info` request owns the policy until it disconnects; every other
connection can still ask `info` (so a launcher can tell BUSY from ABSENT
in milliseconds instead of timing out and loading a competing model) but
gets ("err", "busy: ...") for anything else.

The ContactPackage never crosses the wire: plans travel as dicts with
`cpk` replaced by an integer token into the server's package store, and a
prev_plan sent back swaps the token for the stored package. Everything else
in a Plan / ObsSnapshot is numpy and pickles cleanly.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

import numpy as np

from phantom.inference.policy import ObsSnapshot, Plan

log = logging.getLogger(__name__)

PHANTOM_AUTHKEY = b"phantom-policy-v1"
DEFAULT_PORT = 7777

# PhantomPolicy attributes a client may reconfigure per launch without a
# model reload (all consumed at replan time, none change tensor shapes).
CONFIGURABLE = ("nfe", "guidance", "k_seeds", "parity_fixes",
                "persistent_noise", "task_text", "drop_video", "close_p",
                "select_by", "agreement_veto", "agreement_shadow",
                "action_time_origin")

#: Settings whose None is a VALUE ("off"), not "leave the server's default".
#: Every other key is dropped when the client sends None (nfe=None means "the
#: checkpoint's own nfe"), so without this list an arm that ran with a
#: threshold would leave it armed on the warm server for the NEXT attach —
#: the silent cross-arm carry-over the condition tags exist to make impossible.
RESET_WHEN_NONE = ("agreement_veto",)


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

    class Busy(ConnectionError):
        """The server exists and answers, but another client owns it."""

    class Absent(ConnectionError):
        """Nothing listens on the address (connection refused)."""

    class Unreachable(ConnectionError):
        """Something listens but did not complete the handshake in time —
        AMBIGUOUS: never fall back to a competing local load on this."""

    @classmethod
    def _open(cls, address, timeout_s):
        # multiprocessing Client has NO timeout: connect in a worker thread
        # and give up (verification 09-01). Since 09-05 a server accepts
        # every connection immediately, so a timeout here means a hung or
        # foreign process, not a busy server.
        from multiprocessing.connection import Client
        box: dict = {}

        def _connect():
            try:
                box["conn"] = Client(address, authkey=PHANTOM_AUTHKEY)
            except Exception as e:      # noqa: BLE001 — reported below
                box["err"] = e

        t = threading.Thread(target=_connect, daemon=True)
        t.start()
        t.join(timeout_s)
        if "conn" in box:
            return box["conn"]
        if "err" in box:
            raise cls.Absent(f"policy server at {address}: {box['err']}")
        raise cls.Unreachable(
            f"policy server at {address} did not complete the handshake "
            f"within {timeout_s:.0f}s — a hung server or a foreign process "
            "on that port. Not falling back to a local model: check "
            "SERVE.sh / the port before launching.")

    @classmethod
    def probe(cls, address: tuple[str, int], timeout_s: float | None = None) -> dict:
        """`info` only — answered even while another client owns the server.
        Raises Absent / Unreachable; never Busy (busy is a FIELD here)."""
        conn = cls._open(address, timeout_s or cls.CONNECT_TIMEOUT_S)
        try:
            conn.send(("info",))
            status, payload = conn.recv()
            if status != "ok":
                raise RuntimeError(f"policy server error on 'info':\n{payload}")
            return payload
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def __init__(self, address: tuple[str, int], config: dict | None = None):
        self._conn = self._open(address, self.CONNECT_TIMEOUT_S)
        self.info = self._call("info")
        if self.info.get("busy"):
            self.close()
            raise self.Busy(
                f"policy server at {address} (ckpt {self.info.get('ckpt')}) is "
                f"BUSY: client #{self.info.get('owner')} has been attached since "
                f"{time.strftime('%H:%M:%S', time.localtime(self.info.get('owner_since') or 0))}"
                " — another run_deploy (possibly Ctrl-Z'd) owns it. Bring it "
                "to the foreground and end it; nothing was sent to the robot.")
        cfg = {k: v for k, v in (config or {}).items()
               if k in CONFIGURABLE and (v is not None or k in RESET_WHEN_NONE)}
        self.info = self._call("configure", cfg)
        effective_origin = self.info.get("effective", {}).get("action_time_origin", "inference_ready")
        if effective_origin is None and str(self.info.get("policy_kind", "phantom")) == "lerobot":
            # the pi0.5 adapter has no timing settings: one delivery semantics, inference_ready
            effective_origin = "inference_ready"
        if effective_origin != cfg.get("action_time_origin", "inference_ready"):
            self.close()
            raise RuntimeError("server action_time_origin differs from explicit client request/default; "
                               "request inference_ready or observation explicitly")
        # mirror the EFFECTIVE policy attrs (nfe=None falls back to the
        # checkpoint default server-side; tags must show the real value)
        for k, v in self.info.get("effective", {}).items():
            setattr(self, k, v)
        self.wrench_baseline_rows = int(self.info.get("wrench_baseline_rows", 0) or 0)
        log.info("remote policy: %s", self.info)

    def _call(self, *msg):
        self._conn.send(msg)
        status, payload = self._conn.recv()
        if status != "ok":
            if "busy: client #" in str(payload):
                # lost the ownership race (two clients saw busy=False and
                # both sent configure): a clear refusal, never a fallback
                self.close()
                raise self.Busy(f"policy server is BUSY — {str(payload).strip()}")
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
    """Server half: owns one loaded PhantomPolicy, lets ONE client at a time
    drive it (the owner), answers `info` to everyone, keeps the last few
    contact packages addressable by token."""

    ACCEPT_FAILURE_LIMIT = 20

    def __init__(self, policy, ckpt: str, keep_packages: int = 4,
                 ckpt_sha: str | None = None, levers: dict | None = None):
        self.policy = policy
        self.ckpt = ckpt
        self.ckpt_sha = ckpt_sha          # digest of the LOADED artifact (#9)
        # inference-latency levers the server was started with (compile / flex / fp8);
        # reported in `info` so every episode records what produced its plans
        self.levers = dict(levers or {})
        self.warmed = False
        self._store: dict[int, object] = {}
        self._next_token = 1
        self._keep = keep_packages
        self._policy_lock = threading.Lock()   # the model is not thread-safe
        self._owner_lock = threading.Lock()
        self._owner: dict | None = None        # {"id", "since"}
        self._conn_seq = 0

    # -- ownership -----------------------------------------------------
    def status(self) -> dict:
        with self._owner_lock:
            o = dict(self._owner) if self._owner else None
        return {"ckpt": self.ckpt, "ckpt_sha": self.ckpt_sha,
                "warmed": self.warmed, "busy": o is not None,
                # checkpoint properties the robot process must mirror
                "wrench_baseline_rows": int(getattr(self.policy, "wrench_baseline_rows", 0) or 0),
                # "phantom" (Cosmos teacher/student, has an ACC head) or
                # "lerobot" (π0.5 adapter: no ACC head — --terminal-veto is
                # a silent no-op and must be refused at attach; 09-10)
                "policy_kind": str(getattr(self.policy, "policy_kind", "phantom")),
                "inference_levers": dict(self.levers),
                # the deploy imagination probe (--null-imagination), owned by
                # the SERVER: a launch adopts it and tags null:<mode>, it is
                # never per-launch configurable (see CONFIGURABLE)
                "null_imagination": str(getattr(self.policy, "null_imagination",
                                                "none") or "none"),
                "owner": o["id"] if o else None,
                "owner_since": o["since"] if o else None}

    def _claim(self, conn_id) -> None:
        with self._owner_lock:
            if self._owner is None:
                self._owner = {"id": conn_id, "since": time.time()}
                log.info("client #%s owns the policy", conn_id)
            elif self._owner["id"] != conn_id:
                o = self._owner
                raise RuntimeError(
                    f"busy: client #{o['id']} owns the policy since "
                    f"{time.strftime('%H:%M:%S', time.localtime(o['since']))}")

    def _release(self, conn_id) -> None:
        with self._owner_lock:
            if self._owner is not None and self._owner["id"] == conn_id:
                self._owner = None
                self._store.clear()
                log.info("client #%s released the policy", conn_id)

    # -- request handlers ----------------------------------------------
    def handle(self, msg: tuple, conn_id=None):
        kind = msg[0]
        if kind == "info":
            return self.status()
        # every other request needs ownership. conn_id None = a caller that
        # drives handle() directly (tests, single-process use) — no claim,
        # so it can never leave a permanent phantom owner behind.
        if conn_id is not None:
            self._claim(conn_id)
        with self._policy_lock:
            return self._handle_owned(msg)

    def _handle_owned(self, msg: tuple):
        kind = msg[0]
        if kind == "configure":
            config = dict(msg[1])
            current_origin = getattr(self.policy, "action_time_origin", "inference_ready")
            if "action_time_origin" in config or current_origin == "observation":
                from phantom.inference.action_timing import validate_action_time_origin
                origin = validate_action_time_origin(config.get("action_time_origin", current_origin))
                seeds = config.get("k_seeds", self.policy.k_seeds)
                if origin == "observation" and seeds != 1:
                    raise ValueError("observation action epoch candidate supports K1 only")
                # Validate the prospective pair before mutating either setting.
                # Restoring legacy K4 must clear observation mode first; opting
                # into observation mode must establish K1 before that setter.
                first = ("action_time_origin", "k_seeds") if origin == "inference_ready" else ("k_seeds", "action_time_origin")
                for key in first:
                    if key in config:
                        setattr(self.policy, key, config.pop(key))
            for k, v in config.items():
                if k in CONFIGURABLE:
                    setattr(self.policy, k, v)
            return {**self.status(),
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
            obs, prev, tcp_pose = msg[1], msg[2], msg[3]
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

    def _serve_conn(self, conn, conn_id) -> None:
        """One client, its own thread: `info` answers immediately even while
        another client owns the policy; owned requests serialize on
        `_policy_lock`. Disconnect releases ownership."""
        try:
            with conn:
                while True:
                    msg = conn.recv()
                    try:
                        conn.send(("ok", self.handle(msg, conn_id)))
                    except Exception:
                        import traceback
                        conn.send(("err", traceback.format_exc()))
        except (EOFError, OSError, ConnectionError):
            pass
        finally:
            self._release(conn_id)
            log.info("client #%s disconnected", conn_id)

    def serve_forever(self, port: int = DEFAULT_PORT, listener=None) -> None:
        import multiprocessing
        from multiprocessing.connection import Listener
        srv = listener or Listener(("127.0.0.1", port), authkey=PHANTOM_AUTHKEY)
        with srv:
            log.info("policy server READY on 127.0.0.1:%d (ckpt %s)",
                     port, self.ckpt)
            print(f"READY ckpt={self.ckpt} port={port}", flush=True)
            accept_failures = 0
            while True:
                try:
                    conn = srv.accept()
                except (EOFError, OSError, ConnectionError,
                        multiprocessing.AuthenticationError) as e:
                    # a client that dies mid-handshake (Ctrl-Z'd / killed
                    # deploy, a probe that timed out) used to take the whole
                    # warm server down with it (rig 2026-09-04 19:47).
                    # A broken LISTENER raises the same way every time:
                    # back off, and give up after a run of failures instead
                    # of spinning at 100% CPU (09-05).
                    accept_failures += 1
                    log.warning("client dropped during accept (%s) — "
                                "serving on (%d in a row)", type(e).__name__,
                                accept_failures)
                    if accept_failures >= self.ACCEPT_FAILURE_LIMIT:
                        raise RuntimeError(
                            f"{accept_failures} consecutive accept() failures "
                            "— the listener itself is broken") from e
                    time.sleep(min(0.05 * accept_failures, 1.0))
                    continue
                accept_failures = 0
                self._conn_seq += 1
                cid = self._conn_seq
                log.info("client #%d connected", cid)
                threading.Thread(target=self._serve_conn, args=(conn, cid),
                                 name=f"policy-client-{cid}", daemon=True).start()
