"""Numpy-only client for an already-running PHANTOM policy server.

The server protocol is phantom.inference.remote's authenticated Python
``multiprocessing.connection`` protocol. It is intended for a trusted local
policy endpoint. Only built-in SimpleNamespace/dict and numpy values cross the
wire, so Isaac's Python need not import the model stack or install simulator
modules into the resident policy server's checkout.

This client never starts a server or launches a hardware deployment process.
When configuration overrides are supplied, the previous effective settings
are restored before a normal close. Inference state/noise is episode-specific
and is reset explicitly by the caller; it is not restorable after simulation.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

AUTHKEY = b"phantom-policy-v1"
CONFIGURABLE = (
    "nfe",
    "guidance",
    "k_seeds",
    "parity_fixes",
    "persistent_noise",
    "task_text",
    "drop_video",
    "close_p",
)
PLAN_FIELDS = (
    "t_created",
    "t0_pose",
    "actions",
    "action_times",
    "sigma",
    "gate",
    "p_evt",
    "latency_s",
    "diag",
)
OBS_FIELDS = (
    "t",
    "rgb",
    "wrist_window",
    "ur_state",
    "gel",
    "fields",
    "contact_state",
    "reactive",
    "prev_chunk",
)


class RemoteSimulationPolicy:
    """Existing resident policy interface with bounded connection/response wait.

    ``timeout_s`` covers waiting for a response after a request has been sent;
    allow enough time for the checkpoint's inference. A timed-out connection
    is closed and cannot be reused (late replies must not become the next
    request's result). ``connect_timeout_s`` also covers the auth handshake
    when another client already occupies the single-client server.
    """

    def __init__(
        self,
        address=("127.0.0.1", 7777),
        *,
        config: dict | None = None,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 6.0,
    ):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not np.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be finite and positive")
        unknown = set(config or {}) - set(CONFIGURABLE)
        if unknown:
            raise ValueError(f"unsupported policy configuration: {sorted(unknown)}")
        self.address = tuple(address)
        self.timeout_s = float(timeout_s)
        self._conn = None
        self._lock = threading.Lock()
        self._restore = {}
        from multiprocessing.connection import Client

        box, ready, abandoned = {}, threading.Event(), threading.Event()
        connect_lock = threading.Lock()

        def connect():
            try:
                conn = Client(self.address, authkey=AUTHKEY)
                with connect_lock:
                    if abandoned.is_set():
                        conn.close()
                    else:
                        box["conn"] = conn
            except Exception as exc:  # noqa: BLE001 - relay handshake failures to the caller
                box["error"] = exc
            finally:
                ready.set()

        worker = threading.Thread(target=connect, daemon=True)
        worker.start()
        if not ready.wait(float(connect_timeout_s)):
            with connect_lock:
                abandoned.set()
                # Cover the completion/timeout boundary race too.
                if "conn" in box:
                    box["conn"].close()
            raise TimeoutError(
                f"policy server {self.address} connection/auth timed out; it may have another client"
            )
        if "error" in box:
            raise ConnectionError(
                f"policy server {self.address}: {box['error']}"
            ) from box["error"]
        self._conn = box["conn"]
        try:
            self.info = self._call("info")
            current = self._call("configure", {})
            self.info.update(current)
            effective = dict(current.get("effective", {}))
            override = {k: v for k, v in (config or {}).items() if v is not None}
            missing = set(override) - set(effective)
            if missing:
                raise RuntimeError(
                    f"server cannot report restorable settings: {sorted(missing)}"
                )
            self._restore = {k: effective[k] for k in override}
            if override:
                self.info.update(self._call("configure", override))
            for key, value in self.info.get("effective", {}).items():
                setattr(self, key, value)
            # Checkpoint preprocessing property, never a configurable override.
            # Older teachers were trained on raw pad wrench; v6 subtracts the
            # per-episode median captured by the observation builder.
            self.wrench_baseline_rows = int(
                self.info.get("wrench_baseline_rows", 0) or 0
            )
        except Exception:
            self.close()
            raise

    def _call(self, *message):
        with self._lock:
            if self._conn is None:
                raise ConnectionError("policy connection is closed")
            self._conn.send(message)
            if not self._conn.poll(self.timeout_s):
                self._conn.close()
                self._conn = None
                raise TimeoutError(
                    f"policy server timed out handling {message[0]!r}; connection closed"
                )
            status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"policy server failed {message[0]!r}: {payload}")
        return payload

    def replan(self, obs, prev_plan, tcp_pose):
        snapshot = SimpleNamespace(**{name: getattr(obs, name) for name in OBS_FIELDS})
        prev = (
            None
            if prev_plan is None
            else (
                {name: getattr(prev_plan, name) for name in PLAN_FIELDS},
                getattr(prev_plan, "_cpk_token", None),
            )
        )
        values, token = self._call(
            "replan", snapshot, prev, np.asarray(tcp_pose, dtype=np.float64)
        )
        missing = set(PLAN_FIELDS) - set(values)
        if missing:
            raise RuntimeError(
                f"policy response is missing plan fields: {sorted(missing)}"
            )
        return SimpleNamespace(
            **{name: values[name] for name in PLAN_FIELDS}, cpk=None, _cpk_token=token
        )

    def remote_reset(self, seed: int | None):
        self._call("reset_episode", None if seed is None else int(seed))

    def reset_episode(self):
        self.remote_reset(None)

    def close(self):
        """Restore changed settings, then release the resident server client slot.

        Restoration cannot be guaranteed after network/server failure; such a
        failure is surfaced to the caller, with the connection still closed.
        """
        try:
            if self._conn is not None and self._restore:
                self._call("configure", self._restore)
                self._restore = {}
        finally:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
