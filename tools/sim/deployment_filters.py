"""Small inference-loop guards mirrored from PHANTOM deployment.

The default GO_ANY preset does not enable TerminalVeto. Its motion-stall
watchdog is unconditional and examines measured TCP versus accepted streamed
TCP targets before each model call. These guards never connect hardware.
"""

from __future__ import annotations

import logging
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

log = logging.getLogger(__name__)


class PlannerStallWatchdog:
    """Mirror PlannerLoop.STALL_* and its consecutive-replan strike rule.

    Call ``check(t, actual_tcp, accepted_target)`` immediately before inference.
    Use current physical feedback and the last IK-accepted target; model-input
    delay experiments must not delay this safety observation. On ``stop_reason``
    call the adapter's ``request_stop`` and skip inference for that replan.
    """

    def __init__(self):
        self.previous_tcp = None
        self.previous_command = None
        self.previous_time = None
        self.strikes = 0
        self.stop_reason = None

    @staticmethod
    def _pose(value, name):
        pose = np.asarray(value, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError(f"{name} must be a finite six-vector")
        return pose

    def check(self, t, actual_tcp, accepted_target):
        if not np.isfinite(t) or (
            self.previous_time is not None and t < self.previous_time
        ):
            raise ValueError("watchdog time must be finite and monotonic")
        actual = self._pose(actual_tcp, "actual_tcp")
        command = (
            None
            if accepted_target is None
            else self._pose(accepted_target, "accepted_target")
        )
        commanded_motion = measured_motion = None
        if (
            self.previous_command is not None
            and command is not None
            and self.previous_tcp is not None
        ):
            commanded_motion = float(
                np.linalg.norm(command[:3] - self.previous_command[:3])
            )
            measured_motion = float(np.linalg.norm(actual[:3] - self.previous_tcp[:3]))
            if commanded_motion > 0.005 and measured_motion < 0.2 * commanded_motion:
                self.strikes += 1
                if self.strikes >= 2:
                    self.stop_reason = "motion_stall"
            else:
                self.strikes = 0
        self.previous_tcp = actual.copy()
        self.previous_command = None if command is None else command.copy()
        self.previous_time = float(t)
        return {
            "t": float(t),
            "stop_reason": self.stop_reason,
            "strikes": self.strikes,
            "commanded_displacement_m": commanded_motion,
            "measured_displacement_m": measured_motion,
            "source": "PlannerLoop: command>5mm and actual<20% for two consecutive replan intervals",
        }


# Native source snapshot: phantom/deploy/planner.py at 60965b85369d2527903c40d97e7b5e919fbc4ff9.
# The four decision/history methods below retain native logic. The sole semantic
# substitution is time.perf_counter() -> the supplied current simulation clock.
NATIVE_VETO_SOURCE_SHA256 = (
    "c8aadbd9359ea7cfb26a9551592e3e9c61d592026a616668ed001146b4496191"
)
HISTORICAL_VETO_SOURCE_SHA256 = (
    "7c71eea057710ca1ea3331e9b52934584c56709bef5427d1f1f37eb08ecc22b7"
)


@dataclass
class TerminalVetoConfig:
    """Deploy-time terminal commitment guard (review P3, 2026-08-28).

    The ACC gate is read-only at deploy — `plan.gate` / `plan.p_evt` are logged
    and nothing in `phantom/deploy/` lets them modify a chunk — while the rig
    closes on air at 65-120 mm and lifts anyway. This is the scripted override
    (2503.23835 gets 100% disturbance resilience from exactly this shape of
    rule; 2410.13124 reports 11.5% phantom grasps even for a tactile-equipped
    diffusion policy, so a scripted veto is defensible rather than a patch).

    Two rules, both OFF unless `--terminal-veto` is passed:

    close-mask   a commanded gripper-close transition is rewritten to HOLD the
                 current aperture unless p_contact > p_close, or the TCP is
                 already inside the task's demo CLOSE band (where closing is
                 what a demo would do): `z <= z_ref + z_margin`, with `z_ref`
                 the demo `tcp_z_min` and `z_margin` sized so the band reaches
                 the per-task demo close p95 (eval/grasp_label.Z_MAX_MM).
                 Measuring it from the already-lowered SAFETY floor with a
                 15 mm margin put the hatch 30-90 mm below every demo close and
                 ~50 mm below the model's own predicted close height, so it
                 never fired (VALIDATION_0830 P0 #5).
    recovery     if a close was EXECUTED and this or the very next replan
                 reports p_evt[none] > p_none, the grasp is phantom: command
                 the task open aperture and forbid any upward z in the chunk,
                 so the policy re-descends instead of lifting nothing. Capped
                 at `max_retries` cycles per episode, then the episode ends
                 with reason `veto_retry_cap` (an uncapped open/re-descend loop
                 is the safety gap the review's completeness critic flagged).

    Both rules read the aperture through the TRAINING close rule
    (train/common.close_index): a rise of `close_rise` above the episode's
    RUNNING MINIMUM, not above the current sample. The rig's terminal phase
    ramps 0.31 -> 0.52 over several replans (per-replan rise 0.02-0.06), so the
    old per-replan rate test never fired on the failure this exists to catch.
    """

    p_close: float = 0.5  # theta_close on p_contact = 1 - p_evt[none]
    p_none: float = 0.9  # p_evt[none] above which a close is phantom
    max_retries: int = 3
    z_floor: float | None = None  # m; the SAFETY floor (tcp_z_min - margin)
    z_ref: float | None = None  # m; demo tcp_z_min the band is measured
    # from (falls back to z_floor)
    z_margin: float = 0.015  # "already in the demo close band" margin
    open_aperture: float = 0.0  # task open aperture (demo start mean)
    # Close detection reuses the TRAINING rule (train/common.close_index):
    # aperture past CLOSE_ABS_POS after rising CLOSE_ABS_RISE from the running
    # minimum of the measured aperture.
    close_pos: float = 0.45
    close_rise: float = 0.15
    # Tactile-grounded phantom / lost-object recovery (rig 09-04, evaluated
    # on all 17 closes of the session: 6/6 phantoms + 5/5 lost objects flagged,
    # 0/6 carried grasps): gripper measured closed AND the TCP has risen more
    # than `phantom_dz` above its height at the close AND the trailing
    # `phantom_window` max of the baseline-corrected pad force is below
    # `phantom_f` on BOTH pads, sustained `phantom_t` -> open + re-descend
    # through the same recovery arm as the p_none rule. A time-only gate has
    # no clean cell (force ramp after a real close spans 0.4-3.2 s); the lift
    # gate defers the question to the moment it becomes answerable.
    phantom_dz: float = 0.03
    phantom_f: float = 2.5
    phantom_t: float = 0.3
    phantom_window: float = 1.0


class TerminalVetoFilter:
    """Optional native terminal veto applied at plan delivery, before submit.

    Pass this object as adapter.plan_filter and call reset() for every episode.
    The adapter must retain the request snapshot until inference delivery,
    expose entered_grip_after(t), clear_grip_latch(), request_stop(reason),
    current rings, and _last_observe_t. Only actually sent closure history may
    arm recovery. Task-specific z_ref/z_floor/open_aperture must be supplied
    from deployment evidence; this class does not infer a task or start pose.

    Implements native close-mask, p_none recovery, and tactile phantom recovery,
    including loaded-pad/floor-only exclusions and retry cap. It does not enable
    any other LEVERS CLI preset or add a policy controller. Simulated pad loads
    remain uncalibrated; matching the rule does not validate its physical input.

    implementation="fd4a032" reproduces that recorded commit's veto decisions:
    it predates loaded-pad exclusions, tactile recovery and grip-latch clearing
    during recovery. The phantom_* settings are unused by that implementation.
    Its settings still must be supplied from the task's historical CLI defaults.
    """

    VETO_RECOVERY_REPLANS = 1

    def __init__(self, hw, config=None, *, wrench_base=None, implementation="live"):
        if implementation not in ("live", "fd4a032"):
            raise ValueError("terminal veto implementation must be live or fd4a032")
        self.implementation = implementation
        self.hw = hw
        self.veto = (
            TerminalVetoConfig()
            if config is None
            else TerminalVetoConfig(**config)
            if isinstance(config, dict)
            else config
        )
        if not isinstance(self.veto, TerminalVetoConfig):
            raise TypeError("config must be TerminalVetoConfig or a matching dict")
        for name, value in asdict(self.veto).items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"terminal veto {name} must be finite")
        if (
            self.veto.max_retries < 0
            or int(self.veto.max_retries) != self.veto.max_retries
        ):
            raise ValueError("terminal veto max_retries must be a nonnegative integer")
        if not 0 <= self.veto.open_aperture <= hw.gripper.max_close_cmd:
            raise ValueError(
                "terminal veto open_aperture exceeds hardware closure limits"
            )
        if not 0 <= self.veto.p_close <= 1 or not 0 <= self.veto.p_none <= 1:
            raise ValueError("terminal veto probabilities must lie in 0..1")
        if (
            min(
                self.veto.z_margin,
                self.veto.close_rise,
                self.veto.phantom_dz,
                self.veto.phantom_f,
                self.veto.phantom_t,
            )
            < 0
            or self.veto.phantom_window <= 0
        ):
            raise ValueError(
                "terminal veto thresholds must be nonnegative and window positive"
            )
        self._wrench_base = deepcopy(wrench_base or {})
        self.reset()

    def reset(self):
        self.state = {
            "closed_idx": None,
            "retries": 0,
            "g_min": None,
            "in_close": False,
            "grip_seen_t": float("-inf"),
            "close_permitted": False,
            "allowed_floor_only": False,
            "closed_floor_only": False,
        }
        self.replan_index = 0
        self.last_record = None
        self._now = None
        self.executor = self.session = None
        self.snapshots = SimpleNamespace(wrench_base=deepcopy(self._wrench_base))

    @staticmethod
    def feedback_source(adapter):
        """Match the native opt-in release bridge without changing defaults."""
        return (
            "current_delivery"
            if getattr(adapter, "release_controller", None) is not None
            else "request_snapshot_historical"
        )

    def _feedback(self, snapshot, adapter, ur, now):
        source = self.feedback_source(adapter)
        if source == "request_snapshot_historical":
            return (
                ur[2 * self.hw.arm.dof : 2 * self.hw.arm.dof + 6].copy(),
                float(ur[-2]),
                {"feedback_source": source},
            )
        feedback = {}
        timestamps = {}
        for name, field, shape in (
            ("arm", "tcp_pose", (6,)),
            ("gripper", "state", (2,)),
        ):
            ring = adapter.rings.get(name)
            times, values = ring.latest(1) if ring is not None else ([], {})
            if not len(times) or field not in values:
                raise RuntimeError(
                    f"terminal veto release profile requires measured {name} feedback"
                )
            value = np.asarray(values[field][-1], dtype=float)
            timestamp = float(times[-1])
            if (
                value.shape != shape
                or not np.isfinite(value).all()
                or not np.isfinite(timestamp)
                or timestamp > now + 1e-9
            ):
                raise ValueError(f"invalid terminal veto delivery {name} feedback")
            feedback[name] = value.copy()
            timestamps[f"feedback_{name}_t_s"] = timestamp
        return (
            feedback["arm"],
            float(feedback["gripper"][0]),
            {"feedback_source": source, **timestamps},
        )

    def __call__(self, plan, snapshot, adapter):
        for method in ("entered_grip_after", "clear_grip_latch", "request_stop"):
            if not callable(getattr(adapter, method, None)):
                raise NotImplementedError(
                    f"native terminal veto requires adapter.{method}()"
                )
        now = getattr(adapter, "_last_observe_t", None)
        if (
            now is None
            or not np.isfinite(now)
            or now < snapshot.t
            or (self._now is not None and now < self._now)
        ):
            raise ValueError(
                "terminal veto requires current monotonic delivery-clock feedback"
            )
        sensors = self.hw.tactile.sensors if self.implementation == "live" else ()
        for sensor in sensors:
            ring = adapter.rings.get(f"tactile_{sensor.name}")
            ts, values = ring.latest(1) if ring is not None else ([], {})
            if not len(ts) or "wrench" not in values:
                raise NotImplementedError(
                    "native terminal veto loaded/empty guards require each pad wrench ring"
                )
        ur = np.asarray(snapshot.ur_state, float)
        if ur.shape != (self.hw.ur_state_dim,) or not np.isfinite(ur).all():
            raise ValueError("terminal veto requires a finite native UR state")
        # Model conditioning remains the captured request snapshot. Only the
        # opt-in native release bridge evaluates veto rules on live delivery
        # TCP/aperture. Safety separately checks feedback freshness each tick.
        veto_tcp, veto_grip, feedback_metadata = self._feedback(
            snapshot, adapter, ur, float(now)
        )
        original = np.asarray(plan.actions)
        if (
            original.ndim != 2
            or original.shape[1] != 7
            or not len(original)
            or not np.isfinite(original).all()
            or not np.isfinite(plan.p_evt).all()
        ):
            raise ValueError(
                "terminal veto requires finite native action/probability arrays"
            )
        self._now = float(now)
        self.executor = adapter
        self.session = SimpleNamespace(rings=adapter.rings)
        filtered = copy(plan)
        filtered.actions = original.copy()
        filtered.diag = deepcopy(getattr(plan, "diag", None) or {})
        apply_veto = (
            self._apply_veto
            if self.implementation == "live"
            else self._apply_veto_historical
        )
        rec = apply_veto(
            filtered,
            veto_tcp,
            veto_grip,
            self.state,
            self.replan_index,
        )
        # Explicit placement variant: the native whole-chunk max test can call
        # an opening prefix "closing" because of a later close sample. Only
        # restore the teacher's opening samples inside the declared release
        # window. Recovery, arm-channel rewrites and retry stops stay native.
        from phantom.deploy.release_controller import restore_policy_openings

        restore_policy_openings(filtered, original, rec, adapter)
        rewritten = rec["action"] in (
            "close_masked",
            "recovery_open",
            "recovery_tactile",
        )
        stop_reason = "veto_retry_cap" if rec["action"] == "retry_cap" else None
        rec.update(
            **feedback_metadata,
            feedback_tcp_pose=veto_tcp.tolist(),
            feedback_gripper=veto_grip,
            snapshot_t_s=float(snapshot.t),
            applied_at_s=self._now,
            replan_index=self.replan_index,
            cpk_invalidated=rewritten,
            stop_reason=stop_reason,
            implementation=self.implementation,
            native_source_sha256=(
                NATIVE_VETO_SOURCE_SHA256
                if self.implementation == "live"
                else HISTORICAL_VETO_SOURCE_SHA256
            ),
        )
        filtered.diag["terminal_veto"] = rec.copy()
        if rewritten:
            filtered.diag["actions_pre_veto"] = original.tolist()
        self.last_record = rec.copy()
        self.replan_index += 1
        if stop_reason is not None:
            adapter.request_stop(stop_reason)
        return filtered

    def _note_executed_close(self, state: dict, n: int) -> None:
        """Latch the replan index of a close the EXECUTOR actually entered.

        Plan acceptance is not execution: a close living in the tail of a chunk
        playback never reached was never commanded, and a chunk the veto itself
        rewrote carries the held aperture rather than the proposal. The
        executor's `_grip_hist` is the deploy-side stand-in for the recorded
        STREAM_ACTIONS gripper channel, i.e. exactly the commands that went
        out."""
        v = self.veto
        fn = getattr(self.executor, "entered_grip_after", None)
        if v is None or fn is None:
            return
        try:
            steps = fn(state["grip_seen_t"])
        except Exception:  # never let telemetry kill an episode
            log.exception("entered_grip_after failed — the veto latch stays cold")
            return
        g_min = state["g_min"]
        for t_step, g in steps:
            state["grip_seen_t"] = max(state["grip_seen_t"], float(t_step))
            if g_min is None:
                continue
            # the TRANSITION into the close is the event, not the state: the
            # running minimum keeps every later step of a held grasp above the
            # rise threshold, and re-arming on those would make the latch
            # unbounded again by another route.
            closed = g > v.close_pos and (g - g_min) > v.close_rise
            if closed and not state["in_close"]:  # noqa: SIM102 - preserve native decision structure
                # arm the latch only for a close the veto PERMITTED: after
                # `close_masked` the executor holds the aperture, but on an
                # open-loop ramp the MEASURED aperture can rise anyway and
                # this read it back as an executed close, arming the
                # phantom-grasp recovery on a close the veto itself prevented
                # (revalidation 2026-08-31 §2 #3).
                if state.get("close_permitted"):
                    state["closed_idx"] = n
                    # a close allowed ONLY by the at_floor hatch carries no
                    # gate evidence, and on the 08-28 statistics (p_none>0.9 on
                    # 80% of replans) the recovery would reopen 5/18 real
                    # grasps: never fire it on a floor-only close.
                    state["closed_floor_only"] = state.get("allowed_floor_only", False)
            state["in_close"] = closed

    def _apply_veto(
        self, plan, tcp_pose: np.ndarray, grip_now: float, state: dict, n: int = 0
    ) -> dict | None:
        """Rewrite `plan.actions` in place per TerminalVeto. Returns the trace
        record (or None when the veto is off).

        Both rules only ever touch the gripper channel and the z channel, and
        both are expressed as a rewrite of the CHUNK, not of the executor: the
        executor keeps its single contract (play the chunk it was given), and
        the trace records exactly the chunk the arm was asked to follow."""
        v = self.veto
        if v is None:
            return None
        a = plan.actions
        z_now = float(tcp_pose[2])
        p_none = float(plan.p_evt[0]) if np.size(plan.p_evt) else 0.0
        p_contact = 1.0 - p_none
        rec = {"p_contact": round(p_contact, 4), "retries": state["retries"]}

        # running minimum of the MEASURED aperture (train/common.close_index's
        # rule), and the executed-close latch it feeds.
        state["g_min"] = (
            grip_now if state["g_min"] is None else min(state["g_min"], grip_now)
        )
        self._note_executed_close(state, n)

        # ---- (b) phantom-grasp recovery ------------------------------------
        # checked FIRST: it reacts to the close executed one replan ago, and
        # only inside that window — outside it the latch is dropped, so a real
        # grasp is never reopened by a late high-p_none frame (a release, a
        # transport frame where the gel loses the object, an egg held lightly).
        idx = state["closed_idx"]
        if idx is not None and (n - idx) > self.VETO_RECOVERY_REPLANS:
            state["closed_idx"] = idx = None
        if idx is not None and p_none > v.p_none and state.get("closed_floor_only"):
            # the only door open on the 08-28 data is the floor hatch; a
            # recovery keyed on p_none would self-cancel those closes (the
            # voters split on whether that is protection or an anti-grasp —
            # unresolvable without a successful rig grasp, so the guard takes
            # the tail risk off the table either way)
            state["closed_idx"] = None
            rec["action"] = "recovery_skipped_floor_close"
            log.info(
                "terminal veto: high p_none (%.2f) after a FLOOR-ONLY "
                "close — recovery suppressed, latch cleared",
                p_none,
            )
            return rec
        if idx is not None and p_none > v.p_none:
            # tactile veto over the model's opinion (verification 09-04): with
            # both pads loaded the object IS held — never open on p_none alone
            held = self._pad_loads(state, v.phantom_window)
            if len(held) >= 2 and all(f >= v.phantom_f for f in held.values()):
                state["closed_idx"] = None
                rec["action"] = "recovery_skipped_loaded"
                rec["pad_load"] = {k: round(f, 2) for k, f in held.items()}
                log.info(
                    "terminal veto: high p_none (%.2f) but both pads loaded "
                    "(%s) — recovery suppressed",
                    p_none,
                    rec["pad_load"],
                )
                return rec
            state["closed_idx"] = None
            state["retries"] += 1
            rec["retries"] = state["retries"]
            if state["retries"] > v.max_retries:
                rec["action"] = "retry_cap"
                return rec
            a[:, 6] = v.open_aperture
            # no lift. NB: REVIEW_SYNTHESIS P3 writes "clamp the next chunk to
            # z >= z_now (no lift)"; taken literally that clamp PERMITS exactly
            # the upward motion its own parenthesis forbids, so the intent —
            # never command a z above where we are, i.e. re-descend or hold —
            # is what is implemented.
            cum = np.cumsum(a[:, 2])
            cum = np.minimum(cum, 0.0)
            a[:, 2] = np.diff(np.concatenate([[0.0], cum]))
            self._invalidate_cpk(plan)
            if hasattr(self.executor, "clear_grip_latch"):
                self.executor.clear_grip_latch()
            rec["action"] = "recovery_open"
            log.warning(
                "terminal veto: phantom grasp (p_none=%.2f) — opening to "
                "%.2f and forbidding lift (retry %d/%d)",
                p_none,
                v.open_aperture,
                state["retries"],
                v.max_retries,
            )
            return rec

        # ---- (b2) tactile-grounded phantom / lost-object recovery -----------
        # z at the moment the MEASURED aperture crossed close_pos upward
        if grip_now > v.close_pos:
            if state.get("close_z") is None:
                state["close_z"] = z_now
        else:
            state["close_z"] = None
            state["phantom_since"] = None
        loads = self._pad_loads(state, v.phantom_window)
        empty = (
            state.get("close_z") is not None
            and z_now - state["close_z"] > v.phantom_dz
            and len(loads) >= 2
            and all(f < v.phantom_f for f in loads.values())
        )
        t_rep = self._now
        if empty:
            if state.get("phantom_since") is None:
                state["phantom_since"] = t_rep
        else:
            state["phantom_since"] = None
        if empty and (t_rep - state["phantom_since"]) >= v.phantom_t:
            rise_mm = (z_now - state["close_z"]) * 1000.0
            state["close_z"] = None
            state["phantom_since"] = None
            state["closed_idx"] = None
            state["retries"] += 1
            rec["retries"] = state["retries"]
            rec["pad_load"] = {k: round(f, 2) for k, f in loads.items()}
            if state["retries"] > v.max_retries:
                rec["action"] = "retry_cap"
                return rec
            a[:, 6] = v.open_aperture
            cum = np.minimum(np.cumsum(a[:, 2]), 0.0)
            a[:, 2] = np.diff(np.concatenate([[0.0], cum]))
            self._invalidate_cpk(plan)
            if hasattr(self.executor, "clear_grip_latch"):
                self.executor.clear_grip_latch()
            rec["action"] = "recovery_tactile"
            log.warning(
                "terminal veto: gripper closed and lifted %.0f mm with "
                "NO pad load (%s) — opening to %.2f and re-descending "
                "(retry %d/%d)",
                rise_mm,
                rec["pad_load"],
                v.open_aperture,
                state["retries"],
                v.max_retries,
            )
            return rec

        # ---- (a) close mask -------------------------------------------------
        g_max = float(np.max(a[:, 6]))
        g_min = state["g_min"] if state["g_min"] is not None else grip_now
        closing = g_max > v.close_pos and (g_max - g_min) > v.close_rise
        if not closing:
            rec["action"] = "none"
            return rec
        z_ref = v.z_ref if v.z_ref is not None else v.z_floor
        at_floor = z_ref is not None and z_now <= z_ref + v.z_margin
        if p_contact > v.p_close or at_floor:
            # the latch is armed later, by `_note_executed_close`, from the
            # gripper steps the executor actually ENTERED — a plan that is
            # accepted but never played is not the close the recovery reacts to.
            rec["action"] = "close_allowed"
            rec["at_floor"] = bool(at_floor)
            state["close_permitted"] = True
            state["allowed_floor_only"] = bool(at_floor and not (p_contact > v.p_close))
            return rec
        a[:, 6] = grip_now  # hold the current aperture
        self._invalidate_cpk(plan)
        state["close_permitted"] = False
        rec["action"] = "close_masked"
        log.warning(
            "terminal veto: close masked (p_contact=%.2f <= %.2f, "
            "z=%.0f mm) — holding aperture %.2f",
            p_contact,
            v.p_close,
            z_now * 1000,
            grip_now,
        )
        return rec

    def _pad_loads(self, state: dict, window_s: float) -> dict[str, float]:
        """Trailing-`window_s` max of the baseline-corrected |fz| per pad from
        the tactile rings; {} when no session/rings (tests, student stubs).
        The baseline is the first replan's reading (pads untouched at the
        start pose) — the left pad idles 0.7-2.2 N above zero."""
        session = self.session or getattr(self.snapshots, "session", None)
        rings = getattr(session, "rings", None)
        if not rings:
            return {}
        rate = float(getattr(self.hw.tactile, "rate_hz", 8.0) or 8.0)
        k = max(1, int(round(window_s * rate)))  # noqa: RUF046 - exact native conversion
        out = {}
        base = state.setdefault("pad_base", {})
        for s_ in self.hw.tactile.sensors:
            ring = rings.get(f"tactile_{s_.name}")
            if ring is None:
                continue
            try:
                ts_t, tac = ring.latest(k)
                w = tac.get("wrench") if hasattr(tac, "get") else None
            except Exception:  # noqa: BLE001, S112 - preserve native telemetry fallback
                continue
            if w is None or not len(ts_t):
                continue
            fz = np.asarray(w, dtype=np.float64).reshape(len(ts_t), -1)[:, 2]
            if s_.name not in base:
                sb = getattr(self.snapshots, "wrench_base", {}) or {}
                base[s_.name] = (
                    float(sb[s_.name][2]) if s_.name in sb else float(fz[-1])
                )
            out[s_.name] = float(np.max(np.abs(fz - base[s_.name])))
        return out

    @staticmethod
    def _invalidate_cpk(plan) -> None:
        """Drop the contact package of a chunk the veto rewrote.

        `plan.cpk` is the model's IMAGINED contact for the chunk it proposed,
        and the next replan feeds it back as `prev_cpk` (policy.py:244). After
        a rewrite it describes motion the arm was never asked to make, so it
        must not condition the next chunk (Codex, 2026-08-30).

        With a policy server the package lives SERVER-side, addressed by
        `_cpk_token` — clearing only `cpk` (always None on the remote path)
        would leave the token alive and the invalidation a silent no-op
        (verification 09-01)."""
        plan.cpk = None
        if hasattr(plan, "_cpk_token"):
            plan._cpk_token = None

    def _apply_veto_historical(
        self, plan, tcp_pose: np.ndarray, grip_now: float, state: dict, n: int = 0
    ) -> dict | None:
        """Rewrite `plan.actions` in place per TerminalVeto. Returns the trace
        record (or None when the veto is off).

        Both rules only ever touch the gripper channel and the z channel, and
        both are expressed as a rewrite of the CHUNK, not of the executor: the
        executor keeps its single contract (play the chunk it was given), and
        the trace records exactly the chunk the arm was asked to follow."""
        v = self.veto
        if v is None:
            return None
        a = plan.actions
        z_now = float(tcp_pose[2])
        p_none = float(plan.p_evt[0]) if np.size(plan.p_evt) else 0.0
        p_contact = 1.0 - p_none
        rec = {"p_contact": round(p_contact, 4), "retries": state["retries"]}

        # running minimum of the MEASURED aperture (train/common.close_index's
        # rule), and the executed-close latch it feeds.
        state["g_min"] = (
            grip_now if state["g_min"] is None else min(state["g_min"], grip_now)
        )
        self._note_executed_close(state, n)

        # ---- (b) phantom-grasp recovery ------------------------------------
        # checked FIRST: it reacts to the close executed one replan ago, and
        # only inside that window — outside it the latch is dropped, so a real
        # grasp is never reopened by a late high-p_none frame (a release, a
        # transport frame where the gel loses the object, an egg held lightly).
        idx = state["closed_idx"]
        if idx is not None and (n - idx) > self.VETO_RECOVERY_REPLANS:
            state["closed_idx"] = idx = None
        if idx is not None and p_none > v.p_none and state.get("closed_floor_only"):
            # the only door open on the 08-28 data is the floor hatch; a
            # recovery keyed on p_none would self-cancel those closes (the
            # voters split on whether that is protection or an anti-grasp —
            # unresolvable without a successful rig grasp, so the guard takes
            # the tail risk off the table either way)
            state["closed_idx"] = None
            rec["action"] = "recovery_skipped_floor_close"
            log.info(
                "terminal veto: high p_none (%.2f) after a FLOOR-ONLY "
                "close — recovery suppressed, latch cleared",
                p_none,
            )
            return rec
        if idx is not None and p_none > v.p_none:
            state["closed_idx"] = None
            state["retries"] += 1
            rec["retries"] = state["retries"]
            if state["retries"] > v.max_retries:
                rec["action"] = "retry_cap"
                return rec
            a[:, 6] = v.open_aperture
            # no lift. NB: REVIEW_SYNTHESIS P3 writes "clamp the next chunk to
            # z >= z_now (no lift)"; taken literally that clamp PERMITS exactly
            # the upward motion its own parenthesis forbids, so the intent —
            # never command a z above where we are, i.e. re-descend or hold —
            # is what is implemented.
            cum = np.cumsum(a[:, 2])
            cum = np.minimum(cum, 0.0)
            a[:, 2] = np.diff(np.concatenate([[0.0], cum]))
            self._invalidate_cpk(plan)
            rec["action"] = "recovery_open"
            log.warning(
                "terminal veto: phantom grasp (p_none=%.2f) — opening to "
                "%.2f and forbidding lift (retry %d/%d)",
                p_none,
                v.open_aperture,
                state["retries"],
                v.max_retries,
            )
            return rec

        # ---- (a) close mask -------------------------------------------------
        g_max = float(np.max(a[:, 6]))
        g_min = state["g_min"] if state["g_min"] is not None else grip_now
        closing = g_max > v.close_pos and (g_max - g_min) > v.close_rise
        if not closing:
            rec["action"] = "none"
            return rec
        z_ref = v.z_ref if v.z_ref is not None else v.z_floor
        at_floor = z_ref is not None and z_now <= z_ref + v.z_margin
        if p_contact > v.p_close or at_floor:
            # the latch is armed later, by `_note_executed_close`, from the
            # gripper steps the executor actually ENTERED — a plan that is
            # accepted but never played is not the close the recovery reacts to.
            rec["action"] = "close_allowed"
            rec["at_floor"] = bool(at_floor)
            state["close_permitted"] = True
            state["allowed_floor_only"] = bool(at_floor and not (p_contact > v.p_close))
            return rec
        a[:, 6] = grip_now  # hold the current aperture
        self._invalidate_cpk(plan)
        state["close_permitted"] = False
        rec["action"] = "close_masked"
        log.warning(
            "terminal veto: close masked (p_contact=%.2f <= %.2f, "
            "z=%.0f mm) — holding aperture %.2f",
            p_contact,
            v.p_close,
            z_now * 1000,
            grip_now,
        )
        return rec
