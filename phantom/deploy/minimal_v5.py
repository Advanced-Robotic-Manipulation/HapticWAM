"""Opt-in native controller port of the minimal-v5 simulator profile.

Historical veto/request feedback plus the existing native release/FINISH bridge.
No sensor synthesis, hardware construction, inference or safety-limit changes.
Defaults remain in PlannerLoop; only explicit selection uses this mixin.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
PROFILE = "minimal_v5"
HISTORICAL_VETO_SOURCE_SHA256 = (
    "7c71eea057710ca1ea3331e9b52934584c56709bef5427d1f1f37eb08ecc22b7"
)
VETO_FIELDS = (
    "p_close",
    "p_none",
    "max_retries",
    "z_floor",
    "z_ref",
    "z_margin",
    "open_aperture",
    "close_pos",
    "close_rise",
)


def validate_profile(profile, *, release_config, veto, mode, hw):
    """Validate explicit enablement before any driver/model is constructed."""
    if profile is None:
        return None
    if profile != PROFILE:
        raise ValueError(f"Unknown placement controller profile: {profile}")
    if mode not in ("teacher", "student", "vision_only"):
        raise ValueError("minimal_v5 controller port supports teacher, student and vision_only modes")
    if veto is None:
        raise ValueError("minimal_v5 requires --terminal-veto")
    if release_config is None or not release_config.finish_after_release:
        raise ValueError(
            "minimal_v5 requires explicit release bounds and finish_after_release=true"
        )
    parameters = {name: getattr(veto, name) for name in VETO_FIELDS}
    if any(value is None or not np.isfinite(value) for value in parameters.values()):
        raise ValueError(
            "minimal_v5 requires finite explicit task veto references/thresholds"
        )
    if not (0 <= veto.p_close <= 1 and 0 <= veto.p_none <= 1):
        raise ValueError("minimal_v5 veto probabilities must lie in 0..1")
    max_close_cmd = float(hw.gripper.max_close_cmd)
    if not np.isfinite(max_close_cmd) or not (0 <= veto.open_aperture <= max_close_cmd):
        raise ValueError(
            "minimal_v5 open_aperture must lie in 0..hw.gripper.max_close_cmd"
        )
    if (
        veto.max_retries < 0
        or int(veto.max_retries) != veto.max_retries
        or min(veto.z_margin, veto.close_rise) < 0
    ):
        raise ValueError("minimal_v5 veto retries/margins must be nonnegative")
    return {
        "id": PROFILE,
        "veto_implementation": "fd4a032",
        "veto_feedback_source": "request_snapshot_historical",
        "historical_veto_source_sha256": HISTORICAL_VETO_SOURCE_SHA256,
        "native_port_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "effective_veto": parameters,
        "effective_release": release_config.to_dict(),
        "sensor_inputs": "Unchanged native live wrist/tactile/RGB streams and checkpoint preprocessing; no simulator proxy",
        "qualification": "Controller-port CPU parity only; physical timing, force transfer and geometry remain unqualified",
        "native_timing_differences": "Actual gripper mailbox acceptance and stale-feedback checks may delay FINISH; post-FINISH observation uses explicit finish_observation_s, not simulator 60-second horizon",
    }


def planner_class(profile, default):
    """Default class identity is retained; no hidden global monkeypatch."""
    if profile is None:
        return default
    if profile != PROFILE:
        raise ValueError(f"Unknown placement controller profile: {profile}")

    class MinimalV5PlannerLoop(HistoricalV5VetoMixin, default):
        pass

    return MinimalV5PlannerLoop


class HistoricalV5VetoMixin:
    """Historical decisions copied verbatim from the hash-audited v1 bridge."""

    request_snapshot_veto = True

    def _apply_veto(self, plan, tcp_pose, grip_now, state, n=0):
        actions = np.asarray(plan.actions)
        pose = np.asarray(tcp_pose)
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or not len(actions)
            or not np.isfinite(actions).all()
            or not np.isfinite(plan.p_evt).all()
            or pose.shape != (6,)
            or not np.isfinite(pose).all()
            or not np.isfinite(grip_now)
        ):
            raise ValueError(
                "minimal_v5 requires finite native plan and request feedback"
            )
        rec = self._apply_veto_historical(plan, tcp_pose, grip_now, state, n)
        if rec is not None:
            rec.update(
                controller_profile=PROFILE,
                implementation="fd4a032",
                feedback_source="request_snapshot_historical",
                feedback_tcp_pose=np.asarray(tcp_pose, dtype=float).tolist(),
                feedback_gripper=float(grip_now),
                native_source_sha256=HISTORICAL_VETO_SOURCE_SHA256,
                cpk_invalidated=rec.get("action") in ("close_masked", "recovery_open"),
            )
        return rec

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
            # no lift. NB: finding P3 writes "clamp the next chunk to
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
