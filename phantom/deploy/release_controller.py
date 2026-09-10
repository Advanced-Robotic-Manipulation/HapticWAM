"""Explicit placement release variant; observes robot feedback, never object state.

The native rig aperture latch is a running maximum and cannot honor a policy
release. This opt-in gate permits a sustained policy opening inside a declared
TCP release volume after a loaded grasp. It does not generate motion or choose
an aperture, and it does not identify task success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from phantom.deploy.unlatched_finish import UnlatchedFinishConfig, UnlatchedFinishObserver


@dataclass(frozen=True)
class PlacementReleaseConfig:
    tcp_min_m: tuple[float, float, float]
    tcp_max_m: tuple[float, float, float]
    open_command_max: float = 0.45
    opening_hold_s: float = 0.2
    unloaded_force_max_n: float = 0.5
    unloaded_hold_s: float = 0.2
    rearm_close_command_min: float = 0.5
    finish_after_release: bool = False
    finish_observation_s: float = 2.0
    unlatched_finish: UnlatchedFinishConfig | None = None

    def __post_init__(self):
        for name in ("tcp_min_m", "tcp_max_m"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite base-frame metres")
            object.__setattr__(self, name, tuple(float(v) for v in value))
        if not np.all(np.asarray(self.tcp_min_m) < np.asarray(self.tcp_max_m)):
            raise ValueError("TCP release bounds must have positive volume")
        if not 0 <= self.open_command_max < self.rearm_close_command_min <= 1:
            raise ValueError(
                "release/rearm closure thresholds must satisfy 0 <= open < close <= 1"
            )
        for name in ("opening_hold_s", "unloaded_hold_s", "unloaded_force_max_n"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.finish_after_release, bool):
            raise TypeError("finish_after_release must be a boolean")
        if not np.isfinite(self.finish_observation_s) or self.finish_observation_s <= 0:
            raise ValueError("finish_observation_s must be finite and positive")
        if isinstance(self.unlatched_finish, dict):
            object.__setattr__(self, "unlatched_finish", UnlatchedFinishConfig(**self.unlatched_finish))
        if self.unlatched_finish is not None:
            if not isinstance(self.unlatched_finish, UnlatchedFinishConfig):
                raise TypeError("unlatched_finish must be a configuration object or dict")
            if not self.finish_after_release:
                raise ValueError("unlatched_finish requires finish_after_release")

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        values = asdict(self)
        if self.unlatched_finish is None:
            values.pop("unlatched_finish")  # preserve legacy serialized config
        return values


class PlacementReleaseController:
    """Latch permission only; every released aperture still comes from policy.

    The gate starts cold and becomes eligible only after the native load latch was
    actually armed. Once committed, loaded pads cannot immediately re-arm it. Both
    pads must first become unloaded and the measured gripper open for a dwell, then
    a subsequent policy closing command permits a new load latch.
    """

    def __init__(self, config: PlacementReleaseConfig):
        self.config = config
        self.unlatched_observer = (
            UnlatchedFinishObserver(config.unlatched_finish) if config.unlatched_finish else None
        )
        self.reset()

    def reset(self):
        self.phase = "unarmed"
        self.opening_since = None
        self.unloaded_since = None
        self.committed_at = None
        self.finished_at = None
        self.last_event = None
        if self.unlatched_observer is not None:
            self.unlatched_observer.reset()

    @property
    def variant(self):
        if self.unlatched_observer is not None:
            return "placement_policy_release_finish_v3_observe_unlatched"
        return (
            "placement_policy_release_finish_v2"
            if self.config.finish_after_release
            else "placement_policy_release_v1"
        )

    @property
    def finished(self):
        return self.phase == "finished"

    def note_latch(self, latch):
        if latch is not None and self.phase == "unarmed":
            self.phase = "holding"
            if self.unlatched_observer is not None:
                self.unlatched_observer.reset("native_latch_owns_release")

    def in_volume(self, tcp):
        xyz = np.asarray(tcp, dtype=float)[:3]
        return bool(
            xyz.shape == (3,)
            and np.isfinite(xyz).all()
            and np.all(xyz >= self.config.tcp_min_m)
            and np.all(xyz <= self.config.tcp_max_m)
        )

    def window_active(self, tcp):
        return self.phase in (
            "holding",
            "releasing",
            "waiting_for_close",
        ) and self.in_volume(tcp)

    @property
    def suppress_latch(self):
        return self.phase in ("releasing", "waiting_for_close", "finished")

    def update(
        self,
        t,
        *,
        tcp,
        policy_grip,
        measured_grip,
        pad_loads,
        eligible,
        accepted_grip=None,
        finish_permitted=True,
        accepted_grip_ack=None,
        feedback_times=None,
        observer_deferred=False,
    ):
        self.last_event = None
        c = self.config
        if self.phase == "unarmed" and self.unlatched_observer is not None:
            if observer_deferred and eligible:
                self.unlatched_observer.reason = "awaiting_causal_executor_tick"
                return self.suppress_latch
            finished = self.unlatched_observer.update(
                t, tcp=tcp, measured=measured_grip, loads=pad_loads,
                eligible=eligible, ack=accepted_grip_ack, samples=feedback_times,
                unloaded_force_max_n=c.unloaded_force_max_n,
                unloaded_hold_s=c.unloaded_hold_s,
                finish_permitted=(finish_permitted and accepted_grip is not None
                                  and accepted_grip_ack is not None
                                  and np.isfinite(accepted_grip)
                                  and np.isclose(accepted_grip, accepted_grip_ack[2], atol=1e-12, rtol=0)),
            )
            if finished:
                self.phase = "finished"
                self.finished_at = float(t)
                self.last_event = "unlatched_release_finished"
        if self.phase == "holding":
            opening = (
                eligible
                and self.window_active(tcp)
                and np.isfinite(policy_grip)
                and policy_grip <= c.open_command_max
            )
            if not opening:
                self.opening_since = None
            elif self.opening_since is None:
                self.opening_since = float(t)
            elif t - self.opening_since >= c.opening_hold_s - 1e-9:
                self.phase = "releasing"
                self.committed_at = float(t)
                self.last_event = "policy_release_committed"
        elif self.phase == "releasing":
            unloaded_open = (
                len(pad_loads) >= 2
                and all(
                    np.isfinite(f) and 0 <= f < c.unloaded_force_max_n
                    for f in pad_loads.values()
                )
                and np.isfinite(measured_grip)
                and measured_grip <= c.open_command_max
            )
            if not unloaded_open:
                self.unloaded_since = None
            elif self.unloaded_since is None:
                self.unloaded_since = float(t)
            elif t - self.unloaded_since >= c.unloaded_hold_s - 1e-9:
                if not c.finish_after_release:
                    self.phase = "waiting_for_close"
                    self.last_event = "measured_unloaded_open"
                elif (
                    finish_permitted
                    and accepted_grip is not None
                    and np.isfinite(accepted_grip)
                    and 0 <= accepted_grip <= c.open_command_max
                ):
                    self.phase = "finished"
                    self.finished_at = float(t)
                    self.last_event = "placement_release_finished"
        elif self.phase == "waiting_for_close":
            if eligible and policy_grip >= c.rearm_close_command_min:
                self.phase = "unarmed"
                self.opening_since = self.unloaded_since = None
                self.last_event = "policy_reclose_permitted"
        return self.suppress_latch

    def stop(self):
        self.opening_since = None
        self.last_event = "safety_preempted"
        if self.unlatched_observer is not None and not self.finished:
            self.unlatched_observer.reset("safety_preempted")

    def diagnostics(self, tcp):
        values = {
            "variant": self.variant,
            "phase": self.phase,
            "window_active": self.window_active(tcp),
            "opening_since_s": self.opening_since,
            "committed_at_s": self.committed_at,
            "finished_at_s": self.finished_at,
            "controller_finished": self.finished,
            "task_success": "not inferred by this controller",
            "unloaded_since_s": self.unloaded_since,
            "suppress_latch": self.suppress_latch,
            "event": self.last_event,
        }
        if self.unlatched_observer is not None:
            values["unlatched_observer"] = self.unlatched_observer.diagnostics(tcp)
        return values


def make_release_controller(config, hw):
    """Shared native/simulator validation; no device or model construction."""
    if config is None:
        return None
    if isinstance(config, dict):
        config = PlacementReleaseConfig.from_dict(config)
    if not isinstance(config, PlacementReleaseConfig):
        raise TypeError("release_config must be a PlacementReleaseConfig or dict")
    if hw.safety.grip_latch_fz_n <= 0:
        raise ValueError("placement release requires the native load latch enabled")
    if config.rearm_close_command_min > hw.gripper.max_close_cmd:
        raise ValueError("release rearm threshold exceeds hardware closure limit")
    return PlacementReleaseController(config)


def original_policy_grip(plan, play_time, action_rate_hz, max_play_steps):
    """Exclude recovery/masked samples from the policy-opening commitment."""
    if plan is None:
        return False
    veto = (getattr(plan, "diag", None) or {}).get("terminal_veto", {})
    action = veto.get("action")
    if action in ("recovery_open", "recovery_tactile", "retry_cap"):
        return False
    if action == "close_masked":
        cap = len(plan.actions)
        if max_play_steps is not None:
            cap = min(cap, max_play_steps)
        index = int(np.clip(play_time * action_rate_hz, 0, cap - 1e-6))
        return index in veto.get("placement_release_passthrough_indices", [])
    return True


def restore_policy_openings(plan, original_actions, veto_record, executor):
    """Only restore original opening rows; native arm/recovery guards remain."""
    mask_fn = getattr(executor, "placement_release_opening_mask", None)
    if (
        not veto_record
        or veto_record.get("action") != "close_masked"
        or not callable(mask_fn)
    ):
        return
    permitted = np.asarray(mask_fn(original_actions[:, 6]), dtype=bool)
    if permitted.shape != (len(original_actions),):
        raise ValueError("placement release opening mask has wrong shape")
    if permitted.any():
        plan.actions[permitted, 6] = original_actions[permitted, 6]
        veto_record["placement_release_passthrough_indices"] = np.flatnonzero(
            permitted
        ).tolist()
        controller = getattr(executor, "release_controller", None)
        veto_record["placement_release_variant"] = (
            controller.variant
            if controller is not None
            else "placement_policy_release_v1"
        )
        plan.cpk = None
        if hasattr(plan, "_cpk_token"):
            plan._cpk_token = None
