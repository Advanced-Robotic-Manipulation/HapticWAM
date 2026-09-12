"""Opt-in relative release permission from actual loaded motor history.

No object state or target synthesis. Permission and subsequent motor acceptance,
measured opening, fresh unload, and optional FINISH are separate events.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from phantom.deploy.unlatched_finish import UnlatchedFinishConfig, UnlatchedFinishObserver


def finite_number(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


@dataclass(frozen=True)
class RelativeReleaseConfig:
    tcp_min_m: tuple[float, float, float]
    tcp_max_m: tuple[float, float, float]
    command_delta_min: float = 12 / 255
    measured_delta_min: float = 6 / 255
    reclose_tolerance: float = 2 / 255
    activity_hold_s: float = .2
    permission_hold_s: float = .2
    history_timeout_s: float = 10.
    feedback_max_age_s: float = .25
    minimum_policy_samples: int = 2

    def __post_init__(self):
        validated = self.observer_config()
        object.__setattr__(self, "tcp_min_m", validated.tcp_min_m)
        object.__setattr__(self, "tcp_max_m", validated.tcp_max_m)
        if (type(self.minimum_policy_samples) is not int or self.minimum_policy_samples < 2
                or type(self.permission_hold_s) not in (float, int)
                or not np.isfinite(self.permission_hold_s) or self.permission_hold_s <= 0):
            raise ValueError("Relative permission needs a positive dwell and at least two distinct played policy samples")

    def observer_config(self):
        fields = {k: v for k, v in asdict(self).items() if k not in ("permission_hold_s", "minimum_policy_samples")}
        return UnlatchedFinishConfig(**fields)


def original_played_sample(plan, play_time, action_rate_hz, max_play_steps, t):
    """Stable original action identity at the index the executor actually plays."""
    from phantom.deploy.release_controller import original_policy_grip

    if not original_policy_grip(plan, play_time, action_rate_hz, max_play_steps):
        return None
    actions = np.asarray(plan.actions, dtype=float)
    original = np.asarray((getattr(plan, "diag", None) or {}).get("actions_pre_veto", actions), dtype=float)
    times = np.asarray(plan.action_times, dtype=float)
    if (actions.ndim != 2 or actions.shape[1] != 7 or original.shape != actions.shape
            or times.shape != (len(actions),) or not len(actions)
            or not np.isfinite(actions).all() or not np.isfinite(original).all()
            or not np.isfinite(times).all() or not np.isfinite([t, play_time, action_rate_hz, plan.t_created]).all()
            or play_time < 0 or action_rate_hz <= 0):
        return None
    cap = min(len(actions), max_play_steps) if max_play_steps else len(actions)
    index = int(np.clip(play_time * action_rate_hz, 0., cap-1e-6))
    if not np.isclose(actions[index, 6], original[index, 6], atol=1e-12, rtol=0):
        return None
    identity = {"t_created": float(plan.t_created), "action_times": times.tolist(), "actions": original.tolist()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"plan_id": digest, "plan_created_t": float(plan.t_created), "action_index": index,
        "original_grip": float(original[index, 6]), "played_at": float(t),
        "original_policy_eligible": True, "absolute_play_cap": int(cap)}


class LoadedPlateauObserver(UnlatchedFinishObserver):
    """The original close/open observer plus a causal loaded-command transition.

    A rigid grasp may accept a stronger motor target without another6/255
    measured close. Keep the originally proven measured-close reference while
    recording a higher actual accepted loaded plateau. Constant motor command
    never raises a reference or refreshes its history age through passive/noisy
    movement. This method never assigns opening, phase or finished state.
    """
    def reset(self, reason="reset"):
        super().reset(reason)
        self.reset_plateau(reason)

    def reset_plateau(self, reason="reset"):
        self.plateau_ack = None
        self.plateau_since = None
        self.plateau_last_samples = None
        self.plateau_samples = 0
        self.plateau_reason = reason

    def update(self, t, *, reference_advance=False, reference_reset_reason=None, native_latch=None,
               loaded_force_min_n=None, **observation):
        if reference_reset_reason is not None:
            self.reset_plateau(reference_reset_reason)
            return False
        if not reference_advance:
            return super().update(t, **observation)
        ref, c = self.reference, self.config
        ack, measured = observation["ack"], observation["measured"]
        loads, samples = observation["loads"], observation["samples"]
        required = {"arm", "gripper"} | {"tactile_"+str(name) for name in loads}
        valid = (self.phase == "grasp_activity" and self.opening is None and self.finished_at is None
            and ref is not None and observation["eligible"] and finite_number(t)
            and ack is not None and len(ack) == 4 and type(ack[0]) is int and ack[3] is True
            and np.isfinite([ack[1], ack[2]]).all() and ack[1] <= t and 0 <= ack[2] <= 1
            and ack[0] > ref["ack_generation"] and ack[1] > ref["accepted_t"]
            and ack[2] > ref["command"]+1e-12
            and native_latch is not None and np.isfinite(native_latch) and native_latch >= ack[2]-1e-12
            and loaded_force_min_n is not None and finite_number(loaded_force_min_n)
            and loaded_force_min_n > 0 and len(loads) >= 2
            and all(np.isfinite(v) and v > loaded_force_min_n for v in loads.values())
            and np.isfinite(measured) and ref["measured"]-c.reclose_tolerance <= measured <= 1
            and samples is not None and required.issubset(samples)
            and all(finite_number(samples[k]) and 0 <= t-samples[k] <= c.feedback_max_age_s
                    and samples[k] >= ack[1] for k in required))
        if not valid:
            self.reset_plateau("no_higher_actual_loaded_plateau")
            return False
        current = {k: float(samples[k]) for k in required}
        if self.plateau_ack != tuple(ack):
            self.reset_plateau("new_actual_ack_plateau")
            self.plateau_ack = tuple(ack)
        if self.plateau_last_samples is not None:
            if any(current[k] < self.plateau_last_samples[k] for k in required):
                self.reset_plateau("plateau_capture_regressed")
                return False
            if not all(current[k] > self.plateau_last_samples[k] for k in required):
                self.plateau_reason = "awaiting_distinct_plateau_feedback"
                return False
        self.plateau_last_samples = current
        self.plateau_samples += 1
        if self.plateau_since is None:
            # As with release permission, old sensor captures cannot backdate
            # the first actual controller observation of these conditions.
            self.plateau_since = float(t)
        supported = min(current.values())-self.plateau_since
        if supported < c.activity_hold_s-1e-9 or self.plateau_samples < 2:
            self.plateau_reason = "actual_loaded_plateau_dwell"
            return False
        # Derived observer transition from real ACK + sustained feedback. Keep
        # initial measured-close proof: no object-state inference or new delta.
        self.reference = {"command": float(ack[2]), "measured": ref["measured"],
            "t": min(current.values()), "ack_generation": ack[0], "accepted_t": float(ack[1])}
        self.last_samples = current
        self.last_ack = (ack[0], float(ack[1]))
        self.reason = "higher_actual_loaded_command_plateau"
        self.reset_plateau("higher_actual_loaded_command_plateau")
        return False


def observer_from_loaded_history(config, history, expected_reference):
    """Replay validated real feedback to hand off a reference without setting phases.

    No permission event is replayed as an ACK or measured opening. The original
    observer itself must derive loaded activity again from the actual history.
    """
    observer = LoadedPlateauObserver(config)
    for row in history:
        observer.update(**deepcopy(row), finish_permitted=False)
    if (observer.phase != "grasp_activity" or observer.opening is not None
            or observer.finished_at is not None or observer.reference != expected_reference):
        raise ValueError("Recorded loaded-reference history does not reproduce the required observer state")
    return observer


class RelativeReleaseGate:
    def __init__(self, config, *, loaded_force_min_n, unloaded_force_max_n, unloaded_hold_s):
        self.config = config
        self.loaded_force_min_n = float(loaded_force_min_n)
        self.unloaded_force_max_n = float(unloaded_force_max_n)
        self.unloaded_hold_s = float(unloaded_hold_s)
        if not np.isfinite([self.loaded_force_min_n, self.unloaded_force_max_n, self.unloaded_hold_s]).all() or min(
                self.loaded_force_min_n, self.unloaded_force_max_n, self.unloaded_hold_s) <= 0:
            raise ValueError("Existing positive latch and unload criteria are required")
        self.reset()

    def reset(self, reason="reset"):
        self.state = "tracking"
        self.reason = reason
        self.reference = self.reference_history = None
        self.reference_observer = None
        self.reference_history_sha256 = None
        self.tracker = LoadedPlateauObserver(self.config.observer_config())
        self.observer = None
        self.history = []
        self._history_hasher = hashlib.sha256()
        self.last_samples = None
        self.last_ack = None
        self.loaded_since = self.pending_since = None
        self.policy_samples = set()
        self.last_policy_sample = None
        self.permission_event = self.release_evidence = None
        self.restore_latch = None
        self.last_event = None
        self.events = []

    @property
    def suppress_latch(self):
        return self.state in ("permitted", "released", "finished")

    @property
    def finished(self):
        return self.state == "finished"

    def in_volume(self, tcp):
        return self.tracker.in_volume(tcp)

    def _event(self, t, name, **values):
        self.last_event = {"t": float(t), "event": name, **deepcopy(values)}
        self.events.append(self.last_event)

    def _reset_plateau(self, t, reason):
        if self.tracker.plateau_ack is not None:
            call = {"t": float(t), "reference_reset_reason": reason}
            self.tracker.update(**call, finish_permitted=False)
            self.history.append(call)
            self._history_hasher.update((json.dumps(finite_json(call), sort_keys=True, allow_nan=False)+"\n").encode())

    def _cancel(self, t, reason):
        if reason in ("invalid_or_stale_feedback", "capture_time_regressed", "invalid_or_regressed_actual_ack"):
            self._reset_plateau(t, reason)
        if self.state == "permitted":
            self._reset_plateau(t, "relative_permission_cancelled")
            self.restore_latch = self.reference["command"]
            self._event(t, "relative_permission_cancelled", reason=reason,
                        restored_previous_accepted_reference=self.restore_latch)
        self.state = "holding" if self.reference is not None else "tracking"
        self.reason = reason
        self.pending_since = None
        self.policy_samples.clear()
        self.observer = None

    def update(self, t, *, tcp, policy_grip, measured_grip, pad_loads, eligible,
               accepted_grip, accepted_grip_ack, feedback_times, native_latch,
               played_sample, permission_allowed, finish_enabled, finish_permitted,
               observer_deferred=False):
        c = self.config
        self.last_event = None
        self.restore_latch = None
        if self.finished:
            return True
        # A proven physical unload is nonterminal in observer-off. Leaving the
        # volume afterwards cannot resurrect the old loaded retention command.
        was_released = self.state == "released"
        if was_released:
            if (eligible and np.isfinite(policy_grip)
                    and isinstance(played_sample, dict) and played_sample.get("original_policy_eligible") is True
                    and policy_grip > self.reference["command"]-c.command_delta_min+c.reclose_tolerance):
                self.reset("original_policy_reclose_after_release")
                self._event(t, "relative_release_rearmed_by_policy_close")
                return False
            if not finish_enabled:
                return True
        if observer_deferred:
            self.reason = "awaiting_causal_executor_tick"
            return self.suppress_latch
        required = {"arm", "gripper"} | {"tactile_"+str(name) for name in pad_loads}
        valid = (finite_number(t)
            and len(pad_loads) >= 2 and len(required) >= 4
            and all(np.isfinite(v) and v >= 0 for v in pad_loads.values())
            and np.asarray(tcp).shape == (6,) and np.isfinite(tcp).all()
            and np.isfinite(measured_grip) and 0 <= measured_grip <= 1
            and feedback_times is not None and required.issubset(feedback_times)
            and all(finite_number(feedback_times[k])
                    and -1e-9 <= t-feedback_times[k] <= c.feedback_max_age_s for k in required))
        if not valid:
            if was_released:
                self.reason = "released_with_invalid_completion_feedback"
                return True
            self._cancel(t, "invalid_or_stale_feedback")
            return self.suppress_latch
        samples = {k: float(feedback_times[k]) for k in required}
        if self.last_samples is not None and any(samples[k] < self.last_samples[k] for k in samples):
            if was_released:
                self.reason = "released_with_regressed_completion_feedback"
                return True
            self._cancel(t, "capture_time_regressed")
            return self.suppress_latch
        distinct = self.last_samples is None or all(samples[k] > self.last_samples[k] for k in samples)
        if distinct:
            self.last_samples = samples
        sample_t = min(samples.values())
        ack = accepted_grip_ack
        ack_valid = (ack is not None and len(ack) == 4 and type(ack[0]) is int and ack[0] > 0
            and np.isfinite([ack[1], ack[2]]).all() and ack[1] <= t+1e-9 and 0 <= ack[2] <= 1
            and accepted_grip is not None and np.isfinite(accepted_grip)
            and np.isclose(accepted_grip, ack[2], atol=1e-12, rtol=0))
        if ack_valid and self.last_ack is not None:
            old = self.last_ack
            ack_valid = (ack[0] >= old[0] and ack[1] >= old[1]
                and (ack[0] != old[0] or tuple(ack) == tuple(old))
                and (ack[0] == old[0] or ack[1] > old[1]))
        if ack is not None and not ack_valid:
            if was_released:
                self.reason = "released_with_invalid_completion_ack"
                return True
            self._cancel(t, "invalid_or_regressed_actual_ack")
            return False
        if ack_valid:
            self.last_ack = tuple(ack)
        observation = dict(t=float(t), tcp=np.asarray(tcp).tolist(), measured=float(measured_grip),
            loads={k: float(v) for k, v in pad_loads.items()}, eligible=bool(eligible), ack=ack, samples=samples,
            unloaded_force_max_n=self.unloaded_force_max_n, unloaded_hold_s=self.unloaded_hold_s)
        if was_released:
            if self.observer is not None and self.observer.update(**observation, finish_permitted=bool(finish_enabled and finish_permitted)):
                self.state = "finished"
                self._event(t, "relative_release_finished", observer_evidence=self.observer.diagnostics(tcp), task_success_inferred=False)
            return True
        if self.reference is None:
            self.tracker.update(**observation, finish_permitted=False)
            # Duplicate/ineligible calls may reset the real observer; retain
            # every call, not only fresh-frame calls, for exact replay proof.
            self.history.append(deepcopy(observation))
            self._history_hasher.update((json.dumps(finite_json(observation), sort_keys=True, allow_nan=False)+"\n").encode())
            if len(self.history) > 20000:
                self.tracker.reset("loaded_history_buffer_limit")
                self.history.clear()
                self._history_hasher = hashlib.sha256()
                self.loaded_since = None
                self.reason = "awaiting_new_close_after_history_limit"
                return False
            loaded = (distinct and ack_valid and ack[3] is True and sample_t >= ack[1]-1e-9
                and native_latch is not None and all(f > self.loaded_force_min_n for f in pad_loads.values()))
            if not loaded and distinct:
                self.loaded_since = None
            elif loaded:
                if self.loaded_since is None:
                    self.loaded_since = sample_t
                if (sample_t-self.loaded_since >= c.activity_hold_s-1e-9
                        and self.tracker.phase == "grasp_activity" and self.tracker.reference is not None):
                    reference = deepcopy(self.tracker.reference)
                    # The live servo path copies only the already-causal small
                    # observer state. Full history replay is an offline audit.
                    self.reference, self.reference_history = reference, tuple(self.history)
                    self.history = []
                    self.reference_observer = deepcopy(self.tracker)
                    self.reference_history_sha256 = self._history_hasher.hexdigest()
                    self.state = "holding"
                    self._event(t, "relative_loaded_reference_observed", reference=reference,
                                capture_times=samples, loaded_force_min_n=self.loaded_force_min_n,
                                source_history_rows=len(self.reference_history), loaded_history_sha256=self.reference_history_sha256)
        if self.reference is None:
            self.reason = "awaiting_actual_loaded_close_reference"
            return False
        if t-self.reference["t"] > c.history_timeout_s:
            self._cancel(t, "loaded_reference_expired")
            # Do not turn an expired old grasp into new loaded history.
            self.reference = self.reference_history = None
            self.reference_observer = None
            self.tracker.reset("loaded_reference_expired")
            self.history.clear()
            self._history_hasher = hashlib.sha256()
            self.loaded_since = None
            return False
        if self.state == "holding":
            advance = dict(observation, reference_advance=True, native_latch=native_latch,
                loaded_force_min_n=self.loaded_force_min_n)
            self.tracker.update(**advance, finish_permitted=False)
            self.history.append(deepcopy(advance))
            self._history_hasher.update((json.dumps(finite_json(advance), sort_keys=True, allow_nan=False)+"\n").encode())
            if self.tracker.reference["command"] > self.reference["command"]+1e-12:
                prior = deepcopy(self.reference)
                self.reference = deepcopy(self.tracker.reference)
                self.reference_history = self.reference_history + tuple(self.history)
                self.history = []
                self.reference_observer = deepcopy(self.tracker)
                self.reference_history_sha256 = self._history_hasher.hexdigest()
                self.pending_since = None
                self.policy_samples.clear()
                self._event(t, "relative_loaded_reference_advanced", previous_reference=prior,
                    reference=self.reference, capture_times=samples, loaded_force_min_n=self.loaded_force_min_n,
                    source_history_rows=len(self.reference_history), loaded_history_sha256=self.reference_history_sha256,
                    measured_reference_preserved=True, reference_source="sustained_actual_acknowledged_loaded_plateau")
        sample_ok = (isinstance(played_sample, dict) and played_sample.get("original_policy_eligible") is True
            and isinstance(played_sample.get("plan_id"), str) and len(played_sample["plan_id"]) == 64
            and type(played_sample.get("action_index")) is int and played_sample["action_index"] >= 0
            and type(played_sample.get("absolute_play_cap")) is int
            and played_sample["action_index"] < played_sample["absolute_play_cap"]
            and finite_number(played_sample.get("plan_created_t")) and played_sample["plan_created_t"] <= t+1e-9
            and np.isclose(played_sample.get("played_at", np.nan), t, atol=1e-9, rtol=0)
            and np.isfinite(policy_grip) and 0 <= policy_grip <= 1
            and np.isclose(played_sample.get("original_grip", np.nan), policy_grip, atol=1e-12, rtol=0))
        if sample_ok and self.last_policy_sample is not None:
            previous = self.last_policy_sample
            sample_ok = (played_sample["plan_created_t"] >= previous["plan_created_t"]
                and (played_sample["plan_id"] != previous["plan_id"] or played_sample["action_index"] >= previous["action_index"]))
        if sample_ok:
            self.last_policy_sample = deepcopy(played_sample)
        opening = (permission_allowed and eligible and ack_valid and ack[3] is True
            and sample_ok and self.in_volume(tcp)
            and self.reference["command"]-policy_grip >= c.command_delta_min-1e-12)
        if not opening:
            self._cancel(t, "relative_policy_request_ineligible_or_outside_band")
            return False
        if self.state != "permitted":
            if native_latch is None:
                self._cancel(t, "native_latch_does_not_own_retention")
                return False
            self.policy_samples.add((played_sample["plan_id"], played_sample["action_index"]))
            if self.pending_since is None:
                self.pending_since = float(t)
            if distinct:
                if (sample_t-self.pending_since >= c.permission_hold_s-1e-9
                        and t-self.pending_since >= c.permission_hold_s-1e-9
                        and len(self.policy_samples) >= c.minimum_policy_samples):
                    self._reset_plateau(t, "reference_frozen_at_release_permission")
                    self.observer = deepcopy(self.reference_observer)
                    self.state = "permitted"
                    self._event(t, "relative_policy_release_permitted", reference=self.reference,
                        actual_ack_at_permission=list(ack),
                        original_played_sample=played_sample, capture_times=samples,
                        dwell_s=sample_t-self.pending_since, policy_dwell_s=t-self.pending_since,
                        feedback_support_s=sample_t-self.pending_since, permission_request_since=self.pending_since,
                        distinct_policy_samples=len(self.policy_samples), loaded_history_sha256=self.reference_history_sha256,
                        history_sha256_convention="newline_join_sorted_json_per_actual_tracker_call_v1",
                        current_volume=True, current_safety_permitted=True,
                        command_target_source="current_original_policy_only", acceptance_observed=False,
                        opening_observed=False, finish_observed=False)
                    self.permission_event = deepcopy(self.last_event)
            if self.state != "permitted":
                self.reason = "awaiting_relative_permission_dwell_and_distinct_policy_samples"
                return False
        # The observer's loaded reference predates permission. An ACK already
        # present when permission was granted cannot prove that this permission
        # caused a submitted opening. Sim reports actual I/O after command()
        # using the same timestamp; its higher generation proves ordering, and
        # only a later observation may consume that ACK. Native I/O stamps are
        # strictly later. Never rewrite an unchanged ACK's causal origin.
        permission_ack = self.permission_event["actual_ack_at_permission"]
        if not (ack[0] > permission_ack[0] and ack[1] >= self.permission_event["t"]
                and t > self.permission_event["t"]
                and (self.observer.opening is not None
                     or self.reference["command"]-ack[2] >= c.command_delta_min-1e-12)):
            self.reason = "awaiting_actual_lower_ack_after_permission"
            return True
        finished = self.observer.update(**observation,
            finish_permitted=bool(finish_enabled and finish_permitted))
        self.reason = self.observer.reason
        proved = (self.observer.opening is not None and self.observer.unloaded_since is not None
            and self.observer.last_samples is not None
            and min(self.observer.last_samples.values())-self.observer.unloaded_since >= self.unloaded_hold_s-1e-9
            and self.observer.phase in ("opening_observed", "finished"))
        if proved:
            self.release_evidence = deepcopy(self.observer.diagnostics(tcp))
            self.state = "finished" if finished else "released"
            self._event(t, "relative_release_finished" if finished else "relative_release_observed_nonterminal",
                        observer_evidence=self.release_evidence, task_success_inferred=False)
        elif self.observer.reference is None:
            self._cancel(t, "relative_completion_observer_invalidated")
        return self.suppress_latch

    def stop(self, t=None):
        if self.state == "released":
            self.reason = "safety_preempted_after_proven_unload"
            self.observer = None
            return
        self._cancel(0. if t is None else t, "safety_preempted")

    def diagnostics(self, tcp):
        return {"variant": "relative_policy_release_permission_v2_loaded_plateau", "state": self.state,
            "reason": self.reason, "in_relative_volume": self.in_volume(tcp),
            "reference": deepcopy(self.reference), "permission_since_s": self.pending_since,
            "permission_event": deepcopy(self.permission_event), "last_event": deepcopy(self.last_event),
            "suppress_latch": self.suppress_latch, "restore_latch": self.restore_latch,
            "observer": self.observer.diagnostics(tcp) if self.observer is not None else None,
            "release_evidence": deepcopy(self.release_evidence), "events": deepcopy(self.events)}
