"""Tactile grasp-success labeling (finding P8).

WHY THIS EXISTS
---------------
The Robotiq `gOBJ` status is **not** a grasp-success signal on this rig.
`phantom/drivers/base.py:70-73` documents the real codes (0 moving, 1 contact
while opening, 2 contact while closing, 3 at requested position); with
position-streamed teleop the fingers reach the commanded aperture on soft
objects and the gripper reports 3, so `obj==2` fires on only 1.6-22.8% of
*successful* Carton/egg/waffles demos while firing 85-100% on the over-squeeze
`*_fail` demos. An "OBJ after close" auto-label would mark ~690/1070 genuine
successes as failures. `obj==2` is therefore an over-squeeze / stall FLAG here
and is **never** part of `grasp_ok` (regression-tested).

The label we use instead is the paper's own point: mechanical object-detect
fails on soft objects, the visuotactile gel does not. Grasp success is
tactile contact SUSTAINED THROUGH THE LIFT.

THE RULE (P8, verbatim thresholds)
----------------------------------
    t_close  = ts_grip[close_index(pos)]           # train/common.py close_index
    contact(t) = max_f mask_frac_f(t) > tau_contact_area
    hold     = [t_close + 0.5 s, t_release)
    c_hold   = mean contact over hold
    grasp_ok = z_close <= Z_MAX[task] and hold >= 2 s
               and c_hold >= 0.8 and lift >= 50 mm while in contact
    stall    = mean(obj == 2 over hold) > 0.5      # FLAG only
    task_ok  = operator verdict                     # place/wipe stays human

TWO CASES THE RULE USED TO GET WRONG (validation 2026-08-30)
------------------------------------------------------------
1. **The hold tail is not guaranteed.** `DeploymentRuntime.run_episode` stops
   the recorder immediately after `planner.run()` returns (runtime.py:241-246),
   so a grasp on one of the last replans — or an episode ended by
   `veto_retry_cap` — has NO 2.5 s tail to measure. `hold 0.4s < 2.0s` there is
   TRUNCATION, not failure, and counting it as a negative biases G2/G3 (a
   16-episode decision). Such an episode is now flagged `hold_truncated` and is
   reported in its own `trunc_*` cell of the confusion table: inconclusive,
   never a negative. (`grasp_ok` still requires a measured hold — an
   unverifiable grasp is not a success either.)
2. **A retry after the terminal veto is a real grasp.** `--terminal-veto`
   forces the gripper open on a suspected phantom grasp and lets the policy
   close again; scoring the FIRST close made every veto retry-success a
   failure — the exact number G2 is decided on. The rule now walks every close
   attempt (close, reopen, close, ...) and is evaluated on the LAST one, with
   `n_close_attempts` / `t_first_close_s` kept for provenance.

DUAL LABEL, NOT A REPLACEMENT
-----------------------------
The rule has **no validated positive class on a policy-driven grasp** — it was
validated on teleop demos and on known-failure rollouts, and there are zero
successful rollouts to date. Every label therefore carries the operator verdict
(`meta.success`, `meta.notes`) alongside `grasp_ok`; report the confusion matrix
(`tools/label_grasps.py --confusion`) and do not freeze the thresholds before
>= 10 human-confirmed policy grasps exist.

VALIDATION (compute3, 2026-08-29, defaults below, contact grid 10 Hz)
--------------------------------------------------------------------
Demos (operator `s`): 87/94 = 92.6% positive — Carton 20/20, whiteboard 21/21,
egg 16/17, waffles 30/36. P8 expected >= 95%; the gap is entirely `c_hold`
(7 episodes at 0.02/0.22/0.42/0.44/0.56/0.64/0.72, six of them waffles, where
the gel loses the soft waffle intermittently mid-lift). `c_hold >= 0.5` would
give 95.7% and `>= 0.4` 97.9% with NO change in the rollout false positives —
recorded here, NOT applied: the thresholds stay as P8 specified them until a
human re-checks those seven videos.
Rollouts: 0/26 on 20260828 and 1/11 on 20260820 = 1/37 = 2.7% false positive.
The one positive (`ep_teacher_whiteboard_1787248612`, operator `f`) measures
c_hold 1.00 and a 208 mm in-contact lift, which contradicts its own auto-relabel
note "outcome=grasp-contact-no-lift" — it needs a human video pass before it is
called either way. Z_MAX headroom over the demo maxima is thin for Carton
(143 vs 144 mm) and waffles (97 vs 103 mm), generous for egg (78/101) and
whiteboard (163/181): re-derive the table after any table or TCP-offset change.

STREAMS USED (and which carries what)
-------------------------------------
`gripper.zarr`            (T, 2) = [position 0..1 (1 = closed), gOBJ 0..3]
                          -> t_close (close_index), release, stall flag.
`arm_tcp_pose.zarr`       (T, 6) = [x, y, z, rx, ry, rz] in METRES
                          -> z_close_mm, lift_mm (x1000).
`tactile_<s>_fields_ds.zarr`  (T, h, w, 8) canonical field stack
                          [disp_x, disp_y, depth, shear_x, shear_y, fx, fy, fz]
                          downsampled to recording.field_ds, recorded at
                          recording.field_ds_rate_hz. This is THE contact
                          stream: `derived.derive_timestep()` -> `mask_frac`
                          (fraction of cells with |depth| > tau_contact_depth,
                          or |fz| > tau_contact_fz when force_calibrated),
                          exactly as `phantom/data/windows.py:296-331` does for
                          training. `prev=None` is passed because only
                          `mask_frac` is read (slip/CoP are unused here and
                          `mask_frac` does not depend on `prev`).
`tactile_<s>_area.zarr`   (T,) = the VENDOR SDK `getContactArea()` in mm^2.
                          Deliberately NOT used: it is an undocumented SDK
                          output, is not what training thresholds, and
                          `derived.area_crosscheck_tol` exists precisely because
                          it only agrees with our own mask to ~30%.
Contact is evaluated on a uniform `contact_rate_hz` grid (default 10 Hz) over
the hold window, nearest `fields_ds` frame per sensor. Training evaluates the
same per-frame predicate on the latent grid (`temporal_comp/fps` = 1 Hz), which
is too coarse to resolve a 2 s hold; the per-frame definition is identical.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader
from phantom.data.schema import (STREAM_ARM_TCP_POSE, STREAM_GRIPPER,
                                 tactile_stream)
from phantom.train.common import close_index

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# thresholds (finding P8) — every one of these is overridable
# ---------------------------------------------------------------------------

#: Per-task ceiling on the TCP z at the moment of close, millimetres.
#: PROVENANCE: the p95 of z_close over the successful teleop demos of that task
#: plus a 15 mm margin, measured 2026-08-28 over all 1115 v5 episodes
#: (finding P8). It is a per-task *table height* proxy, so it is
#: only valid for the rig geometry those demos were recorded on: re-derive it
#: (tools/rig_trace_decompose.py demos <task_dir>) after any table/TCP-offset
#: change. Keys are the raw task names; a `*_fail` suffix is stripped.
Z_MAX_MM: dict[str, float] = {
    "waffles": 103.0,
    "Carton": 144.0,
    "egg": 101.0,
    "whiteboard": 181.0,
}

HOLD_START_S = 0.5        # hold window opens this long after the close command
MIN_HOLD_S = 2.0          # hold must last at least this long
MIN_C_HOLD = 0.8          # mean tactile contact fraction over the hold
MIN_LIFT_MM = 50.0        # z gain while in contact after the close
STALL_FRAC = 0.5          # mean(obj == 2) over hold above this => over-squeeze
RELEASE_DROP = 0.10       # gripper position drop below the plateau = release
CONTACT_RATE_HZ = 10.0    # sampling grid for contact / z over the hold window


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------

@dataclass
class GraspLabel:
    """One episode's dual label: the tactile rule AND the operator verdict."""

    episode: str
    task: str
    # --- the rule ---
    t_close: float | None          # s since episode start; None = never closed
    z_close_mm: float | None       # TCP z at close
    hold_s: float                  # length of [t_close+0.5, t_release)
    c_hold: float                  # mean tactile contact fraction over hold
    lift_mm: float                 # max z gain over z_close while in contact
    stall: bool                    # mean(obj==2) over hold > 0.5 — FLAG ONLY
    grasp_ok: bool
    reasons: list[str] = field(default_factory=list)
    # the recording ENDED inside the hold window (no reopen was seen and the
    # hold is short) — see the module docstring
    hold_truncated: bool = False
    # ... and nothing ELSE disqualifies the grasp, so the rule has no verdict:
    # the confusion table counts these apart instead of as negatives. A
    # truncated episode that also closed above Z_MAX is still a plain failure.
    inconclusive: bool = False
    # close attempts in this episode; > 1 means the gripper reopened and closed
    # again (the --terminal-veto retry). The rule is evaluated on the LAST one.
    n_close_attempts: int = 1
    t_first_close_s: float | None = None
    # --- the human ---
    operator_success: bool | None = None   # meta.success ('s'/'f'/skipped)
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    policy: str = ""
    # --- provenance ---
    z_max_mm: float | None = None
    obj2_frac_hold: float = 0.0
    n_contact_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nearest(ts: np.ndarray, t: float) -> int:
    """Index of the sample nearest `t` (same convention as WindowSampler)."""
    i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
    if i > 0 and abs(ts[i - 1] - t) <= abs(ts[i] - t):
        i -= 1
    return i


def _release_after(pos: np.ndarray, g_ts: np.ndarray, ci: int, t_end: float,
                   hold_start_s: float = HOLD_START_S
                   ) -> tuple[float, int | None]:
    """(t_release, release index) for the close at `ci`.

    Release = the gripper position falling `RELEASE_DROP` below the plateau it
    held after the close. Index None means it never reopened before the
    recording ended, and `t_release` is then the end of the recording — the
    truncation case the caller has to distinguish from a real release."""
    t_close = float(g_ts[ci])
    plateau_sel = (g_ts >= t_close) & (g_ts <= t_close + 2.0)
    plateau = float(pos[plateau_sel].max()) if plateau_sel.any() else float(pos[ci])
    after = np.nonzero((g_ts > t_close + hold_start_s)
                       & (pos < plateau - RELEASE_DROP))[0]
    if len(after):
        return float(g_ts[after[0]]), int(after[0])
    return float(t_end), None


def close_attempts(pos: np.ndarray, g_ts: np.ndarray, t_end: float,
                   hold_start_s: float = HOLD_START_S) -> list[tuple[int, float, bool]]:
    """Every (close index, t_release, released) in the episode, in order.

    A `--terminal-veto` episode closes, is forced open, and closes again; the
    P8 rule is about whether the object is HELD, so it is the LAST attempt that
    decides (P8 auto-label bug, validation 2026-08-30). Each attempt is found
    by re-running `close_index` on the samples after the previous release, so
    the close rule (and its Carton max-relative fallback) is unchanged."""
    out: list[tuple[int, float, bool]] = []
    start = 0
    while start < len(pos):
        ci = close_index(pos[start:])
        if ci is None:
            break
        ci += start
        t_release, ri = _release_after(pos, g_ts, ci, t_end, hold_start_s)
        out.append((ci, t_release, ri is not None))
        if ri is None or ri <= ci:
            break
        start = ri + 1
    return out


def _z_max_for(task: str, table: dict[str, float]) -> float | None:
    return table.get(task, table.get(task.removesuffix("_fail")))


def _tactile_sensors(reader: EpisodeReader, hw) -> list[str]:
    return [s.name for s in hw.tactile.sensors
            if reader.has(tactile_stream(s.name, "fields_ds"))]


def contact_series(reader: EpisodeReader, hw, times: np.ndarray,
                   sensors: list[str]) -> np.ndarray:
    """(len(times),) bool — the TRAINING frame-level contact predicate.

    `contact(t) = max_f mask_frac_f(t) > tau_contact_area`, with `mask_frac`
    from `derived.derive_timestep` on the nearest `fields_ds` frame — the same
    call and the same threshold as `phantom/data/windows.py:296-331`.
    """
    if len(times) == 0 or not sensors:
        return np.zeros(len(times), dtype=bool)
    tau = hw.derived.tau_contact_area
    out = np.zeros(len(times), dtype=bool)
    for sname in sensors:
        stream = tactile_stream(sname, "fields_ds")
        ts = reader.ts(stream)
        data = reader.data(stream)
        for k, t in enumerate(times):
            if out[k]:
                continue          # already in contact on another finger
            frame = np.asarray(data[_nearest(ts, float(t))], dtype=np.float32)
            mf = dv.derive_timestep(frame, None, 1.0, hw)["mask_frac"]
            out[k] = mf > tau
    return out


# ---------------------------------------------------------------------------
# the labeler
# ---------------------------------------------------------------------------

def label_episode(ep_dir: str | Path, hw, task: str | None = None, *,
                  z_max_table: dict[str, float] | None = None,
                  contact_rate_hz: float = CONTACT_RATE_HZ,
                  min_hold_s: float = MIN_HOLD_S,
                  min_c_hold: float = MIN_C_HOLD,
                  min_lift_mm: float = MIN_LIFT_MM) -> GraspLabel:
    """Label one recorded episode with the P8 tactile grasp rule.

    `task` defaults to `meta.task`. Returns a GraspLabel carrying both the rule
    verdict (`grasp_ok` + `reasons`) and the operator verdict
    (`operator_success`, `notes`) so the two can be crosstabbed — the rule has
    never been validated against a true positive produced by the policy.
    """
    ep = Path(ep_dir)
    reader = EpisodeReader(ep)
    meta = reader.meta
    task = task or meta.task or ""
    table = Z_MAX_MM if z_max_table is None else z_max_table
    z_max = _z_max_for(task, table)

    lab = GraspLabel(
        episode=ep.name, task=task, t_close=None, z_close_mm=None,
        hold_s=0.0, c_hold=0.0, lift_mm=0.0, stall=False, grasp_ok=False,
        operator_success=meta.success, notes=meta.notes,
        tags=list(meta.tags or []), policy=meta.policy or "",
        z_max_mm=z_max)

    for stream in (STREAM_GRIPPER, STREAM_ARM_TCP_POSE):
        if not reader.has(stream):
            lab.reasons.append(f"missing_stream:{stream}")
            return lab

    try:
        # a truncated recording can leave <stream>.zarr with chunk dirs but no
        # .zarray metadata (seen on incoming_recovery/ep_waffles_1787394038_009,
        # which is also missing arm_tcp_pose entirely) — that is a QC finding,
        # not a crash.
        grip = np.asarray(reader.data(STREAM_GRIPPER)[:], dtype=np.float64)
        g_ts = reader.ts(STREAM_GRIPPER)
        tcp = np.asarray(reader.data(STREAM_ARM_TCP_POSE)[:], dtype=np.float64)
        t_ts = reader.ts(STREAM_ARM_TCP_POSE)
    except Exception as e:                                 # noqa: BLE001
        lab.reasons.append(f"unreadable_stream:{type(e).__name__}")
        return lab
    if len(g_ts) == 0 or len(t_ts) == 0:
        lab.reasons.append("empty_streams")
        return lab

    pos, obj = grip[:, 0], (grip[:, 1] if grip.shape[1] > 1 else np.zeros(len(grip)))
    z_mm = tcp[:, 2] * 1000.0
    t0 = float(min(g_ts[0], t_ts[0]))

    # ---- close ---------------------------------------------------------
    # every attempt, then the LAST one: after a --terminal-veto reopen the
    # grasp that counts is the one the gripper ended up holding
    t_end = float(min(g_ts[-1], t_ts[-1]))
    attempts = close_attempts(pos, g_ts, t_end)
    if not attempts:
        lab.reasons.append("never_closed")
        return lab
    lab.n_close_attempts = len(attempts)
    lab.t_first_close_s = round(float(g_ts[attempts[0][0]]) - t0, 3)
    ci, t_release, released = attempts[-1]
    t_close = float(g_ts[ci])
    lab.t_close = round(t_close - t0, 3)
    z_close = float(z_mm[_nearest(t_ts, t_close)])
    lab.z_close_mm = round(z_close, 1)

    # ---- hold window: [t_close + 0.5 s, t_release) ----------------------
    # release = the gripper reopening (position falling RELEASE_DROP below the
    # closed plateau); otherwise the episode ends while still closed, and that
    # end is TRUNCATION, not a release (`released` is False).
    hold_start = t_close + HOLD_START_S
    hold_s = max(0.0, t_release - hold_start)
    lab.hold_s = round(hold_s, 2)

    # ---- tactile contact over the hold ---------------------------------
    sensors = _tactile_sensors(reader, hw)
    if not sensors:
        lab.reasons.append("no_tactile_stream")
    n = int(max(0.0, hold_s) * float(contact_rate_hz))
    times = (hold_start + np.arange(n) / float(contact_rate_hz)
             if n > 0 else np.zeros(0))
    contact = contact_series(reader, hw, times, sensors)
    lab.n_contact_samples = int(len(times))
    lab.c_hold = round(float(contact.mean()), 3) if len(times) else 0.0

    # ---- lift while in contact -----------------------------------------
    if contact.any():
        z_grid = np.array([z_mm[_nearest(t_ts, float(t))] for t in times[contact]])
        lab.lift_mm = round(float(max(0.0, (z_grid - z_close).max())), 1)

    # ---- over-squeeze flag (NEVER part of grasp_ok) ---------------------
    hold_sel = (g_ts >= hold_start) & (g_ts < t_release)
    lab.obj2_frac_hold = (round(float((obj[hold_sel] == 2).mean()), 3)
                          if hold_sel.any() else 0.0)
    lab.stall = bool(lab.obj2_frac_hold > STALL_FRAC)

    # ---- verdict --------------------------------------------------------
    if z_max is None:
        lab.reasons.append(f"z_max_unknown_task:{task}")
    elif z_close > z_max:
        lab.reasons.append(f"z_close {z_close:.0f}mm > {z_max:.0f}mm")
    if hold_s < min_hold_s:
        # a short hold with NO reopen is the recording running out, not the
        # gripper letting go: distinct reason, distinct confusion cell
        lab.hold_truncated = not released
        lab.reasons.append(
            f"hold_truncated {hold_s:.1f}s < {min_hold_s:.1f}s "
            f"(recording ends {t_end - t_close:.1f}s after the close, "
            f"still closed)" if lab.hold_truncated
            else f"hold {hold_s:.1f}s < {min_hold_s:.1f}s")
    if lab.c_hold < min_c_hold:
        lab.reasons.append(f"c_hold {lab.c_hold:.2f} < {min_c_hold:.2f}")
    if lab.lift_mm < min_lift_mm:
        lab.reasons.append(f"lift {lab.lift_mm:.0f}mm < {min_lift_mm:.0f}mm")
    lab.grasp_ok = not lab.reasons
    # Abstain only when the truncation is the ONLY thing in the way: hold,
    # c_hold and lift all shrink with the cut-off window, but z_close is
    # measured at the close instant and a close above Z_MAX is a failure however
    # long the recording ran.
    lab.inconclusive = bool(
        lab.hold_truncated
        and all(reason_key(r) in ("hold_truncated", "c_hold", "lift")
                for r in lab.reasons))
    return lab


def reason_key(reason: str) -> str:
    """'c_hold 0.12 < 0.80' -> 'c_hold' (for aggregate condition counts)."""
    return reason.split(" ")[0].split(":")[0]


def confusion(labels: list[GraspLabel]) -> dict[str, int]:
    """Rule-vs-operator counts. `op_none` = operator never labeled it.

    `trunc_*` is a THIRD row, not part of `no_*`: those episodes ended inside
    the hold window with nothing else against them (`inconclusive`), so the rule
    has no verdict and counting them as negatives understates the policy
    (validation 2026-08-30). `ok_* + no_* + trunc_*` is still every label."""
    out = {"ok_s": 0, "ok_f": 0, "ok_none": 0,
           "no_s": 0, "no_f": 0, "no_none": 0,
           "trunc_s": 0, "trunc_f": 0, "trunc_none": 0}
    for l in labels:
        v = {True: "s", False: "f", None: "none"}[l.operator_success]
        pre = "ok_" if l.grasp_ok else ("trunc_" if l.inconclusive else "no_")
        out[pre + v] += 1
    return out
