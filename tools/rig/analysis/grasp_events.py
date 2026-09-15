#!/usr/bin/env python3
"""Per-take grasp-event tracker for deploy rollouts (rig session 2026-09-15).

WHY THIS EXISTS
---------------
`phantom/eval/grasp_label.py` answers one question — "did the tactile rule see
a grasp sustained through a lift?" — and answers it with a single boolean plus
reasons.  The run-sheet question is different and finer grained: for every take
we want to know *where in the pick the policy fell over*.  Specifically whether
it

  * never commanded a close at all,
  * closed on air / on the table (the **under-grasp**: a close command with no
    pad load behind it),
  * closed on the object but never got it off the table,
  * lifted it and then lost it (**drop**),
  * lifted it and put it down in the crate (**placed**), or
  * put it down having squashed it (**placed_crushed**).

Those six are what the per-arm ablation table needs, and the tactile rule
collapses the middle four into one `grasp_ok=False`.

WHICH CHANNEL IS THE "CLOSE COMMAND"
------------------------------------
`gripper.zarr` is (T, 2) = [MEASURED position 0..1 (1 = closed), Robotiq gOBJ
0..3] — see `phantom/data/schema.py:44` and `phantom/drivers/base.py:105-120`.
There is **no command column in that stream**.  The commanded aperture is
`actions.zarr[:, 6]`, the absolute gripper command the executor entered
(action_dim 7 = 6 EE deltas + 1 absolute gripper), sampled on the ~12.6 Hz
action grid.  This script therefore detects CLOSE COMMANDS on `actions[:, 6]`
and falls back to the measured position only when `actions.zarr` is missing
(reported per episode in the `close_src` column).  The two agree closely on
this session's takes — the measured position tracks the command within ~0.02
once the fingers stop moving — but "the policy commanded a close and nothing
was there" is a statement about the COMMAND, so the command is what we count.

The close predicate itself is the one already validated in
`phantom/train/common.py:close_index` (cmd > 0.45 after rising > 0.15 from its
running minimum).  The max-relative FALLBACK in `close_index` is deliberately
NOT reproduced here: it exists so Carton's wide 0.40-0.47 grasp is not missed,
and on waffles it would manufacture a "close" out of a mid-reach aperture
adjustment on takes that never closed at all (several `dp` takes top out at
0.40 with zero pad load).  Missing a close is the safer error for this table.

PAD LOAD, AND WHY IT IS BASELINE-CORRECTED
------------------------------------------
Pad load = ||wrench[:, :3]|| from `tactile_<side>_wrench.zarr`, minus the
per-episode zero taken before the first close.  This correction is not
cosmetic: on this rig the LEFT pad sits at a standing 1.0-2.5 N with the
gripper wide open while the RIGHT pad sits at ~0.04 N, so an uncorrected
threshold is ~1.5 N stricter on the right pad than on the left.

`tactile_<side>_area.zarr` is reported for context but never thresholded: it is
the vendor SDK's `getContactArea()` and `phantom/eval/grasp_label.py` documents
that it only agrees with our own contact mask to ~30%.

CLASSES
-------
never_closed        no close command in the take
closed_on_air       close command(s), but no close produced two-pad load
                    (THE UNDER-GRASP) — includes a momentary touch that never
                    became a sustained two-pad hold
grasped_no_lift     sustained two-pad hold, but TCP z never rose far enough
grasped_dropped     lifted while loaded, then lost contact while still high
grasped_held_at_end held the object to the end of the recording (no release
                    was recorded) — inconclusive, NOT a drop and NOT a place
placed              lifted, then released low over the crate
placed_crushed      placed, with a peak pad load above the clean-placed range

`crush` is also emitted as its OWN column, so a take that squashed the pack and
then dropped it still shows up as a crush even though its class is not
`placed_crushed`.

USAGE
-----
    python tools/rig/analysis/grasp_events.py <deploy folder> \
        --hw configs/hardware.nuc.yaml --csv out.csv [--md out.md]

Read-only.  Only numpy + zarr are imported at module scope, so the file also
runs standing alone from /tmp on the rig NUC with the repo off sys.path.
`--hw` is accepted for interface parity with the other rig tools and is used
only by the optional `--rule` cross-check against `phantom.eval.grasp_label`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# thresholds — every one is a CLI flag; the provenance is in the docstrings
# ---------------------------------------------------------------------------

#: Close command predicate, verbatim from phantom/train/common.py:356-358 so
#: the two stay in sync.  (The max-relative fallback there is NOT used; see
#: the module docstring.)
CLOSE_ABS_POS = 0.45
CLOSE_ABS_RISE = 0.15

#: A close is considered undone (the gripper reopened, so a later close is a
#: SECOND attempt) when the channel falls this far below the plateau it held.
#: Same value and same meaning as grasp_label.RELEASE_DROP, widened to 0.15
#: because the command channel is noisier than the measured position.
REOPEN_DROP = 0.15

#: Plateau is measured over this window after the close.
PLATEAU_S = 2.0

#: Per-pad load, newtons, above which that pad is "in contact".  BOTH pads must
#: exceed it.  PROVENANCE (20260915_experiment, 61 finalized takes): the
#: baseline-corrected whole-take peak of min(left, right) is <= 4.9 N for every
#: take the operator scored a failure with no grasp, and >= 6.1 N for every
#: take the operator scored a success.  5.0 N sits in that empty gap.  It is
#: also not a tuned number: `--sweep` re-runs the classification across
#: 2.0-6.0 N and the class counts are flat over 2.5-6.0 N.  The script
#: re-prints both on every run (`--md`), so they are re-derivable, not frozen.
CONTACT_N = 5.0

#: Peak pad load over the hold, newtons, above which the take is flagged as a
#: crush.  PROVENANCE: over the operator-clean successes of this session the
#: peak of max(left, right) tops out at 15.1 N; the two `crushed`-tagged takes
#: that actually closed measure 17.0 N and 21.4 N.  16.0 N separates them.
#: (The team-lead's prior was 18 N — that value keeps the 21.4 N take and
#: loses the 17.0 N one; both numbers are printed in the summary.)
CRUSH_N = 16.0

#: Window after a close command in which that close's pad load is measured.
PAD_WINDOW_S = 1.5

#: Two-pad contact must last this long to count as a hold.  Below it the take
#: is a touch, not a grasp, and is reported as `closed_on_air`.
MIN_HOLD_S = 0.4

#: Dropouts shorter than this are bridged when measuring the hold.  The gel
#: intermittently loses a soft waffle mid-lift (documented in
#: phantom/eval/grasp_label.py's validation notes), and the wrench stream runs
#: at only ~8 Hz, so a single missing sample must not split a hold in two.
CONTACT_GAP_S = 0.6

#: TCP z rise over z_close, mm, that counts as having got the object up.
LIFT_MIN_MM = 30.0

#: TCP z, mm, at or below which losing contact is a PLACEMENT rather than a
#: drop.  PROVENANCE: the deployed placement controller's own release gate,
#: `stop_state.placement_descent.release_gate_z_m` = 0.16 m, read per episode
#: when present; this is the fallback for episodes with no such block.
PLACE_Z_MM = 160.0

#: Table height, mm, used for "height above table".  Read per episode from
#: `meta.deploy_overrides.z_floor_m` when present.
Z_FLOOR_MM = 31.5

#: Resampling grid for the two-pad contact series.
CONTACT_RATE_HZ = 20.0

CLASSES = ["never_closed", "closed_on_air", "grasped_no_lift",
           "grasped_dropped", "grasped_held_at_end", "placed", "placed_crushed"]


# ---------------------------------------------------------------------------
# episode IO (deliberately zarr-direct: this file must run off-repo)
# ---------------------------------------------------------------------------

def load_stream(ep: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(data, ts) for `<ep>/<name>.zarr`, or None when the stream is absent.

    A truncated recording can leave a `.zarr` directory with chunk files but no
    `.zarray` metadata; that is a QC finding, not a crash, so it reads as a
    missing stream.
    """
    path = ep / f"{name}.zarr"
    if not path.exists():
        return None
    try:
        g = zarr.open(str(path), mode="r")
        return np.asarray(g["data"][:]), np.asarray(g["ts"][:], dtype=np.float64)
    except Exception:                                          # noqa: BLE001
        return None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:                                          # noqa: BLE001
        return {}


def tag_map(tags) -> dict[str, str]:
    """['label:pi05', 'seed:101', 'nfe1'] -> {'label': 'pi05', 'seed': '101'}."""
    out: dict[str, str] = {}
    for t in tags or []:
        if ":" in t:
            k, v = t.split(":", 1)
            out[k] = v
    return out


def at_time(ts: np.ndarray, values: np.ndarray, t: float):
    """Value of a stream at time `t`, nearest sample (WindowSampler convention)."""
    i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
    if i > 0 and abs(ts[i - 1] - t) <= abs(ts[i] - t):
        i -= 1
    return values[i]


# ---------------------------------------------------------------------------
# close-command detection
# ---------------------------------------------------------------------------

@dataclass
class CloseEvent:
    """One gripper CLOSE command and what the pads felt right after it."""

    index: int                      # 1-based, in time order within the take
    t_s: float                      # seconds since episode start
    cmd: float                      # commanded aperture at the close
    x_mm: float
    y_mm: float
    z_mm: float
    height_above_table_mm: float
    pad_left_n: float               # peak baseline-corrected load, PAD_WINDOW_S
    pad_right_n: float
    both_contact: bool              # min(left, right) >= contact_n
    area_left: float                # vendor SDK contact area, context only
    area_right: float
    t_reopen_s: float | None        # None = still closed at end of recording


def find_closes(cmd: np.ndarray, ts: np.ndarray, *,
                close_pos: float = CLOSE_ABS_POS,
                close_rise: float = CLOSE_ABS_RISE,
                reopen_drop: float = REOPEN_DROP,
                plateau_s: float = PLATEAU_S) -> list[tuple[int, int | None]]:
    """Every (close index, reopen index) in the channel, in time order.

    A close fires at the first sample above `close_pos` that has risen
    `close_rise` above its running minimum.  The gripper is then considered
    reopened at the first later sample `reopen_drop` below the plateau it held
    over the next `plateau_s`, and the search restarts from there — so a
    close / forced-open / close-again take (`--terminal-veto`) reports two
    attempts rather than one.
    """
    cmd = np.asarray(cmd, dtype=np.float64)
    if cmd.size == 0:
        return []
    out: list[tuple[int, int | None]] = []
    start = 0
    while start < len(cmd):
        seg = cmd[start:]
        run_min = np.minimum.accumulate(seg)
        hit = np.nonzero((seg > close_pos) & (seg - run_min > close_rise))[0]
        if not len(hit):
            break
        ci = int(hit[0]) + start
        t_close = float(ts[ci])
        sel = (ts >= t_close) & (ts <= t_close + plateau_s)
        plateau = float(cmd[sel].max()) if sel.any() else float(cmd[ci])
        after = np.nonzero((ts > t_close) & (cmd < plateau - reopen_drop))[0]
        ri = int(after[0]) if len(after) else None
        out.append((ci, ri))
        if ri is None or ri <= ci:
            break
        start = ri + 1
    return out


# ---------------------------------------------------------------------------
# pad load
# ---------------------------------------------------------------------------

def pad_load(wrench: np.ndarray, ts: np.ndarray, t_zero: float | None
             ) -> np.ndarray:
    """||force|| per sample, with the pre-close zero removed.

    The zero is the median force vector over the samples before `t_zero` (the
    first close), capped at 16 samples and requiring at least 3; with fewer
    than 3 the take's first 3 samples are used.  Subtracting the median VECTOR
    (not the median norm) is what removes a standing bias direction.
    """
    f = np.asarray(wrench[:, :3], dtype=np.float64)
    if t_zero is None:
        pre = f[:8]
    else:
        pre = f[ts < t_zero]
    if len(pre) < 3:
        pre = f[:3]
    base = np.median(pre, axis=0) if len(pre) else np.zeros(3)
    return np.linalg.norm(f - base, axis=1)


def longest_run(mask: np.ndarray, grid: np.ndarray, gap_s: float
                ) -> tuple[int, int] | None:
    """(start, end) indices of the longest True run in `mask`, bridging gaps.

    A False stretch shorter than `gap_s` between two True stretches is filled
    in first, so one missing tactile sample does not split a hold.
    """
    if not mask.any():
        return None
    m = mask.copy()
    idx = np.nonzero(m)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and (grid[b] - grid[a]) <= gap_s:
            m[a:b] = True
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return max(runs, key=lambda r: grid[r[1]] - grid[r[0]])


# ---------------------------------------------------------------------------
# per-episode analysis
# ---------------------------------------------------------------------------

@dataclass
class TakeRow:
    """One take's row in the grasp-event table."""

    episode: str = ""
    arm: str = ""                   # `label:<row>` tag — the experiment arm
    cell: str = ""                  # `seed:<n>` tag — the start-pose cell
    policy: str = ""
    verdict: str = ""               # operator: s / f / crushed / unlabeled
    cls: str = ""
    crush: bool = False
    n_closes: int = 0
    close_src: str = ""             # actions[:,6] (command) or gripper[:,0]
    t_first_close_s: float | None = None
    z_first_close_mm: float | None = None
    h_first_close_mm: float | None = None
    t_grasp_close_s: float | None = None    # the close that produced the hold
    z_grasp_close_mm: float | None = None
    pad_peak_n: float = 0.0         # max(L, R) peak over the hold
    pad_peak_min_n: float = 0.0     # min(L, R) peak over the hold
    pad_peak_take_n: float = 0.0    # max(L, R) peak anywhere in the take
    pad_peak_take_min_n: float = 0.0  # min(L, R) peak — what the rule thresholds
    hold_s: float = 0.0
    lift_mm: float = 0.0
    drop: bool = False
    released: bool = False
    z_release_mm: float | None = None
    y_release_m: float | None = None
    ctrl_releases: int = 0          # placement controller's own release count
    obj2_frac_hold: float = 0.0     # Robotiq gOBJ==2 over the hold — FLAG only
    stop_reason: str = ""
    n_replans: int = 0
    agree: bool = True
    disagreement: str = ""
    notes: str = ""
    rule_grasp_ok: str = ""         # optional phantom.eval.grasp_label check
    closes: list[CloseEvent] = field(default_factory=list)

    def csv_dict(self) -> dict:
        d = asdict(self)
        d.pop("closes", None)
        return d


def operator_verdict(meta: dict) -> str:
    """s / f / crushed / crushed_then_failed / unlabeled, from meta + tags."""
    tags = meta.get("tags") or []
    for t in tags:
        if t in ("crushed", "crushed_then_failed"):
            return t
    success = meta.get("success")
    if success is None:
        return "unlabeled"
    return "s" if success else "f"


def analyse_episode(ep: Path, *, contact_n: float = CONTACT_N,
                    crush_n: float = CRUSH_N,
                    pad_window_s: float = PAD_WINDOW_S,
                    min_hold_s: float = MIN_HOLD_S,
                    gap_s: float = CONTACT_GAP_S,
                    lift_min_mm: float = LIFT_MIN_MM,
                    place_z_mm: float = PLACE_Z_MM,
                    rate_hz: float = CONTACT_RATE_HZ) -> TakeRow:
    """Classify one recorded take.  Never raises on a malformed episode."""
    meta = read_json(ep / "meta.json")
    stop = read_json(ep / "stop.json")
    tags = tag_map(meta.get("tags"))
    row = TakeRow(
        episode=ep.name,
        arm=tags.get("label", "?"),
        cell=tags.get("seed", "?"),
        policy=str(meta.get("policy") or ""),
        verdict=operator_verdict(meta),
        notes=str(meta.get("notes") or "").replace("\n", " ")[:120],
        stop_reason=str(stop.get("stopped_reason") or ""),
        n_replans=int(stop.get("n_replans") or 0),
    )

    grip = load_stream(ep, "gripper")
    tcp = load_stream(ep, "arm_tcp_pose")
    acts = load_stream(ep, "actions")
    lw = load_stream(ep, "tactile_left_wrench")
    rw = load_stream(ep, "tactile_right_wrench")
    la = load_stream(ep, "tactile_left_area")
    ra = load_stream(ep, "tactile_right_area")
    if tcp is None or (grip is None and acts is None) or lw is None or rw is None:
        row.cls = "unreadable"
        row.agree = False
        row.disagreement = "missing streams"
        return row

    tcp_d, tcp_ts = tcp
    z_mm = tcp_d[:, 2] * 1000.0

    # --- the close-command channel ---------------------------------------
    if acts is not None and acts[0].ndim == 2 and acts[0].shape[1] >= 7:
        cmd, cmd_ts = np.asarray(acts[0][:, 6], dtype=np.float64), acts[1]
        row.close_src = "actions[:,6] (command)"
    else:
        cmd, cmd_ts = np.asarray(grip[0][:, 0], dtype=np.float64), grip[1]
        row.close_src = "gripper[:,0] (measured position)"

    t0 = float(min(x[1][0] for x in (grip, tcp, acts, lw, rw)
                   if x is not None and len(x[1])))
    t_end = float(max(tcp_ts[-1], cmd_ts[-1]))

    z_floor_mm = Z_FLOOR_MM
    try:
        z_floor_mm = float(meta["deploy_overrides"]["z_floor_m"]) * 1000.0
    except Exception:                                          # noqa: BLE001
        pass

    closes = find_closes(cmd, cmd_ts)
    row.n_closes = len(closes)

    # --- pad load, zeroed before the first close --------------------------
    t_zero = float(cmd_ts[closes[0][0]]) if closes else None
    lf = pad_load(lw[0], lw[1], t_zero)
    rf = pad_load(rw[0], rw[1], t_zero)
    row.pad_peak_take_n = round(float(max(lf.max(initial=0.0),
                                          rf.max(initial=0.0))), 2)
    row.pad_peak_take_min_n = round(float(min(lf.max(initial=0.0),
                                              rf.max(initial=0.0))), 2)

    # --- per-close detail --------------------------------------------------
    for k, (ci, ri) in enumerate(closes, start=1):
        t_close = float(cmd_ts[ci])
        w = (lw[1] >= t_close) & (lw[1] <= t_close + pad_window_s)
        w2 = (rw[1] >= t_close) & (rw[1] <= t_close + pad_window_s)
        lpk = float(lf[w].max()) if w.any() else 0.0
        rpk = float(rf[w2].max()) if w2.any() else 0.0
        xyz = at_time(tcp_ts, tcp_d[:, :3], t_close) * 1000.0
        row.closes.append(CloseEvent(
            index=k, t_s=round(t_close - t0, 2), cmd=round(float(cmd[ci]), 3),
            x_mm=round(float(xyz[0]), 1), y_mm=round(float(xyz[1]), 1),
            z_mm=round(float(xyz[2]), 1),
            height_above_table_mm=round(float(xyz[2]) - z_floor_mm, 1),
            pad_left_n=round(lpk, 2), pad_right_n=round(rpk, 2),
            both_contact=bool(min(lpk, rpk) >= contact_n),
            area_left=round(float(at_time(la[1], la[0], t_close)), 1)
            if la is not None else float("nan"),
            area_right=round(float(at_time(ra[1], ra[0], t_close)), 1)
            if ra is not None else float("nan"),
            t_reopen_s=round(float(cmd_ts[ri]) - t0, 2) if ri is not None else None,
        ))

    if not closes:
        row.cls = "never_closed"
        return row

    row.t_first_close_s = row.closes[0].t_s
    row.z_first_close_mm = row.closes[0].z_mm
    row.h_first_close_mm = row.closes[0].height_above_table_mm

    # --- two-pad contact series -------------------------------------------
    t_start = float(cmd_ts[closes[0][0]])
    n = int(max(0.0, t_end - t_start) * rate_hz)
    if n < 2:
        row.cls = "closed_on_air"
        return row
    grid = t_start + np.arange(n) / rate_hz
    l_on = np.array([at_time(lw[1], lf, t) for t in grid]) >= contact_n
    r_on = np.array([at_time(rw[1], rf, t) for t in grid]) >= contact_n
    both = l_on & r_on

    run = longest_run(both, grid, gap_s)
    hold_s = 0.0 if run is None else float(grid[run[1]] - grid[run[0]])
    if run is None or hold_s < min_hold_s:
        # a close command with no sustained two-pad load behind it: the
        # under-grasp.  A momentary touch lands here too, by design.
        row.cls = "closed_on_air"
        row.hold_s = round(hold_s, 2)
        return row

    i0, i1 = run
    t_hold0, t_hold1 = float(grid[i0]), float(grid[i1])
    row.hold_s = round(hold_s, 2)

    # the close that produced this hold = the last one at or before it
    grasp_ci = closes[0][0]
    for ci, _ in closes:
        if float(cmd_ts[ci]) <= t_hold0 + pad_window_s:
            grasp_ci = ci
    t_grasp = float(cmd_ts[grasp_ci])
    z_close = float(at_time(tcp_ts, z_mm, t_grasp))
    row.t_grasp_close_s = round(t_grasp - t0, 2)
    row.z_grasp_close_mm = round(z_close, 1)

    # --- pad peaks over the hold ------------------------------------------
    lsel = (lw[1] >= t_hold0) & (lw[1] <= t_hold1)
    rsel = (rw[1] >= t_hold0) & (rw[1] <= t_hold1)
    lpk = float(lf[lsel].max()) if lsel.any() else 0.0
    rpk = float(rf[rsel].max()) if rsel.any() else 0.0
    row.pad_peak_n = round(max(lpk, rpk), 2)
    row.pad_peak_min_n = round(min(lpk, rpk), 2)
    row.crush = bool(row.pad_peak_n > crush_n)

    # --- lift while loaded --------------------------------------------------
    z_hold = np.array([at_time(tcp_ts, z_mm, t) for t in grid[i0:i1 + 1]])
    row.lift_mm = round(float(max(0.0, (z_hold - z_close).max())), 1)

    # --- gOBJ over-squeeze flag (never part of the class) -------------------
    if grip is not None and grip[0].shape[1] > 1:
        gsel = (grip[1] >= t_hold0) & (grip[1] <= t_hold1)
        if gsel.any():
            row.obj2_frac_hold = round(float((grip[0][gsel, 1] == 2).mean()), 3)

    # --- release: placement or drop ----------------------------------------
    pd_block = (stop.get("stop_state") or {}).get("placement_descent") or {}
    row.ctrl_releases = int(pd_block.get("releases") or 0)
    gate_z_mm = place_z_mm
    if pd_block.get("release_gate_z_m") is not None:
        gate_z_mm = float(pd_block["release_gate_z_m"]) * 1000.0

    # contact ended before the recording did (allow one grid step of slack)
    row.released = bool(t_hold1 < t_end - max(2.0 / rate_hz, 0.3))
    if row.released:
        xyz_r = at_time(tcp_ts, tcp_d[:, :3], t_hold1)
        row.z_release_mm = round(float(xyz_r[2]) * 1000.0, 1)
        row.y_release_m = round(float(xyz_r[1]), 4)
    elif row.ctrl_releases > 0 and pd_block.get("released_at_z_m") is not None:
        # the controller opened the gripper but the pads were still loaded at
        # the cut — trust the controller's own record
        row.released = True
        row.z_release_mm = round(float(pd_block["released_at_z_m"]) * 1000.0, 1)
        row.y_release_m = round(float(pd_block.get("released_at_y_m") or 0.0), 4)

    placed = (row.ctrl_releases > 0
              or (row.z_release_mm is not None and row.z_release_mm <= gate_z_mm))

    if row.lift_mm < lift_min_mm:
        row.cls = "grasped_no_lift"
    elif not row.released:
        row.cls = "grasped_held_at_end"
    elif placed:
        row.cls = "placed_crushed" if row.crush else "placed"
    else:
        row.cls = "grasped_dropped"
        row.drop = True
    return row


# ---------------------------------------------------------------------------
# operator cross-check
# ---------------------------------------------------------------------------

def cross_check(row: TakeRow) -> TakeRow:
    """Set `agree` / `disagreement` against the operator's own verdict.

    The operator verdict is the ground truth for the paper; this flags the
    takes where the sensor story and the human story do not line up, which are
    the ones worth re-watching on video.
    """
    v, c = row.verdict, row.cls
    why = []
    if v == "s":
        if c not in ("placed",):
            why.append(f"operator success but sensors say {c}")
    elif v == "f":
        if c in ("placed", "placed_crushed"):
            why.append(f"operator failure but sensors say {c}")
    elif v in ("crushed", "crushed_then_failed"):
        if not row.crush:
            why.append(f"tagged {v} but peak pad load only {row.pad_peak_n} N")
    if row.crush and v not in ("crushed", "crushed_then_failed"):
        why.append(f"peak pad load {row.pad_peak_n} N over crush threshold, "
                   f"no crush tag")
    row.agree = not why
    row.disagreement = "; ".join(why)
    return row


# ---------------------------------------------------------------------------
# optional cross-check against the repo's tactile rule
# ---------------------------------------------------------------------------

def rule_check(ep: Path, hw_path: str) -> str:
    """`phantom.eval.grasp_label` verdict as a short string, or why not."""
    try:
        from phantom.config.hardware import load_hardware
        from phantom.eval.grasp_label import label_episode
        hw = load_hardware(hw_path)
        lab = label_episode(str(ep), hw)
        return (f"ok={lab.grasp_ok} hold={lab.hold_s} c={lab.c_hold} "
                f"lift={lab.lift_mm} n={lab.n_close_attempts}")
    except Exception as e:                                     # noqa: BLE001
        return f"rule_unavailable:{type(e).__name__}"


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def per_arm_table(rows: list[TakeRow]) -> str:
    arms = sorted({r.arm for r in rows})
    present = [c for c in CLASSES if any(r.cls == c for r in rows)]
    present += sorted({r.cls for r in rows} - set(CLASSES))
    head = "| arm | n | " + " | ".join(present) + " |"
    sep = "|" + "---|" * (len(present) + 2)
    lines = [head, sep]
    for a in arms:
        sub = [r for r in rows if r.arm == a]
        cnt = Counter(r.cls for r in sub)
        lines.append(f"| `{a}` | {len(sub)} | "
                     + " | ".join(str(cnt.get(c, 0)) for c in present) + " |")
    cnt = Counter(r.cls for r in rows)
    lines.append(f"| **all** | **{len(rows)}** | "
                 + " | ".join(f"**{cnt.get(c, 0)}**" for c in present) + " |")
    return "\n".join(lines)


def threshold_evidence(rows: list[TakeRow], contact_n: float, crush_n: float
                       ) -> str:
    """The separation the thresholds were chosen on, recomputed from this run."""
    def peaks(pred):
        # min(left, right), because BOTH pads must clear the threshold
        return sorted(r.pad_peak_take_min_n for r in rows if pred(r))

    ok = peaks(lambda r: r.verdict == "s")
    bad = peaks(lambda r: r.verdict == "f")
    placed_clean = sorted(r.pad_peak_n for r in rows if r.cls == "placed")
    crushed = sorted(r.pad_peak_take_n for r in rows
                     if r.verdict in ("crushed", "crushed_then_failed"))
    below = sum(x < contact_n for x in bad)
    out = [
        f"- **Contact threshold `{contact_n:.1f} N` per pad, both pads.** "
        f"Whole-take peak of min(left, right), baseline corrected — the "
        f"quantity the rule thresholds: operator successes span "
        f"{ok[0]:.1f}-{ok[-1]:.1f} N (n={len(ok)}), operator failures "
        f"{bad[0]:.1f}-{bad[-1]:.1f} N (n={len(bad)}), of which {below} fall "
        f"below the threshold — the takes that never loaded the pack at all.",
        f"- **Crush threshold `{crush_n:.1f} N`** on the peak of "
        f"max(left, right) over the hold. Clean `placed` takes peak at "
        + (f"{placed_clean[0]:.1f}-{placed_clean[-1]:.1f} N "
           f"(n={len(placed_clean)})" if placed_clean else "n/a")
        + "; operator `crushed` tags measure "
        + (", ".join(f"{x:.1f}" for x in crushed) if crushed else "n/a") + " N.",
        f"- **Close command** read from `actions[:, 6]` with the "
        f"`phantom/train/common.py` predicate (> {CLOSE_ABS_POS} after a "
        f"> {CLOSE_ABS_RISE} rise); the max-relative fallback there is not used.",
        f"- **Hold** = longest two-pad contact run >= {MIN_HOLD_S} s, "
        f"dropouts under {CONTACT_GAP_S} s bridged. **Lift** = TCP z rise over "
        f"z at the close, >= {LIFT_MIN_MM} mm to count. **Placement** = "
        f"release at or below the controller's own `release_gate_z_m`, or a "
        f"`placement_descent.releases` count above zero.",
    ]
    return "\n".join(out)


def sweep_table(eps: list[Path], contact_ns, crush_n: float) -> str:
    """Class counts as a function of the contact threshold.

    A threshold is only worth trusting if the answer does not move when it
    does. This re-runs the whole classification at each value so the report
    carries its own sensitivity check.
    """
    lines = ["| contact N | placed | closed_on_air | grasped_no_lift | "
             "grasped_dropped | operator successes classed placed |",
             "|---|---|---|---|---|---|"]
    for cn in contact_ns:
        rows = [cross_check(analyse_episode(e, contact_n=cn, crush_n=crush_n))
                for e in eps]
        c = Counter(r.cls for r in rows)
        n_s = sum(r.verdict == "s" for r in rows)
        ok = sum(r.cls in ("placed", "placed_crushed")
                 for r in rows if r.verdict == "s")
        lines.append(f"| {cn:.1f} | {c['placed'] + c['placed_crushed']} | "
                     f"{c['closed_on_air']} | {c['grasped_no_lift']} | "
                     f"{c['grasped_dropped']} | {ok}/{n_s} |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[TakeRow], src: Path,
                   contact_n: float, crush_n: float,
                   sweep: str = "") -> None:
    dis = [r for r in rows if not r.agree]
    lines = [
        "# Grasp events — rig deploy takes",
        "",
        f"Source: `{src}` — {len(rows)} takes.",
        "",
        "## Per-arm class counts",
        "",
        per_arm_table(rows),
        "",
        "Class meanings are in the module docstring of "
        "`tools/rig/analysis/grasp_events.py`. `closed_on_air` is the "
        "under-grasp: the policy commanded a close and the pads felt nothing.",
        "",
        "## Thresholds and how they were chosen",
        "",
        threshold_evidence(rows, contact_n, crush_n),
        "",
    ]
    if sweep:
        lines += ["### Sensitivity to the contact threshold", "", sweep, "",
                  "The classification is what it is over this whole range, so "
                  "the chosen value is not a tuned number.", ""]
    lines += [
        "## Crush flags (independent of class)",
        "",
    ]
    crushes = [r for r in rows if r.crush]
    if crushes:
        lines += ["| take | arm | verdict | class | peak pad N |", "|---|---|---|---|---|"]
        lines += [f"| `{r.episode}` | {r.arm} | {r.verdict} | {r.cls} | "
                  f"{r.pad_peak_n:.1f} |" for r in crushes]
    else:
        lines.append("None.")
    lines += ["", f"## Disagreements with the operator verdict ({len(dis)})", ""]
    if dis:
        lines += ["| take | arm | cell | verdict | class | why |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| `{r.episode}` | {r.arm} | {r.cell} | {r.verdict} | "
                  f"{r.cls} | {r.disagreement} |" for r in dis]
    else:
        lines.append("None.")
    lines += ["", "## Per-arm outcome rates", "",
              "| arm | n | operator success | sensor `placed` | "
              "under-grasp (`closed_on_air`) | never closed |",
              "|---|---|---|---|---|---|"]
    for a in sorted({r.arm for r in rows}):
        sub = [r for r in rows if r.arm == a]
        n = len(sub)
        s = sum(r.verdict == "s" for r in sub)
        p = sum(r.cls in ("placed", "placed_crushed") for r in sub)
        u = sum(r.cls == "closed_on_air" for r in sub)
        nc = sum(r.cls == "never_closed" for r in sub)
        lines.append(f"| `{a}` | {n} | {s} ({100*s/n:.0f}%) | {p} ({100*p/n:.0f}%) "
                     f"| {u} ({100*u/n:.0f}%) | {nc} ({100*nc/n:.0f}%) |")
    lines += [
        "",
        "## Known limits",
        "",
        "- **A light grasp can read as `closed_on_air`.** The gel under-reads "
        "a soft waffle pack: a take can carry the pack through a full lift at "
        "a sustained 1-2.5 N on both pads, under any usable contact "
        "threshold. Lowering the threshold to reach those takes does not "
        "help — the sensitivity table above shows the classification is flat "
        "down to 2.5 N — so treat `closed_on_air` as *no measurable pad load*, "
        "not as proof the gripper caught nothing.",
        "- **`grasped_held_at_end` is inconclusive, not a failure.** The "
        "recorder stops as soon as the planner returns, so a take that was "
        "still holding at the cut has no release to classify. Counting those "
        "as drops would understate the arm.",
        "- **The crush flag is a peak-load flag, not a damage detector.** It "
        "fires on any take that squeezed harder than a clean placement ever "
        "did, whether or not the pack was visibly damaged, and it cannot see "
        "damage done by pressing down with the gripper body rather than "
        "pinching (one take tagged `crushed` never commanded a close at all).",
        "- **The operator verdict remains the ground truth.** These classes "
        "say where in the pick the sensors lost the story; they have not been "
        "validated against video.",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def episode_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("ep_*") if (p / "meta.json").exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-take grasp-event tracker for deploy rollouts.")
    ap.add_argument("folder", type=Path, help="deploy folder holding ep_*/")
    ap.add_argument("--hw", default="configs/hardware.nuc.yaml",
                    help="hardware config (used only by --rule)")
    ap.add_argument("--csv", type=Path, required=True, help="per-take CSV out")
    ap.add_argument("--closes-csv", type=Path, default=None,
                    help="optional per-close-event CSV out")
    ap.add_argument("--md", type=Path, default=None,
                    help="optional per-arm summary markdown out")
    ap.add_argument("--contact-n", type=float, default=CONTACT_N)
    ap.add_argument("--crush-n", type=float, default=CRUSH_N)
    ap.add_argument("--pad-window-s", type=float, default=PAD_WINDOW_S)
    ap.add_argument("--min-hold-s", type=float, default=MIN_HOLD_S)
    ap.add_argument("--lift-min-mm", type=float, default=LIFT_MIN_MM)
    ap.add_argument("--rule", action="store_true",
                    help="also run phantom.eval.grasp_label on each take")
    ap.add_argument("--sweep", action="store_true",
                    help="add a contact-threshold sensitivity table to --md")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    eps = episode_dirs(a.folder)
    if not eps:
        print(f"no ep_*/ with meta.json under {a.folder}", file=sys.stderr)
        return 2

    rows: list[TakeRow] = []
    for ep in eps:
        r = analyse_episode(ep, contact_n=a.contact_n, crush_n=a.crush_n,
                            pad_window_s=a.pad_window_s,
                            min_hold_s=a.min_hold_s,
                            lift_min_mm=a.lift_min_mm)
        if a.rule:
            r.rule_grasp_ok = rule_check(ep, a.hw)
        rows.append(cross_check(r))

    a.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(TakeRow().csv_dict().keys())
    with a.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r.csv_dict())

    if a.closes_csv:
        a.closes_csv.parent.mkdir(parents=True, exist_ok=True)
        with a.closes_csv.open("w", newline="") as fh:
            cf = ["episode", "arm", "cell", "verdict"] + list(
                asdict(CloseEvent(0, 0, 0, 0, 0, 0, 0, 0, 0, False, 0, 0, None)).keys())
            w = csv.DictWriter(fh, fieldnames=cf)
            w.writeheader()
            for r in rows:
                for c in r.closes:
                    w.writerow({"episode": r.episode, "arm": r.arm,
                                "cell": r.cell, "verdict": r.verdict,
                                **asdict(c)})

    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        sweep = (sweep_table(eps, (2.0, 2.5, 3.0, 4.0, a.contact_n, 6.0),
                             a.crush_n) if a.sweep else "")
        write_markdown(a.md, rows, a.folder, a.contact_n, a.crush_n, sweep)

    if not a.quiet:
        print(per_arm_table(rows))
        print()
        dis = [r for r in rows if not r.agree]
        print(f"{len(rows)} takes, {len(dis)} disagreeing with the operator:")
        for r in dis:
            print(f"  {r.episode}  [{r.arm}/cell {r.cell}]  "
                  f"verdict={r.verdict} class={r.cls} — {r.disagreement}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
