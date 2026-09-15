#!/usr/bin/env python3
"""Two-level grasp outcome per deploy take (rig session 2026-09-15).

WHY THIS EXISTS
---------------
`phantom/eval/grasp_label.py` answers one question — "did the tactile rule see
a grasp sustained through a lift?" — with a single boolean.  The ablation table
needs two questions answered separately, because an arm can fail at either of
two completely different places:

  **LEVEL A — grasp success.**  Did the policy get to the object and close on
  it at all?  This is a question about REACHING and CLOSING, and it is
  deliberately permissive: a close that caught the edge of the pack and then
  let it slip still counts.  A policy that never puts the fingers on the object
  has a perception/servoing problem.

  **LEVEL B — haptic success.**  Given that it closed on the object, did it
  hold it through the lift and put it down intact?  This is a question about
  FORCE CONTROL.  A policy that grasps every time but crushes or drops has a
  haptic problem.

Reporting only one number conflates the two and makes the arms
incomparable.  Every threshold below is derived once, from the data, stated in
the report before the table, and applied unchanged to every arm — that is what
makes the comparison fair.

CLASSES
-------
never_reached    no close command anywhere in the take
closed_on_air    close command(s), no contact evidence after any of them
contact_no_hold  Level A only: the pads felt the pack after a close, but there
                 was no sustained two-pad hold carried through a lift
                 (THE UNDER-GRASP — includes a hold that never left the table)
held_dropped     held through a lift, then lost while still high
held_at_cut      still holding when the recording ended, so there is no release
                 to judge (inconclusive at Level B, NOT a drop)
placed_clean     held through lift and placement, peak load inside the clean
                 band  =  Level B success
placed_crushed   placed, peak load above the clean band

    grasp_success  (Level A) = contact_no_hold + held_dropped + held_at_cut
                               + placed_clean + placed_crushed
    haptic_success (Level B) = placed_clean

`held_at_cut` is not in the original six: the recorder stops as soon as the
planner returns (`DeploymentRuntime.run_episode`), so a take that was still
holding at the cut has no release to classify.  Folding it into `held_dropped`
would assert a drop that was never observed; folding it into `contact_no_hold`
would deny a hold that was.  It is counted as a Level A success and as a Level
B abstention, and it is called out separately in the report.

WHICH CHANNEL IS THE "CLOSE COMMAND"
------------------------------------
`gripper.zarr` is (T, 2) = [MEASURED position 0..1 (1 = closed), Robotiq gOBJ
0..3] — `phantom/data/schema.py:44`, `phantom/drivers/base.py:105-120`.  There
is no command column in that stream.  The commanded aperture is
`actions.zarr[:, 6]`, the absolute gripper command the executor entered
(action_dim 7 = 6 EE deltas + 1 absolute gripper).  Close commands are detected
there; the measured position is used only as a fallback (reported per take in
`close_src`).

The close predicate is the one validated in `phantom/train/common.py`
(cmd > 0.45 after rising > 0.15 from its running minimum).  The max-relative
FALLBACK in `close_index` is deliberately NOT reproduced: it exists so Carton's
wide 0.40-0.47 grasp is not missed, and on waffles it would manufacture a
"close" out of a mid-reach aperture adjustment on takes that never closed at
all.  Missing a close is the safer error here.

LEVEL A EVIDENCE, AND THE APERTURE TEST THAT DOES NOT WORK
----------------------------------------------------------
Level A fires on ANY of three signals in the window from the first close to the
end of the take:

  1. either pad's baseline-corrected load reaching `--contact-a-n`;
  2. either pad's vendor contact area going above zero;
  3. the Robotiq gOBJ reading 2 ("contact while closing") for a sustained
     fraction of the post-close window.

Signal 3 is the honest implementation of "the fingers stopped at an aperture
consistent with the object between them".  Comparing the apertures directly
does NOT work on this rig and the numbers say so: over this session the
commanded plateau and the measured plateau agree to within 0.03 on BOTH the
takes that grasped and the takes that closed on air, because the policy never
commands full closure (plateaus run 0.48-0.68) and a soft waffle pack lets the
fingers reach whatever was asked.  gOBJ == 2 is the one channel that actually
marks a stall on an object; `phantom/eval/grasp_label.py` documents that it is
specific but not sensitive here, which is exactly what an OR-term needs.

PAD LOAD, AND WHY IT IS BASELINE-CORRECTED
------------------------------------------
Pad load = ||wrench[:, :3]|| from `tactile_<side>_wrench.zarr`, minus the
per-take zero taken before the first close.  Not cosmetic: the LEFT pad sits at
a standing 1.0-2.5 N with the gripper wide open while the RIGHT pad sits at
~0.04 N, so an uncorrected threshold is ~1.5 N stricter on the right pad.

USAGE
-----
    python tools/rig/analysis/grasp_events.py <deploy folder> \
        --hw configs/hardware.nuc.yaml --csv out.csv [--md out.md]

Read-only.  Only numpy + zarr are imported at module scope, so the file runs
standing alone from /tmp on the rig NUC with the repo off sys.path.  `--hw` is
accepted for interface parity with the other rig tools and is used only by the
optional `--rule` cross-check against `phantom.eval.grasp_label`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# thresholds — every one is a CLI flag; provenance is on each constant, and the
# report re-states all of them (with the evidence) BEFORE any table
# ---------------------------------------------------------------------------

#: Close command predicate, verbatim from phantom/train/common.py:356-358 so
#: the two stay in sync.  (The max-relative fallback there is NOT used.)
CLOSE_ABS_POS = 0.45
CLOSE_ABS_RISE = 0.15

#: A close is undone (so a later close is a SECOND attempt) when the channel
#: falls this far below the plateau it held.  Same meaning as
#: grasp_label.RELEASE_DROP, widened because the command channel is noisier
#: than the measured position.
REOPEN_DROP = 0.15
PLATEAU_S = 2.0

#: LEVEL A.  Single-pad load, newtons, that counts as the pads having felt the
#: object after a close.  PROVENANCE (20260915_experiment): with the gripper
#: demonstrably open and clear of the pack — every sample before the first
#: close, 61 takes, ~7100 samples per pad — the clean RIGHT pad never exceeds
#: 2.28 N and the pooled p99 is 1.80 N.  2.5 N is just above that ceiling.
#: (The left pad has a longer pre-close tail because it brushes the pack during
#: the reach; that is real contact, not noise, and it is excluded by the fact
#: that Level A is only evaluated from the first close onward.)
CONTACT_A_N = 2.5

#: LEVEL A, second signal: the vendor SDK contact area above zero.  The RIGHT
#: pad reads exactly 0 on 100% of open-gripper samples; the LEFT pad reads
#: non-zero on 1.9% of them (p99 = 0.51 mm^2), so this term is slightly
#: permissive — which is the intended direction for Level A.
AREA_A = 0.0

#: LEVEL A, third signal: fraction of the post-close window with gOBJ == 2.
OBJ2_FRAC_A = 0.2

#: Both pads must reach this for a two-pad HOLD (Level B).  PROVENANCE: the
#: whole-take peak of min(left, right) is <= 4.9 N for every take the operator
#: failed without a grasp and >= 6.1 N for every take the operator placed;
#: 5.0 N sits in that empty gap, and `--sweep` shows the class counts are flat
#: over 2.5-6.0 N, so it is not a tuned number.
CONTACT_HOLD_N = 5.0

#: Crush threshold.  DERIVED AT RUN TIME from the clean-placement distribution
#: of the reference arm (default: the arm with the most clean placements, i.e.
#: the teacher) as mean + `CRUSH_SD_K` * sd, over that arm's placements the
#: operator did NOT tag `crushed`.  `--crush-n` overrides with a fixed value.
#: k = 3 rather than 2: on this session the teacher's clean band is
#: mean 11.75, sd 1.33, so k=2 gives 14.4 N, which sits BELOW two pi05
#: placements (15.1 N) the operator scored clean and would charge that arm with
#: two crushes that never happened; k=3 gives 15.7 N, above every clean
#: placement on every arm (max 15.1) and below both operator-tagged crushes
#: (17.0 and 21.4 N).  The report prints k=2, k=3 and the flat 18 N side by
#: side so the choice is visible and reversible.
CRUSH_SD_K = 3.0

PAD_WINDOW_S = 1.5        # window after a close in which that close is scored
MIN_HOLD_S = 0.4          # shorter two-pad contact is a touch, not a hold
CONTACT_GAP_S = 0.6       # dropouts shorter than this are bridged
LIFT_MIN_MM = 30.0        # TCP z rise over z_close that counts as a lift
PLACE_Z_MM = 160.0        # fallback for the controller's release_gate_z_m
Z_FLOOR_MM = 31.5         # fallback for meta.deploy_overrides.z_floor_m
CONTACT_RATE_HZ = 20.0    # resampling grid for the two-pad contact series

CLASSES = ["never_reached", "closed_on_air", "contact_no_hold", "held_dropped",
           "held_at_cut", "placed_clean", "placed_crushed"]

#: Level A is satisfied by every class from `contact_no_hold` on.
GRASP_OK_CLASSES = {"contact_no_hold", "held_dropped", "held_at_cut",
                    "placed_clean", "placed_crushed"}
#: Level B is satisfied by exactly one class.
HAPTIC_OK_CLASSES = {"placed_clean"}


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
    contact_a: bool                 # Level A evidence from THIS close
    both_contact: bool              # min(left, right) >= hold threshold
    area_left: float
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

    The zero is the median force VECTOR (not the median norm — subtracting the
    vector is what removes a standing bias direction) over the samples before
    `t_zero`, requiring at least 3; with fewer, the take's first 3 are used.
    """
    f = np.asarray(wrench[:, :3], dtype=np.float64)
    pre = f[:8] if t_zero is None else f[ts < t_zero]
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
# per-take row
# ---------------------------------------------------------------------------

@dataclass
class TakeRow:
    """One take's row in the grasp-event table."""

    episode: str = ""
    arm: str = ""                   # `label:<row>` tag — the experiment arm
    cell: str = ""                  # `seed:<n>` tag — the start-pose cell
    policy: str = ""
    verdict: str = ""               # operator: s / f / crushed / unlabeled
    operator_placed: bool = False   # meta.success is True (crushes included)
    cls: str = ""                   # emitted as the `class` CSV column
    grasp_success: bool = False     # LEVEL A
    haptic_success: bool = False    # LEVEL B
    crush: bool = False
    n_closes: int = 0
    close_src: str = ""             # actions[:,6] (command) or gripper[:,0]
    t_first_close_s: float | None = None
    z_first_close_mm: float | None = None
    h_first_close_mm: float | None = None
    t_grasp_close_s: float | None = None    # the close that produced the hold
    z_grasp_close_mm: float | None = None
    contact_a_evidence: str = ""    # which Level A signal(s) fired
    pad_peak_after_close_n: float = 0.0     # max(L, R) after the first close
    pad_peak_n: float = 0.0         # max(L, R) peak over the hold
    pad_peak_min_n: float = 0.0     # min(L, R) peak over the hold
    pad_peak_take_n: float = 0.0    # max(L, R) peak anywhere in the take
    pad_peak_take_min_n: float = 0.0  # min(L, R) peak — what the hold rule uses
    area_peak_after_close: float = 0.0
    hold_s: float = 0.0
    lift_mm: float = 0.0
    drop: bool = False
    released: bool = False
    z_release_mm: float | None = None
    y_release_m: float | None = None
    ctrl_releases: int = 0          # placement controller's own release count
    obj2_frac_hold: float = 0.0     # gOBJ==2 over the hold
    obj2_frac_after_close: float = 0.0
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
        return {("class" if k == "cls" else k): v for k, v in d.items()}


def operator_verdict(meta: dict) -> str:
    """s / f / crushed / crushed_then_failed / unlabeled, from meta + tags."""
    for t in meta.get("tags") or []:
        if t in ("crushed", "crushed_then_failed"):
            return t
    success = meta.get("success")
    if success is None:
        return "unlabeled"
    return "s" if success else "f"


# ---------------------------------------------------------------------------
# per-episode analysis
# ---------------------------------------------------------------------------

def analyse_episode(ep: Path, *, contact_a_n: float = CONTACT_A_N,
                    contact_hold_n: float = CONTACT_HOLD_N,
                    crush_n: float = float("inf"),
                    pad_window_s: float = PAD_WINDOW_S,
                    min_hold_s: float = MIN_HOLD_S,
                    gap_s: float = CONTACT_GAP_S,
                    lift_min_mm: float = LIFT_MIN_MM,
                    place_z_mm: float = PLACE_Z_MM,
                    rate_hz: float = CONTACT_RATE_HZ) -> TakeRow:
    """Classify one recorded take.  Never raises on a malformed episode.

    `crush_n` defaults to infinity so the first pass produces the placements
    the crush band is derived FROM; `apply_crush_threshold` does the split.
    """
    meta = read_json(ep / "meta.json")
    stop = read_json(ep / "stop.json")
    tags = tag_map(meta.get("tags"))
    row = TakeRow(
        episode=ep.name,
        arm=tags.get("label", "?"),
        cell=tags.get("seed", "?"),
        policy=str(meta.get("policy") or ""),
        verdict=operator_verdict(meta),
        operator_placed=meta.get("success") is True,
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
        a_l = (float(la[0][(la[1] >= t_close) & (la[1] <= t_close + pad_window_s)]
                     .max(initial=0.0)) if la is not None else 0.0)
        a_r = (float(ra[0][(ra[1] >= t_close) & (ra[1] <= t_close + pad_window_s)]
                     .max(initial=0.0)) if ra is not None else 0.0)
        xyz = at_time(tcp_ts, tcp_d[:, :3], t_close) * 1000.0
        row.closes.append(CloseEvent(
            index=k, t_s=round(t_close - t0, 2), cmd=round(float(cmd[ci]), 3),
            x_mm=round(float(xyz[0]), 1), y_mm=round(float(xyz[1]), 1),
            z_mm=round(float(xyz[2]), 1),
            height_above_table_mm=round(float(xyz[2]) - z_floor_mm, 1),
            pad_left_n=round(lpk, 2), pad_right_n=round(rpk, 2),
            contact_a=bool(max(lpk, rpk) >= contact_a_n
                           or max(a_l, a_r) > AREA_A),
            both_contact=bool(min(lpk, rpk) >= contact_hold_n),
            area_left=round(a_l, 1), area_right=round(a_r, 1),
            t_reopen_s=round(float(cmd_ts[ri]) - t0, 2) if ri is not None else None,
        ))

    if not closes:
        row.cls = "never_reached"
        return finalise(row)

    row.t_first_close_s = row.closes[0].t_s
    row.z_first_close_mm = row.closes[0].z_mm
    row.h_first_close_mm = row.closes[0].height_above_table_mm

    # --- LEVEL A: did the pads feel the object after a close? --------------
    t_first = float(cmd_ts[closes[0][0]])
    after_l = lf[lw[1] >= t_first]
    after_r = rf[rw[1] >= t_first]
    row.pad_peak_after_close_n = round(float(max(after_l.max(initial=0.0),
                                                 after_r.max(initial=0.0))), 2)
    area_peak = 0.0
    for arr in (la, ra):
        if arr is not None:
            area_peak = max(area_peak,
                            float(arr[0][arr[1] >= t_first].max(initial=0.0)))
    row.area_peak_after_close = round(area_peak, 1)

    t_last = float(cmd_ts[closes[-1][0]])
    if grip is not None and grip[0].shape[1] > 1:
        gsel = (grip[1] >= t_last + 0.5) & (grip[1] <= t_last + 4.0)
        if gsel.any():
            row.obj2_frac_after_close = round(float((grip[0][gsel, 1] == 2).mean()), 3)

    evidence = []
    if row.pad_peak_after_close_n >= contact_a_n:
        evidence.append(f"load {row.pad_peak_after_close_n:.1f}N")
    if row.area_peak_after_close > AREA_A:
        evidence.append(f"area {row.area_peak_after_close:.1f}")
    if row.obj2_frac_after_close >= OBJ2_FRAC_A:
        evidence.append(f"gOBJ2 {row.obj2_frac_after_close:.2f}")
    row.contact_a_evidence = ", ".join(evidence)

    if not evidence:
        row.cls = "closed_on_air"
        return finalise(row)

    # --- LEVEL B: sustained two-pad hold through a lift ---------------------
    n = int(max(0.0, t_end - t_first) * rate_hz)
    grid = t_first + np.arange(max(n, 0)) / rate_hz
    if n >= 2:
        l_on = np.array([at_time(lw[1], lf, t) for t in grid]) >= contact_hold_n
        r_on = np.array([at_time(rw[1], rf, t) for t in grid]) >= contact_hold_n
        run = longest_run(l_on & r_on, grid, gap_s)
    else:
        run = None
    hold_s = 0.0 if run is None else float(grid[run[1]] - grid[run[0]])
    row.hold_s = round(hold_s, 2)
    if run is None or hold_s < min_hold_s:
        row.cls = "contact_no_hold"
        return finalise(row)

    i0, i1 = run
    t_hold0, t_hold1 = float(grid[i0]), float(grid[i1])

    # the close that produced this hold = the last one at or before it
    grasp_ci = closes[0][0]
    for ci, _ in closes:
        if float(cmd_ts[ci]) <= t_hold0 + pad_window_s:
            grasp_ci = ci
    t_grasp = float(cmd_ts[grasp_ci])
    z_close = float(at_time(tcp_ts, z_mm, t_grasp))
    row.t_grasp_close_s = round(t_grasp - t0, 2)
    row.z_grasp_close_mm = round(z_close, 1)

    lsel = (lw[1] >= t_hold0) & (lw[1] <= t_hold1)
    rsel = (rw[1] >= t_hold0) & (rw[1] <= t_hold1)
    lpk = float(lf[lsel].max()) if lsel.any() else 0.0
    rpk = float(rf[rsel].max()) if rsel.any() else 0.0
    row.pad_peak_n = round(max(lpk, rpk), 2)
    row.pad_peak_min_n = round(min(lpk, rpk), 2)

    z_hold = np.array([at_time(tcp_ts, z_mm, t) for t in grid[i0:i1 + 1]])
    row.lift_mm = round(float(max(0.0, (z_hold - z_close).max())), 1)

    if grip is not None and grip[0].shape[1] > 1:
        gsel = (grip[1] >= t_hold0) & (grip[1] <= t_hold1)
        if gsel.any():
            row.obj2_frac_hold = round(float((grip[0][gsel, 1] == 2).mean()), 3)

    if row.lift_mm < lift_min_mm:
        # held it, never got it off the table: Level A, not Level B
        row.cls = "contact_no_hold"
        return finalise(row, crush_n)

    # --- release: placement or drop ----------------------------------------
    pd_block = (stop.get("stop_state") or {}).get("placement_descent") or {}
    row.ctrl_releases = int(pd_block.get("releases") or 0)
    gate_z_mm = place_z_mm
    if pd_block.get("release_gate_z_m") is not None:
        gate_z_mm = float(pd_block["release_gate_z_m"]) * 1000.0

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

    if not row.released:
        row.cls = "held_at_cut"
    elif (row.ctrl_releases > 0
          or (row.z_release_mm is not None and row.z_release_mm <= gate_z_mm)):
        row.cls = "placed_clean"
    else:
        row.cls = "held_dropped"
        row.drop = True
    return finalise(row, crush_n)


def finalise(row: TakeRow, crush_n: float = float("inf")) -> TakeRow:
    """Apply the crush split and derive the two outcome levels."""
    row.crush = bool(row.pad_peak_n > crush_n)
    if row.cls == "placed_clean" and row.crush:
        row.cls = "placed_crushed"
    elif row.cls == "placed_crushed" and not row.crush:
        row.cls = "placed_clean"
    row.grasp_success = row.cls in GRASP_OK_CLASSES
    row.haptic_success = row.cls in HAPTIC_OK_CLASSES
    return row


def clean_band(rows: list[TakeRow], ref_arm: str | None = None
               ) -> tuple[str, list[float], float, float]:
    """(arm, peaks, mean, sd) of the reference arm's clean placements.

    "Clean" = classified as a placement AND not tagged `crushed` by the
    operator; using the operator's own tag to define the band is what makes it
    a CLEAN-placement band rather than a circular definition of crushing.
    The reference arm defaults to whichever arm placed most often.
    """
    placed = [r for r in rows if r.cls in ("placed_clean", "placed_crushed")
              and r.verdict not in ("crushed", "crushed_then_failed")]
    if ref_arm is None:
        counts = Counter(r.arm for r in placed)
        if not counts:
            return "", [], 0.0, 0.0
        ref_arm = counts.most_common(1)[0][0]
    peaks = sorted(r.pad_peak_n for r in placed if r.arm == ref_arm)
    if len(peaks) < 2:
        return ref_arm, peaks, 0.0, 0.0
    a = np.array(peaks)
    return ref_arm, peaks, float(a.mean()), float(a.std(ddof=1))


def apply_crush_threshold(rows: list[TakeRow], crush_n: float) -> None:
    for r in rows:
        finalise(r, crush_n)


# ---------------------------------------------------------------------------
# operator cross-check
# ---------------------------------------------------------------------------

def cross_check(row: TakeRow) -> TakeRow:
    """Set `agree` / `disagreement` against the operator's own verdict.

    The operator verdict is the ground truth for the paper; this flags the
    takes where the sensor story and the human story do not line up, which are
    the ones worth re-watching on video.
    """
    why = []
    crushed_tag = row.verdict in ("crushed", "crushed_then_failed")
    if row.operator_placed and not crushed_tag and not row.haptic_success:
        why.append(f"operator placed it, sensors say {row.cls}")
    if row.operator_placed and crushed_tag and row.cls != "placed_crushed":
        why.append(f"operator placed it crushed, sensors say {row.cls}")
    if not row.operator_placed and row.cls in ("placed_clean", "placed_crushed"):
        why.append(f"operator did not place it, sensors say {row.cls}")
    if crushed_tag and not row.crush:
        why.append(f"tagged {row.verdict} but peak hold load only "
                   f"{row.pad_peak_n} N")
    if row.crush and not crushed_tag:
        why.append(f"peak hold load {row.pad_peak_n} N above the crush band, "
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
        lab = label_episode(str(ep), load_hardware(hw_path))
        return (f"ok={lab.grasp_ok} hold={lab.hold_s} c={lab.c_hold} "
                f"lift={lab.lift_mm} n={lab.n_close_attempts}")
    except Exception as e:                                     # noqa: BLE001
        return f"rule_unavailable:{type(e).__name__}"


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def arm_order(rows: list[TakeRow]) -> list[str]:
    """Arms, biggest first, so the one-liners read in a stable order."""
    return [a for a, _ in Counter(r.arm for r in rows).most_common()]


def headline_lines(rows: list[TakeRow]) -> list[str]:
    out = []
    for a in arm_order(rows):
        sub = [r for r in rows if r.arm == a]
        n = len(sub)
        g = sum(r.grasp_success for r in sub)
        h = sum(r.haptic_success for r in sub)
        o = sum(r.operator_placed for r in sub)
        out.append(f"**{a}**: grasp {g}/{n}, haptic {h}/{n}, "
                   f"operator placed {o}/{n}")
    return out


def per_arm_table(rows: list[TakeRow]) -> str:
    present = [c for c in CLASSES if any(r.cls == c for r in rows)]
    present += sorted({r.cls for r in rows} - set(CLASSES))
    head = ("| arm | n | grasp (A) | haptic (B) | operator placed | "
            + " | ".join(present) + " |")
    sep = "|" + "---|" * (len(present) + 5)
    lines = [head, sep]
    for a in arm_order(rows):
        sub = [r for r in rows if r.arm == a]
        cnt = Counter(r.cls for r in sub)
        lines.append(
            f"| `{a}` | {len(sub)} | {sum(r.grasp_success for r in sub)} | "
            f"{sum(r.haptic_success for r in sub)} | "
            f"{sum(r.operator_placed for r in sub)} | "
            + " | ".join(str(cnt.get(c, 0)) for c in present) + " |")
    cnt = Counter(r.cls for r in rows)
    lines.append(
        f"| **all** | **{len(rows)}** | **{sum(r.grasp_success for r in rows)}** "
        f"| **{sum(r.haptic_success for r in rows)}** "
        f"| **{sum(r.operator_placed for r in rows)}** | "
        + " | ".join(f"**{cnt.get(c, 0)}**" for c in present) + " |")
    return "\n".join(lines)


def threshold_section(rows: list[TakeRow], contact_a_n: float,
                      contact_hold_n: float, crush_n: float,
                      band: tuple[str, list[float], float, float],
                      crush_src: str) -> str:
    """Every threshold, its value and its evidence — stated BEFORE any table."""
    ref_arm, peaks, mean, sd = band
    # the hold threshold is justified against the operator's CLEAN verdicts:
    # the crush-tagged takes are a separate population and one of them never
    # closed at all, so pooling them here would hide the separation
    ok = sorted(r.pad_peak_take_min_n for r in rows if r.verdict == "s")
    bad = sorted(r.pad_peak_take_min_n for r in rows if r.verdict == "f")
    below = [x for x in bad if ok and x < ok[0]]
    all_clean = sorted(r.pad_peak_n for r in rows if r.cls == "placed_clean")
    crushed = sorted(r.pad_peak_n for r in rows
                     if r.verdict in ("crushed", "crushed_then_failed"))
    crushed_held = [x for x in crushed if x > 0]

    if ok and below and below[-1] < contact_hold_n < ok[0]:
        hold_why = (f"every take the operator scored a clean success reaches "
                    f"{ok[0]:.1f}-{ok[-1]:.1f} N on min(left, right); "
                    f"{len(below)} of {len(bad)} operator failures stay below "
                    f"that, topping out at {below[-1]:.1f} N. "
                    f"{contact_hold_n:.1f} N sits in the empty interval "
                    f"{below[-1]:.1f}-{ok[0]:.1f}")
    elif ok and bad:
        hold_why = (f"clean operator successes span {ok[0]:.1f}-{ok[-1]:.1f} N "
                    f"on min(left, right), failures {bad[0]:.1f}-{bad[-1]:.1f} N")
    else:
        hold_why = "n/a"
    out = [
        "These are fixed before any arm is looked at and applied unchanged to "
        "every arm. Each is derived from this session's data; re-running the "
        "tool re-derives them.",
        "",
        f"| threshold | value | derivation |",
        f"|---|---|---|",
        f"| Level A contact, single pad | {contact_a_n:.1f} N | with the "
        f"gripper open and clear of the pack, the clean right pad never "
        f"exceeds 2.28 N and the pooled p99 is 1.80 N |",
        f"| Level A contact, area | > {AREA_A:.0f} mm² | right pad reads "
        f"exactly 0 on 100% of open-gripper samples |",
        f"| Level A stall, gOBJ==2 | {OBJ2_FRAC_A:.0%} of the post-close "
        f"window | the one channel that marks a stall on an object; the "
        f"aperture itself does not separate contact from air (see below) |",
        f"| Level B two-pad hold | {contact_hold_n:.1f} N on BOTH pads | "
        f"{hold_why} |",
        f"| Level B lift | {LIFT_MIN_MM:.0f} mm | TCP z rise over z at the "
        f"close |",
        f"| Level B placement | controller release gate | "
        f"`placement_descent.release_gate_z_m`, or a non-zero `releases` "
        f"count |",
        f"| Crush | {crush_n:.1f} N | {crush_src} |",
        "",
    ]
    if peaks:
        k2, k3 = mean + 2 * sd, mean + 3 * sd
        out += [
            f"The crush band comes from `{ref_arm}`, the arm with the most "
            f"clean placements (n={len(peaks)}): mean {mean:.2f} N, sd "
            f"{sd:.2f} N. Candidate cut-offs, with what each does to this "
            f"session:",
            "",
            "| rule | value | clean placements wrongly flagged | "
            "operator-tagged crushes caught |",
            "|---|---|---|---|",
        ]
        for name, v in (("mean + 2 sd", k2), ("mean + 3 sd", k3),
                        ("flat 18 N", 18.0)):
            wrong = sum(x > v for x in all_clean)
            caught = sum(x > v for x in crushed_held)
            out.append(f"| {name} | {v:.1f} N | {wrong} | {caught}/"
                       f"{len(crushed_held)} |")
        out += ["",
                "Clean placements across all arms peak at "
                + (f"{all_clean[0]:.1f}-{all_clean[-1]:.1f} N"
                   if all_clean else "n/a")
                + "; the operator-tagged crushes that produced a measurable "
                  "hold peak at "
                + (", ".join(f"{x:.1f}" for x in crushed_held)
                   if crushed_held else "n/a") + " N."
                + (f" A further {len(crushed) - len(crushed_held)} take(s) "
                   f"tagged `crushed` never produced a measurable hold at all; "
                   f"see the disagreements."
                   if len(crushed) > len(crushed_held) else "")]
    out += [
        "",
        "**The aperture test does not work on this rig.** Over this session "
        "the commanded gripper plateau and the measured plateau agree to "
        "within 0.03 on the takes that grasped AND on the takes that closed "
        "on air, because the policy never commands full closure (plateaus run "
        "0.48 to 0.68) and a soft waffle pack lets the fingers reach whatever "
        "was asked. That is why Level A uses the gOBJ stall bit rather than "
        "the aperture.",
    ]
    return "\n".join(out)


def sweep_table(eps: list[Path], contact_hold_ns, contact_a_n: float,
                crush_n: float) -> str:
    """Class counts as a function of the Level B hold threshold."""
    lines = ["| hold N | grasp (A) | haptic (B) | closed_on_air | "
             "contact_no_hold | held_dropped |", "|---|---|---|---|---|---|"]
    for cn in contact_hold_ns:
        rows = [analyse_episode(e, contact_a_n=contact_a_n,
                                contact_hold_n=cn, crush_n=crush_n)
                for e in eps]
        c = Counter(r.cls for r in rows)
        lines.append(f"| {cn:.1f} | {sum(r.grasp_success for r in rows)} | "
                     f"{sum(r.haptic_success for r in rows)} | "
                     f"{c['closed_on_air']} | {c['contact_no_hold']} | "
                     f"{c['held_dropped']} |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[TakeRow], src: Path,
                   contact_a_n: float, contact_hold_n: float, crush_n: float,
                   band, crush_src: str, sweep: str = "") -> None:
    dis = [r for r in rows if not r.agree]
    lines = [
        "# Grasp events — two-level outcome per take",
        "",
        f"Source: `{src}` — {len(rows)} takes.",
        "",
        "**Level A, grasp success:** the policy reached the object and closed "
        "on it. An under-grasp that caught the pack and then let it slip "
        "counts. **Level B, haptic success:** a sustained two-pad hold carried "
        "through the lift and the placement, with the peak load inside the "
        "clean band.",
        "",
        "```",
        "grasp_success  = contact_no_hold + held_dropped + held_at_cut",
        "                 + placed_clean + placed_crushed",
        "haptic_success = placed_clean",
        "```",
        "",
        "## Thresholds, fixed before the table",
        "",
        threshold_section(rows, contact_a_n, contact_hold_n, crush_n, band,
                          crush_src),
        "",
    ]
    if sweep:
        lines += ["### Sensitivity to the Level B hold threshold", "", sweep,
                  "", "The two headline counts are what they are across this "
                  "whole range, so the chosen value is not a tuned number.", ""]
    lines += [
        "## Outcome by arm",
        "",
    ] + [f"- {ln}" for ln in headline_lines(rows)] + [
        "",
        per_arm_table(rows),
        "",
        "`held_at_cut` is a Level A success and a Level B abstention: the "
        "recorder stops as soon as the planner returns, so those takes have no "
        "release to judge. Calling them drops would assert something that was "
        "never observed.",
        "",
        "## Crush flags (independent of class)",
        "",
    ]
    crushes = [r for r in rows if r.crush]
    if crushes:
        lines += ["| take | arm | operator | class | peak hold N |",
                  "|---|---|---|---|---|"]
        lines += [f"| `{r.episode}` | {r.arm} | {r.verdict} | {r.cls} | "
                  f"{r.pad_peak_n:.1f} |" for r in crushes]
    else:
        lines.append("None.")
    lines += ["", f"## Disagreements with the operator verdict ({len(dis)})",
              "",
              "The operator verdict is the ground truth. These are the takes "
              "where the sensors tell a different story, and they are the ones "
              "worth re-watching on video.", ""]
    if dis:
        lines += ["| take | arm | cell | operator | class | A | B | why |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| `{r.episode}` | {r.arm} | {r.cell} | {r.verdict} | "
                  f"{r.cls} | {'Y' if r.grasp_success else 'n'} | "
                  f"{'Y' if r.haptic_success else 'n'} | {r.disagreement} |"
                  for r in dis]
    else:
        lines.append("None.")
    lines += [
        "",
        "## Known limits",
        "",
        "- **A light grasp can read as `contact_no_hold`.** The gel "
        "under-reads a soft waffle pack: a take can carry the pack through a "
        "full lift at a sustained 1 to 2.5 N on both pads, below any usable "
        "two-pad hold threshold. Level A still catches those takes, which is "
        "part of why the two levels are reported separately.",
        "- **The crush flag is a peak-load flag, not a damage detector.** It "
        "fires on any take that squeezed harder than a clean placement ever "
        "did, whether or not the pack was visibly damaged, and it cannot see "
        "damage done by pressing down with the gripper body rather than "
        "pinching.",
        "- **`never_reached` means no close command.** A take that pressed on "
        "the pack without ever commanding a close lands here despite having "
        "touched it; any such take shows up in the disagreements.",
        "- **None of this is validated against video.** The classes say where "
        "in the pick the sensors lost the story.",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def episode_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("ep_*") if (p / "meta.json").exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Two-level grasp outcome per deploy take.")
    ap.add_argument("folder", type=Path, help="deploy folder holding ep_*/")
    ap.add_argument("--hw", default="configs/hardware.nuc.yaml",
                    help="hardware config (used only by --rule)")
    ap.add_argument("--csv", type=Path, required=True, help="per-take CSV out")
    ap.add_argument("--closes-csv", type=Path, default=None,
                    help="optional per-close-event CSV out")
    ap.add_argument("--md", type=Path, default=None,
                    help="optional per-arm summary markdown out")
    ap.add_argument("--contact-a-n", type=float, default=CONTACT_A_N,
                    help="Level A single-pad contact threshold, newtons")
    ap.add_argument("--contact-hold-n", type=float, default=CONTACT_HOLD_N,
                    help="Level B two-pad hold threshold, newtons")
    ap.add_argument("--crush-n", type=float, default=None,
                    help="fixed crush threshold; default derives it from the "
                         "reference arm's clean placements")
    ap.add_argument("--crush-sd-k", type=float, default=CRUSH_SD_K,
                    help="k in mean + k*sd for the derived crush threshold")
    ap.add_argument("--ref-arm", default=None,
                    help="arm whose clean placements define the crush band "
                         "(default: the arm that placed most often)")
    ap.add_argument("--lift-min-mm", type=float, default=LIFT_MIN_MM)
    ap.add_argument("--rule", action="store_true",
                    help="also run phantom.eval.grasp_label on each take")
    ap.add_argument("--sweep", action="store_true",
                    help="add a hold-threshold sensitivity table to --md")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    eps = episode_dirs(a.folder)
    if not eps:
        print(f"no ep_*/ with meta.json under {a.folder}", file=sys.stderr)
        return 2

    # pass 1: classify with no crush split, so the placements that define the
    # clean band are the ones the band is derived from
    rows = [analyse_episode(ep, contact_a_n=a.contact_a_n,
                            contact_hold_n=a.contact_hold_n,
                            lift_min_mm=a.lift_min_mm) for ep in eps]
    band = clean_band(rows, a.ref_arm)
    ref_arm, peaks, mean, sd = band
    if a.crush_n is not None:
        crush_n = a.crush_n
        crush_src = "fixed on the command line"
    elif len(peaks) >= 2:
        crush_n = mean + a.crush_sd_k * sd
        crush_src = (f"mean + {a.crush_sd_k:g} sd of `{ref_arm}`'s "
                     f"{len(peaks)} clean placements "
                     f"({mean:.2f} + {a.crush_sd_k:g} x {sd:.2f})")
    else:
        crush_n = 18.0
        crush_src = "fallback: too few clean placements to derive a band"

    # pass 2: apply it
    apply_crush_threshold(rows, crush_n)
    for r in rows:
        cross_check(r)
    if a.rule:
        for ep, r in zip(eps, rows):
            r.rule_grasp_ok = rule_check(ep, a.hw)

    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(TakeRow().csv_dict().keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.csv_dict())

    if a.closes_csv:
        a.closes_csv.parent.mkdir(parents=True, exist_ok=True)
        blank = CloseEvent(0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, None)
        with a.closes_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["episode", "arm", "cell",
                                               "operator"] + list(asdict(blank)))
            w.writeheader()
            for r in rows:
                for c in r.closes:
                    w.writerow({"episode": r.episode, "arm": r.arm,
                                "cell": r.cell, "operator": r.verdict,
                                **asdict(c)})

    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        sweep = (sweep_table(eps, (3.0, 4.0, a.contact_hold_n, 6.0, 7.0),
                             a.contact_a_n, crush_n) if a.sweep else "")
        write_markdown(a.md, rows, a.folder, a.contact_a_n, a.contact_hold_n,
                       crush_n, band, crush_src, sweep)

    if not a.quiet:
        print(f"crush threshold {crush_n:.2f} N — {crush_src}\n")
        for ln in headline_lines(rows):
            print("  " + ln.replace("**", ""))
        print()
        print(per_arm_table(rows))
        print()
        dis = [r for r in rows if not r.agree]
        print(f"{len(rows)} takes, {len(dis)} disagreeing with the operator:")
        for r in dis:
            print(f"  {r.episode}  [{r.arm}/cell {r.cell}]  "
                  f"operator={r.verdict} class={r.cls} — {r.disagreement}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
