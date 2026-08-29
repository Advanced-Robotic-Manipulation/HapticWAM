"""Offline replay: run a checkpoint on the RIG'S OWN recorded states.

`terminal_eval` conditions every input on a demo that is already descending at
a demo-placed object, so it measures the terminal PRIOR, not perception-driven
commit (REVIEW_SYNTHESIS P1). This tool instead rebuilds, for every accepted
replan of a recorded deploy episode, the exact `ObsSnapshot` that
`SnapshotBuilder.build()` saw at that instant (`t = trace[i]["t"] +
meta.clock_calibration["offset"]`, the alignment tools/rig_trace_decompose.py
uses), pushes it through the SAME `PhantomPolicy._batch_from_obs` +
`rf.sample` path deploy uses, and compares K fresh samples against the chunk
the rig actually played.

Per replan it reports head_dz (cum z over steps 0-8, mm), tail_dz, chunk_dz,
close_step (first step with gripper > 0.45), grip_max, the across-seed std,
the trace's own values, and whether the trace falls inside the K-seed spread
(E0 / GATE G0: if the replay cannot reproduce the rig, nothing downstream is
interpretable — the recorded episodes carry `seed:none`, so the trace can only
be checked against the spread).

E0 on compute3 (5090), one day of rig episodes:

    cd ~/phantom-icra-2027 && python tools/replay_rig.py \
        --ckpt runs/teacher_v5_batch0822/v5_6.pt \
        --hardware configs/hardware.nuc.yaml \
        --episodes data/episodes/deploy/20260828/ep_* \
        --seeds 8 --persistent-noise --prev-chunk proposal --prev-cpk chained \
        --out runs/replay_e0.json

E3 (is the intent loop live?) is the same command with
`--prev-chunk measured` / `--prev-chunk zeros`; E2 sweeps `--nfe`.
CPU smoke (no weights, random tiny backbone): add `--tiny` and drop `--ckpt`.

Known fidelity gaps, all in the replay's favour to state plainly:
 - the K seeds denoise as ONE batch of K, so a seed is "an independent noise
   draw", not the rig's (unrecorded) seed; `--persistent-noise` holds that
   K-batch draw across the episode the way deploy holds its single draw;
 - deploy stamps `snap.t` BEFORE it reads the rings, so the rebuilt arm row
   can be one 125 Hz sample older than the one the rig used;
 - `reactive` is 0 at the first replan here; on the rig the ring warm-up had
   already primed `SnapshotBuilder._prev_fields`;
 - `--prev-chunk proposal` replays the TRACE's previous chunk (what deploy
   conditioned on), not this run's own previous sample, so a replan's
   conditioning never drifts away from the recorded episode.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader
from phantom.data.schema import (STREAM_ARM_FT, STREAM_ARM_Q, STREAM_ARM_QD,
                                 STREAM_ARM_TCP_POSE, STREAM_ARM_TCP_SPEED,
                                 STREAM_CAMERA_SCENE, STREAM_GRIPPER,
                                 NormStats, tactile_stream)
from phantom.inference.policy import ObsSnapshot, PhantomPolicy, Plan
from phantom.train import common as C
from phantom.train.builder import build_model

log = logging.getLogger("replay_rig")

CLOSE_THR = 0.45          # same aperture rule as terminal_eval / close_index
HEAD_STEP = 8             # "head" = cum z over steps 0..8 of the 16-step chunk


class RigEpisode:
    """Recorded deploy episode: zarr streams + planner_trace.json.

    Row selection reproduces `SnapshotBuilder.build()`: the LATEST row with
    ts <= t (deploy read `ring.latest(1)`), never the nearest row. The one
    unavoidable slack is that deploy stamped `snap.t` BEFORE reading the
    rings, so a sample that landed during the read could be one row newer than
    what this rebuild picks; at 125 Hz that is <= 8 ms of arm state.
    """

    def __init__(self, path: Path, hw):
        self.path = Path(path)
        self.hw = hw
        self.reader = EpisodeReader(self.path)
        self.meta = self.reader.meta
        self.trace = json.loads((self.path / "planner_trace.json").read_text())
        self.offset = float((self.meta.clock_calibration or {}).get("offset", 0.0))
        self._ts: dict[str, np.ndarray] = {}

    def ts(self, stream: str) -> np.ndarray:
        t = self._ts.get(stream)
        if t is None:
            t = self.reader.ts(stream)
            self._ts[stream] = t
        return t

    def t_master(self, i: int) -> float:
        """Zarr-clock time of replan i (rig_trace_decompose.py:41,60)."""
        return float(self.trace[i]["t"]) + self.offset

    def last_leq(self, stream: str, t: float) -> int:
        return int(np.clip(np.searchsorted(self.ts(stream), t, side="right") - 1,
                           0, len(self.ts(stream)) - 1))

    def nearest(self, stream: str, t: float) -> int:
        ts = self.ts(stream)
        i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        if i > 0 and (ts[i] - t) >= (t - ts[i - 1]):
            i -= 1
        return i

    def row(self, stream: str, t: float) -> np.ndarray:
        return np.asarray(self.reader.data(stream)[self.last_leq(stream, t)])

    # ------------------------------------------------------------------
    def snapshot(self, t: float, prev_fields: np.ndarray | None,
                 *, teacher: bool) -> tuple[ObsSnapshot, np.ndarray | None]:
        """The ObsSnapshot deploy built at zarr time `t` (planner.py:77-185).

        Returns (snapshot, fields_ds stack) — the stack is the next replan's
        `_prev_fields`, which is how deploy derives `reactive`.
        """
        hw = self.hw
        rgb = self.row(STREAM_CAMERA_SCENE, t)          # JPEG decoded by the reader

        # wrist F/T window: anchored at the newest arm SAMPLE time, resampled
        # onto linspace(t_ft - window_s, t_ft, window_len) with np.interp
        ts_a = self.ts(STREAM_ARM_FT)
        j = self.last_leq(STREAM_ARM_FT, t)
        t_ft = float(ts_a[j])
        lo = max(0, int(np.searchsorted(ts_a, t_ft - 2.0 * hw.wrist_ft.window_s)) - 1)
        ft = np.asarray(self.reader.data(STREAM_ARM_FT)[lo:j + 1], dtype=np.float64)
        grid = np.linspace(t_ft - hw.wrist_ft.window_s, t_ft, hw.wrist_ft.window_len)
        wrist = np.stack([np.interp(grid, ts_a[lo:j + 1], ft[:, k])
                          for k in range(ft.shape[1])], axis=-1).astype(np.float32)

        ur_state = np.concatenate([
            self.row(STREAM_ARM_Q, t), self.row(STREAM_ARM_QD, t),
            self.row(STREAM_ARM_TCP_POSE, t), self.row(STREAM_ARM_TCP_SPEED, t),
            self.row(STREAM_GRIPPER, t),
        ]).astype(np.float32)

        snap = ObsSnapshot(t=t, rgb=rgb, wrist_window=wrist, ur_state=ur_state)
        if not teacher:
            return snap, None

        dt_field = 1.0 / hw.recording.field_ds_rate_hz
        fields, gels, contact, ds_now = [], [], [], []
        for s in hw.tactile.sensors:
            fields.append(np.asarray(self.row(tactile_stream(s.name, "keyframes"), t),
                                     dtype=np.float32))
            gels.append(self.row(tactile_stream(s.name, "infer_img"), t))
            fs = tactile_stream(s.name, "fields_ds")
            i = self.last_leq(fs, t)
            data = self.reader.data(fs)
            cur = np.asarray(data[i], dtype=np.float32)
            prev = np.asarray(data[max(i - 1, 0)], dtype=np.float32)
            d = dv.derive_timestep(cur, prev, dt_field, hw)
            contact.append(np.concatenate([
                self.row(tactile_stream(s.name, "wrench"), t),
                [self.row(tactile_stream(s.name, "area"), t)],
                np.nan_to_num(d["cop"], nan=0.0), [d["slip"]], [d["mask_frac"]],
            ]).astype(np.float32))
            ds_now.append(cur)
        snap.fields = np.stack(fields)
        snap.gel = np.stack(gels)
        snap.contact_state = np.stack(contact)
        ds_now = np.stack(ds_now)
        if prev_fields is not None:
            snap.reactive = dv.reactive_score(ds_now, prev_fields)
        return snap, ds_now

    # ------------------------------------------------------------------
    def measured_prev_chunk(self, t: float) -> np.ndarray:
        """The prev_chunk TRAINING saw: 16 measured Δ-TCP rows on the action
        grid ending at t (windows.py:277 reads the recorded `actions` stream,
        which data_collect/session.py:733-750 writes as
        `pose_delta(prev_tcp, tcp)` of the MEASURED pose plus the gripper
        command). Reconstructed here from arm_tcp_pose + gripper so it works
        on deploy episodes too, where `actions` holds executor COMMANDS."""
        hw = self.hw
        H, rate = hw.control.chunk_horizon, hw.control.action_rate_hz
        pose = self.reader.data(STREAM_ARM_TCP_POSE)
        out = np.zeros((H, hw.control.action_dim), dtype=np.float32)
        for k in range(H):
            g = t - (H - k) / rate
            cur = np.asarray(pose[self.nearest(STREAM_ARM_TCP_POSE, g)])
            prv = np.asarray(pose[self.nearest(STREAM_ARM_TCP_POSE, g - 1.0 / rate)])
            out[k, :6] = dv.pose_delta(prv, cur)
            out[k, 6] = float(np.asarray(
                self.reader.data(STREAM_GRIPPER)[self.nearest(STREAM_GRIPPER, g)])[0])
        return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def chunk_metrics(a: np.ndarray) -> dict:
    """a: (H, 7) DENORMALIZED chunk (Δ-EE metres + gripper)."""
    cum = np.cumsum(a[:, :3], axis=0) * 1000.0
    grip = a[:, 6]
    hit = np.nonzero(grip > CLOSE_THR)[0]
    return {"head_dz": float(cum[HEAD_STEP, 2]),
            "tail_dz": float(cum[-1, 2] - cum[HEAD_STEP, 2]),
            "chunk_dz": float(cum[-1, 2]),
            "head_dxy": float(np.linalg.norm(cum[HEAD_STEP, :2])),
            "close_step": float(hit[0]) if len(hit) else float(len(grip)),
            "grip_max": float(grip.max())}


KEYS = ("head_dz", "tail_dz", "chunk_dz", "head_dxy", "close_step", "grip_max")


def summarize(seed_metrics: list[dict], trace_m: dict) -> dict:
    row: dict = {}
    for k in KEYS:
        v = np.array([m[k] for m in seed_metrics], dtype=np.float64)
        row[k] = float(v.mean())
        row[f"{k}_std"] = float(v.std())
        row[f"trace_{k}"] = trace_m[k]
    hz = np.array([m["head_dz"] for m in seed_metrics])
    row["head_dz_err"] = float(trace_m["head_dz"] - hz.mean())
    row["trace_in_spread"] = bool(hz.min() <= trace_m["head_dz"] <= hz.max())
    return row


# ---------------------------------------------------------------------------
# policy + sampling
# ---------------------------------------------------------------------------

def build_policy(args, hw) -> PhantomPolicy:
    paths = load_paths()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if dev == "cpu" else torch.bfloat16
    if args.tiny:
        pm = build_model(hw, paths, student=False, tiny=True, load_base=False,
                         device=dev, dtype=dt)
        norm = NormStats.identity()
    else:
        from phantom.config.model import PhantomModelConfig
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = PhantomModelConfig.from_dict(payload["configs"]["model"])
        pm = build_model(hw, paths, student=mc.student, mc=mc, device=dev, dtype=dt,
                         inference=True)
        # deploy loads EMA (parity fix 2026-08-27); the replay must too
        C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw, load_ema=True,
                                  payload=payload)
        ns = payload["norm_stats"]
        norm = NormStats(
            mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
            std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})
    return PhantomPolicy(pm, norm, nfe=args.nfe, guidance=args.guidance,
                         persistent_noise=args.persistent_noise)


def _tile(batch: dict, k: int) -> dict:
    """B=1 -> B=k, so all seeds denoise in ONE forward (independent noise per
    batch element; rf.sample draws randn over the whole x0)."""
    out = {}
    for key, v in batch.items():
        if torch.is_tensor(v):
            out[key] = v.repeat(k, *([1] * (v.dim() - 1)))
        elif isinstance(v, list):
            out[key] = list(v) * k
        else:
            out[key] = v
    return out


def replay_episode(policy: PhantomPolicy, ep: RigEpisode, args) -> dict:
    hw = policy.hw
    teacher = not policy.pm.layout.student
    # the rig recorded its own condition per replan (provenance, 2026-08-20):
    # sample at the requested nfe/guidance but say so when they differ
    diag = (ep.trace[0].get("diag") or {}) if ep.trace else {}
    if diag and (diag.get("nfe"), diag.get("guidance")) != (args.nfe, args.guidance):
        log.warning("%s was RECORDED at nfe=%s guidance=%s; replaying at "
                    "nfe=%s guidance=%s", ep.path.name, diag.get("nfe"),
                    diag.get("guidance"), args.nfe, args.guidance)
    policy.nfe, policy.guidance = int(args.nfe), float(args.guidance)
    policy.task_text = ep.meta.text or ep.meta.task
    policy.reset_episode()
    prev_fields = None
    prev_cpk = None
    prev_actions = None          # last ACCEPTED trace chunk (deploy's prev_plan)
    rows = []
    for i, r in enumerate(ep.trace):
        t = ep.t_master(i)
        try:
            snap, prev_fields = ep.snapshot(t, prev_fields, teacher=teacher)
        except (KeyError, IndexError, FileNotFoundError) as e:
            log.warning("%s replan %d: cannot rebuild the snapshot (%s)",
                        ep.path.name, i, e)
            continue
        if not r.get("accepted", True):
            continue             # deploy advances its feedback state only on accepts
        if args.prev_chunk == "proposal":
            pa = prev_actions
        elif args.prev_chunk == "measured":
            pa = ep.measured_prev_chunk(t) if i else None
        else:
            pa = None
        stub = None if pa is None else Plan(
            t_created=t, t0_pose=snap.ur_state[2 * hw.arm.dof:2 * hw.arm.dof + 6],
            actions=np.asarray(pa, dtype=np.float32),
            action_times=np.zeros(1), sigma=np.zeros(1), gate=0.0,
            p_evt=np.zeros(1), cpk=None)
        batch = _tile(policy._batch_from_obs(snap, stub), args.seeds)
        with torch.no_grad():
            pred = policy.rf.sample(
                batch, nfe=policy.nfe, guidance_scale=policy.guidance,
                prev_cpk=prev_cpk if args.prev_cpk == "chained" else None,
                reuse_noise=policy.persistent_noise)
        acts = [np.asarray(policy.norm.denormalize(
            "action", pred.actions_B_H_A[k].float().cpu()), dtype=np.float64)
            for k in range(args.seeds)]
        prev_cpk = pred.cpk.detach()
        prev_actions = np.asarray(r["actions"], dtype=np.float32)
        row = summarize([chunk_metrics(a) for a in acts],
                        chunk_metrics(prev_actions))
        row.update(episode=ep.path.name, replan=i, t=t, gate=r.get("gate"),
                   latency_s=r.get("latency_s"))
        rows.append(row)
        log.info("%s replan %2d: head_dz %7.1f +-%5.1f mm (trace %7.1f, in-spread %s) "
                 "close_step %.1f grip_max %.2f", ep.path.name, i, row["head_dz"],
                 row["head_dz_std"], row["trace_head_dz"], row["trace_in_spread"],
                 row["close_step"], row["grip_max"])
    return {"episode": ep.path.name, "task": ep.meta.task,
            "success": ep.meta.success, "n_replans": len(rows),
            "nfe": policy.nfe, "guidance": policy.guidance, "rows": rows,
            "summary": episode_summary(rows)}


SUMMARY_KEYS = ("head_dz", "head_dz_std", "trace_head_dz", "chunk_dz",
                "close_step", "grip_max", "trace_in_spread")


def episode_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {k: float(np.mean([r[k] for r in rows])) for k in SUMMARY_KEYS}
    out["abs_head_dz_err"] = float(np.mean([abs(r["head_dz_err"]) for r in rows]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt")
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--nfe", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--persistent-noise", action="store_true")
    ap.add_argument("--prev-chunk", choices=("proposal", "measured", "zeros"),
                    default="proposal")
    ap.add_argument("--prev-cpk", choices=("chained", "none"), default="chained")
    ap.add_argument("--tiny", action="store_true",
                    help="random tiny backbone, identity norm stats — CPU smoke "
                         "test of the replay plumbing, NOT an evaluation")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert args.ckpt or args.tiny, "--ckpt is required unless --tiny"

    hw = load_hardware(args.hardware, quiet=True)
    policy = build_policy(args, hw)
    policy.rf._gen = torch.Generator().manual_seed(args.seed)

    out = {"ckpt": args.ckpt or "tiny", "hardware": args.hardware,
           "seeds": args.seeds, "nfe": args.nfe, "guidance": args.guidance,
           "prev_chunk": args.prev_chunk, "prev_cpk": args.prev_cpk,
           "persistent_noise": bool(args.persistent_noise), "episodes": []}
    for path in args.episodes:
        p = Path(path)
        if not (p / "planner_trace.json").exists():
            log.warning("%s: no planner_trace.json — skipped", p)
            continue
        out["episodes"].append(replay_episode(policy, RigEpisode(p, hw), args))

    print(f"\n{'episode':44s} {'n':>3s} {'head_dz':>8s} {'seed_sd':>8s} "
          f"{'trace_dz':>8s} {'|err|':>7s} {'in_spr':>6s} {'close':>6s} {'grip':>5s}")
    for e in out["episodes"]:
        s = e["summary"]
        if not s:
            print(f"{e['episode']:44s}   0   (no accepted replans replayed)")
            continue
        print(f"{e['episode']:44s} {e['n_replans']:3d} {s['head_dz']:8.1f} "
              f"{s['head_dz_std']:8.1f} {s['trace_head_dz']:8.1f} "
              f"{s['abs_head_dz_err']:7.1f} {s['trace_in_spread']:6.2f} "
              f"{s['close_step']:6.1f} {s['grip_max']:5.2f}")
    rows = [r for e in out["episodes"] for r in e["rows"]]
    if rows:
        out["summary"] = episode_summary(rows)
        print("\nOVERALL " + json.dumps(out["summary"]))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
