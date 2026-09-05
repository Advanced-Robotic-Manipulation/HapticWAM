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
interpretable).

E0 on compute3 (5090), one day of rig episodes:

    cd ~/phantom-icra-2027 && python tools/replay_rig.py \
        --ckpt runs/teacher_v5_batch0822/v5_6.pt \
        --hardware configs/hardware.nuc.yaml \
        --episodes data/episodes/deploy/20260828/ep_* \
        --seeds 8 --persistent-noise --prev-chunk proposal --prev-cpk chained \
        --merge-lora --out runs/replay_e0.json

`--merge-lora` belongs in that line: `run_deploy.build_policy` ALWAYS folds the
LoRA into the bf16 base weights, so without it E0 replays a numerically
different network from the rig.

E3 (is the intent loop live?) is the same command with
`--prev-chunk measured` / `--prev-chunk zeros`; E2 sweeps `--nfe`.
Session-4 **Arm B** episodes (`parity:on` in `meta.tags`) must be replayed with
`--parity-fixes`, which reproduces all four deploy parity switches (below);
without it the replay silently conditions Arm B on the legacy intent channel —
the very channel E3/P2 measures. The tag is checked and a mismatch is a loud
warning.
CPU smoke (no weights, random tiny backbone): add `--tiny`; `--tiny --ckpt
<tiny.pt>` loads that checkpoint's weights and norm stats into the tiny
backbone instead of running random init with identity norms.

SEEDING (2026-08-30). Every episode is reseeded before it is replayed, so a
result never depends on the position of an episode in `--episodes` nor on
which other episodes were dropped:

  default            seed = --seed + a stable per-EPISODE offset derived from
                     the episode directory name (never the list index)
  --seed-from-meta   seed = the `seed:<n>` tag `run_deploy` records per episode
                     since ba61354. This reproduces the rig's ACTUAL noise draw
                     for a post-fix episode exactly (manual_seed(n) +
                     reset_episode_noise(), no warm-up replan), so the trace can
                     be checked against the SAMPLE rather than only against the
                     K-seed spread. `tools/replay_deploy_path.py --deploy-rng`
                     remains the way to reproduce a PRE-fix trace (2026-08-29
                     and earlier), where the rig ran the generator's
                     constructor seed through one warm-up replan and no
                     `seed:<n>` tag exists.
                     K comes from the episode's own `kseeds:<K>` tag: deploy
                     hands rf.sample a B=1 batch and expands to K INSIDE it, so
                     the replay does the same (`k_seeds=K`, not a pre-tiled
                     B=K batch — build_x0 draws at B, so tiling first shifts the
                     whole stream and reproduces nothing). The trace is then
                     scored against row `diag.k_pick`, the chunk the rig's
                     selector actually executed (`k_pick != 0` in 310/400
                     recorded replans), while the K-spread is still reported for
                     E2. Passing a `--seeds` that disagrees with `kseeds:<K>`
                     is refused.

--parity-fixes reproduces `SnapshotBuilder(parity_fixes=True)` +
`PhantomPolicy.replan`'s parity branch, by CALLING the deploy code rather than
restating it (`SnapshotBuilder.prev_chunk_from_history`):
 1. prev_chunk = the measured/executed past (`--prev-chunk measured`, the
    default under this flag) instead of the policy's own last proposal;
 2. contact-state dt = the MEASURED fields_ds inter-frame dt;
 3. reactive = the two CONSECUTIVE fields_ds frames at t, not the last two
    replans;
 4. prev_cpk_step = round(previous replan's latency / latent_dt).

Known fidelity gaps, all in the replay's favour to state plainly:
 - without `--seed-from-meta` the batch is pre-tiled to B=K, so build_x0 draws
   at B=K and a seed is "an independent noise draw", not the rig's;
   `--persistent-noise` holds that K-batch draw across the episode the way
   deploy holds its single draw. WITH `--seed-from-meta` the expansion moves
   inside rf.sample (`k_seeds=K`) and this gap closes;
 - deploy stamps `snap.t` BEFORE it reads the rings, so the rebuilt arm row
   can be one 125 Hz sample older than the one the rig used;
 - `reactive` is 0 at the first replan without `--parity-fixes`; on the rig the
   ring warm-up had already primed `SnapshotBuilder._prev_fields`;
 - `--prev-chunk proposal` replays the TRACE's previous chunk (what deploy
   conditioned on), not this run's own previous sample, so a replan's
   conditioning never drifts away from the recorded episode;
 - on a `--terminal-veto` replan the trace's `actions` are the VETO's
   arithmetic, not a model sample. The comparison therefore prefers the
   trace's `actions_pre_veto` (recorded since F9) and flags the row
   `trace_vetoed` when it had to fall back to the rewritten chunk;
   conditioning still uses the post-veto chunk, which is what deploy carried
   forward as `prev_plan`.
"""

from __future__ import annotations

import argparse
import json
import logging
import zlib
from pathlib import Path

import numpy as np

from phantom.deploy.planner import VETO_REWRITE_ACTIONS
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ARM_FT, STREAM_ARM_Q,
                                 STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                                 STREAM_ARM_TCP_SPEED, STREAM_CAMERA_SCENE,
                                 STREAM_GRIPPER, NormStats, tactile_stream)
from phantom.inference.policy import (ObsSnapshot, PhantomPolicy, Plan,
                                      _cpk_row)
from phantom.train import common as C
from phantom.train.builder import build_model

log = logging.getLogger("replay_rig")

CLOSE_THR = 0.45          # same aperture rule as terminal_eval / close_index
HEAD_STEP = 8             # "head" = cum z over steps 0..8 of the 16-step chunk


def episode_seed(base: int, ep_name: str) -> int:
    """Sampling seed for ONE episode: `base` + a stable per-episode offset.

    The offset is a CRC32 of the episode directory name, not the episode's
    index in `--episodes`: replays are compared across checkpoints, and an
    invalid episode dropped for one checkpoint used to shift every later
    episode's noise (`main()` seeded once per RUN, so results depended on the
    list and its order — validation 2026-08-30). Same episode, same base seed,
    same numbers, whatever else is on the command line."""
    return int(base) + int(zlib.crc32(ep_name.encode()) % 1_000_003)


def meta_seed(meta) -> int | None:
    """The `seed:<n>` tag `run_deploy` writes per episode (ba61354), or None.

    `seed:none` (pre-fix, or `--seed` omitted before that commit) reads as
    None: there is no recorded draw to reproduce."""
    for tag in (meta.tags or []):
        if isinstance(tag, str) and tag.startswith("seed:"):
            v = tag.split(":", 1)[1]
            try:
                return int(v)
            except ValueError:
                return None
    return None


def meta_kseeds(meta) -> int | None:
    """`kseeds:<K>` from meta.tags (run_deploy:630), else None.

    K is the number of chunks the rig's selector actually sampled in ONE
    denoise; reproducing the executed chunk needs the same K (rf.build_x0
    draws at B=K, so a different K shifts the whole noise stream) and the
    per-replan `diag.k_pick` that says which of the K was executed."""
    for tag in (meta.tags or []):
        if isinstance(tag, str) and tag.startswith("kseeds:"):
            try:
                return max(1, int(tag.split(":", 1)[1]))
            except ValueError:
                return None
    return None


def meta_parity(meta) -> bool | None:
    """`parity:on|off` from meta.tags (run_deploy always tags it), else None."""
    for tag in (meta.tags or []):
        if tag == "parity:on":
            return True
        if tag == "parity:off":
            return False
    return None


class RigEpisode:
    """Recorded deploy episode: zarr streams + planner_trace.json.

    Row selection reproduces `SnapshotBuilder.build()`: the LATEST row with
    ts <= t (deploy read `ring.latest(1)`), never the nearest row. The one
    unavoidable slack is that deploy stamped `snap.t` BEFORE reading the
    rings, so a sample that landed during the read could be one row newer than
    what this rebuild picks; at 125 Hz that is <= 8 ms of arm state.
    """

    jpeg_quality: int | None = None      # set by main() from --jpeg-quality
    # set by main() from the CHECKPOINT (configs.train.wrench_baseline_rows):
    # the per-episode wrench zero offset the model was trained without
    wrench_baseline_rows: int = 0

    def __init__(self, path: Path, hw):
        self.path = Path(path)
        self.hw = hw
        self._wbase: dict[str, np.ndarray] = {}

        self.reader = EpisodeReader(self.path)
        self.meta = self.reader.meta
        self.trace = json.loads((self.path / "planner_trace.json").read_text())
        self.offset = float((self.meta.clock_calibration or {}).get("offset", 0.0))
        self._ts: dict[str, np.ndarray] = {}
        self._sb = None
        self._poses: np.ndarray | None = None

    def wrench_baseline(self, stream: str) -> np.ndarray:
        """Same statistic as WindowSampler._EpisodeCache.wrench_baseline:
        median of the episode's first N wrench rows (0 rows -> zeros)."""
        n = int(self.wrench_baseline_rows or 0)
        if n <= 0:
            return np.zeros(6, np.float32)
        if stream not in self._wbase:
            w = np.asarray(self.reader.data(stream)[:n], dtype=np.float32)
            self._wbase[stream] = (np.median(w, axis=0).astype(np.float32)
                                   if len(w) else np.zeros(6, np.float32))
        return self._wbase[stream]

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
                 *, teacher: bool, parity: bool = False
                 ) -> tuple[ObsSnapshot, np.ndarray | None]:
        """The ObsSnapshot deploy built at zarr time `t` (planner.py:77-185).

        Returns (snapshot, fields_ds stack) — the stack is the next replan's
        `_prev_fields`, which is how deploy derives `reactive` WITHOUT
        `--parity-fixes`.

        `parity=True` reproduces `SnapshotBuilder(parity_fixes=True)`: the
        measured contact-state dt, the consecutive-frame `reactive`, and the
        measured/executed `prev_chunk` on the snapshot itself (which is what
        `PhantomPolicy._batch_from_obs` prefers over `prev_plan.actions`).
        """
        hw = self.hw
        rgb = self.row(STREAM_CAMERA_SCENE, t)          # JPEG decoded by the reader
        if self.jpeg_quality is not None:
            # image-fragility probe: the rig saw the RAW ring frame, the
            # recording is a JPEG re-encode; re-encoding again at quality Q
            # measures how much the chunk moves per unit of image degradation
            import cv2
            ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
            assert ok
            rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()

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
        if parity:
            # deploy sets this inside build() under --parity-fixes; the policy
            # then prefers it over prev_plan.actions
            snap.prev_chunk = self.measured_prev_chunk(t)
        if not teacher:
            return snap, None

        dt_field = 1.0 / hw.recording.field_ds_rate_hz
        fields, gels, contact, ds_now, ds_prev = [], [], [], [], []
        for s in hw.tactile.sensors:
            fields.append(np.asarray(self.row(tactile_stream(s.name, "keyframes"), t),
                                     dtype=np.float32))
            gels.append(self.row(tactile_stream(s.name, "infer_img"), t))
            fs = tactile_stream(s.name, "fields_ds")
            i = self.last_leq(fs, t)
            data = self.reader.data(fs)
            cur = np.asarray(data[i], dtype=np.float32)
            prev = np.asarray(data[max(i - 1, 0)], dtype=np.float32)
            dt_use = dt_field
            if parity and i >= 1:
                # planner.py: under parity training's MEASURED inter-frame dt
                # is used, and derive_timestep divides the tangential flow by it
                ts_f = self.ts(fs)
                dt_use = float(max(float(ts_f[i]) - float(ts_f[i - 1]), 1e-6))
            d = dv.derive_timestep(cur, prev, dt_use, hw)
            wst = tactile_stream(s.name, "wrench")
            contact.append(np.concatenate([
                self.row(wst, t) - self.wrench_baseline(wst),
                [self.row(tactile_stream(s.name, "area"), t)],
                np.nan_to_num(d["cop"], nan=0.0), [d["slip"]], [d["mask_frac"]],
            ]).astype(np.float32))
            ds_now.append(cur)
            ds_prev.append(prev)
        snap.fields = np.stack(fields)
        snap.gel = np.stack(gels)
        snap.contact_state = np.stack(contact)
        ds_now = np.stack(ds_now)
        if parity:
            # training's reactive: the two CONSECUTIVE fields_ds frames at t
            # (~1/30 s apart), not the last two REPLANS (~1 s apart)
            snap.reactive = dv.reactive_score(ds_now, np.stack(ds_prev))
        elif prev_fields is not None:
            snap.reactive = dv.reactive_score(ds_now, prev_fields)
        return snap, ds_now

    # ------------------------------------------------------------------
    def _builder(self):
        """The DEPLOY `SnapshotBuilder`, reused for the parity `prev_chunk`.

        Constructed with `session=None` on purpose: `prev_chunk_from_history`
        touches only `self.hw` and `self.executor`, and calling deploy's own
        method is the point — a second implementation here is exactly how the
        replay drifted away from the arm it is supposed to score."""
        if self._sb is None:
            from phantom.deploy.planner import SnapshotBuilder
            self._sb = SnapshotBuilder(self.hw, None, "teacher",
                                       parity_fixes=True,
                                       executor=_RecordedGripperCmds(self))
        return self._sb

    def poses(self) -> np.ndarray:
        if self._poses is None:
            self._poses = np.asarray(self.reader.data(STREAM_ARM_TCP_POSE)[:],
                                     dtype=np.float64)
        return self._poses

    def measured_prev_chunk(self, t: float) -> np.ndarray | None:
        """The prev_chunk deploy builds under `--parity-fixes`, and the one
        TRAINING saw: 16 chained `pose_delta` rows of the MEASURED TCP on the
        action grid ending at t, plus the gripper COMMAND that was executed.

        Delegates to `SnapshotBuilder.prev_chunk_from_history` — the recorded
        streams stand in for the rings (`arm["tcp_pose"]` and the executor's
        gripper history, `_RecordedGripperCmds`). Returns None when the
        recording cannot supply two distinct arm rows, exactly as deploy does
        (the caller then keeps the normalized-zero first-replan conditioning).
        """
        grip_now = float(np.asarray(self.row(STREAM_GRIPPER, t))[0])
        out = self._builder().prev_chunk_from_history(
            t, self.ts(STREAM_ARM_TCP_POSE), {"tcp_pose": self.poses()}, grip_now)
        return None if out is None else np.asarray(out, dtype=np.float32)


class _RecordedGripperCmds:
    """`executor.gripper_cmd_at(times)` replayed off the recording.

    `ChunkExecutor` reports every newly-entered action-grid step into
    STREAM_ACTIONS (executor.py:233) and remembers its gripper command; the
    recorded rows ARE that history, so a zero-order hold over them is the same
    function. Without the stream (older episodes) returns None and
    `prev_chunk_from_history` substitutes the measured aperture."""

    def __init__(self, ep: "RigEpisode"):
        self.ep = ep

    def gripper_cmd_at(self, times) -> np.ndarray | None:
        if not self.ep.reader.has(STREAM_ACTIONS):
            return None
        ts = self.ep.ts(STREAM_ACTIONS)
        if not len(ts):
            return None
        g = np.asarray(self.ep.reader.data(STREAM_ACTIONS)[:], dtype=np.float64)[:, 6]
        times = np.asarray(times, dtype=np.float64)
        idx = np.searchsorted(ts, times, side="right") - 1
        return np.where(idx >= 0, g[np.clip(idx, 0, len(g) - 1)], np.nan)


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


def summarize(seed_metrics: list[dict], trace_m: dict,
              k_pick: int | None = None) -> dict:
    row: dict = {}
    for k in KEYS:
        v = np.array([m[k] for m in seed_metrics], dtype=np.float64)
        row[k] = float(v.mean())
        row[f"{k}_std"] = float(v.std())
        row[f"trace_{k}"] = trace_m[k]
    hz = np.array([m["head_dz"] for m in seed_metrics])
    row["head_dz_err"] = float(trace_m["head_dz"] - hz.mean())
    row["trace_in_spread"] = bool(hz.min() <= trace_m["head_dz"] <= hz.max())
    # per-seed values: the across-seed DISTRIBUTION is the point of E2 (mode
    # collapse / bimodality) and the rig's own unrecorded seed must be placed in it
    for k in ("head_dz", "chunk_dz", "close_step", "grip_max"):
        row[f"seed_{k}"] = [float(m[k]) for m in seed_metrics]
    # ... and, when the trace recorded WHICH of the K the selector executed
    # (diag.k_pick), score the trace against THAT row. The K-spread above stays:
    # it is E2's signal, but it is not the reproduction check.
    if k_pick is not None and 0 <= int(k_pick) < len(seed_metrics):
        kp = int(k_pick)
        row["k_pick"] = kp
        for k in KEYS:
            row[f"pick_{k}"] = float(seed_metrics[kp][k])
        row["head_dz_pick_err"] = float(trace_m["head_dz"]
                                        - seed_metrics[kp]["head_dz"])
    return row


# ---------------------------------------------------------------------------
# policy + sampling
# ---------------------------------------------------------------------------

def build_policy(args, hw) -> PhantomPolicy:
    paths = load_paths()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if dev == "cpu" else torch.bfloat16
    payload = None
    if args.ckpt:
        from phantom.config.model import PhantomModelConfig
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = PhantomModelConfig.from_dict(payload["configs"]["model"])
    if args.tiny:
        # --tiny --ckpt used to SILENTLY drop the checkpoint and run a random
        # backbone with identity norm stats (validation 2026-08-30). Honour it:
        # the tiny model is built from the CHECKPOINT'S model config, its
        # trainable state (LoRA + phantom modules) is loaded and its norm stats
        # are used, so the smoke run denormalizes like the run it stands in
        # for. The FROZEN base stays random init (load_base=False) — that is
        # what --tiny means, and it is why --tiny is a plumbing smoke test and
        # never an evaluation.
        pm = build_model(hw, paths, student=mc.student if payload else False,
                         tiny=True, load_base=False, mc=mc if payload else None,
                         device=dev, dtype=dt)
        norm = NormStats.identity()
        if payload is not None:
            try:
                C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw,
                                          load_ema=True, payload=payload)
            except Exception as e:                          # noqa: BLE001
                raise SystemExit(
                    f"--tiny --ckpt {args.ckpt}: the checkpoint does not fit "
                    f"the tiny backbone ({type(e).__name__}: {e}). --tiny is "
                    f"the CPU smoke path; drop --ckpt to run a random tiny "
                    f"backbone, or drop --tiny to evaluate this checkpoint.")
            norm = _norm_from(payload)
        if getattr(args, "merge_lora", False):
            log.warning("--merge-lora is ignored under --tiny (load_base=False, "
                        "so there is no base weight to fold into)")
    else:
        pm = build_model(hw, paths, student=mc.student, mc=mc, device=dev, dtype=dt,
                         inference=True)
        # deploy loads EMA (parity fix 2026-08-27); the replay must too
        C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw, load_ema=True,
                                  payload=payload)
        if getattr(args, "merge_lora", False):
            # deploy folds the LoRA into the bf16 base weights (run_deploy
            # build_policy); replaying with the same fold tests whether that
            # cast is what separates the rig from the unmerged replay
            from phantom.backbone import loader as bl
            bl.merge_lora(pm.rf.net)
        norm = _norm_from(payload)
    policy = PhantomPolicy(pm, norm, nfe=args.nfe, guidance=args.guidance,
                         persistent_noise=args.persistent_noise,
                         parity_fixes=bool(getattr(args, "parity_fixes", False)))
    from phantom.train.common import wrench_baseline_rows_of
    policy.wrench_baseline_rows = wrench_baseline_rows_of(payload)
    return policy


def _norm_from(payload: dict) -> NormStats:
    ns = payload["norm_stats"]
    return NormStats(
        mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
        std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})


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


def seed_for_episode(ep: RigEpisode, args) -> tuple[int, str]:
    """(seed, provenance) for this episode — see the module docstring."""
    if args.seed_from_meta:
        n = meta_seed(ep.meta)
        if n is None:
            raise SystemExit(
                f"{ep.path.name}: --seed-from-meta, but meta.tags carries no "
                f"`seed:<n>` (tags={list(ep.meta.tags or [])}). Episodes "
                f"recorded before ba61354 have no recorded draw; replay them "
                f"without the flag (K-seed spread) or use "
                f"tools/replay_deploy_path.py --deploy-rng.")
        return int(n), "meta"
    return episode_seed(args.seed, ep.path.name), "name"


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
    # --seed-from-meta reproduces the rig's own draw, so it must also reproduce
    # the rig's own K: rf.build_x0 draws at B=K before the K-seed expansion, so
    # replaying a kseeds:4 episode at K!=4 shifts the whole noise stream and
    # reproduces nothing (validation 2026-08-31).
    rec_kseeds = meta_kseeds(ep.meta)
    seeds = int(args.seeds)
    use_k_seeds = False          # expand inside rf.sample, the way deploy does
    if args.seed_from_meta and rec_kseeds is not None:
        if getattr(args, "seeds_explicit", False) and seeds != rec_kseeds:
            raise SystemExit(
                f"{ep.path.name}: --seed-from-meta with --seeds {seeds}, but the "
                f"episode was recorded at `kseeds:{rec_kseeds}`. build_x0 draws "
                f"at B=K, so a different K reproduces nothing. Pass "
                f"--seeds {rec_kseeds} or drop --seeds and let the recorded tag "
                f"decide.")
        seeds = rec_kseeds
        use_k_seeds = True
    elif args.seed_from_meta and seeds > 1:
        log.warning("%s: --seed-from-meta with --seeds %d and no `kseeds:` tag "
                    "— the K seeds are drawn as one batch, so no element is the "
                    "rig's recorded draw. Pass --seeds 1.", ep.path.name, seeds)
    rec_parity = meta_parity(ep.meta)
    if rec_parity is not None and rec_parity != bool(args.parity_fixes):
        log.warning("%s was RECORDED with parity:%s but is being replayed with "
                    "--parity-fixes %s — the replay is conditioning on a "
                    "DIFFERENT intent channel than the rig did",
                    ep.path.name, "on" if rec_parity else "off",
                    "ON" if args.parity_fixes else "OFF")
    policy.nfe, policy.guidance = int(args.nfe), float(args.guidance)
    policy.task_text = ep.meta.text or ep.meta.task
    # Per-EPISODE seeding: the run used to seed once, before the loop, so every
    # episode after the first started wherever the previous one's replans left
    # the generator and the numbers moved with the --episodes list (validation
    # 2026-08-30). reset_episode() clears the held noise draw as well.
    seed, seed_src = seed_for_episode(ep, args)
    policy.rf._gen = torch.Generator().manual_seed(int(seed))
    policy.reset_episode()
    log.info("%s: seed %d (%s)", ep.path.name, seed,
             "recorded seed:<n>" if seed_src == "meta" else "base + episode name")
    prev_fields = None
    prev_cpk = None
    prev_actions = None          # last ACCEPTED trace chunk (deploy's prev_plan)
    prev_latency = None          # previous ACCEPTED replan's latency (prev_cpk_step)
    n_vetoed = 0
    rows = []
    for i, r in enumerate(ep.trace):
        t = ep.t_master(i)
        try:
            snap, prev_fields = ep.snapshot(t, prev_fields, teacher=teacher,
                                            parity=args.parity_fixes)
        except (KeyError, IndexError, FileNotFoundError) as e:
            log.warning("%s replan %d: cannot rebuild the snapshot (%s)",
                        ep.path.name, i, e)
            continue
        if not r.get("accepted", True):
            continue             # deploy advances its feedback state only on accepts
        pa = None
        if args.prev_chunk == "proposal":
            pa = prev_actions
        elif args.prev_chunk == "measured":
            # under --parity-fixes the snapshot already carries it, for EVERY
            # replan including the first (deploy builds it there too)
            pa = (snap.prev_chunk if snap.prev_chunk is not None
                  else (ep.measured_prev_chunk(t) if i else None))
        if args.prev_chunk != "measured":
            # snapshot(parity=True) set the measured chunk on the snapshot and
            # _batch_from_obs prefers it; the E3 swaps must still reach the batch
            snap.prev_chunk = None
        else:
            snap.prev_chunk = pa
        stub = None if pa is None else Plan(
            t_created=t, t0_pose=snap.ur_state[2 * hw.arm.dof:2 * hw.arm.dof + 6],
            actions=np.asarray(pa, dtype=np.float32),
            action_times=np.zeros(1), sigma=np.zeros(1), gate=0.0,
            p_evt=np.zeros(1), cpk=None)
        b1 = policy._batch_from_obs(snap, stub)
        # deploy takes ONE B=1 batch through build_x0 and expands to K inside
        # rf.sample. _tile expands FIRST, so build_x0 draws at B=K and the
        # generator stream no longer matches the rig's.
        k_arg = 1
        if use_k_seeds:
            batch, k_arg = b1, seeds
        else:
            batch = _tile(b1, seeds)
        # deploy's parity branch (policy.py) realigns which future step of the
        # one-replan-old package ACC summarises
        cpk_step = 0
        if args.parity_fixes and prev_cpk is not None and prev_latency:
            cpk_step = int(round(float(prev_latency) / policy.latent_dt))
        with torch.no_grad():
            pred = policy.rf.sample(
                batch, nfe=policy.nfe, guidance_scale=policy.guidance,
                prev_cpk=prev_cpk if args.prev_cpk == "chained" else None,
                prev_cpk_step=cpk_step,
                reuse_noise=policy.persistent_noise,
                k_seeds=k_arg)
        acts = [np.asarray(policy.norm.denormalize(
            "action", pred.actions_B_H_A[k].float().cpu()), dtype=np.float64)
            for k in range(seeds)]
        # which of the K the rig's selector actually executed (policy.py:234)
        k_pick = (r.get("diag") or {}).get("k_pick") if use_k_seeds else None
        if k_pick is not None and not (0 <= int(k_pick) < seeds):
            log.warning("%s replan %d: diag.k_pick=%s is out of range for K=%d",
                        ep.path.name, i, k_pick, seeds)
            k_pick = None
        # deploy feeds the NEXT replan the package of the chunk it SELECTED
        # (policy._cpk_row), keeping prev_cpk at B=1; chaining the whole K-batch
        # would grow the ACC input by a factor of K at every replan.
        prev_cpk = (_cpk_row(pred.cpk, int(k_pick or 0)) if use_k_seeds
                    else pred.cpk.detach())
        # CONDITIONING carries the chunk deploy actually kept as prev_plan —
        # post-veto, since _apply_veto rewrites plan.actions in place. The
        # COMPARISON instead wants the model's own sample: on a vetoed replan
        # trace["actions"] is the veto's arithmetic (a scripted aperture and a
        # zeroed z), so trace_in_spread against it is uninformative (G0).
        prev_actions = np.asarray(r["actions"], dtype=np.float32)
        pre_veto = r.get("actions_pre_veto")
        # _apply_veto writes a record on EVERY replan while the veto is on, so its
        # mere existence is not a rewrite: only close_masked / recovery_open touch
        # plan.actions (planner.py:606 writes actions_pre_veto on exactly those).
        veto_act = (r.get("terminal_veto") or {}).get("action")
        vetoed = veto_act in VETO_REWRITE_ACTIONS or pre_veto is not None
        trace_chunk = np.asarray(pre_veto if pre_veto is not None else r["actions"],
                                 dtype=np.float64)
        prev_latency = r.get("latency_s")
        row = summarize([chunk_metrics(a) for a in acts],
                        chunk_metrics(trace_chunk), k_pick=k_pick)
        if k_pick is not None:
            kp = int(k_pick)
            # the reproduction check proper: the executed chunk, element-wise
            row["pick_abs_err"] = float(np.abs(acts[kp] - trace_chunk).max())
            errs = [float(np.abs(a - trace_chunk).max()) for a in acts]
            row["best_seed"] = int(np.argmin(errs))
            row["best_abs_err"] = float(min(errs))
        row.update(episode=ep.path.name, replan=i, t=t, gate=r.get("gate"),
                   latency_s=r.get("latency_s"),
                   # what the trace_* columns were measured on
                   trace_source="actions_pre_veto" if pre_veto is not None else "actions",
                   trace_vetoed=bool(vetoed),
                   # ... and whether that comparison is interpretable at all
                   trace_comparable=bool(pre_veto is not None or not vetoed))
        if not row["trace_comparable"]:
            n_vetoed += 1
        rows.append(row)
        log.info("%s replan %2d: head_dz %7.1f +-%5.1f mm (trace %7.1f, in-spread %s) "
                 "close_step %.1f grip_max %.2f%s", ep.path.name, i, row["head_dz"],
                 row["head_dz_std"], row["trace_head_dz"], row["trace_in_spread"],
                 row["close_step"], row["grip_max"],
                 "" if row["trace_comparable"] else "  [VETOED chunk, trace_* is "
                 "the veto's arithmetic — no actions_pre_veto in this trace]")
    if n_vetoed:
        log.warning("%s: %d/%d replayed replans compare against a VETO-REWRITTEN "
                    "chunk (pre-F9 trace); their trace_in_spread is not a G0 "
                    "signal", ep.path.name, n_vetoed, len(rows))
    return {"episode": ep.path.name, "task": ep.meta.task,
            "success": ep.meta.success, "n_replans": len(rows),
            "nfe": policy.nfe, "guidance": policy.guidance,
            "seed": int(seed), "seed_source": seed_src,
            "seeds": int(seeds), "recorded_kseeds": rec_kseeds,
            "k_seeds_expansion": bool(use_k_seeds),
            "parity_fixes": bool(args.parity_fixes),
            "recorded_parity": rec_parity,
            "n_uncomparable_vetoed": n_vetoed, "rows": rows,
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
    ap.add_argument("--seeds", type=int, default=None,
                    help="K sampled chunks per replan (default 8). Under "
                         "--seed-from-meta this defaults to the episode's "
                         "recorded `kseeds:<K>` tag, and a value that "
                         "disagrees with that tag is refused")
    ap.add_argument("--persistent-noise", action="store_true")
    ap.add_argument("--prev-chunk", choices=("proposal", "measured", "zeros"),
                    default=None,
                    help="intent-channel source. Default: `measured` under "
                         "--parity-fixes (what deploy builds there), else "
                         "`proposal` (the legacy deploy behaviour)")
    ap.add_argument("--prev-cpk", choices=("chained", "none"), default="chained")
    ap.add_argument("--parity-fixes", action="store_true",
                    help="replay the Session-4 ARM B construction: measured "
                         "prev_chunk (via SnapshotBuilder itself), measured "
                         "contact dt, consecutive-frame reactive, and the "
                         "aligned prev_cpk_step. Required for episodes tagged "
                         "`parity:on`")
    ap.add_argument("--tiny", action="store_true",
                    help="tiny backbone — CPU smoke test of the replay "
                         "plumbing, NOT an evaluation. Without --ckpt: random "
                         "init + identity norm stats; with --ckpt: that "
                         "(tiny) checkpoint's weights and norm stats")
    ap.add_argument("--seed", type=int, default=1000,
                    help="BASE seed; each episode adds a stable offset derived "
                         "from its directory name, so results do not depend on "
                         "the --episodes list or its order")
    ap.add_argument("--seed-from-meta", action="store_true",
                    help="seed each episode from its recorded `seed:<n>` tag "
                         "instead — reproduces the rig's ACTUAL noise draw for "
                         "episodes recorded after ba61354. K comes from the "
                         "episode's `kseeds:<K>` tag and the expansion happens "
                         "inside rf.sample (as deploy does), so the executed "
                         "chunk is reproduced at row `diag.k_pick`. Refuses "
                         "episodes with no seed tag (use replay_deploy_path "
                         "--deploy-rng for pre-fix traces)")
    ap.add_argument("--jpeg-quality", type=int, default=None,
                    help="re-encode the scene frame at this JPEG quality (image-fragility probe)")
    ap.add_argument("--merge-lora", action="store_true",
                    help="fold LoRA into the base weights exactly as run_deploy does")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    RigEpisode.jpeg_quality = args.jpeg_quality
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert args.ckpt or args.tiny, "--ckpt is required unless --tiny"
    if args.prev_chunk is None:
        args.prev_chunk = "measured" if args.parity_fixes else "proposal"
    # K is per-EPISODE under --seed-from-meta (its `kseeds:<K>` tag), so remember
    # whether the user pinned it: an explicit disagreement is a hard failure,
    # silence means "take the recorded K". See replay_episode.
    args.seeds_explicit = args.seeds is not None
    if args.seeds is None:
        args.seeds = 8
    if args.seeds < 1:
        raise SystemExit("--seeds must be >= 1")

    hw = load_hardware(args.hardware, quiet=True)
    policy = build_policy(args, hw)
    RigEpisode.wrench_baseline_rows = int(getattr(policy, "wrench_baseline_rows", 0) or 0)
    # NB: no run-level seeding here — replay_episode seeds PER EPISODE, so the
    # numbers do not depend on the --episodes list or its order

    out = {"ckpt": args.ckpt or "tiny", "hardware": args.hardware,
           "seeds": args.seeds, "nfe": args.nfe, "guidance": args.guidance,
           "prev_chunk": args.prev_chunk, "prev_cpk": args.prev_cpk,
           "parity_fixes": bool(args.parity_fixes),
           "seed_base": args.seed, "seed_from_meta": bool(args.seed_from_meta),
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
