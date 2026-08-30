"""Terminal-phase offline eval: does the sampled chunk COMMIT to the grasp?

The openloop dir_cos metric is transport-dominated and could not see the
rig's 3-6 cm grasp miss (13/13 episodes closed 35-80 mm above grasp height).
This evaluates windows anchored in the last 1.5 s before the demo's first
gripper close, sampled exactly like deploy, and reports:

  endpoint_err_mm       |cum(pred dpos) - cum(gt dpos)| at chunk end (xyz)
  z_end_err_mm          pred z-at-end - gt z-at-end  (POSITIVE = ends HIGH)
  close_step_err        (first step pred gripper > 0.45) - (same for gt); +ve = late
  commit_ratio          |pred descent| / |gt descent| over the chunk
  pred_close_height_mm  absolute TCP z at the PREDICTED close (window-start z
                        + cum dz up to that step) — the rig's actual failure
                        mode (closing 65-120 mm high) expressed in mm

    python tools/terminal_eval.py --ckpt <pt> --data <root>/tasks \
        --hardware configs/hardware.nuc.yaml [--nfe 5] [--guidance 1.0]
Bars (from demos): endpoint < 15 mm, |z_end_err| < 10 mm, commit_ratio ~ 1.

Conditioning ablations (REVIEW_SYNTHESIS P1.2 / experiment E9) — `--null`.

THE GT CONTACT PACKAGE IS ZEROED IN EVERY MODE BUT `contact_gt`, INCLUDING THE
DEFAULT. This is the one thing the E9 table was read backwards on (validation
2026-08-30): `--null none` is not "the model gets everything", it is the DEPLOY
condition — `events` and every `cpk_*` batch key zeroed, so the CONTACT frames
are co-denoised from noise like every other generated frame. Two orthogonal
switches describe every mode:

  package   what the batch's GT contact package holds: ZEROED or KEPT
  frames    what the CONTACT frames do during denoising: CO-DENOISED from
            noise (free) or cond-PINNED (held at their x0 every step)

  mode          package  frames        what it measures
  none          zeroed   co-denoised   the deploy condition — the DEFAULT
  tactile       zeroed   co-denoised   gel / fields / contact_state / reactive
                                       zeroed: the streams the "sensor-free"
                                       student's layout drops. wrist F/T is
                                       deliberately KEPT (`source: ur_internal`,
                                       present in both arms of the contrast).
  wrist         zeroed   co-denoised   the wrist F/T window zeroed
  prev_cpk      zeroed   co-denoised   the PREVIOUS replan's package (the ACC
                                       intent channel) zeroed; the previous
                                       window's chunk itself untouched
  obs           zeroed   co-denoised   `_null_obs_batch` (gel/fields/
                                       contact_state/wrist/ur_state/reactive/
                                       text) plus a blacked video conditioning
                                       frame — the classifier-free null branch.
                                       Blacking the pixels is safe HERE only
                                       because sampling encodes frame 0 alone
                                       (build_x0(encode_gen=False)), so the
                                       causal-VAE bleed `_null_obs_batch`
                                       avoids cannot happen.
  contact_zero  zeroed   PINNED        P7's control arm: the CONTACT frames are
                                       pinned to the ZERO package, so the head
                                       is handed "no contact, and it is not
                                       allowed to imagine any". vs `none` this
                                       isolates PINNING; vs `contact_gt` it
                                       isolates the package's CONTENT — the
                                       "cond-pinned to GT **vs zeros**" pair
                                       REVIEW_SYNTHESIS P7 actually asked for.
  contact_gt    KEPT     PINNED        the OPPOSITE of a null and the P7 probe:
                                       privileged future contact, held through
                                       every denoise step. Deploy NEVER has
                                       this; a large shift away from `none`
                                       means the ACTION head leans on contact
                                       tokens it will not have.
  all           zeroed   co-denoised   obs + prev_cpk (every perceived channel
                                       and the intent package)

Every run records its own semantics in the JSON (`summary["null_semantics"]`),
so a table built from these files can never be relabelled by hand again.

Interpretation (REVIEW_SYNTHESIS §2 E9, gate G1b): if tactile-null lands on
top of the real run (|Δ endpoint| < 3 mm and |Δ close_step| < 1 step), the
tactile-teacher premise does not survive and the paper pivots.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.data.schema import NormStats
from phantom.data.windows import WindowSampler
from phantom.model.rf import PhantomRectifiedFlow
from phantom.model.sequence import FrameGroup
from phantom.train import common as C
from phantom.train.builder import build_model

NULL_MODES = ("none", "tactile", "wrist", "prev_cpk", "obs", "contact_zero",
              "contact_gt", "all")
METRICS = ("endpoint_err_mm", "z_end_err_mm", "commit_ratio", "close_step_err",
           "pred_close_height_mm")


def close_steps(gt_grip: np.ndarray, pr_grip: np.ndarray) -> tuple[int | None, int]:
    """(gt close step, pred close step) within one action chunk.

    The GT close is found with the training rule (close_index: absolute 0.45
    rule, max-relative fallback for wide grasps). The PREDICTION is then
    judged against the ABSOLUTE aperture the GT reached at its close (minus
    a 0.05 margin) — applying close_index's max-relative fallback to a
    16-step predicted chunk credited any >0.10 aperture rise as "closed"
    (delta review 2026-08-27). A prediction that never reaches that aperture
    scores len(chunk) (= "did not close in the chunk")."""
    from phantom.train.common import close_index
    gi = close_index(gt_grip)
    if gi is None:
        return None, len(pr_grip)
    thr = float(gt_grip[gi]) - 0.05
    hit = np.nonzero(pr_grip >= thr)[0]
    return int(gi), (int(hit[0]) if len(hit) else len(pr_grip))


# ---------------------------------------------------------------------------
# conditioning ablations
# ---------------------------------------------------------------------------

def null_batch(batch: dict, mode: str) -> dict:
    """Shallow copy of `batch` with the channels named by `mode` zeroed.

    Only what the model PERCEIVES is touched; targets (action_chunk) and the
    intent chunk (prev_chunk) are never nulled here. `prev_cpk` and
    `contact_gt` are not batch edits — see `nulls_prev_cpk` / `pins_contact`."""
    if mode not in NULL_MODES:
        raise SystemExit(f"--null {mode!r} not in {NULL_MODES}")
    out = dict(batch)

    def zero(*keys):
        for k in keys:
            if k in out and torch.is_tensor(out[k]):
                out[k] = torch.zeros_like(out[k])

    if mode in ("tactile",):
        # the streams the student layout has no frames for, PLUS `reactive`:
        # it is `derived.reactive_score` of two consecutive fields_ds frames,
        # i.e. a tactile-derived scalar. Leaving it live made "tactile nulled"
        # a partial null (validation 2026-08-30, F12).
        zero("gel", "fields", "contact_state", "reactive")
    if mode in ("wrist",):
        zero("wrist")
    if mode in ("obs", "all"):
        out = PhantomRectifiedFlow._null_obs_batch(out)
        zero("video")          # safe at sampling: only frame 0 is encoded
    return out


def nulls_prev_cpk(mode: str) -> bool:
    return mode in ("prev_cpk", "all")


def pins_contact(mode: str) -> bool:
    """Are the CONTACT frames cond-PINNED (held at their x0 every step)?

    True for both P7 arms: `contact_gt` pins them to the GT package,
    `contact_zero` pins them to the ZERO package. Every other mode leaves them
    co-denoised from noise, which is what deploy does."""
    return mode in ("contact_gt", "contact_zero")


def keeps_gt_package(mode: str) -> bool:
    """Does the batch keep the GT contact package (`events` + `cpk_*`)?

    ONLY `contact_gt`. Every other mode — the default `none` included — zeroes
    it, because deploy has no privileged future contact. Read `--null none`
    as "real observations, NO GT contact package", never as "everything on"."""
    return mode == "contact_gt"


def null_semantics(mode: str) -> dict:
    """The two orthogonal switches of `mode`, recorded into every JSON.

    The E9 premise table was published with `none` and `contact_gt` swapped in
    the reading (validation 2026-08-30); shipping the semantics beside the
    numbers makes that unrepeatable."""
    if mode not in NULL_MODES:
        raise SystemExit(f"--null {mode!r} not in {NULL_MODES}")
    frames = {True: "cond_pinned", False: "co_denoised_from_noise"}[pins_contact(mode)]
    pin_target = ("gt_package" if keeps_gt_package(mode) else
                  "zero_package") if pins_contact(mode) else None
    zeroed = sorted(_zeroed_keys(mode))
    return {"mode": mode,
            "gt_contact_package": "kept" if keeps_gt_package(mode) else "zeroed",
            "contact_frames": frames,
            "contact_frames_pinned_to": pin_target,
            "batch_streams_zeroed": zeroed,
            # under contact_zero the PREVIOUS window is sampled with the same
            # pinned-zero CONTACT frames, so the intent package it hands the
            # terminal window is zero by construction — part of the arm, not a
            # separate null
            "prev_cpk": ("zeroed" if nulls_prev_cpk(mode) else
                         "zero_by_pinning" if mode == "contact_zero" else
                         "as_sampled"),
            "is_deploy_condition": mode == "none"}


def _zeroed_keys(mode: str) -> set[str]:
    """Which OBSERVATION streams `null_batch` zeroes for `mode` (doc only)."""
    if mode == "tactile":
        return {"gel", "fields", "contact_state", "reactive"}
    if mode == "wrist":
        return {"wrist"}
    if mode in ("obs", "all"):
        return {"gel", "fields", "contact_state", "wrist", "ur_state",
                "reactive", "text", "video"}
    return set()


def zero_package(pkg):
    """Copy of a ContactPackage with every tensor field zeroed."""
    fields = {f.name: getattr(pkg, f.name) for f in dataclasses.fields(pkg)}
    return type(pkg)(**{k: (torch.zeros_like(v) if torch.is_tensor(v) else v)
                        for k, v in fields.items()})


def contact_pinned_layout(layout):
    """`layout` with the CONTACT frames added to the FRAME_REPLACE cond mask,
    i.e. held at their x0 through every denoise step.

    x0 is whatever the batch's `cpk_*` keys hold: the GT package under
    `--null contact_gt`, zeros under `--null contact_zero`."""
    class _ContactPinned(type(layout)):
        def cond_mask_T(self):
            m = super().cond_mask_T()
            m[self.frame_slice(FrameGroup.CONTACT)] = True
            return m
    vals = {f.name: getattr(layout, f.name) for f in dataclasses.fields(layout)}
    return _ContactPinned(**vals)


def ur_z_moments(ns: dict, z_idx: int) -> tuple[float, float]:
    """(mean, std) of the TCP-z channel of `ur_state`; identity if the
    checkpoint carries no ur_state stats (then ur_state is unnormalized)."""
    m, s = ns.get("mean", {}).get("ur_state"), ns.get("std", {}).get("ur_state")
    if m is None or s is None:
        return 0.0, 1.0
    return (float(np.asarray(m, dtype=np.float64)[z_idx]),
            float(np.asarray(s, dtype=np.float64)[z_idx]))


def resolve_episodes(data_root: Path, split: str) -> list[Path] | None:
    """Episode dirs for `split`, or a hard failure.

    `manifest_split` falls back to EVERY episode when the manifest is missing
    (common.py:370) — silently turning "val" into "train + val". A checkpoint
    decision made on that number is worthless, so refuse (REVIEW_SYNTHESIS
    P1). `--split all` is the deliberate opt-in."""
    if split == "all":
        return None
    mf = Path(data_root).parent / "manifests" / "all.jsonl"
    if not mf.exists():
        raise SystemExit(
            f"no manifest at {mf} — refusing to evaluate every episode under "
            f"{data_root} as if it were held out. Point --data at a "
            f"<root>/tasks whose ../manifests/all.jsonl exists, or pass "
            f"--split all to opt in deliberately.")
    eps = C.manifest_split(Path(data_root), split)
    if not eps:
        raise SystemExit(f"manifest {mf} has no {split!r} episodes")
    return eps


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _finite(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])


def _mean(rows, key):
    v = _finite(rows, key)
    return float(v.mean()) if len(v) else float("nan")


def _median(rows, key):
    v = _finite(rows, key)
    return float(np.median(v)) if len(v) else float("nan")


def _seed_std(rows, key):
    """Mean over windows of the ACROSS-SEED std of `key`.

    A window is (episode, t0); every seed contributed one row. With one seed
    this is 0 by construction and reported as NaN instead."""
    by_win: dict[tuple, list[float]] = {}
    for r in rows:
        v = r.get(key, np.nan)
        if np.isfinite(v):
            by_win.setdefault((r["episode"], round(float(r["t0"]), 4)), []).append(float(v))
    stds = [float(np.std(v, ddof=1)) for v in by_win.values() if len(v) > 1]
    return float(np.mean(stds)) if stds else float("nan")


def block(rows: list[dict]) -> dict:
    """mean / median / across-seed std for every metric over `rows`."""
    out = {}
    for k in METRICS:
        out[k] = _mean(rows, k)
        out["median_" + k] = _median(rows, k)
        out["seed_std_" + k] = _seed_std(rows, k)
    return out


def summarize(rows: list[dict], **meta) -> dict:
    tasks = sorted({r["task"] for r in rows})
    summary = {**meta, "n": len(rows),
               "n_windows": len({(r["episode"], round(float(r["t0"]), 4)) for r in rows}),
               "n_episodes": len({r["episode"] for r in rows})}
    summary.update(block(rows))
    summary["per_task"] = {t: {**block([r for r in rows if r["task"] == t]),
                              "n": sum(r["task"] == t for r in rows)}
                           for t in tasks}
    return summary


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--nfe", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lead-s", type=float, default=None,
                    help="anchor t0 this many seconds before the first close "
                         "(default: the chunk duration, so the chunk ENDS at "
                         "the close — no post-grasp lift inside the window)")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap the (task-sorted) window index; default is ALL")
    ap.add_argument("--split", default="val", choices=("val", "train", "all"),
                    help="manifest split; 'all' deliberately bypasses the manifest")
    ap.add_argument("--null", default="none", choices=NULL_MODES,
                    help="conditioning ablation. THE GT CONTACT PACKAGE IS "
                         "ZEROED IN EVERY MODE BUT contact_gt, THE DEFAULT "
                         "INCLUDED. none=deploy condition (real obs, package "
                         "zeroed, CONTACT frames co-denoised); "
                         "tactile=gel/fields/contact_state/reactive zeroed; "
                         "wrist=F/T window zeroed; prev_cpk=previous replan's "
                         "package zeroed; obs=classifier-free null + black "
                         "video; contact_zero=CONTACT frames cond-PINNED to "
                         "the ZERO package (P7's control arm); "
                         "contact_gt=package KEPT and CONTACT frames "
                         "cond-PINNED to it (privileged future contact, the "
                         "opposite of a null); all=obs+prev_cpk. Every run "
                         "writes summary['null_semantics'] saying exactly "
                         "this for the mode it ran.")
    ap.add_argument("--persistent-noise", action="store_true",
                    help="hold the initial noise draw fixed across the replans "
                         "of one window (deploy's reuse_noise)")
    ap.add_argument("--no-ema", dest="ema", action="store_false", default=True,
                    help="GO scripts deploy with --ema; match that by default")
    ap.add_argument("--tiny", action="store_true",
                    help="tiny random-init backbone (tests only)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    # say what this condition IS before any number is printed — the E9 table
    # was published with two of these rows read backwards (2026-08-30)
    print("null semantics: " + json.dumps(null_semantics(args.null)))

    dev, dt = ("cuda" if torch.cuda.is_available() else "cpu"), torch.bfloat16
    if dev == "cpu" or args.tiny:
        dt = torch.float32
    hw = load_hardware(args.hardware)
    paths = load_paths()
    from phantom.config.model import PhantomModelConfig
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = PhantomModelConfig.from_dict(payload["configs"]["model"])
    # student=mc.student, NOT a hardcoded False: builder.py:48 asserts the two
    # agree, so a student checkpoint (D9/D12 evaluate one offline) used to die
    # at "mc.student must agree with the student flag" (F12). `inference=True`
    # matches replay_rig.build_policy (activation checkpointing off).
    pm = build_model(hw, paths, student=mc.student, tiny=args.tiny, mc=mc,
                     device=dev, dtype=dt, inference=True)
    C.load_phantom_checkpoint(Path(args.ckpt), pm.rf, hw=hw, load_ema=args.ema, payload=payload)
    ns = payload["norm_stats"]
    a_mean = np.asarray(ns["mean"]["action"], dtype=np.float64)
    a_std = np.asarray(ns["std"]["action"], dtype=np.float64)
    z_idx = 2 * hw.arm.dof + 2          # q, qd, tcp_pose[x,y,Z], ... (hardware.py:505)
    z_mean, z_std = ur_z_moments(ns, z_idx)
    norm = NormStats(mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
                     std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})

    data_root = Path(args.data)
    # the sampler must build the layout's OWN streams: a student checkpoint has
    # no gel/fields/contact_state frames (F12)
    sampler = WindowSampler(hw, pm.bb, norm, student=mc.student, seed=0)
    val_eps = resolve_episodes(data_root, args.split)
    ds = C.WindowDataset(data_root, sampler, episodes=val_eps, windows_per_episode=1,
                         resample=False, seed=0)
    pm.rf.eval()
    if pins_contact(args.null):
        pm.rf.layout = contact_pinned_layout(pm.rf.layout)
    rows = []
    chunk_s = hw.control.chunk_horizon / hw.control.action_rate_hz
    lead = args.lead_s if args.lead_s is not None else chunk_s

    def make_batch(ep, t0):
        item = sampler.sample(ep, t0)
        batch = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).unsqueeze(0)
            elif isinstance(v, str):
                batch[k] = [v]
            elif isinstance(v, (int, float, np.floating)):
                batch[k] = torch.tensor([v])
            else:
                batch[k] = v
        batch = C.to_device(batch, dev, dt)
        if not keeps_gt_package(args.null):
            # no privileged future contact package — deploy never has it, so
            # THE DEFAULT (`--null none`) ZEROES IT TOO. `contact_zero` also
            # lands here and then pins the CONTACT frames to these zeros;
            # `contact_gt` is the only mode that keeps the GT package.
            for k in list(batch):
                if k == "events" or k.startswith("cpk_"):
                    batch[k] = torch.zeros_like(batch[k])
        return item, null_batch(batch, args.null)

    skipped = 0
    for wi in ds.index[: args.max_episodes]:
        tc = ds._close_time(wi.episode)
        if tc is None:
            skipped += 1
            continue
        t0 = tc - lead
        if not (wi.lo <= t0 <= wi.hi):
            skipped += 1           # chunk would not span the close — not a terminal window
            continue
        item, batch = make_batch(wi.episode, t0)
        gt = batch["action_chunk"][0].float().cpu().numpy().astype(np.float64) * a_std + a_mean
        # absolute TCP z at the window start, for pred_close_height_mm
        ur0 = item["ur_state"]
        ur0 = ur0.numpy() if torch.is_tensor(ur0) else np.asarray(ur0)
        z0 = float(np.asarray(ur0, dtype=np.float64)[z_idx] * z_std + z_mean)
        preds = []
        with torch.no_grad():
            for s in range(args.seeds):
                pm.rf._gen = torch.Generator().manual_seed(1000 + s)
                pm.rf.reset_episode_noise()
                # steady-state deploy path: every rig replan after the first
                # passes the TRUE previous package — reproduce it with a prior
                # window one chunk earlier (first-replan path otherwise)
                prev_cpk = None
                t_prev = t0 - chunk_s
                if t_prev >= wi.lo:
                    _, pb = make_batch(wi.episode, t_prev)
                    prev_cpk = pm.rf.sample(pb, nfe=args.nfe, guidance_scale=args.guidance,
                                            reuse_noise=args.persistent_noise).cpk
                    if nulls_prev_cpk(args.null):
                        prev_cpk = zero_package(prev_cpk)
                p = pm.rf.sample(batch, nfe=args.nfe, guidance_scale=args.guidance,
                                 prev_cpk=prev_cpk, reuse_noise=args.persistent_noise)
                preds.append((s, p.actions_B_H_A[0].float().cpu().numpy().astype(np.float64)
                              * a_std + a_mean))
        for s, pr in preds:
            cg, cp = np.cumsum(gt[:, :3], axis=0), np.cumsum(pr[:, :3], axis=0)
            end_err = float(np.linalg.norm(cp[-1] - cg[-1]) * 1000)
            z_err = float((cp[-1, 2] - cg[-1, 2]) * 1000)
            gt_desc = float(-cg[-1, 2]); pr_desc = float(-cp[-1, 2])
            commit = pr_desc / gt_desc if abs(gt_desc) > 2e-3 else float("nan")
            gt_i, pr_i = close_steps(gt[:, 6], pr[:, 6])
            rows.append({"episode": wi.episode.name, "task": item.get("text", "?"),
                         "t0": float(t0), "seed": int(s), "z_start_mm": z0 * 1000,
                         "endpoint_err_mm": end_err, "z_end_err_mm": z_err,
                         "commit_ratio": commit,
                         "close_step_err": (pr_i - gt_i) if gt_i is not None else float("nan"),
                         # absolute height at the close each side predicts
                         "pred_close_height_mm": ((z0 + cp[pr_i, 2]) * 1000
                                                  if pr_i < len(pr) else float("nan")),
                         "gt_close_height_mm": ((z0 + cg[gt_i, 2]) * 1000
                                                if gt_i is not None else float("nan")),
                         "gt_close_aperture": float(gt[gt_i, 6]) if gt_i is not None else float("nan"),
                         "pr_max_aperture": float(pr[:, 6].max())})
    if not rows:
        print("no windows with a gripper close found"); return 1
    print(f"episodes skipped (no close / chunk cannot span the close): {skipped}")
    summary = summarize(rows, nfe=args.nfe, guidance=args.guidance, seeds=args.seeds,
                        null=args.null, null_semantics=null_semantics(args.null),
                        student=bool(mc.student), split=args.split,
                        persistent_noise=bool(args.persistent_noise),
                        max_episodes=args.max_episodes, ema=bool(args.ema),
                        skipped=skipped)
    # gt_close_height_mm is a property of the demos, not of a condition —
    # reported once so a --null run can be read against it
    summary["gt_close_height_mm"] = _mean(rows, "gt_close_height_mm")
    summary["median_gt_close_height_mm"] = _median(rows, "gt_close_height_mm")
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
