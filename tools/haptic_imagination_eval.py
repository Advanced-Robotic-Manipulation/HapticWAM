"""Haptic imagination on held-out windows: how well a checkpoint PREDICTS the
contact package the pads then measured, off-policy, without a rig.

`tools/tactile_prediction_eval.py` scores the same quantity at DEPLOY time
(it needs `planner_cpk.npz`, i.e. episodes a policy actually drove). This
tool asks the same question of a checkpoint alone, on the val split: draw a
window, hide every privileged future (`events`, `cpk_*` zeroed exactly as
`train/common.evaluate_sampled` does), sample the joint sequence, and score
the model's imagined contact package against the package the recorded pads
measured over the same latent grid (`phantom/eval/tactile_prediction.
observed_package` — the deploy metric's own construction of "what was felt").

A STUDENT has no pad frame in its observation; it must still predict the
contact frames, and the target is the measured pads. That is the point: the
number says whether the sensor-free policy imagines touch as well as the
teacher, which senses it.

On top of the deploy metric (d_fz RMSE / skill vs persistence, mask IoU, CoP,
slip, event accuracy) this adds the two decision-level numbers:

  event P/R/F1   per-step in-contact classification (predicted event band vs
                 the measured event labels), reported for every window and
                 for the TERMINAL windows alone (t0 inside the pre-close
                 band — the closing decision).
  onset timing   signed error of the predicted contact-onset step vs the
                 measured one, in ms. NOTE the latent grid is coarse
                 (latent_dt = temporal_comp / fps = 1.0 s here), so this is
                 quantized to whole latent steps; it is a lead/lag sign and
                 magnitude, not a millisecond-accurate latency.
  veto signal    the ACC readout the TerminalVeto actually reads —
                 p_contact = 1 - p_evt[none] vs the measured event at the
                 first latent step, at the deployed threshold p_close = 0.5
                 — plus the ACC gate vs its contact-within-lookahead label.

    python tools/haptic_imagination_eval.py --selftest          # metric layer
    python tools/haptic_imagination_eval.py --ckpt runs/teacher/x/teacher_001500.pt \\
        --data-root data/phantom-demos/tasks --label teacher_v5_ftA \\
        --windows-file out/hie/windows.json --seeds 4 --nfe 1 --out-dir out/hie

`--windows-file` is written on the first run and REUSED afterwards, so every
checkpoint is scored on identical (episode, t0) anchors.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("hie")

# event ontology is imported lazily in main(); the selftest needs it too but
# phantom.config.model is torch-free.
from phantom.config.model import EVENTS, EVENT_IDX, N_EVENTS      # noqa: E402

IN_CONTACT = (EVENT_IDX["onset"], EVENT_IDX["hold"], EVENT_IDX["slip"])
HEADLINE = ("tpe_dfz_skill", "tpe_dfz_skill_contact", "tpe_mask_iou",
            "tpe_event_acc", "tpe_dfz_rmse", "tpe_dfz_rmse_persist",
            "tpe_cop_err", "tpe_slip_err")


# ---------------------------------------------------------------------------
# decision-level metrics (pure numpy; `rows` are per-window records)
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f = (2 * p * r / (p + r)) if (p + r) and np.isfinite(p) and np.isfinite(r) and (p + r) > 0 \
        else (0.0 if (tp + fp + fn) else float("nan"))
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def contact_event_prf(rows: list[dict]) -> dict:
    """Per-step in-contact classification over the predicted event band.

    Positive = the event label says the pads are in contact at the end of
    that latent step (onset / hold / slip); negative = none / release.
    Every (window, seed, step) is one sample.
    """
    tp = fp = fn = tn = 0
    for r in rows:
        for ep_, eo in zip(r["ev_pred"], r["ev_obs"]):
            p, o = ep_ in IN_CONTACT, eo in IN_CONTACT
            tp += p and o
            fp += p and not o
            fn += (not p) and o
            tn += (not p) and (not o)
    out = _prf(int(tp), int(fp), int(fn))
    n = tp + fp + fn + tn
    out["accuracy"] = (tp + tn) / n if n else float("nan")
    out["n_steps"] = int(n)
    out["positive_rate_obs"] = (tp + fn) / n if n else float("nan")
    return out


def onset_window_prf(rows: list[dict]) -> dict:
    """Window-level onset detection: did the model imagine a contact ONSET
    inside the horizon where the pads recorded one (and only there)?"""
    tp = fp = fn = tn = 0
    for r in rows:
        p = EVENT_IDX["onset"] in r["ev_pred"]
        o = EVENT_IDX["onset"] in r["ev_obs"]
        tp += p and o
        fp += p and not o
        fn += (not p) and o
        tn += (not p) and (not o)
    out = _prf(int(tp), int(fp), int(fn))
    out["n_windows"] = int(tp + fp + fn + tn)
    return out


def onset_timing(rows: list[dict], latent_dt: float) -> dict:
    """Signed onset-step error in ms on windows where both the prediction and
    the measurement contain an onset. Quantized to latent_dt (see module
    docstring): report it as lead/lag, not as a latency."""
    d = []
    for r in rows:
        if EVENT_IDX["onset"] in r["ev_pred"] and EVENT_IDX["onset"] in r["ev_obs"]:
            kp = list(r["ev_pred"]).index(EVENT_IDX["onset"])
            ko = list(r["ev_obs"]).index(EVENT_IDX["onset"])
            d.append((kp - ko) * latent_dt * 1000.0)
    a = np.asarray(d, dtype=float)
    if not a.size:
        return {"n": 0, "mean_ms": float("nan"), "median_ms": float("nan"),
                "mae_ms": float("nan"), "exact_step_frac": float("nan")}
    return {"n": int(a.size), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "mae_ms": float(np.abs(a).mean()),
            "exact_step_frac": float((a == 0).mean())}


def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUROC with tie handling; NaN when one class is absent."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if not s.size or y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, float)
    sr = s[order]
    i = 0
    while i < s.size:                       # average ranks within ties
        j = i
        while j + 1 < s.size and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def veto_signal(rows: list[dict], p_close: float = 0.5, p_none: float = 0.9) -> dict:
    """The ACC readout the TerminalVeto reads, against its training target.

    p_contact = 1 - p_evt[none] at the deployed threshold p_close, scored
    against the measured event at the FIRST latent step (`events[:, 0]`, the
    ACC event target in rf.training_step). Also the raw ACC gate g against
    `gate_label` (contact within event_lookahead_s), and how often the
    recovery rule's p_evt[none] > p_none would fire on a window whose pads
    were in fact in contact (a false phantom-grasp call)."""
    pc, lab, g, gl = [], [], [], []
    for r in rows:
        if r.get("p_evt") is None:
            continue
        pc.append(1.0 - float(r["p_evt"][EVENT_IDX["none"]]))
        lab.append(bool(r["ev_obs"][0] in IN_CONTACT))
        if r.get("gate") is not None:
            g.append(float(r["gate"]))
            gl.append(float(r["gate_label"]))
    if not pc:
        return {"n": 0}
    pc, lab = np.asarray(pc), np.asarray(lab, bool)
    pred = pc > p_close
    out = _prf(int((pred & lab).sum()), int((pred & ~lab).sum()), int((~pred & lab).sum()))
    out.update({
        "n": int(pc.size),
        "accuracy": float((pred == lab).mean()),
        "auroc": _auroc(pc, lab),
        "p_contact_mean_when_contact": float(pc[lab].mean()) if lab.any() else float("nan"),
        "p_contact_mean_when_free": float(pc[~lab].mean()) if (~lab).any() else float("nan"),
        "false_phantom_rate": float(((1.0 - pc) > p_none)[lab].mean()) if lab.any()
        else float("nan"),
        "contact_rate_obs": float(lab.mean()),
    })
    if g:
        g, gl = np.asarray(g), np.asarray(gl)
        gp, gy = g > 0.5, gl > 0.5
        out["gate"] = {"auroc": _auroc(g, gy), "accuracy": float((gp == gy).mean()),
                       "f1": _prf(int((gp & gy).sum()), int((gp & ~gy).sum()),
                                  int((~gp & gy).sum()))["f1"],
                       "label_rate": float(gy.mean()), "mean_g": float(g.mean())}
    return out


def bootstrap_stat(rows: list[dict], fn, n_boot: int = 2000, seed: int = 0,
                   key: str = "f1") -> tuple[float, float]:
    """Percentile CI of a rate-style statistic by resampling WINDOWS."""
    if len(rows) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), size=len(rows))
        v = fn([rows[i] for i in idx]).get(key, float("nan"))
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


# ---------------------------------------------------------------------------
# selftest: a FAKE prediction with hand-computable answers
# ---------------------------------------------------------------------------

def selftest() -> int:
    import phantom.eval.tactile_prediction as TP

    Tc, F, H, W = 3, 2, 4, 4
    rng = np.random.default_rng(0)
    d_fz = rng.normal(size=(Tc, F, H, W)).astype(np.float32)
    mask = (rng.uniform(size=(Tc, F, H, W)) > 0.5).astype(np.float32)
    obs = {
        "d_fz": d_fz, "mask": mask,
        "mask0": mask[0].copy(),
        "cop": rng.uniform(-1, 1, (Tc, F, 2)).astype(np.float32),
        "cop0": rng.uniform(-1, 1, (F, 2)).astype(np.float32),
        "slip": rng.uniform(size=(Tc, F)).astype(np.float32),
        "slip0": rng.uniform(size=(F,)).astype(np.float32),
        "wrench": rng.normal(size=(Tc, F, 6)).astype(np.float32),
        "wrench0": rng.normal(size=(F, 6)).astype(np.float32),
        "event": np.array([EVENT_IDX["none"], EVENT_IDX["onset"], EVENT_IDX["hold"]]),
        "contact": np.array([False, True, True]), "contact0": False,
    }
    onehot = np.eye(N_EVENTS, dtype=np.float32)[obs["event"]]
    perfect = {"d_fz": obs["d_fz"].copy(), "mask": obs["mask"].copy(),
               "cop": obs["cop"].copy(), "slip": obs["slip"].copy(),
               "event": onehot, "wrench": obs["wrench"].copy()}
    s = TP.summarize_steps({k: v[None] for k, v in TP.score_replan(perfect, obs).items()},
                           Tc=Tc, n_total=1, n_scored=1)
    ok = True

    def check(name: str, got, want, tol=1e-6):
        nonlocal ok
        good = (want is None and not np.isfinite(got)) or abs(got - want) <= tol
        ok = ok and good
        print(f"  [{'ok ' if good else 'FAIL'}] {name}: {got!r} (want {want!r})")

    print("selftest: perfect prediction")
    check("dfz skill", s["tpe_dfz_skill"], 1.0)
    check("mask IoU", s["tpe_mask_iou"], 1.0)
    check("event acc", s["tpe_event_acc"], 1.0)

    print("selftest: persistence-equal prediction (zero deltas, frozen state)")
    persist = {"d_fz": np.zeros_like(obs["d_fz"]),
               "mask": np.repeat(obs["mask0"][None], Tc, 0),
               "cop": np.repeat(obs["cop0"][None], Tc, 0),
               "slip": np.repeat(obs["slip0"][None], Tc, 0),
               "event": np.eye(N_EVENTS, dtype=np.float32)[
                   np.full(Tc, EVENT_IDX["none"])],
               "wrench": np.repeat(obs["wrench0"][None], Tc, 0)}
    sp = TP.summarize_steps({k: v[None] for k, v in TP.score_replan(persist, obs).items()},
                            Tc=Tc, n_total=1, n_scored=1)
    check("dfz skill == 0", sp["tpe_dfz_skill"], 0.0)
    check("mask IoU == persistence", sp["tpe_mask_iou"], sp["tpe_mask_iou_persist"])

    print("selftest: decision-level metrics")
    # window A: predicted onset one step LATE, window B: onset missed entirely
    rows = [
        {"ev_pred": [EVENT_IDX["none"], EVENT_IDX["none"], EVENT_IDX["onset"]],
         "ev_obs": [EVENT_IDX["none"], EVENT_IDX["onset"], EVENT_IDX["hold"]],
         "p_evt": [0.2, 0.5, 0.2, 0.05, 0.05], "gate": 0.9, "gate_label": 1.0,
         "terminal": True},
        {"ev_pred": [EVENT_IDX["none"]] * 3,
         "ev_obs": [EVENT_IDX["none"], EVENT_IDX["onset"], EVENT_IDX["hold"]],
         "p_evt": [0.9, 0.04, 0.03, 0.02, 0.01], "gate": 0.2, "gate_label": 1.0,
         "terminal": True},
    ]
    # in-contact steps: obs positives = 4 (steps 1,2 of both windows);
    # predicted positives = 1 (window A step 2) -> tp 1, fp 0, fn 3
    c = contact_event_prf(rows)
    check("contact tp", c["tp"], 1)
    check("contact fn", c["fn"], 3)
    check("contact fp", c["fp"], 0)
    check("contact f1", c["f1"], 2 * 1.0 * 0.25 / 1.25)
    o = onset_window_prf(rows)
    check("onset window tp", o["tp"], 1)
    check("onset window fn", o["fn"], 1)
    t = onset_timing(rows, latent_dt=1.0)
    check("onset timing n", t["n"], 1)
    check("onset timing mean_ms (+1 step late)", t["mean_ms"], 1000.0)
    v = veto_signal(rows)
    # window A: p_contact 0.8 > 0.5, obs event[0] = none -> fp
    # window B: p_contact 0.1, obs none -> tn ; no positives in the label
    check("veto n", v["n"], 2)
    check("veto fp", v["fp"], 1)
    check("veto accuracy", v["accuracy"], 0.5)
    check("gate auroc is NaN (one class)", v["gate"]["auroc"], None)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# window plan
# ---------------------------------------------------------------------------

def build_window_plan(sampler, episodes: list[Path], *, per_episode: int,
                      terminal_frac: float, seed: int,
                      grasp_window_s=(1.5, 0.2)) -> list[dict]:
    """Deterministic (episode, t0, terminal) anchors, shared by every
    checkpoint. `terminal_frac` of each episode's windows are drawn from the
    pre-close commit band (train/common.WindowDataset's own band)."""
    from phantom.train.common import close_index

    rng = np.random.default_rng(seed)
    plan: list[dict] = []
    for ep in episodes:
        try:
            lo, hi = sampler.valid_range(ep)
        except Exception as e:                                   # short episode
            log.warning("skip %s: %s", ep.name, e)
            continue
        tc = None
        try:
            import zarr
            g = zarr.open(str(Path(ep) / "gripper.zarr"), mode="r")
            i = close_index(np.asarray(g["data"][:, 0], dtype=np.float64))
            if i is not None:
                tc = float(np.asarray(g["ts"][:], dtype=np.float64)[i])
        except Exception as e:                                   # noqa: BLE001
            log.warning("no close time for %s (%s)", ep.name, e)
        n_term = int(round(per_episode * terminal_frac))
        for j in range(per_episode):
            want_term = j < n_term and tc is not None
            if want_term:
                a = max(lo, tc - grasp_window_s[0])
                b = min(hi, tc - grasp_window_s[1])
                if b > a:
                    plan.append({"episode": str(ep), "t0": float(rng.uniform(a, b)),
                                 "terminal": True})
                    continue
            plan.append({"episode": str(ep), "t0": float(rng.uniform(lo, hi)),
                         "terminal": False})
    return plan


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run the metric layer on a fake prediction and exit")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--label", default=None, help="row name in the output (default: ckpt stem)")
    ap.add_argument("--data-root", type=Path, default=None, help="<dataset>/tasks")
    ap.add_argument("--split", default="val")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--windows-file", type=Path, default=None,
                    help="reuse (or write) the shared window plan")
    ap.add_argument("--episodes", type=int, default=40, help="val episodes to draw from")
    ap.add_argument("--per-episode", type=int, default=2)
    ap.add_argument("--terminal-frac", type=float, default=0.5)
    ap.add_argument("--plan-seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=4, help="noise seeds per window")
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N windows")
    ap.add_argument("--dump-rows", action="store_true",
                    help="also write <label>_rows.json: per-window predicted/observed "
                         "event bands, ACC readouts and per-window scores, so any "
                         "subset (e.g. windows with an onset) can be scored offline")
    ap.add_argument("--prev-cpk", choices=("none", "earlier"), default="none",
                    help="none (default): first-replan path — ACC two_pass runs the "
                         "model's OWN anticipation, exactly as deploy's first replan. "
                         "earlier: steady-state deploy path — sample the window one "
                         "action chunk earlier and feed ITS package as prev_cpk "
                         "(tools/terminal_eval.py's convention).")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path("out/hie"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.selftest:
        return selftest()
    if not args.ckpt or not args.data_root:
        ap.error("--ckpt and --data-root are required (or pass --selftest)")

    import torch
    from phantom.backbone import loader as bl
    from phantom.config.hardware import load_hardware
    from phantom.config.paths import load_paths
    from phantom.config.model import PhantomModelConfig
    from phantom.data.episode_store import EpisodeReader
    from phantom.data.schema import NormStats
    from phantom.data.windows import WindowSampler
    from phantom.deploy.cpk_trace import denormalize_package
    from phantom.train import common as C
    from phantom.train.builder import build_model
    import phantom.eval.tactile_prediction as TP

    t_start = time.time()
    hw = load_hardware(args.hardware)
    paths = load_paths()
    label = args.label or args.ckpt.stem

    payload = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    saved_mc = payload.get("configs", {}).get("model")
    mc = PhantomModelConfig.from_dict(saved_mc) if isinstance(saved_mc, dict) else None
    student = bool(mc.student) if mc is not None else False
    log.info("%s: student=%s acc=%s mask_wrist=%s video_attend=%s", label, student,
             getattr(getattr(mc, "acc", None), "self_anticipation", None),
             getattr(mc, "mask_wrist", None), getattr(mc, "video_attend", None))

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    pm = build_model(hw, paths, student=student, mc=mc, device=args.device,
                     dtype=dtype, inference=True)
    payload = C.load_phantom_checkpoint(args.ckpt, pm.rf, hw=hw,
                                        load_ema=not args.no_ema, payload=payload)
    norm = NormStats.identity()
    if payload.get("norm_stats"):
        ns = payload["norm_stats"]
        norm = NormStats(mean={k: np.asarray(v, dtype=np.float32) for k, v in ns["mean"].items()},
                         std={k: np.asarray(v, dtype=np.float32) for k, v in ns["std"].items()})
    else:
        stats = args.data_root / "norm_stats.json"
        if stats.exists():
            norm = NormStats.load(stats)
            log.warning("checkpoint carries no norm stats — using %s", stats)
    bl.merge_lora(pm.rf.net)
    rf = pm.rf
    rf.eval()

    sampler = WindowSampler(hw, pm.bb, norm, student=student, seed=0)
    Tc = pm.bb.t_video - 1
    latent_dt = sampler.latent_dt
    chunk_s = hw.control.chunk_horizon / hw.control.action_rate_hz

    # ---- window plan (shared across checkpoints) --------------------------
    if args.windows_file and args.windows_file.exists():
        plan = json.loads(args.windows_file.read_text())["windows"]
        log.info("window plan: %d windows from %s", len(plan), args.windows_file)
    else:
        eps = C.manifest_split(args.data_root, args.split)
        if eps is None:
            raise SystemExit(f"no manifest for split {args.split!r} under {args.data_root}")
        eps = sorted(eps)[:args.episodes] if args.episodes else sorted(eps)
        plan = build_window_plan(sampler, eps, per_episode=args.per_episode,
                                 terminal_frac=args.terminal_frac, seed=args.plan_seed)
        if args.windows_file:
            args.windows_file.parent.mkdir(parents=True, exist_ok=True)
            args.windows_file.write_text(json.dumps(
                {"split": args.split, "plan_seed": args.plan_seed, "windows": plan}, indent=1))
        log.info("window plan: %d windows over %d episodes (written to %s)",
                 len(plan), len(eps), args.windows_file)
    if args.limit:
        plan = plan[:args.limit]

    # ---- score -------------------------------------------------------------
    rows: list[dict] = []                 # one per (window, seed)
    per_window_steps: list[dict] = []     # steps pooled over seeds, per window
    skipped: list[tuple[str, str]] = []
    for wi, w in enumerate(plan):
        ep = Path(w["episode"])
        t0 = float(w["t0"])
        try:
            item = sampler.sample(ep, t0)
        except Exception as e:                                   # noqa: BLE001
            skipped.append((ep.name, f"sample: {type(e).__name__}: {e}"))
            continue
        try:
            obs = TP.observed_package(TP._Streams(EpisodeReader(ep), hw), hw, t0,
                                      latent_dt, Tc)
        except Exception as e:                                   # noqa: BLE001
            skipped.append((ep.name, f"observed: {type(e).__name__}: {e}"))
            continue
        if obs is None:
            skipped.append((ep.name, "observed package leaves the recording"))
            continue

        def to_batch(item: dict) -> dict:
            b: dict = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    b[k] = v.unsqueeze(0)
                elif isinstance(v, np.ndarray):
                    b[k] = torch.from_numpy(v).unsqueeze(0)
                elif isinstance(v, str):
                    b[k] = [v]
                elif isinstance(v, (int, float, np.floating)):
                    b[k] = torch.tensor([v])
                else:
                    b[k] = v
            for k, v in b.items():
                if torch.is_tensor(v):
                    v = v.to(rf.device)
                    if v.is_floating_point():
                        v = v.to(rf.dtype)
                    b[k] = v
            # NO privileged future (evaluate_sampled's rule; the deploy policy
            # zeroes the same placeholders). ACC's prev-cpk summary still comes
            # from the model's own two_pass anticipation, not from these.
            for k in list(b):
                if k == "events" or k.startswith("cpk_"):
                    b[k] = torch.zeros_like(b[k])
            return b

        batch: dict = {}
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
        for k, v in batch.items():
            if torch.is_tensor(v):
                v = v.to(rf.device)
                if v.is_floating_point():
                    v = v.to(rf.dtype)
                batch[k] = v
        gate_label = float(item["gate_label"])
        # NO privileged future: the GT contact package and events are what the
        # ACC self-anticipation input would otherwise leak (evaluate_sampled's
        # rule — deploy never has them).
        for k in list(batch):
            if k == "events" or k.startswith("cpk_"):
                batch[k] = torch.zeros_like(batch[k])

        prev_batch = None
        if args.prev_cpk == "earlier":
            t_prev = t0 - chunk_s
            try:
                lo, _hi = sampler.valid_range(ep)
            except Exception:                                    # noqa: BLE001
                lo = float("inf")
            if t_prev >= lo:
                try:
                    prev_batch = to_batch(sampler.sample(ep, t_prev))
                except Exception as e:                           # noqa: BLE001
                    skipped.append((ep.name, f"prev window: {type(e).__name__}: {e}"))

        seed_steps = []
        for s in range(max(1, args.seeds)):
            rf._gen = torch.Generator().manual_seed(1000 * s + wi)
            with torch.no_grad():
                prev_cpk = (rf.sample(prev_batch, nfe=args.nfe).cpk
                            if prev_batch is not None else None)
                pred = rf.sample(batch, nfe=args.nfe, prev_cpk=prev_cpk)
            pkg = denormalize_package(pred.cpk, norm, hw)
            st = TP.score_replan(pkg, obs)
            seed_steps.append({k: v[None] for k, v in st.items()})
            p_evt = (pred.acc.p_evt[0].float().cpu().numpy().tolist()
                     if pred.acc is not None else None)
            gate = (float(pred.acc.g[0]) if pred.acc is not None else None)
            rows.append({
                "episode": ep.name, "t0": t0, "seed": s, "terminal": bool(w["terminal"]),
                "task": item.get("task", ""),
                "ev_pred": [int(k) for k in np.argmax(pkg["event"], axis=-1)],
                "ev_obs": [int(k) for k in obs["event"]],
                "p_evt": p_evt, "gate": gate, "gate_label": gate_label,
                # measured structure of the window, for offline subsetting
                "obs_contact0": bool(obs["contact0"]),
                "obs_contact": [bool(x) for x in obs["contact"]],
                "obs_has_onset": bool(EVENT_IDX["onset"] in [int(k) for k in obs["event"]]),
                "obs_any_change": bool(any(
                    a != b for a, b in zip([bool(obs["contact0"])]
                                           + [bool(x) for x in obs["contact"]],
                                           [bool(x) for x in obs["contact"]]))),
                "prev_cpk_used": bool(prev_batch is not None),
            })
        per_window_steps.append({
            "terminal": bool(w["terminal"]),
            "episode": ep.name, "t0": t0,
            "obs_has_onset": bool(EVENT_IDX["onset"] in [int(k) for k in obs["event"]]),
            "obs_any_change": bool(any(
                a != b for a, b in zip([bool(obs["contact0"])] + [bool(x) for x in obs["contact"]],
                                       [bool(x) for x in obs["contact"]]))),
            "steps": {k: np.concatenate([d[k] for d in seed_steps]) for k in seed_steps[0]},
        })
        if (wi + 1) % 5 == 0:
            log.info("%s: %d/%d windows (%.0fs)", label, wi + 1, len(plan),
                     time.time() - t_start)

    if not per_window_steps:
        raise SystemExit(f"{label}: no scoreable windows ({skipped[:3]})")

    # ---- aggregate ---------------------------------------------------------
    def pooled_summary(sel) -> dict:
        chunks = [w["steps"] for w in per_window_steps if sel(w)]
        if not chunks:
            return {}
        pooled = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        n = int(pooled["dfz_se"].shape[0])
        return TP.summarize_steps(pooled, Tc=Tc,
                                  fz_unit_to_N=float(getattr(hw.tactile,
                                                             "dist_force_unit_to_N", 0.0) or 0.0),
                                  n_total=n, n_scored=n)

    per_window_summaries = [TP.summarize_steps(w["steps"], Tc=Tc, n_total=1, n_scored=1)
                            for w in per_window_steps]
    ci = {}
    for k in HEADLINE:
        vals = [s.get(k, np.nan) for s in per_window_summaries]
        lo, hi = TP.bootstrap_ci(vals, n_boot=args.boot, seed=0)
        ci[k] = {"mean_of_windows": float(np.nanmean(vals)) if np.isfinite(vals).any()
                 else float("nan"), "ci_lo": lo, "ci_hi": hi}

    term = [r for r in rows if r["terminal"]]
    out = {
        "label": label, "ckpt": str(args.ckpt), "student": student,
        "n_windows": len(per_window_steps), "n_rows": len(rows), "seeds": args.seeds,
        "nfe": args.nfe, "Tc": Tc, "latent_dt": latent_dt,
        "ema": not args.no_ema,
        "pooled": pooled_summary(lambda w: True),
        "pooled_terminal": pooled_summary(lambda w: w["terminal"]),
        "window_ci": ci,
        "contact_event_prf": contact_event_prf(rows),
        "contact_event_prf_terminal": contact_event_prf(term) if term else {},
        "onset_window_prf": onset_window_prf(rows),
        "onset_window_prf_terminal": onset_window_prf(term) if term else {},
        "onset_timing": onset_timing(rows, latent_dt),
        "onset_timing_terminal": onset_timing(term, latent_dt) if term else {},
        "veto_signal": veto_signal(rows),
        "veto_signal_terminal": veto_signal(term) if term else {},
        "skipped": skipped,
        "runtime_s": round(time.time() - t_start, 1),
    }
    out["contact_event_f1_ci"] = bootstrap_stat(rows, contact_event_prf,
                                                n_boot=min(args.boot, 500), seed=0)
    if term:
        out["contact_event_f1_ci_terminal"] = bootstrap_stat(
            term, contact_event_prf, n_boot=min(args.boot, 500), seed=0)

    def dflt(o):
        if isinstance(o, (np.floating, np.integer)):
            v = o.item()
            return None if isinstance(v, float) and not math.isfinite(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, float) and not math.isfinite(o):
            return None
        raise TypeError(type(o).__name__)

    out["prev_cpk"] = args.prev_cpk
    out["prev_cpk_windows"] = int(sum(1 for r in rows if r.get("prev_cpk_used")) /
                                  max(1, args.seeds))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dump_rows:
        per_window = [{k: v for k, v in w.items() if k != "steps"}
                      | {"summary": TP.summarize_steps(w["steps"], Tc=Tc,
                                                       n_total=1, n_scored=1),
                         "steps": {k: v.tolist() for k, v in w["steps"].items()}}
                      for w in per_window_steps]
        (args.out_dir / f"{label}_rows.json").write_text(
            json.dumps({"label": label, "rows": rows, "windows": per_window},
                       indent=1, default=dflt), encoding="utf-8")
    f = args.out_dir / f"{label}.json"
    f.write_text(json.dumps(out, indent=1, default=dflt), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("label", "n_windows", "n_rows", "runtime_s")},
                     default=dflt))
    p = out["pooled"]
    print(f"  d_fz RMSE {p.get('tpe_dfz_rmse'):.4f} / persist "
          f"{p.get('tpe_dfz_rmse_persist'):.4f}  skill {p.get('tpe_dfz_skill'):.3f}")
    print(f"  mask IoU {p.get('tpe_mask_iou'):.3f}  event acc {p.get('tpe_event_acc'):.3f}")
    print(f"  contact-event F1 {out['contact_event_prf']['f1']:.3f}  "
          f"onset timing {out['onset_timing']['mean_ms']} ms (n={out['onset_timing']['n']})")
    print(f"  veto p_contact F1 {out['veto_signal'].get('f1')}  "
          f"AUROC {out['veto_signal'].get('auroc')}")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
