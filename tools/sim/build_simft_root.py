"""Assemble the fine-tune data root: sim expert episodes + real deploy rollouts + val hold-out.

Layout produced (what train_teacher --data <root>/tasks expects):

    <root>/tasks/<task>/<episode>/   symlinked sim episodes and val episodes; STAGED copies
                                     of deploy rollouts (streams symlinked, actions.zarr +
                                     meta.json copied, then re-derived to measured delta-EE
                                     so the originals are never touched)
    <root>/manifests/all.jsonl       split=train: sim + deploy, split=val: the val demos
    <root>/norm_stats.json           the init checkpoint's stats (train_teacher refuses a
                                     mismatch), passed in with --norm-stats

Deploy rollouts enter only if finalized, judged (success not None) and not tagged
contaminated/unlabeled; successes supervise actions, failures the contact heads only
(schema.is_failure_demo). --real-fraction caps the number of real rollouts so the mix
stays near the requested share of real windows (episodes weighted equally).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from phantom.config.hardware import load_hardware  # noqa: E402
from phantom.data.schema import NON_TRAINING_TAGS, STREAM_ACTIONS, STREAM_ACTIONS_PLAN, EpisodeMeta  # noqa: E402
from rederive_rollout_actions import rederive_actions  # noqa: E402

log = logging.getLogger("build_simft_root")


def _usable_rollout(ep: Path) -> tuple[bool, str]:
    try:
        meta = EpisodeMeta.load(ep / "meta.json")
    except Exception as e:
        return False, f"meta: {e}"
    if str(meta.status or "finalized") != "finalized":
        return False, "not finalized"
    if set(str(t) for t in meta.tags) & set(NON_TRAINING_TAGS):
        return False, "non-training tag"
    if meta.success is None:
        return False, "unjudged"
    for s in ("camera_scene_color", "arm_ft", "actions", "tactile_left_fields_ds", "tactile_right_fields_ds"):
        if not (ep / f"{s}.zarr").exists():
            return False, f"missing {s}"
    return True, "ok"


def stage_rollout(src: Path, dst: Path, hw) -> str:
    """Streams symlinked, actions.zarr + meta.json copied, actions re-derived in the copy."""
    dst.mkdir(parents=True, exist_ok=False)
    for item in src.iterdir():
        if item.name in ("meta.json", f"{STREAM_ACTIONS}.zarr", f"{STREAM_ACTIONS_PLAN}.zarr"):
            if item.is_dir():
                shutil.copytree(item, dst / item.name)
            else:
                shutil.copy2(item, dst / item.name)
        else:
            os.symlink(item.resolve(), dst / item.name)
    return rederive_actions(dst, hw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--task", default="waffles")
    ap.add_argument("--sim-root", type=Path, nargs="+", required=True, help="export root(s) (tasks/<task>/ep_sim_*)")
    ap.add_argument("--real-root", type=Path, nargs="*", default=[],
                    help="root(s) of real TELEOP demos laid out as tasks/<task>/ep_* (e.g. the NUC copy); "
                         "symlinked as-is, split=train; failure demos keep their <task>_fail meta")
    ap.add_argument("--tasks", nargs="*", default=None, help="tasks to include (default: every tasks/<task> found)")
    ap.add_argument("--deploy-roots", type=Path, nargs="*", default=[], help="dirs holding ep_* deploy rollouts (task = --task only)")
    ap.add_argument("--val-root", type=Path, default=None, help="data root whose manifests/all.jsonl val rows are reused")
    ap.add_argument("--norm-stats", type=Path, required=True)
    ap.add_argument("--hardware", default="configs/hardware.nuc.yaml")
    ap.add_argument("--max-real", type=int, default=None, help="cap on deploy rollouts (successes kept first)")
    ap.add_argument("--max-sim", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    hw = load_hardware(args.hardware, quiet=True)
    root = args.root
    rows = []
    task_names = args.tasks or sorted({d.name for r in list(args.sim_root) + list(args.real_root)
                                       for d in (r / "tasks").glob("*") if d.is_dir()})
    for task in task_names:
        tasks = root / "tasks" / task
        tasks.mkdir(parents=True, exist_ok=True)
        sim_eps = []
        for sroot in args.sim_root:
            sim_eps += sorted(p for p in (sroot / "tasks" / task).glob("ep_sim_*") if (p / "meta.json").exists())
        sim_eps = [p for p in sim_eps if EpisodeMeta.load(p / "meta.json").status == "finalized"]
        if args.max_sim:
            sim_eps = sim_eps[: args.max_sim]
        for ep in sim_eps:
            dst = tasks / ep.name
            if not dst.exists():
                os.symlink(ep.resolve(), dst)
            rows.append({"episode": ep.name, "task": task, "success": True, "failure_demo": False, "split": "train",
                         "session": "sim_expert", "path": f"tasks/{task}/{ep.name}"})
        n_real = 0
        for rroot in args.real_root:
            for ep in sorted((rroot / "tasks" / task).glob("ep_*")):
                if not (ep / "meta.json").exists():
                    continue
                meta = EpisodeMeta.load(ep / "meta.json")
                if str(meta.status or "finalized") != "finalized":
                    continue
                dst = tasks / ep.name
                if not dst.exists():
                    os.symlink(ep.resolve(), dst)
                fail = str(meta.task).endswith("_fail") or meta.success is False
                rows.append({"episode": ep.name, "task": meta.task, "success": bool(meta.success),
                             "failure_demo": fail, "split": "train", "session": "real_teleop",
                             "path": f"tasks/{task}/{ep.name}"})
                n_real += 1
        log.info("%s: sim episodes %d, real teleop episodes %d", task, len(sim_eps), n_real)

    task = args.task
    tasks = root / "tasks" / task
    tasks.mkdir(parents=True, exist_ok=True)

    cands = []
    for droot in args.deploy_roots:
        for ep in sorted(droot.glob(f"ep_*{args.task}_*")):
            ok, why = _usable_rollout(ep)
            if ok:
                cands.append(ep)
            else:
                log.debug("skip %s: %s", ep.name, why)
    cands.sort(key=lambda p: (not EpisodeMeta.load(p / "meta.json").success, p.name))
    if args.max_real is not None:
        cands = cands[: args.max_real]
    n_succ = 0
    for ep in cands:
        dst = tasks / ep.name
        if not dst.exists():
            status = stage_rollout(ep, dst, hw)
            log.debug("%s: %s", ep.name, status)
        meta = EpisodeMeta.load(dst / "meta.json")
        n_succ += bool(meta.success)
        rows.append({"episode": ep.name, "task": args.task, "success": bool(meta.success),
                     "failure_demo": not bool(meta.success), "split": "train", "session": "deploy",
                     "path": f"tasks/{args.task}/{ep.name}"})
    log.info("deploy rollouts: %d (%d successes)", len(cands), n_succ)

    if args.val_root:
        mf = args.val_root / "manifests" / "all.jsonl"
        for l in mf.read_text().splitlines():
            r = json.loads(l)
            if r.get("split") != "val" or r.get("task") not in task_names:
                continue
            src = (args.val_root / r["path"]).resolve()
            vdir = root / "tasks" / r["task"]
            vdir.mkdir(parents=True, exist_ok=True)
            dst = vdir / src.name
            if not dst.exists():
                os.symlink(src, dst)
            rows.append({**r, "path": f"tasks/{r['task']}/{src.name}"})
        log.info("val episodes: %d", sum(r["split"] == "val" for r in rows))

    (root / "manifests").mkdir(exist_ok=True)
    (root / "manifests" / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copy2(args.norm_stats, root / "norm_stats.json")
    log.info("manifest rows: %d (train %d, val %d) -> %s", len(rows), sum(r["split"] == "train" for r in rows),
             sum(r["split"] == "val" for r in rows), root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
