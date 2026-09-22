"""Regenerate configs/start_poses.yaml from a dataset root (per-task demo
start-pose distribution, START joint configuration, TCP envelope and z floor).

    python tools/gen_start_poses.py /path/to/phantom-episodes/tasks [--out FILE]

Run on the box that holds the FULL dataset. Success tasks only —
`*_fail` episodes are never deployed, and any episode whose meta.json marks it
a failure demo (``success is False`` / ``failure_demo`` / a ``_fail`` task) is
dropped even if it was symlinked into a success task directory.

EVERY per-task block is computed over the SAME episode set: the run asserts
len(tcp) == len(gripper) == len(q) == len(z) and emits ``q_n`` alongside ``n``
so a consumer can prove it. The 2026-08-28 file had q_n = 17..37 while n = 250
(the joint/envelope/floor stats came from a subset, the TCP mean/std
from the full 250) — a STOP hitbox and a z floor derived from 7% of the demos.
``phantom.deploy.start_pose.load_start_stats`` now refuses such a file.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import zarr

TASKS = ("Carton", "egg", "waffles", "whiteboard")


def is_success_episode(ep: str) -> bool:
    """False for deliberate-failure demos (never deployed, never a start pose).

    A missing/unreadable meta.json is NOT silently accepted: an episode we
    cannot classify is dropped, because a failure demo folded into the START
    distribution would teach the homing routine the rig's own failure mode.
    """
    try:
        m = json.loads(Path(ep, "meta.json").read_text())
    except Exception as ex:                    # noqa: BLE001
        print(f"skip {ep}: unreadable meta.json ({ex})", file=sys.stderr)
        return False
    task = str(m.get("task", ""))
    if task.endswith("_fail") or m.get("failure_demo") or m.get("success") is False:
        return False
    if "deliberate_failure" in (m.get("tags") or []):
        return False
    return True


def task_stats(root: str, task: str) -> dict:
    """Every statistic for one task, over ONE episode set."""
    eps = [e for e in sorted(glob.glob(f"{root}/{task}/ep_*"))
           if os.path.isdir(e) and is_success_episode(e)]
    poses, grips, qs, zmins, lo, hi, used = [], [], [], [], [], [], []
    for e in eps:
        try:
            tcp = np.asarray(zarr.open(e + "/arm_tcp_pose.zarr")["data"])
            q0 = np.asarray(zarr.open(e + "/arm_q.zarr")["data"][0])
            g0 = float(np.ravel(zarr.open(e + "/gripper.zarr")["data"][0])[0])
        except Exception as ex:                # noqa: BLE001 — skip broken episode, keep stats honest
            print(f"skip {e}: {ex}", file=sys.stderr)
            continue
        if tcp.ndim != 2 or tcp.shape[0] == 0 or tcp.shape[1] < 3 or not np.all(np.isfinite(tcp)):
            print(f"skip {e}: bad arm_tcp_pose {getattr(tcp, 'shape', None)}", file=sys.stderr)
            continue
        if not np.all(np.isfinite(q0)) or not np.isfinite(g0):
            print(f"skip {e}: non-finite arm_q/gripper start", file=sys.stderr)
            continue
        poses.append(tcp[0])
        zmins.append(float(tcp[:, 2].min()))
        lo.append(tcp[:, :3].min(0))
        hi.append(tcp[:, :3].max(0))
        qs.append(q0)
        grips.append(g0)
        used.append(e)
    P, G, Q, Z = np.array(poses), np.array(grips), np.array(qs), np.array(zmins)
    # the whole point of this rewrite: one episode set behind every number
    assert len(P) == len(G) == len(Q) == len(Z) == len(lo) == len(hi), (
        f"{task}: ragged stats — tcp {len(P)}, gripper {len(G)}, q {len(Q)}, "
        f"z {len(Z)}, envelope {len(lo)}/{len(hi)}")
    assert len(P) >= 20, f"{task}: only {len(P)} episodes — refusing thin stats"
    assert Q.ndim == 2, f"{task}: arm_q start vectors are ragged ({Q.dtype})"
    return {"n": len(P), "P": P, "G": G, "Q": Q, "Z": Z,
            "lo": np.array(lo).min(0), "hi": np.array(hi).max(0), "eps": used}


def render(root: str, per_task: dict[str, dict], note: str = "") -> str:
    fmt = lambda a: "[" + ", ".join(f"{x:.4f}" for x in a) + "]"  # noqa: E731
    total = sum(s["n"] for s in per_task.values())
    lines = [
        "# Per-task demo START-pose distributions - generated from the FULL dataset",
        "# (success episodes only; *_fail tasks and failure demos are excluded).",
        "# Regenerate with tools/gen_start_poses.py after any dataset change.",
        "# tcp: [x, y, z, rx, ry, rz]  (m / axis-angle rad, UR base frame)",
        "# Used by run_deploy (homing is default-on for a real arm; --no-home skips,",
        "# --max-start-sigma gates episode start, --allow-ood-start overrides).",
        "# EVERY block below comes from the SAME episode set per task: tcp_mean/std,",
        "# gripper_mean/std, q_mean/q_std (start joint vector, rad), tcp_z_min (lowest",
        "# demo TCP z, m -> run_deploy's z no-go floor) and tcp_min/tcp_max (demo TCP",
        "# envelope over ALL frames -> run_deploy's STOP hitbox). q_n == n proves it;",
        "# load_start_stats refuses a file where they disagree.",
        "# The policy reads RAW joints, so a wrapped wrist / flipped IK branch gates OUT",
        "# even when the TCP pose is fine (16/26 rig episodes ran like that on 08-28).",
    ]
    if note:
        lines += ["# " + l for l in note.splitlines()]
    lines += [
        f"generated: {time.strftime('%Y-%m-%d')}",
        f"source: {root}",
        f"n_episodes_total: {total}",
        "tasks:",
    ]
    for task, s in per_task.items():
        P, G, Q, Z = s["P"], s["G"], s["Q"], s["Z"]
        lines += [
            f"  {task}:",
            f"    n: {s['n']}",
            f"    tcp_mean: {fmt(P.mean(0))}",
            f"    tcp_std:  {fmt(P.std(0))}",
            f"    gripper_mean: {G.mean():.3f}",
            f"    gripper_std: {G.std():.3f}",
            f"    q_n: {len(Q)}",
            f"    q_mean: {fmt(Q.mean(0))}",
            f"    q_std:  {fmt(Q.std(0))}",
            f"    tcp_z_min: {Z.min():.4f}",
            f"    tcp_min: {fmt(s['lo'])}",
            f"    tcp_max: {fmt(s['hi'])}",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="<dataset>/tasks (holds <task>/ep_*)")
    ap.add_argument("--out", default=None,
                    help="output yaml (default: <repo>/configs/start_poses.yaml)")
    ap.add_argument("--note", default="",
                    help="extra provenance lines for the header comment")
    ap.add_argument("--json-out", default="",
                    help="also dump the per-task episode lists + raw stats here")
    args = ap.parse_args(argv)

    per_task = {t: task_stats(args.root, t) for t in TASKS}
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "configs" / "start_poses.yaml")
    out.write_text(render(args.root, per_task, args.note))
    for t, s in per_task.items():
        print(f"{t}: n={s['n']} q_n={len(s['Q'])} z_min={s['Z'].min():.4f} "
              f"tcp_min={np.round(s['lo'], 4).tolist()} tcp_max={np.round(s['hi'], 4).tolist()}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {t: {"n": s["n"], "episodes": [os.path.basename(e) for e in s["eps"]],
                 "tcp_mean": s["P"].mean(0).tolist(), "tcp_std": s["P"].std(0).tolist(),
                 "q_mean": s["Q"].mean(0).tolist(), "q_std": s["Q"].std(0).tolist(),
                 "tcp_z_min": float(s["Z"].min()),
                 "tcp_min": s["lo"].tolist(), "tcp_max": s["hi"].tolist()}
             for t, s in per_task.items()}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
