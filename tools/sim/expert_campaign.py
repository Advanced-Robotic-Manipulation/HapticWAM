"""Scripted-expert data campaign: N randomised Isaac episodes -> training episodes.

Each episode draws (seeded by --seed + index):
  * a start state from --start-pool (real arm starts: deploy recordings + val demos),
    stratified: --band-frac of the episodes from the deployed band (y <= -0.28, z >= 0.31)
  * packet pose jitter (centre +-2 cm, yaw +-10 deg), lighting +-15 %, one of --textures
  * the expert's own per-episode jitter (rotation targets, place offset) via --seed

then runs tools/sim/launch_waffles.sh --policy-server scripted (no policy server), scores
the trial (score_trial.py) and exports a PLACED trial with export_expert_episode.py.
A ledger (jsonl) records every episode; `--lane k --lanes L` handles indices i % L == k so
L processes share one campaign. Nothing is deleted: failed trials stay under --raw-root.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PY = "/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python"
DEFAULT_EPISODE = "/home/physicalai/phantom-icra-2027/sim/waffles/evidence/fit/ep_waffles_1787395928_000"
DEFAULT_TACTILE_BASELINE = ("/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_pick_place_v1/"
                            "sept4_no_contact_sensor_baseline.npz")


def draw_episode(i: int, seed: int, pool: list[dict], band: np.ndarray, band_frac: float, scene: dict,
                 textures: list[str], holdout: set[str], *, packet_xy_sigma: float, packet_xy_max: float,
                 yaw_sigma_deg: float, yaw_max_deg: float, light_frac: float) -> dict:
    rng = np.random.default_rng([seed, i])
    use_band = rng.uniform() < band_frac
    idx = [k for k in range(len(pool)) if band[k] == use_band and pool[k]["episode"] not in holdout]
    if not idx:
        idx = [k for k in range(len(pool)) if pool[k]["episode"] not in holdout]
    start = pool[int(rng.choice(idx))]
    cfg = json.loads(json.dumps(scene))
    dxy = np.clip(rng.normal(0.0, packet_xy_sigma, 2), -packet_xy_max, packet_xy_max)
    dyaw = float(np.clip(rng.normal(0.0, yaw_sigma_deg), -yaw_max_deg, yaw_max_deg))
    cfg["waffle"]["center"] = [float(cfg["waffle"]["center"][0] + dxy[0]),
                               float(cfg["waffle"]["center"][1] + dxy[1]), float(cfg["waffle"]["center"][2])]
    cfg["waffle"]["yaw"] = float(cfg["waffle"]["yaw"] + math.radians(dyaw))
    if textures:
        cfg["waffle"]["texture"] = str(rng.choice(textures))
    for key in ("dome_intensity", "key_intensity"):
        cfg["lighting"][key] = float(cfg["lighting"][key] * float(rng.uniform(1 - light_frac, 1 + light_frac)))
    return {"index": i, "seed": int(seed + i), "start": start, "band": bool(use_band), "scene": cfg,
            "draw": {"packet_dxy_m": [float(v) for v in dxy], "packet_dyaw_deg": dyaw,
                     "texture": cfg["waffle"].get("texture"),
                     "dome": cfg["lighting"]["dome_intensity"], "key": cfg["lighting"]["key_intensity"]}}


def run_episode(ep: dict, args, raw_root: Path) -> dict:
    name = f"ep{ep['index']:04d}__{ep['start']['episode']}"
    out = raw_root / name
    out.mkdir(parents=True, exist_ok=True)
    scene_path = out / "scene.json"
    scene_path.write_text(json.dumps(ep["scene"], indent=1))
    state = {k: ep["start"][k] for k in ("q", "measured_tcp_pose", "gripper", "wrist_ft", "t_master")}
    state["provenance"] = {"source": ep["start"]["source"], "episode": ep["start"]["episode"]}
    state_path = out / "initial_state.json"
    state_path.write_text(json.dumps(state, indent=1))
    (out / "draw.json").write_text(json.dumps({k: v for k, v in ep.items() if k != "scene"}, indent=1, default=str))
    inputs = Path(args.inputs)
    cmd = [str(Path(args.runtime) / "tools/sim/launch_waffles.sh"), "--mode", "policy",
           "--episode", args.episode, "--output", str(out), "--config", str(scene_path),
           "--duration", str(args.duration), "--seed", str(ep["seed"]),
           "--policy-server", "scripted", "--policy-mode", "teacher",
           "--policy-config", str(inputs / "policy_config__K4_ir.json"),
           "--hardware-config", args.hardware_input, "--ignore-episode-overrides",
           "--max-play-steps", "10", "--grip-play-steps", "10",
           "--observation-delay-s", "0", "--inference-delay-add-s", "0",
           "--tactile", "measured_baseline_proxy", "--tactile-baseline", args.tactile_baseline,
           "--wrist", "gripper_contact_proxy", "--gel-contact-coverage", "manifold_patch_v2",
           "--skip-stage-export", "--policy-initial-state", str(state_path), "--policy-delivery-clock", "native",
           "--placement-release-config", str(inputs / "placement_release.json"),
           "--boundary-projection-config", str(inputs / "boundary_projection__D3.json"),
           "--servo-reach-limiter", "--servo-constraint-hold-s", "2.5", "--experimental-adaptive-policy",
           "--save-policy-observations", "--record-gel-contacts", "--record-packet-support",
           "--record-robot-environment-contacts",
           "--no-progress-stop-s", "25"]
    if args.expert_params:
        cmd += ["--expert-params", args.expert_params]
    t0 = time.time()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with open(out / "isaac.log", "w") as log:
        rc = subprocess.run(cmd, cwd=args.runtime, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    score_cmd = [args.python, str(Path(args.driver) / "score_trial.py"), "--trial", str(out),
                 "--thresholds", str(inputs / "thresholds.json"), "--runtime", args.runtime,
                 "--output", str(out / "trial_result.json")]
    subprocess.run(score_cmd, cwd=args.runtime, stdout=open(out / "score.log", "w"), stderr=subprocess.STDOUT, env=env)
    result = json.loads((out / "trial_result.json").read_text()) if (out / "trial_result.json").exists() else {}
    row = {"index": ep["index"], "trial": str(out), "rc": rc, "wall_s": round(time.time() - t0, 1),
           "stage": result.get("stage_name"), "stop": result.get("stop_reason"), "status": result.get("status"),
           "duration_s": result.get("duration_s"), "band": ep["band"], "start": ep["start"]["episode"],
           "draw": ep["draw"], "exported": None}
    if result.get("stage_name") == "placed" and not args.no_export:
        exp_cmd = [args.python, str(Path(args.runtime) / "tools/sim/export_expert_episode.py"),
                   "--trial", str(out), "--out-root", args.out_root, "--hardware", args.hardware,
                   "--idle-tactile-from", args.idle_tactile_from, "--task", args.task]
        p = subprocess.run(exp_cmd, cwd=args.runtime, capture_output=True, text=True, env=env)
        (out / "export.log").write_text(p.stdout + p.stderr)
        row["exported"] = p.returncode == 0
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="campaign size (global episode indices 0..n-1)")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--runtime", required=True, help="frozen repo tree the Isaac run uses")
    ap.add_argument("--inputs", required=True, help="zoo inputs dir (scene.json, policy_config, thresholds, ...)")
    ap.add_argument("--driver", required=True, help="dir with score_trial.py")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-root", required=True, help="episode export root (tasks/<task>/ep_sim_...)")
    ap.add_argument("--start-pool", required=True)
    ap.add_argument("--scene", default=None, help="base scene json (default <inputs>/scene.json)")
    ap.add_argument("--hardware-input", required=True, help="sim hardware yaml for the run")
    ap.add_argument("--hardware", default="configs/hardware.nuc.yaml", help="real config for the export")
    ap.add_argument("--idle-tactile-from", required=True)
    ap.add_argument("--textures", nargs="*", default=["assets/sim/waffles/packet_top.png",
                                                      "assets/sim/waffles/packet_green_top.png"])
    ap.add_argument("--holdout", nargs="*", default=[], help="start-pool episode names never drawn")
    ap.add_argument("--band-frac", type=float, default=0.6)
    ap.add_argument("--packet-xy-sigma", type=float, default=0.01)
    ap.add_argument("--packet-xy-max", type=float, default=0.02)
    ap.add_argument("--yaw-sigma-deg", type=float, default=5.0)
    ap.add_argument("--yaw-max-deg", type=float, default=10.0)
    ap.add_argument("--light-frac", type=float, default=0.15)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--task", default="waffles")
    ap.add_argument("--expert-params", default=None)
    ap.add_argument("--episode", default=DEFAULT_EPISODE)
    ap.add_argument("--tactile-baseline", default=DEFAULT_TACTILE_BASELINE)
    ap.add_argument("--python", default=DEFAULT_PY)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the draws, run nothing")
    args = ap.parse_args(argv)

    pool_d = json.loads(Path(args.start_pool).read_text())
    pool = pool_d["states"]
    tcp = np.array([s["measured_tcp_pose"] for s in pool])
    band = (tcp[:, 1] <= -0.28) & (tcp[:, 2] >= 0.31)
    scene = json.loads(Path(args.scene or (Path(args.inputs) / "scene.json")).read_text())
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    ledger = raw_root / f"ledger_lane{args.lane}.jsonl"
    done = set()
    if ledger.exists():
        done = {json.loads(l)["index"] for l in ledger.read_text().splitlines() if l.strip()}
    fast_fail = 0
    for i in range(args.n):
        if i % args.lanes != args.lane or i in done:
            continue
        ep = draw_episode(i, args.seed, pool, band, args.band_frac, scene, args.textures, set(args.holdout),
                          packet_xy_sigma=args.packet_xy_sigma, packet_xy_max=args.packet_xy_max,
                          yaw_sigma_deg=args.yaw_sigma_deg, yaw_max_deg=args.yaw_max_deg,
                          light_frac=args.light_frac)
        if args.dry_run:
            print(json.dumps({"index": i, "start": ep["start"]["episode"], "band": ep["band"], **ep["draw"]}))
            continue
        row = run_episode(ep, args, raw_root)
        with open(ledger, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("index", "stage", "stop", "wall_s", "exported")}), flush=True)
        # a launch that dies in seconds is a configuration error, not a sim outcome:
        # stop the lane instead of burning the whole index range
        fast_fail = fast_fail + 1 if (row["rc"] != 0 and row["wall_s"] < 60) else 0
        if fast_fail >= 3:
            print("three consecutive fast failures — lane aborted", file=sys.stderr, flush=True)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
