#!/usr/bin/env python3
"""Build the frozen inputs and a stage study for the sim zoo evaluation (design: docs/results/sim_zoo_20260912/DESIGN.md).

Inputs are copied byte-for-byte from the audited genuine-eight v4 campaign
(scene, hardware, thresholds, controller configs with the rate backoff) and the
frozen 20-start pool; only the policy configs (recipes) are new. Nothing runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CAMPAIGN = REPO / "docs/results/teacher_recovery_20260910/finish/genuine8_rate_v4_frozen"
POOL = REPO / "docs/results/teacher_followthrough_20260909/design/qualification_campaign/frozen"
REMOTE_ROOT = "/dev/shm/phantom_sim_zoo_"
BOX = "/home/physicalai/phantom-icra-2027/phantom/"
PREPARED_EPISODE = "/home/physicalai/phantom-icra-2027/sim/waffles/evidence/fit/ep_waffles_1787395928_000"
TACTILE_BASELINE = "/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_pick_place_v1/sept4_no_contact_sensor_baseline.npz"

MODELS = {
    "v6": dict(checkpoint=BOX + "runs/teacher_v6/teacher_020000.pt", kind="phantom", system="teacher", policy_mode="teacher",
               inputs="camera+pads+wrist", role="teacher, clean retrain"),
    "ftA": dict(checkpoint=BOX + "runs/teacher_v5_ftA/BEST.pt", kind="phantom", system="teacher", policy_mode="teacher",
                inputs="camera+pads+wrist", role="teacher fine-tuned on rig grasps"),
    "stu_ftA_r2": dict(checkpoint=BOX + "runs/student_ftA_r2/student_002000.pt", kind="phantom", system="student",
                       policy_mode="student", inputs="camera+proprio", role="paper's sensor-free student"),
    "ctl_ftA": dict(checkpoint=BOX + "runs/control_ftA/teacher_001200.pt", kind="phantom", system="student",
                    policy_mode="student", inputs="camera+proprio", role="no-distillation control"),
    "stu_v6": dict(checkpoint=BOX + "runs/student_v6/student_001500.pt", kind="phantom", system="student",
                   policy_mode="student", inputs="camera+proprio", role="student of v6"),
    "A_visiononly": dict(checkpoint=BOX + "runs/cosmos_visiononly_v1/teacher_006000.pt", kind="phantom", system="auto",
                         policy_mode="vision_only", inputs="camera", role="vision-only baseline"),
    "pi05": dict(checkpoint=BOX + "runs/pi05_20k/pretrained_model", kind="lerobot", system="lerobot",
                 policy_mode="student", inputs="camera+proprio+text", role="external VLA baseline"),
}
BASE_RECIPE = dict(nfe=1, guidance=1.0, parity_fixes=True, persistent_noise=True, task_text="waffles",
                   drop_video=False, close_p=0.5, use_ema=True, terminal_veto=True)
RECIPES = {
    "K4_ir": dict(BASE_RECIPE, k_seeds=4, action_time_origin="inference_ready"),
    "K1_ir": dict(BASE_RECIPE, k_seeds=1, action_time_origin="inference_ready"),
    "K4_obs": dict(BASE_RECIPE, k_seeds=4, action_time_origin="observation"),
    "K1_obs": dict(BASE_RECIPE, k_seeds=1, action_time_origin="observation"),
    # placement-phase controller treatments (sim zoo follow-up 2026-09-12): same K4_ir recipe,
    # different boundary-projection config (see docs/results/sim_zoo_20260912/README.md, "next steps")
    "K4_ir_D1": dict(BASE_RECIPE, k_seeds=4, action_time_origin="inference_ready",
                     boundary_projection_config="boundary_projection__D1.json"),
    "K4_ir_D2": dict(BASE_RECIPE, k_seeds=4, action_time_origin="inference_ready",
                     boundary_projection_config="boundary_projection__D2.json"),
    "pi05": dict(nfe=10, guidance=1.0, k_seeds=1, parity_fixes=False, persistent_noise=False, task_text="pick up the waffles",
                 drop_video=False, close_p=0.5, use_ema=True, terminal_veto=False, action_time_origin="inference_ready",
                 # the sim's adaptive contract requires action_time_origin explicitly; the box pi0.5 server (main)
                 # predates the key, so runtime v13's client honours an explicit inference_ready for lerobot servers
                 max_play_steps=16, grip_play_steps=16),
}
CONFIGURABLE = ("nfe", "guidance", "k_seeds", "parity_fixes", "persistent_noise", "task_text", "drop_video", "close_p",
                "action_time_origin")
FIXED = dict(horizon_s=40.0, no_progress_stop_s=25.0, max_play_steps=10, grip_play_steps=10, servo_constraint_hold_s=2.5,
             robot_usd=None, waffle_placement="as collected (scene.json)", reach_limiter="bounded_v1 (elbow>=0.40 rad, 1.0 rad/s)")
# one start per stratum (the first interleaved quartet of the frozen qualification schedule)
STARTS_A = ["ep_waffles_1785594294_001_open", "ep_waffles_1785594503_010_open",
            "ep_waffles_1787393283_000_open", "ep_waffles_1787394107_011_open"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_inputs(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    design = json.loads((CAMPAIGN / "v6_relative_on_rate_v4/campaign.json").read_text())
    profile = design["adapter_profile"]
    for name in ("scene.json", "hardware_input.yaml"):
        shutil.copyfile(CAMPAIGN / name, out / name)
    (out / "thresholds.json").write_text(json.dumps(design["thresholds"], indent=2) + "\n")
    for field in ("terminal_veto", "placement_release", "boundary_projection"):
        (out / f"{field}.json").write_text(json.dumps(profile[field], indent=2) + "\n")
    assert profile["boundary_projection"]["rate_solver_backoff"] == "fk_rate_interior_v1"
    (out / "initial_states").mkdir()
    for p in sorted((POOL / "initial_states").glob("*.json")):
        shutil.copyfile(p, out / "initial_states" / p.name)
    shutil.copyfile(POOL / "initial_state_lineage.json", out / "initial_state_lineage.json")
    base = json.loads((out / "boundary_projection.json").read_text())
    # D1: solver inset 2 mm (>> the 0.2-0.3 mm FK/IK round-trip error that tripped final_envelope at the z ceiling)
    (out / "boundary_projection__D1.json").write_text(json.dumps(dict(base, solver_inset_m=0.002), indent=2) + "\n")
    # D2: D1 + upper_y_projection_v3 (raw-request band 3 cm; executed pose still projected onto the plane)
    (out / "boundary_projection__D2.json").write_text(
        json.dumps(dict(base, solver_inset_m=0.002, variant="upper_y_projection_v3", maximum_excursion_m=0.03), indent=2) + "\n")
    for rid, recipe in RECIPES.items():
        (out / f"policy_config__{rid}.json").write_text(
            json.dumps({k: recipe[k] for k in CONFIGURABLE if k in recipe}, indent=2) + "\n")
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "input_manifest.json").write_text(json.dumps({"source_campaign": str(CAMPAIGN), "source_pool": str(POOL),
                                                        "files": manifest}, indent=2) + "\n")
    return manifest


def stage_a1(seeds):
    trials = []
    lanes = {"K4_ir": 0, "K1_ir": 0, "K4_obs": 1, "K1_obs": 1}
    for recipe in ("K4_ir", "K1_ir", "K4_obs", "K1_obs"):
        for start in STARTS_A:
            for seed in seeds:
                trials.append(dict(id=f"A1__v6__{recipe}__{start}__seed{seed}", model="v6", recipe=recipe, start=start,
                                   seed=seed, lane=lanes[recipe]))
    return trials


def all_starts():
    return sorted(p.stem for p in (POOL / "initial_states").glob("*.json"))


def stage_b(seeds, recipe, models):
    """7 models x the A1 identities under the winning recipe; pi0.5 keeps its own recipe."""
    lanes = {"v6": 0, "ftA": 0, "stu_ftA_r2": 0, "ctl_ftA": 1, "stu_v6": 1, "A_visiononly": 1}
    trials = []
    for model in models:
        rid = "pi05" if model == "pi05" else recipe
        for start in STARTS_A:
            for seed in seeds:
                # pi0.5 is split by seed so both lanes carry 28 trials (each lane owns its own pi0.5 server)
                lane = lanes[model] if model != "pi05" else (0 if seed == seeds[0] else 1)
                trials.append(dict(id=f"B__{model}__{rid}__{start}__seed{seed}", model=model, recipe=rid, start=start,
                                   seed=seed, lane=lane))
    return trials


def stage_c(seed, setups):
    """finalists x all 20 starts x one fresh seed; one setup per lane."""
    trials = []
    for lane, (model, recipe) in enumerate(setups):
        for start in all_starts():
            trials.append(dict(id=f"C__{model}__{recipe}__{start}__seed{seed}", model=model, recipe=recipe, start=start,
                               seed=seed, lane=lane % 2))
    return trials


def stage_e1(seed, setups):
    """placement-fix treatments: model:recipe setups x all 20 starts x the Stage C seed; one setup per lane."""
    return [dict(t, id="E1" + t["id"][1:]) for t in stage_c(seed, setups)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["A1", "B", "C", "E1"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-shas", type=Path, required=True, help="json {checkpoint path: sha256}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[910501, 910502])
    parser.add_argument("--recipe", default=None, help="stage B: winning recipe id from A1")
    parser.add_argument("--models", nargs="+", default=list(MODELS), help="stage B: models to run")
    parser.add_argument("--setups", nargs="+", default=[], help="stage C: model:recipe finalists")
    args = parser.parse_args()
    shas = json.loads(args.checkpoint_shas.read_text())
    inputs_dir = args.out / "inputs"
    manifest = build_inputs(inputs_dir) if not inputs_dir.exists() else json.loads((inputs_dir / "input_manifest.json").read_text())["files"]
    models = {}
    for mid, m in MODELS.items():
        models[mid] = dict(m, sha256=shas.get(m["checkpoint"], "lerobot_directory" if m["kind"] == "lerobot" else None))
        if models[mid]["sha256"] is None:
            raise SystemExit("missing checkpoint sha for " + mid)
    if args.stage == "A1":
        trials = stage_a1(args.seeds)
    elif args.stage == "B":
        assert args.recipe in RECIPES, "stage B needs --recipe"
        trials = stage_b(args.seeds, args.recipe, args.models)
    elif args.stage == "C":
        assert args.setups, "stage C needs --setups model:recipe ..."
        trials = stage_c(args.seeds[0], [tuple(x.split(":")) for x in args.setups])
    else:
        assert args.setups, "stage E1 needs --setups model:recipe ..."
        trials = stage_e1(args.seeds[0], [tuple(x.split(":")) for x in args.setups])
    study = dict(study_id=f"sim_zoo_20260912_{args.stage}", stage=args.stage,
                 runtime=REMOTE_ROOT + "runtime_v14_20260912", inputs=REMOTE_ROOT + "inputs_20260912",
                 raw=REMOTE_ROOT + f"raw_20260912/{args.stage}", prepared_episode=PREPARED_EPISODE,
                 tactile_baseline=TACTILE_BASELINE, fixed=FIXED, models=models, recipes=RECIPES, trials=trials,
                 input_manifest_sha256=sha(inputs_dir / "input_manifest.json"), planned_trials=len(trials))
    path = args.out / f"study_{args.stage}.json"
    path.write_text(json.dumps(study, indent=2) + "\n")
    print(json.dumps({"study": str(path), "trials": len(trials), "inputs_files": len(manifest)}))


if __name__ == "__main__":
    main()
