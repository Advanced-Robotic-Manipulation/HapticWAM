"""Trial runner for the §8 protocol: campaign YAML -> physical trials ->
append-only ledger CSV. Resumable — on restart, completed (task, system,
seed, trial) cells are skipped. Sized for the ~2,100 + 960 trial campaign.

Campaign YAML schema:
    name: main_eval
    tasks: [fragile_grasp, slippery_place, insertion, wipe, regrasp]
    systems: [teacher, student, vision_only, no_distill, drop_tactile]
    seeds: [0, 1, 2]
    trials_per_cell: 20
    occlusion_tasks: [fragile_grasp, slippery_place]
    max_replans: 30
    checkpoints: {teacher: <path>, student: <path>, ...}
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import yaml

from phantom.config.hardware import HardwareConfig
from phantom.deploy.runtime import DeploymentRuntime
from phantom.inference.policy import PhantomPolicy

log = logging.getLogger(__name__)

LEDGER_FIELDS = ["timestamp", "campaign", "task", "system", "seed", "trial",
                 "occlusion", "episode_path", "success", "damage",
                 "stopped_reason", "n_replans", "notes"]


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, LEDGER_FIELDS).writeheader()

    def done(self, task: str, system: str, seed: int, trial: int,
             occlusion: bool) -> bool:
        return any(r["task"] == task and r["system"] == system
                   and int(r["seed"]) == seed and int(r["trial"]) == trial
                   and r["occlusion"] == str(occlusion) for r in self.rows)

    def append(self, row: dict) -> None:
        self.rows.append(row)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, LEDGER_FIELDS).writerow(row)


def run_campaign(campaign_yaml: Path, hw: HardwareConfig,
                 policies: dict[str, PhantomPolicy], out_root: Path) -> Path:
    cfg = yaml.safe_load(Path(campaign_yaml).read_text(encoding="utf-8"))
    campaign = cfg["name"]
    ledger = Ledger(out_root / campaign / "ledger.csv")
    occl_tasks = set(cfg.get("occlusion_tasks", []))

    for system in cfg["systems"]:
        policy = policies[system]
        with DeploymentRuntime(hw, policy, mode=system,
                               out_root=out_root / campaign / system) as rt:
            for task in cfg["tasks"]:
                occl_variants = [False] + ([True] if task in occl_tasks else [])
                for occlusion in occl_variants:
                    for seed in cfg["seeds"]:
                        for trial in range(cfg["trials_per_cell"]):
                            if ledger.done(task, system, seed, trial, occlusion):
                                continue
                            _run_one(rt, ledger, campaign, task, system, seed,
                                     trial, occlusion, cfg.get("max_replans", 30))
    return ledger.path


def _run_one(rt: DeploymentRuntime, ledger: Ledger, campaign: str, task: str,
             system: str, seed: int, trial: int, occlusion: bool,
             max_replans: int) -> None:
    print(f"\n=== {campaign} | {system} | {task} | seed {seed} | trial {trial} "
          f"| occlusion={'ON — set up curtain/lights' if occlusion else 'off'} ===")
    input("Reset the scene per the task checklist, then press Enter to start...")
    tags = ["occlusion"] if occlusion else []
    res = rt.run_episode(task=task, tags=tags, max_replans=max_replans,
                         policy_name=system)
    success = _ask_yn("Success?")
    damage = _ask_yn("Damage/breakage?")
    notes = input("Notes (Enter to skip): ").strip()
    ledger.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "campaign": campaign,
        "task": task, "system": system, "seed": seed, "trial": trial,
        "occlusion": str(occlusion),
        "episode_path": str(res.episode_path or ""),
        "success": str(success), "damage": str(damage),
        "stopped_reason": res.stopped_reason or "",
        "n_replans": res.n_replans, "notes": notes,
    })


def _ask_yn(prompt: str) -> bool:
    while True:
        a = input(f"{prompt} [y/n] ").strip().lower()
        if a in ("y", "n"):
            return a == "y"
