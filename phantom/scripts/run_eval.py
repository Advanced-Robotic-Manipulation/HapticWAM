"""[Linux/5090] Evaluation campaign runner + aggregation (pipeline.md §8).

Run trials (resumable; ledger CSV is the durable record):
    python -m phantom.scripts.run_eval --campaign configs/eval_campaign.yaml
Aggregate only:
    python -m phantom.scripts.run_eval --campaign ... --aggregate-only \
        [--episode-metrics] [--tau-obj 1.0]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.eval.aggregate import aggregate, write_report
from phantom.eval.trial_runner import run_campaign
from phantom.scripts.run_deploy import build_policy

log = logging.getLogger("run_eval")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--episode-metrics", action="store_true")
    ap.add_argument("--tau-obj", type=float, default=1.0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()
    campaign_cfg = yaml.safe_load(Path(args.campaign).read_text(encoding="utf-8"))
    out_root = paths.runs_root / "eval"
    ledger = out_root / campaign_cfg["name"] / "ledger.csv"

    if not args.aggregate_only:
        checkpoints = campaign_cfg.get("checkpoints", {})
        policies = {}
        for system in campaign_cfg["systems"]:
            # text/task: build_policy conditions the policy on (text or task);
            # per-trial task strings are set by run_campaign at episode time,
            # so the build-time default is empty (was: AttributeError — this
            # script has been unrunnable since the policy grew task_text)
            pol_args = SimpleNamespace(
                system=system, ckpt=checkpoints.get(system, ""), nfe=None,
                drop_video=False, ema=True, tiny=args.tiny, device=args.device,
                text="", task="")
            policies[system] = build_policy(pol_args, hw, paths)
        ledger = run_campaign(Path(args.campaign), hw, policies, out_root)

    result = aggregate(ledger, hw, tau_obj=args.tau_obj,
                       compute_episode_metrics=args.episode_metrics)
    report = write_report(result, ledger.parent / "report")
    log.info("report: %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
