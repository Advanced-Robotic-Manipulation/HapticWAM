"""Program (3b): DAgger round driver (pipeline.md §5 — 'mandatory, new in v2').

One round = (a) student rollouts were recorded on the rig (scripts/
run_dagger_round.py — the rig still wears the sensors, so every rollout
episode contains FULL teacher inputs); (b) this driver relabels them with the
teacher and re-invokes HID distillation with the rollouts merged in.

    python -m phantom.train.dagger_driver --round 1 \
        --teacher-ckpt <teacher.pt> --student-ckpt <student_r0.pt> \
        --demos <episodes_root> --rollouts <rollout_root>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.dagger.manifest import write_manifest
from phantom.dagger.relabel import relabel_root
from phantom.train import distill_hid

log = logging.getLogger("dagger")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--skip-relabel", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute", default=None, help="forwarded to distill_hid")
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    paths = load_paths()

    if not args.skip_relabel:
        n = relabel_root(Path(args.rollouts), Path(args.teacher_ckpt), hw, paths,
                         device=args.device)
        log.info("relabeled %d rollout episodes", n)

    manifest = write_manifest(
        paths.runs_root / "dagger" / f"round_{args.round}.json",
        demos=Path(args.demos), rollouts=Path(args.rollouts), round_k=args.round)
    log.info("manifest: %s", manifest)

    distill_args = ["--teacher-ckpt", args.teacher_ckpt,
                    "--data", args.demos,
                    "--extra-data", args.rollouts,
                    "--dagger-round", str(args.round),
                    "--device", args.device]
    if args.max_steps is not None:
        distill_args += ["--max-steps", str(args.max_steps)]
    if args.compute is not None:
        distill_args += ["--compute", args.compute]
    return distill_hid.main(distill_args)


if __name__ == "__main__":
    raise SystemExit(main())
