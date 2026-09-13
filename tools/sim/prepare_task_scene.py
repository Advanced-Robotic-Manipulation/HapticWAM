#!/usr/bin/env python3
"""Export a real Carton or egg recording for the shared Isaac scene runner.

Use the PHANTOM recording environment (numpy, scipy, zarr, OpenCV). The
underlying exporter preserves native timestamps and measured robot motion,
requires explicit all-real driver provenance, and never edits the episode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

def main(argv: list[str] | None = None) -> None:
    from tools.sim.prepare_waffles import export_episode

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str.lower, choices=("carton", "egg"), required=True)
    parser.add_argument("--episode", type=Path, required=True, help="Original real episode directory")
    parser.add_argument("--out", type=Path, required=True, help="New directory outside the source episode")
    parser.add_argument("--split", choices=("fit", "heldout"), default="heldout")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--measured-only", action="store_true",
                        help="Explicitly omit proposal/action audit streams; preserve measured motion/RGB/native clocks")
    args = parser.parse_args(argv)
    if not 0 < args.fps <= 125:
        parser.error("--fps must be in (0, 125]")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    export_episode(
        args.episode,
        args.out,
        args.split,
        args.fps,
        args.max_seconds,
        allowed_task=args.task,
        exclude_actions=args.measured_only,
    )


if __name__ == "__main__":
    main()
