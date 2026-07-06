"""Offline derived-channel pass over recorded episodes (recomputable whenever
the derived.* thresholds change after the bench).

    python -m phantom.scripts.postprocess_episodes --data <episodes_root> [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from phantom.config.hardware import load_hardware
from phantom.recording.postprocess import postprocess_root


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    hw = load_hardware(args.hardware)
    n = postprocess_root(Path(args.data), hw, overwrite=args.overwrite)
    print(f"postprocessed {n} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
