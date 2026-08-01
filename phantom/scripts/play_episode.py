"""Play back / verify a recorded episode from the command line.

    python -m phantom.scripts.play_episode <episode_dir>
        [--config configs/data_collect.yaml] [--speed 1.0] [--verify-only]

The same verifier + player the collect panel embeds (docs/data_collect_app.md):
prints the recording-verification report (exit code 1 if the episode is
incomplete), then — unless --verify-only — streams every recorded modality
into a rerun web viewer at the ports from configs/data_collect.yaml. The
viewer stays up after playback so the data can be scrubbed; Ctrl-C exits.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path


def _print_report(rep: dict) -> None:
    meta = rep.get("meta", {})
    print(f"episode {rep['name']}  mode={rep['mode']}  "
          f"status={meta.get('status', '?')}  task={meta.get('task', '?')}")
    for s in rep["streams"]:
        flags = "; ".join(list(s["issues"]) + list(s["warns"]))
        print(f"  {s['name']:<30} {s['rows']:>7} rows  {s['dur_s']:>7.1f} s  "
              f"{s['hz']:>7.1f} Hz  {s['shape']:<14} {flags}")
    for m in rep["missing"]:
        print(f"  MISSING stream: {m}")
    for i in rep["issues"]:
        print(f"  ISSUE: {i}")
    for w in rep.get("warns", []):
        print(f"  WARN: {w}")
    print("verdict:", "OK — everything recorded" if rep["ok"]
          else "ISSUES FOUND")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="phantom.scripts.play_episode")
    ap.add_argument("episode", help="episode directory (contains meta.json)")
    ap.add_argument("--config", default=None,
                    help="data_collect yaml (default configs/data_collect.yaml)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (0.1–16)")
    ap.add_argument("--verify-only", action="store_true",
                    help="print the verification report and exit (no viewer)")
    args = ap.parse_args(argv)

    ep = Path(args.episode)
    if not (ep / "meta.json").is_file():
        print(f"error: {ep} is not an episode directory (no meta.json)")
        return 2

    from phantom.data_collect.config import load_collect
    from phantom.data_collect.playback import (EpisodePlayer, PlayerView,
                                               verify_episode)

    cc, hw = load_collect(args.config)
    rep = verify_episode(ep, hw=hw)
    _print_report(rep)
    if args.verify_only:
        return 0 if rep["ok"] else 1

    view = PlayerView(hw, web_port=cc.rerun.web_port,
                      ws_port=cc.rerun.grpc_port)
    try:
        view.start()
    except RuntimeError as e:
        print(f"error: {e}")
        return 2
    print(f"[play_episode] rerun viewer: {view.viewer_url}")

    last_line = ""

    def on_status(st: dict) -> None:
        nonlocal last_line
        line = f"{st['state']:>8}  {st['pos_s']:5.0f}/{st['dur_s']:.0f} s"
        if line != last_line:
            last_line = line
            print(f"[play_episode] {line}" +
                  (f"  ({st['msg']})" if st.get("msg") else ""))

    player = EpisodePlayer(view, on_status=on_status)
    player.play(ep, speed=args.speed, hw=hw)
    try:
        while player.active():
            time.sleep(0.5)
        st = player.status()
        if st["state"] == "error":
            print(f"[play_episode] playback ERROR: {st.get('msg', '')}")
            return 2
        print("[play_episode] playback finished — viewer stays up for "
              "scrubbing, Ctrl-C to exit")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        player.stop()
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
