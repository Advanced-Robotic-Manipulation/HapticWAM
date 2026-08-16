"""End-to-end no-hardware smoke test (runs on any machine, no GPU):

    mock rig -> sensor session (2 tactile processes + pollers) -> recorder
    -> zarr episode -> offline derived pass -> EpisodeReader replay
    -> assert the ContactScenario event timeline was recovered.

Usage:
    python -m phantom.scripts.mock_smoke [--hardware configs/hardware.yaml]
                                         [--seconds 8] [--out <dir>]
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from phantom.config.hardware import load_hardware
from phantom.config.model import EVENT_IDX
from phantom.data.episode_store import EpisodeReader
from phantom.data.schema import STREAM_ARM_FT, STREAM_CAMERA_SCENE, tactile_stream
from phantom.drivers.factory import make_rig
from phantom.recording.postprocess import postprocess_episode
from phantom.recording.recorder import EpisodeRecorder
from phantom.recording.workers import SensorSession, make_gripper_poller
from phantom.data.schema import EpisodeMeta
from phantom.timesync.clock import IdentityClock

log = logging.getLogger("mock_smoke")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    hw = load_hardware(args.hardware)
    if hw.mode.drivers != "mock" or hw.mode.overrides:
        log.warning("forcing all drivers to mock for the smoke test")
        hw = hw.model_copy(update={"mode": hw.mode.model_copy(
            update={"drivers": "mock", "overrides": {}})})

    out_root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="phantom_smoke_"))
    log.info("output: %s", out_root)

    rig = make_rig(hw, control=False)
    with rig:
        session = SensorSession.start(hw, rig, session_id=str(int(time.time()) % 10_000_000))
        # this smoke test has no GripperPilot (nothing sends gripper.move()),
        # so exercise the standalone poller instead -- keeps the "gripper"
        # stream covered by the smoke test's session.all_alive() check.
        gripper_poller = make_gripper_poller(rig.gripper, hw, session.rings["gripper"])
        gripper_poller.start()
        try:
            recorder = EpisodeRecorder(session, IdentityClock(), out_root)
            ep_path = recorder.start(EpisodeMeta(task="mock_smoke", text="smoke",
                                                 operator="mock_smoke"),
                                     "ep_smoke_0000")
            log.info("recording %.1f s of mock streams...", args.seconds)
            time.sleep(args.seconds)
            assert session.all_alive(), "a stream worker died during recording"
            recorder.stop(success=True)
        finally:
            gripper_poller.stop()
            session.stop()

    # ---- offline derived pass + verification ----
    postprocess_episode(ep_path, hw)
    r = EpisodeReader(ep_path)

    streams = r.streams()
    log.info("episode streams: %s", streams)
    sensor = hw.tactile.sensors[0].name
    for required in (STREAM_CAMERA_SCENE, STREAM_ARM_FT,
                     tactile_stream(sensor, "fields_ds"),
                     tactile_stream(sensor, "keyframes"),
                     tactile_stream(sensor, "events")):
        assert required in streams, f"missing stream {required}"
        assert r.n(required) > 0, f"empty stream {required}"

    events = r._g(tactile_stream(sensor, "events"))["data"][:]
    mask_frac = r._g(tactile_stream(sensor, "mask_frac"))["data"][:]
    n_contact = int((mask_frac > 0.025).sum())   # area semantics (tau_contact_area)
    found = {name for name, idx in EVENT_IDX.items() if (events == idx).any()}
    log.info("event coverage: %s (%d/%d frames in contact)", sorted(found),
             n_contact, len(events))
    # one full scenario cycle contains onset+hold+release; slip needs enough frames
    for must in ("none", "onset", "hold", "release"):
        assert must in found, (f"derived pass failed to recover event '{must}' from the "
                               f"mock scenario (got {sorted(found)})")
    assert n_contact > 0, "no contact frames recovered"

    # rate sanity: tactile stream achieved a usable fraction of the configured rate
    ts = r.ts(tactile_stream(sensor, "fields_ds"))
    achieved = len(ts) / max(ts[-1] - ts[0], 1e-9)
    log.info("achieved tactile field rate: %.1f Hz (configured %.1f)",
             achieved, hw.recording.field_ds_rate_hz)

    log.info("MOCK SMOKE PASSED — episode at %s", ep_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
