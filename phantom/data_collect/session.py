"""CollectApp — the unified teleop + data-collection session.

One session = rig up -> [episodes...] -> rig down -> offload to the external
drive. The web panel (collect/panel.py) is the operator's only interface;
the Echo device's own start/stop flag doubles as the episode toggle.

Thread layout (who owns what):

  panel HTTP threads     buttons -> queue; /api/safeguard -> Safeguard.configure
  runner thread          run_session(): bring-up, MAIN LOOP (10 Hz), teardown,
                         offload
  echo reader thread     ~100 Hz serial poll -> cached filtered q + gripper
  streamer thread        125 Hz: leader target -> tracker -> servo_j  (the
                         device-rate command path — never touched by the loop)
  gripper pilot thread   <=50 Hz: mailbox -> gripper.move (continuous)
  safeguard thread       50 Hz ring scan -> on_trip: streamer.hold() +
                         gripper.open_now() (time-critical part only)
  sensor workers/pollers phantom.recording.workers (process per DM-Tac)
  recorder drain thread  rings -> zarr every 0.25 s
  rerun thread           15 Hz live view

The MAIN LOOP does bookkeeping only: buttons, episode lifecycle, absolute
action logging, safeguard/arm-guard bookkeeping, panel state. It is never in
the motion path (that was the reference stack's fatal mistake).

Recorded data (requirement 1 — everything absolute):
  arm_q / arm_qd / arm_tcp_pose / arm_tcp_speed / arm_ft   absolute state @ RTDE rate
  gripper (pos, obj 0..3)                                  absolute @ 100 Hz
  camera_scene_color 640x480 RGB                           @ 30 Hz
  tactile_{left,right}_* (mode-dependent, see below)       @ tactile rate
  actions_abs = [q_target(6) | gripper_cmd(1)]             absolute commanded action

Modes (requirement 4): full = all DM-Tac modalities PHANTOM needs (field
stack, keyframes, gel image, wrench, area); lite = wrench (+area) only —
fields/keyframes/gel are neither recorded nor visualized.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from phantom.data.derived import pose_delta
from phantom.data.schema import (STREAM_ACTIONS, STREAM_ACTIONS_ABS,
                                 EpisodeMeta)
from phantom.drivers.factory import make_rig
from phantom.recording.recorder import EpisodeRecorder
from phantom.recording.workers import SensorSession
from phantom.teleop.echo import EchoTeleop
from phantom.timesync.clock import IdentityClock, MasterClock

from phantom.data_collect.config import CollectConfig, SafeguardConfig
from phantom.data_collect.gripper import GripperPilot
from phantom.data_collect.offload import offload_session
from phantom.data_collect.safeguard import ArmGuard, TactileSafeguard
from phantom.data_collect.teleop import DirectServoStreamer, TrackPhase

log = logging.getLogger(__name__)

_LITE_DROP = ("_fields_ds", "_keyframes", "_infer_img", "_raw_img")


def lite_stream_filter(stream: str) -> bool:
    """lite mode keeps every stream except the heavy tactile ones."""
    return not any(stream.endswith(suffix) for suffix in _LITE_DROP)


class CollectEcho(EchoTeleop):
    """EchoTeleop + live gripper-tick calibration (panel Calibrate button).

    The base class maps raw squeeze ticks -> 0..1 through the configured
    open/closed ticks; here the endpoints are instance attributes that a
    running calibration capture can update without restarting anything."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.open_tick = float(cfg.gripper_open_tick)
        self.closed_tick = float(cfg.gripper_closed_tick)
        self.last_raw_tick: float = 0.0     # written by the reader thread

    def _gripper_01(self, tick: float) -> float:
        self.last_raw_tick = tick
        span = self.closed_tick - self.open_tick
        if abs(span) < 1e-9:
            return 0.0
        return float(np.clip((tick - self.open_tick) / span, 0.0, 1.0))

    def latest_gripper01(self) -> float | None:
        """Freshest EMA'd squeeze value for the GripperPilot's device-rate
        pull (None until the first device sample)."""
        with self._lock:
            return self._gripper if self._have_sample else None

    def calibrate(self, seconds: float = 3.0) -> tuple[float, float]:
        """Capture min/max raw ticks while the operator does one full
        open->squeeze cycle; apply as the new endpoints."""
        lo, hi = float("inf"), float("-inf")
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            t = self.last_raw_tick
            lo, hi = min(lo, t), max(hi, t)
            time.sleep(0.02)
        if hi - lo < 10:    # no real squeeze happened; keep current mapping
            raise RuntimeError(
                f"calibration saw a tick span of only {hi - lo:.0f} — do a full "
                "open -> squeeze cycle during the 3 s capture")
        self.open_tick, self.closed_tick = lo, hi
        return lo, hi


class CollectApp:
    """One instance per process; run_session() is called by CollectRunner in
    a worker thread, once per session."""

    def __init__(self, cc: CollectConfig, hw, panel):
        self.cc = cc
        self.hw = hw
        self.panel = panel
        self.safeguard: TactileSafeguard | None = None   # live during a session
        self.last_staging_dir: Path | None = None

    # ------------------------------------------------------------------
    def run_session(self, s) -> None:
        """s: SimpleNamespace(task, text, operator, mode, target_episodes).

        Thin wrapper: bring-up + main loop + teardown happen in
        _run_session_active (which raises on a fault); THIS finally guarantees
        the session is offloaded on EVERY exit path — clean quit, arm fault,
        safeguard stop, or a bring-up error — so episodes are never stranded in
        local staging by a failure."""
        cc, panel = self.cc, self.panel
        staging = Path(cc.storage.staging_root) / \
            f"{time.strftime('%Y%m%d_%H%M%S')}_{s.task}"
        staging.mkdir(parents=True, exist_ok=True)
        self.last_staging_dir = staging
        panel.state.update(staging_dir=str(staging), mode=s.mode)
        try:
            self._run_session_active(s, staging)
        finally:
            # Devices are released by _run_session_active's teardown before we
            # get here. Guarded so an offload error can never mask the original
            # fault, and an empty/never-recorded session just no-ops inside.
            panel.state.update(phase="stopping",
                               busy_detail="saving / offloading episodes")
            try:
                self.offload(staging)
            except Exception:
                log.exception("auto-offload failed")

    def _run_session_active(self, s, staging) -> None:
        """Rig bring-up, the 10 Hz main loop, and full teardown. Raises on a
        fault (motion-thread death, bring-up error); run_session's finally
        offloads regardless."""
        cc, hw, panel = self.cc, self.hw, self.panel

        # busy_detail: a short human-readable line the operator watches during
        # the multi-second bring-up / teardown (panel spinner sub-line). Each
        # meaningful step overwrites it; kept as the LAST step on an error so
        # the operator sees where it died.
        panel.state.update(busy_detail="connecting arm")
        rig = make_rig(hw, control=True)
        rig.worker_owned_tactile = True    # real DM-Tac is single-open
        with rig:
            # Pads-free tactile baseline: sensors zero their no-contact
            # reference when the workers connect — the gripper must be OPEN
            # first or every later reading carries a phantom force.
            panel.state.update(
                busy_detail="gripper activation (~10 s calibration cycle)")
            rig.gripper.open()
            deadline = time.perf_counter() + 10.0
            while time.perf_counter() < deadline:
                gst = rig.gripper.get_state()
                if not gst.moving and gst.position <= 0.08:
                    break
                time.sleep(0.2)
            else:
                log.warning("gripper did not settle open in 10 s (pos=%.2f) — "
                            "tactile baseline may be poisoned", gst.position)
            time.sleep(0.5)    # gel relax before the reference grab

            panel.state.update(busy_detail="clock sync")
            clock = (IdentityClock() if hw.mode.resolve("arm") == "mock"
                     else MasterClock.calibrate(rig.arm))
            panel.state.update(
                busy_detail="sensors + camera (open + warmup)")
            session = SensorSession.start(
                hw, rig, session_id=str(int(time.time()) % 10_000_000))
            recorder = EpisodeRecorder(
                session, clock, staging,
                stream_filter=lite_stream_filter if s.mode == "lite" else None)

            echo = CollectEcho(hw.teleop.echo)
            streamer = DirectServoStreamer(hw, rig.arm, echo, cc.teleop,
                                           rings=session.rings)
            # the pilot pulls the leader squeeze at its own rate — the session
            # loop is not in the gripper path (latency, staleness). The tactile
            # reader feeds the force-limited grasp (same ring path as the
            # safeguard / panel force readout).
            pilot = GripperPilot(hw, rig.gripper, cc.gripper, leader=echo,
                                 tactile_force=self._tactile_force_reader(session),
                                 ring=session.rings["gripper"],
                                 feedback_rate_hz=hw.gripper.feedback_rate_hz)
            arm_guard = ArmGuard(hw, session.rings)
            # safeguard settings HANDOFF: thresholds edited in the panel
            # between sessions must survive into this session (the panel
            # stashes them in its state when no live safeguard exists)
            snap = panel.state.snapshot()
            sg_cfg = SafeguardConfig(
                enabled=bool(snap.get("safeguard_enabled",
                                      cc.safeguard.enabled)),
                force_limit_n=float(snap.get("safeguard_force_limit_n",
                                             cc.safeguard.force_limit_n)),
                depth_limit=float(snap.get("safeguard_depth_limit",
                                           cc.safeguard.depth_limit)),
                release_gripper=cc.safeguard.release_gripper)
            safeguard = TactileSafeguard(
                hw, session.rings, sg_cfg,
                on_trip=lambda info: self._trip_now(streamer, pilot, info))
            self.safeguard = safeguard      # panel POST /api/safeguard target
            panel.state.update(safeguard_enabled=safeguard.enabled,
                               safeguard_force_limit_n=safeguard.force_limit_n,
                               safeguard_depth_limit=safeguard.depth_limit)

            viz = None
            native_viewer = None
            try:
                panel.state.update(busy_detail="starting rerun")
                from phantom.data_collect.rerun_view import (
                    CollectRerun, NativeViewer, session_viewer_url)
                viz = CollectRerun(hw, session.rings, mode=s.mode,
                                   rate_hz=cc.rerun.rate_hz,
                                   web_port=cc.rerun.web_port,
                                   ws_port=cc.rerun.grpc_port,
                                   server_memory_limit=cc.rerun.server_memory_limit,
                                   camera_jpeg_quality=cc.rerun.camera_jpeg_quality,
                                   blueprint_path=cc.rerun.blueprint_path,
                                   camera_rate_hz=cc.rerun.camera_rate_hz,
                                   camera_viz_downscale=cc.rerun.camera_viz_downscale)
                viz.start()
                if cc.rerun.viewer == "native":
                    # native wgpu window on the rig display (~0.3 cores vs ~3.5
                    # for the browser WASM viewer). Do NOT embed the web viewer
                    # in the panel — the iframe would burn those cores anyway.
                    native_viewer = NativeViewer(cc.rerun.grpc_port)
                    native_viewer.start()
                    panel.state.update(rerun_url="", viewer_native=True)
                elif cc.rerun.viewer == "off":
                    # No viewer window at all (setup/verification is done —
                    # this is a production collection run): the logger child
                    # still runs and keeps polling the rings, so the ~0.3 core
                    # the native window's rendering costs is the only thing
                    # saved, but that is exactly the CPU the 125 Hz control
                    # loop and the tactile workers want back during a campaign.
                    panel.state.update(rerun_url="", viewer_native=False)
                else:
                    # per-session nonce so the iframe reloads onto the fresh
                    # server (identical URLs never reload -> stale -> spinner)
                    panel.state.update(rerun_url=session_viewer_url(
                        viz.viewer_url, int(time.time() * 1000)),
                        viewer_native=False)
            except RuntimeError as e:
                log.warning("rerun disabled: %s", e)
                viz = None

            try:
                panel.state.update(busy_detail="engaging teleop")
                echo.start()
                streamer.start()
                pilot.start()
                safeguard.start()
                # bring-up complete — the main loop is now live
                panel.state.update(phase="running", busy_detail="")
                self._loop(s, rig, session, recorder, echo, streamer, pilot,
                           arm_guard, safeguard, viz)
            finally:
                # Teardown is multi-second; keep the operator informed. EVERY
                # step is guarded so a failure in one can't skip the rest — a
                # half-done teardown used to leave the RTDE control script /
                # single-controller slot or the sensors/serial held, and the
                # NEXT Start then failed. Order matters: stop the streamer/pilot
                # (they drive servoJ) BEFORE `with rig` exit disconnects the arm,
                # or a servoJ can race the disconnect.
                panel.state.update(phase="stopping",
                                   busy_detail="releasing devices")
                self.safeguard = None
                self._safe_stop("record drain", lambda: recorder.stop(abort=True))
                self._safe_stop("tactile safeguard", safeguard.stop)
                self._safe_stop("gripper pilot", pilot.stop)
                self._safe_stop("arm streamer", streamer.stop)
                self._safe_stop("echo leader", echo.stop)
                panel.state.update(busy_detail="closing sensors")
                if native_viewer is not None:
                    self._safe_stop("native viewer", native_viewer.stop)
                if viz is not None:
                    self._safe_stop("rerun", viz.stop)
                self._safe_stop("sensor workers", session.stop)

    # ------------------------------------------------------------------
    def offload(self, staging: Path | None = None) -> None:
        cc, panel = self.cc, self.panel
        staging = staging or self.last_staging_dir
        if staging is None or not Path(staging).exists():
            panel.state.update(offload_msg="nothing to offload",
                               busy_detail="no episodes to offload")
            return
        n_eps = len(sorted(p for p in Path(staging).glob("ep_*") if p.is_dir()))
        if n_eps == 0:
            panel.state.update(offload_msg="nothing to offload",
                               busy_detail="no episodes to offload")
            return
        panel.state.update(offload_running=True, offload_msg="offloading…",
                           offload_done=0, offload_total=n_eps,
                           busy_detail=f"offloading {n_eps} episodes to drive…")

        def progress(done, total, name):
            panel.state.update(
                offload_done=done, offload_total=total,
                offload_msg=f"copying {name}" if name else "verifying",
                busy_detail=f"offloading {done}/{total} to drive…" if name
                else "verifying offload…")

        res = offload_session(Path(staging), Path(cc.storage.external_drive),
                              min_free_gb=cc.storage.min_free_gb,
                              verify=cc.storage.verify,
                              keep_local=cc.storage.keep_local,
                              require_separate_device=
                              cc.storage.require_separate_device,
                              progress=progress)
        if res.ok:
            done_msg = (f"offload done: {res.episodes} episodes, "
                        f"{res.bytes_moved / 1e9:.2f} GB")
        else:
            # offload_session leaves the local copy untouched on any failure
            done_msg = f"offload FAILED (kept local): {res.error}"
        panel.state.update(offload_running=False, offload_msg=res.describe(),
                           offload_ok=res.ok, busy_detail=done_msg)
        if not res.ok:
            log.error(res.error)

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_stop(what: str, fn) -> None:
        """Run one teardown step, logging (never raising) on failure so the
        remaining steps still run — a single leaked device/interface (RTDE
        control slot, sensor worker, serial port) is what breaks the next Start.
        """
        try:
            fn()
        except Exception:
            log.exception("teardown: %s failed to stop cleanly", what)

    @staticmethod
    def _file_unlabeled(recorder, panel, path, name: str, why: str) -> None:
        """File a stopped-but-never-judged episode as NOT training-ready.

        Such an episode is already finalized on disk with success=None. Left
        that way it is indistinguishable from a judged demo: is_failure_demo()
        is False for success=None, so WindowSampler hands it action_weight 1.0
        and a take the operator never accepted fully supervises action
        imitation. Marking it status='aborted' + tag 'unlabeled' keeps every
        byte on disk (the offload still copies it to the drive) while
        list_episodes() and the hub uploader both skip it. Recoverable by hand
        — edit meta.json — if the take turns out to be worth keeping.
        """
        try:
            recorder.relabel(path, status="aborted", tags=["unlabeled"],
                             notes=f"no operator verdict ({why})")
        except Exception:
            log.exception("could not mark %s as unlabeled", name)
        panel.state.add_episode(name=name, outcome="unjudged",
                                tags=["unlabeled"])
        log.warning("episode %s got no operator verdict (%s) — filed as "
                    "unlabeled, excluded from training", name, why)

    def _trip_now(self, streamer, pilot, info) -> None:
        """Time-critical part of a safeguard trip — runs ON the safeguard
        thread: freeze the arm, open the gripper. Episode bookkeeping happens
        in the main loop (single owner of the recorder)."""
        streamer.hold()
        if self.cc.safeguard.release_gripper:
            pilot.open_now()

    # ------------------------------------------------------------------
    def _loop(self, s, rig, session, recorder, echo, streamer, pilot,
              arm_guard, safeguard, viz) -> None:
        cc, hw, panel = self.cc, self.hw, self.panel

        def note(text, level="INFO"):
            if viz is not None:
                viz.log_event(text, level=level)

        period = 1.0 / hw.control.action_rate_hz
        recording = False
        # a stopped-but-not-yet-judged episode: finalized on disk (no verdict),
        # waiting for the operator to press success / fail / discard
        pending_path = None
        pending_name = ""
        quit_armed = False     # one "End session" already refused (no verdict)
        trip_handled = False
        prev_tcp = None        # previous MEASURED TCP pose (Delta-EE reference)
        arm_hold = ""          # "" | "pstop" | "wrench"
        ep_count = 0        # episodes KEPT (drives the panel count/target)
        ep_seq = 0          # monotonic naming index (never reused after a discard)
        ep_name = ""
        tick_hz = 0.0
        calib_thread = None
        log.info("session %s (mode=%s): panel controls collection", s.task, s.mode)

        while True:
            t0 = time.perf_counter()
            cmd = echo.poll()          # device button edge + gripper (display)
            buttons = dict(cmd.buttons)
            buttons.update(panel.pop_buttons())

            for worker in (streamer, pilot):
                if worker.error is not None:
                    cause = worker.error
                    # Arm fault (servoJ rejected / protective stop / RTDE drop).
                    # Cleanly abort ONLY the episode being written right now: keep
                    # its partial data on disk as 'aborted' (offload still takes
                    # it), mark it in the panel. Earlier episodes keep the
                    # operator's verdicts; a fault BETWEEN episodes aborts nothing.
                    # Then raise -> full teardown + auto-offload (run_session).
                    if recording:
                        panel.state.update(
                            busy_detail="arm fault — stopping episode")
                        recorder.stop(abort=True)
                        recording = False
                        panel.state.add_episode(name=ep_name, outcome="aborted",
                                                tags=["arm_fault"])
                        panel.state.update(recording=False,
                                           busy_detail="saved episode (aborted)")
                    if pending_path is not None:
                        # a fault cannot wait for a verdict: the episode the
                        # operator had not judged yet must not be shipped as a
                        # full-weight demo by the auto-offload that follows
                        self._file_unlabeled(recorder, panel, pending_path,
                                             pending_name, "motion thread died")
                        pending_path, pending_name = None, ""
                    raise RuntimeError(f"motion thread died: {cause}")

            # ---- quit -----------------------------------------------------
            if buttons.get("quit") and pending_path is not None and (
                    buttons.get("success") or buttons.get("fail")
                    or buttons.get("abort")):
                # A verdict and "End session" landed in the SAME 100 ms pop
                # (both cards are on screen at once). The quit branch runs
                # first, so handling it here would throw the operator's
                # judgment away: re-queue the quit instead and let the verdict
                # below apply — it always clears pending_path, so the next
                # tick quits cleanly.
                panel.push_button("quit")
                buttons["quit"] = False
            if buttons.get("quit"):
                if pending_path is not None and not quit_armed:
                    # "End session" while an episode still awaits a verdict:
                    # refuse ONCE and say so. Ending here used to ship the
                    # unjudged take as an ordinary full-weight demo (it is
                    # already finalized on disk with success=None, which
                    # nothing downstream distinguishes from an accepted one).
                    # A second press is honoured — the session must never be
                    # strandable on the rig — and files it as unlabeled.
                    quit_armed = True
                    panel.state.update(
                        phase="running",
                        error=f"{pending_name} has no verdict — press Success "
                              "/ Fail / Discard, or press End session again "
                              "to file it as unlabeled (excluded from "
                              "training)",
                        episode_phase="awaiting_verdict",
                        episode_detail="verdict needed before the session ends")
                    note("End session refused: episode still awaiting a "
                         "verdict", "WARN")
                    continue
                if recording:
                    recorder.stop(abort=True)
                if pending_path is not None:
                    self._file_unlabeled(recorder, panel, pending_path,
                                         pending_name, "session ended")
                    pending_path, pending_name = None, ""
                log.info("session quit")
                return

            # ---- safeguard trip bookkeeping (motion already frozen by the
            #      safeguard thread; this is the single-owner recorder part)
            trip = safeguard.tripped
            if trip is not None and not trip_handled:
                trip_handled = True
                if recording:
                    recorder.stop(success=False,
                                  notes=f"safeguard:{trip.sensor}:{trip.kind}")
                    recording = False
                    panel.state.add_episode(name=ep_name, outcome="safeguard",
                                            tags=[trip.kind])
                    panel.state.update(episode_phase="idle", episode_detail="")
                panel.state.update(
                    recording=False, safeguard_tripped=True,
                    safeguard_msg=trip.describe())
                note(trip.describe(), "ERROR")
            if trip is None and trip_handled:
                # latch cleared by DISABLING the safeguard from the panel —
                # that must also release the frozen motion, or the rig is
                # stuck with no button that can free it
                trip_handled = False
                pilot.release()
                if not arm_hold:
                    streamer.resume()
                panel.state.update(safeguard_tripped=False, safeguard_msg="")
                note("safeguard disabled by operator — motion released", "WARN")

            # ---- resume button (the ONLY way out of a tactile trip) -------
            if buttons.get("resume_safeguard") and trip is not None:
                # ORDER MATTERS: re-arm motion FIRST, clear the latch LAST.
                # The moment reset() clears the latch the 50 Hz safeguard
                # thread may re-trip (pad still loaded); with motion already
                # re-armed, that fresh trip's hold()/open_now() land on live
                # objects and stick — the reversed order let a re-trip be
                # silently overridden while the latch stayed stuck.
                pilot.release()
                if not arm_hold:            # an active ARM hold keeps priority
                    streamer.resume()
                trip_handled = False
                panel.state.update(safeguard_tripped=False, safeguard_msg="")
                safeguard.reset()
                log.info("safeguard resumed from the panel")
                note("safeguard resumed by operator")

            # ---- arm guard (auto-resume, reference behaviour) -------------
            if trip is None:           # tactile trip owns the hold when active
                kind = arm_guard.check()
                if kind and not arm_hold:
                    arm_hold = kind
                    streamer.hold()
                    if recording:
                        # Do NOT auto-file this as a failure: finalize without a
                        # verdict and hand it to the operator, who can discard
                        # it (the usual choice after a protective stop) or keep
                        # it as a fail. Clearing the pendant then lifts the hold
                        # and the next episode starts normally - no session
                        # restart.
                        pending_path = recorder.stop(
                            success=None, notes=f"arm_guard:{kind}")
                        pending_name = pending_path.name if pending_path else ""
                        recording = False
                        panel.state.update(
                            episode_phase="awaiting_verdict",
                            episode_detail=f"{kind}: discard or keep this episode")
                    panel.state.update(recording=False, arm_hold=kind)
                    note(f"ARM GUARD: {kind}", "ERROR")
                elif arm_hold and not kind and arm_guard.recovered():
                    if streamer.phase is TrackPhase.FAULT:
                        # a protective stop killed the control script: recovery
                        # is explicit (panel Re-engage), never automatic
                        panel.state.update(
                            arm_hold=arm_hold,
                            busy_detail="pendant clear - press Re-engage arm")
                    else:
                        arm_hold = ""
                        streamer.resume()
                        panel.state.update(arm_hold="")
                        note("arm guard cleared — teleop re-engaging")

            # ---- operator re-engage after an arm fault --------------------
            if buttons.get("reengage"):
                # Publish BEFORE the blocking call: reengage() stalls this loop
                # for up to ~10 s, but the panel's SSE thread keeps serving, so
                # the operator sees a spinner instead of a dead UI.
                panel.state.update(reengaging=True, error="",
                                   busy_detail="rebuilding the UR control "
                                               "script (up to ~10 s)…")
                ok, why = streamer.reengage()
                panel.state.update(reengaging=False, busy_detail="")
                if ok:
                    arm_hold = ""
                    panel.state.update(arm_hold="", error="")
                    note("arm re-engaged by the operator")
                else:
                    panel.state.update(error=f"re-engage failed: {why}")
                    note(f"re-engage failed: {why}", "ERROR")

            # ---- gripper calibration --------------------------------------
            if buttons.get("calibrate_gripper") and calib_thread is None:
                def _calib():
                    panel.state.update(calibrating=True,
                                       calib_msg="squeeze fully, then release…")
                    try:
                        lo, hi = echo.calibrate(3.0)
                        panel.state.update(
                            calib_msg=f"ticks: open={lo:.0f} closed={hi:.0f} — "
                                      "copy into configs/data_collect.yaml to persist")
                    except RuntimeError as e:
                        panel.state.update(calib_msg=str(e))
                    finally:
                        panel.state.update(calibrating=False)
                calib_thread = threading.Thread(target=_calib, daemon=True)
                calib_thread.start()
            if calib_thread is not None and not calib_thread.is_alive():
                calib_thread = None

            # ---- episode lifecycle (state machine) -----------------------
            #   idle -> [start_stop] -> recording
            #   recording -> [start_stop]  -> finalizing -> awaiting_verdict
            #             -> [discard]      -> finalizing -> idle (aborted)
            #   awaiting_verdict -> [success|fail|discard] -> saving -> idle
            # The verdict is applied AFTER the episode is stopped and finalized;
            # success/fail/discard only apply while a pending episode exists.
            blocked = trip is not None or arm_hold
            end_success = buttons.get("success")
            end_fail = buttons.get("fail")
            end_discard = buttons.get("abort")
            toggle = buttons.get("start_stop")
            if recording and end_discard:
                # discard straight from recording: data kept on disk, aborted
                panel.state.update(episode_phase="finalizing",
                                   episode_detail="discarding episode")
                # operator judgment — unlike the arm-fault abort, a discard
                # SHOULD physically remove the take (explicit delete flag)
                recorder.stop(abort=True, delete=True)
                recording = False
                ep_count = max(0, ep_count - 1)   # discarded: does not count
                panel.state.add_episode(name=ep_name, outcome="discarded")
                panel.state.update(episode_phase="idle", episode_detail="",
                                   ep_count=ep_count)
                note("episode discarded (deleted)", "WARN")
            elif recording and toggle:
                # stop -> finalize WITHOUT a verdict; operator judges next
                panel.state.update(episode_phase="finalizing",
                                   episode_detail="finalizing episode")
                path = recorder.stop(success=None)
                recording = False
                pending_path, pending_name = path, path.name
                log.info("recorded %s — awaiting verdict", pending_name)
                note(f"episode recorded: {pending_name} — awaiting verdict")
                panel.state.update(episode_phase="awaiting_verdict",
                                   episode_detail="awaiting verdict")
            elif pending_path is not None and (end_success or end_fail
                                               or end_discard):
                # apply the operator verdict to the finalized episode
                panel.state.update(episode_phase="saving",
                                   episode_detail="saving episode")
                if end_discard:
                    recorder.relabel(pending_path, discard=True)
                    ep_count = max(0, ep_count - 1)   # discarded: does not count
                    panel.state.update(ep_count=ep_count)
                    outcome = "discarded"
                else:
                    recorder.relabel(pending_path, success=bool(end_success))
                    outcome = "success" if end_success else "fail"
                panel.state.add_episode(name=pending_name, outcome=outcome)
                note(f"episode {pending_name}: {outcome}")
                pending_path, pending_name = None, ""
                # the verdict clears a refused End-session: the next press
                # ends the session normally, and the banner goes away
                quit_armed = False
                panel.state.update(episode_phase="idle", episode_detail="",
                                   error="")
                if (not end_discard and s.target_episodes
                        and ep_count >= s.target_episodes):
                    # target hit on an accepted verdict: close the session the
                    # same way 'quit' does — the runner tears down, offloads,
                    # and the panel is ready to configure the next session
                    log.info("session target of %d episodes reached — closing",
                             s.target_episodes)
                    note(f"target {s.target_episodes}/{s.target_episodes} "
                         "reached — saving session")
                    return
            elif (not recording and pending_path is None and toggle
                  and not blocked):
                if s.target_episodes and ep_count >= s.target_episodes:
                    # never record past the target: the session is about to
                    # close (or a discard just reopened headroom — then this
                    # guard no longer matches and recording proceeds)
                    note(f"session target ({s.target_episodes}) already "
                         "reached — no further episodes", "WARN")
                    continue
                if not session.all_alive():
                    raise RuntimeError("a stream worker died — restart the session")
                meta = EpisodeMeta(task=s.task, text=s.text or s.task,
                                   operator=s.operator, policy="teleop",
                                   tags=[s.mode])
                if hw.wrist_ft.bias_on_episode_start:
                    try:
                        # RTDEControlInterface is not thread-safe: serialize
                        # against the streamer's 125 Hz servo_j
                        with streamer.ctl_lock:
                            rig.arm.zero_ft()
                    except Exception:
                        log.exception("zero_ft failed (continuing)")
                path = recorder.start(
                    meta, f"ep_{s.task}_{int(time.time())}_{ep_seq:03d}")
                ep_name = path.name
                ep_seq += 1
                ep_count += 1
                recording = True
                # start each episode's Delta-EE chain fresh: carrying the
                # previous episode's last pose would make the first action a
                # delta spanning the whole inter-episode gap
                prev_tcp = None
                log.info("RECORDING -> %s", ep_name)
                note(f"RECORDING {ep_name}")
                panel.state.update(episode_phase="recording",
                                   episode_detail="recording…",
                                   t_ep_start=time.time())

            # ---- worker death mid-episode: a truncated episode must never
            #      be finalized as a normal save -----------------------------
            if recording and not session.all_alive():
                recorder.stop(success=False, notes="worker_died")
                recording = False
                panel.state.add_episode(name=ep_name, outcome="worker_died")
                panel.state.update(episode_phase="idle", episode_detail="")
                raise RuntimeError(
                    "a sensor worker/poller died mid-episode — episode saved "
                    "as failure; restart the session")

            # ---- action logging: BOTH conventions -------------------------
            #   actions_abs : ABSOLUTE [q_target(6) | gripper(1)] -- what the
            #                 Echo leader actually commanded (requirement 1).
            #   actions     : canonical Delta-EE [dpose(6) | gripper(1)],
            #                 derived from consecutive MEASURED TCP poses via
            #                 derived.pose_delta -- the convention
            #                 deploy/executor.py replays (cumsum in rotvec
            #                 space) and the ONLY action stream WindowSampler
            #                 reads. Same derivation record_episodes.py uses
            #                 for this joint-space leader. One tick late by
            #                 construction; actions and states carry
            #                 independent timestamps and windows sample by
            #                 nearest ts, so the lag is not a correctness bug.
            if recording:
                q_cmd = streamer.last_cmd
                grip_cmd = pilot.last_sent
                grip = float(grip_cmd if grip_cmd is not None else cmd.gripper)
                t_act = time.perf_counter()
                if q_cmd is not None:
                    recorder.record_action(
                        t_act,
                        np.concatenate([q_cmd, [grip]]).astype(np.float32),
                        stream=STREAM_ACTIONS_ABS)
                tcp = self._tcp_pose(session)
                if tcp is not None:
                    if prev_tcp is not None:
                        recorder.record_action(
                            t_act,
                            np.concatenate([pose_delta(prev_tcp, tcp),
                                            [grip]]).astype(np.float32),
                            stream=STREAM_ACTIONS)
                    prev_tcp = tcp

            # ---- panel live state ----------------------------------------
            dt = time.perf_counter() - t0
            tick_hz = 0.9 * tick_hz + 0.1 / max(dt, period) if tick_hz \
                else 1.0 / period
            panel.state.update(
                recording=recording, episode=ep_name if recording else "",
                ep_count=ep_count, tick_hz=tick_hz,
                track_phase=streamer.phase.value,
                workspace_hold=streamer.in_workspace_hold,
                wrench_N=self._wrench_n(session),
                tactile_force_N=self._tactile_n(session),
                gripper=float(cmd.gripper),
                safeguard_enabled=safeguard.enabled,
                safeguard_force_limit_n=safeguard.force_limit_n,
                safeguard_depth_limit=safeguard.depth_limit)

            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    # ------------------------------------------------------------------
    def _tcp_pose(self, session):
        """Latest MEASURED TCP pose (6,) from the arm ring, or None.

        Read from the ring rather than polling the arm directly: the arm
        poller already fills it at rtde_receive_hz and it is the same data the
        recorder drains, so this adds no RTDE traffic and no lock contention
        with the 125 Hz servo path."""
        ring = session.rings.get("arm")
        if ring is None:
            return None
        ts, d = ring.latest(1)
        if not len(ts):
            return None
        return np.asarray(d["tcp_pose"][0], dtype=np.float64).copy()

    def _wrench_n(self, session) -> float:
        ring = session.rings.get("arm")
        if ring is None:
            return 0.0
        ts, d = ring.latest(1)
        return float(np.linalg.norm(np.asarray(d["ft"][0])[:3])) if len(ts) else 0.0

    def _tactile_n(self, session) -> float:
        """Max resultant fingertip force across sensors (panel display)."""
        peak = 0.0
        for s in self.hw.tactile.sensors:
            ring = session.rings.get(f"tactile_{s.name}")
            if ring is None:
                continue
            ts, d = ring.latest(1)
            if len(ts):
                w = np.asarray(d["wrench"][0], dtype=np.float32).reshape(-1)
                if w.size >= 3:
                    peak = max(peak, float(np.linalg.norm(w[:3])))
        return peak

    def _tactile_force_reader(self, session):
        """Callable for the gripper's force-limited grasp: returns
        (peak_force_n, freshest_age_s) — max resultant force over both pads and
        the freshest sample's age (perf_counter clock, as the safeguard uses) —
        or None if no pad has produced a sample yet. The pilot itself decides
        staleness (grasp_stale_s), so this stays a pure read of the same rings."""
        sensors = self.hw.tactile.sensors
        rings = session.rings

        def read():
            now = time.perf_counter()
            peak = 0.0
            freshest_age = None
            for s in sensors:
                ring = rings.get(f"tactile_{s.name}")
                if ring is None:
                    continue
                ts, d = ring.latest(1)
                if not len(ts):
                    continue
                age = now - float(ts[0])
                freshest_age = age if freshest_age is None else min(freshest_age, age)
                w = np.asarray(d["wrench"][0], dtype=np.float32).reshape(-1)
                if w.size >= 3:
                    peak = max(peak, float(np.linalg.norm(w[:3])))
            if freshest_age is None:
                return None
            return peak, freshest_age

        return read
