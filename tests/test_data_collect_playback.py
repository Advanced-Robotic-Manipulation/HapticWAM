"""collect/playback.py — the recording verifier, episode scanning, the
player loop (stub rerun view), and session/playback mutual exclusion."""

import json
import time
import types
import urllib.request

import numpy as np
import pytest
import zarr

from phantom.data.episode_store import EpisodeWriter
from phantom.data.schema import EpisodeMeta
from phantom.data_collect.panel import CollectPanel, CollectPanelState, CollectRunner
from phantom.data_collect.playback import (EpisodePlayer, PlaybackController,
                                           _expect_hz, _split_tactile,
                                           expected_streams, verify_episode)
from phantom_test_utils import make_small_hw


@pytest.fixture(scope="module")
def hw():
    return make_small_hw()


# ---------------------------------------------------------------------------
# episode fabrication
# ---------------------------------------------------------------------------

def _write_episode(root, hw, *, mode="lite", name="ep_demo_100_000",
                   dur=2.0, finalize=True, drop=(), t0=100.0):
    meta = EpisodeMeta(task="demo", tags=[mode])
    w = EpisodeWriter(root / name, hw, meta)

    def add(stream, hz, shape, dtype=np.float32, fill=0.0):
        if stream in drop:
            return
        n = max(2, int(round(dur * hz)))
        ts = np.linspace(t0, t0 + dur, n)
        w.append(stream, ts, np.full((n, *shape), fill, dtype=dtype))

    r = hw.arm.rtde_receive_hz
    add("arm_q", r, (6,)), add("arm_qd", r, (6,))
    add("arm_tcp_pose", r, (6,)), add("arm_tcp_speed", r, (6,))
    add("arm_ft", r, (6,))
    add("gripper", hw.gripper.feedback_rate_hz, (2,))
    add("actions_abs", hw.control.action_rate_hz, (7,))
    # collect writes BOTH: absolute q_target AND the canonical Delta-EE
    # actions WindowSampler consumes (session.py records them together)
    add("actions", hw.control.action_rate_hz, (7,))
    add("camera_scene_color", hw.cameras.scene.fps, (60, 80, 3), np.uint8)
    for s in hw.tactile.sensors:
        add(f"tactile_{s.name}_wrench", hw.recording.field_ds_rate_hz, (6,))
        add(f"tactile_{s.name}_area", hw.recording.field_ds_rate_hz, (1,))
        if mode == "full":
            add(f"tactile_{s.name}_fields_ds", hw.recording.field_ds_rate_hz,
                (24, 32, 8), np.float16)
            add(f"tactile_{s.name}_keyframes", hw.recording.keyframe_rate_hz,
                (48, 64, 1), np.uint8)
            add(f"tactile_{s.name}_infer_img", hw.recording.infer_img_rate_hz,
                (60, 80, 1), np.uint8)
    if finalize:
        w.finalize(success=True)
    return root / name


# ---------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------

def test_verify_ok_lite(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="lite")
    rep = verify_episode(ep, hw=hw)
    assert rep["ok"], (rep["issues"], rep["missing"],
                       [(s["name"], s["issues"]) for s in rep["streams"]])
    assert rep["mode"] == "lite" and rep["missing"] == []
    names = {s["name"] for s in rep["streams"]}
    assert "actions_abs" in names and "camera_scene_color" in names
    arm_q = next(s for s in rep["streams"] if s["name"] == "arm_q")
    assert arm_q["rows"] > 100 and abs(arm_q["dur_s"] - 2.0) < 0.1
    assert not arm_q["warns"], arm_q["warns"]


def test_verify_ok_full(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="full")
    rep = verify_episode(ep, hw=hw)
    assert rep["ok"], (rep["issues"], rep["missing"])
    names = {s["name"] for s in rep["streams"]}
    for s in hw.tactile.sensors:
        assert f"tactile_{s.name}_fields_ds" in names
        assert f"tactile_{s.name}_keyframes" in names


def test_verify_detects_missing_stream(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="full",
                        drop=("tactile_left_keyframes",))
    rep = verify_episode(ep, hw=hw)
    assert not rep["ok"]
    assert "tactile_left_keyframes" in rep["missing"]


def test_verify_detects_nonmonotonic_and_nan(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="lite", drop=("arm_ft",))
    g = zarr.open_group(str(ep / "arm_ft.zarr"), mode="a")
    data = np.zeros((10, 6), dtype=np.float32)
    data[3, 2] = np.nan
    ts = np.linspace(100.0, 102.0, 10)
    ts[5] = 99.0                                  # non-monotonic
    g.create_dataset("data", data=data, chunks=(16, 6))
    g.create_dataset("ts", data=ts, chunks=(16,))
    rep = verify_episode(ep, hw=hw)
    assert not rep["ok"]
    ft = next(s for s in rep["streams"] if s["name"] == "arm_ft")
    assert any("monotonic" in i for i in ft["issues"])
    assert any("non-finite" in i for i in ft["issues"])


def test_verify_detects_empty_and_length_mismatch(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="lite")
    g = zarr.open_group(str(ep / "weird.zarr"), mode="a")
    g.create_dataset("data", shape=(0, 3), dtype="f4", chunks=(4, 3))
    g.create_dataset("ts", shape=(0,), dtype="f8", chunks=(4,))
    g2 = zarr.open_group(str(ep / "mismatch.zarr"), mode="a")
    g2.create_dataset("data", data=np.zeros((5, 2), dtype="f4"), chunks=(4, 2))
    g2.create_dataset("ts", data=np.linspace(0, 1, 3), chunks=(4,))
    rep = verify_episode(ep, hw=hw)
    assert not rep["ok"]
    by = {s["name"]: s for s in rep["streams"]}
    assert any("empty" in i for i in by["weird"]["issues"])
    assert any("mismatch" in i for i in by["mismatch"]["issues"])


def test_verify_flags_unfinalized(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, finalize=False)
    rep = verify_episode(ep, hw=hw)
    assert not rep["ok"]
    assert any("finalized" in i for i in rep["issues"])


def test_verify_report_is_json_serializable(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="full")
    json.dumps(verify_episode(ep, hw=hw))       # SSE ships it — must not raise


def test_verify_flags_nan_timestamps_and_stays_valid_json(tmp_path, hw):
    """Review fix: NaN ts must be a hard issue AND must never leak literal
    NaN into the report (the panel SSE JSON.parse would die on it)."""
    ep = _write_episode(tmp_path, hw, mode="lite", drop=("arm_ft",))
    g = zarr.open_group(str(ep / "arm_ft.zarr"), mode="a")
    ts = np.linspace(100.0, 102.0, 10)
    ts[4] = np.nan
    g.create_dataset("data", data=np.zeros((10, 6), dtype="f4"), chunks=(16, 6))
    g.create_dataset("ts", data=ts, chunks=(16,))
    rep = verify_episode(ep, hw=hw)
    assert not rep["ok"]
    ft = next(s for s in rep["streams"] if s["name"] == "arm_ft")
    assert any("non-finite timestamps" in i for i in ft["issues"])
    json.dumps(rep, allow_nan=False)        # raises if any NaN/Inf leaked


def test_verify_flags_zero_duration(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="lite", drop=("arm_ft",))
    g = zarr.open_group(str(ep / "arm_ft.zarr"), mode="a")
    g.create_dataset("data", data=np.zeros((10, 6), dtype="f4"), chunks=(16, 6))
    g.create_dataset("ts", data=np.full(10, 100.0), chunks=(16,))
    rep = verify_episode(ep, hw=hw)
    assert not rep["ok"]
    ft = next(s for s in rep["streams"] if s["name"] == "arm_ft")
    assert any("zero duration" in i for i in ft["issues"])


def test_verify_warns_single_row_and_low_rate(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="lite", dur=3.0,
                        drop=("arm_ft", "arm_qd"))
    g = zarr.open_group(str(ep / "arm_ft.zarr"), mode="a")
    g.create_dataset("data", data=np.zeros((1, 6), dtype="f4"), chunks=(16, 6))
    g.create_dataset("ts", data=np.array([101.0]), chunks=(16,))
    g2 = zarr.open_group(str(ep / "arm_qd.zarr"), mode="a")   # 2 Hz vs 500 Hz
    g2.create_dataset("data", data=np.zeros((6, 6), dtype="f4"), chunks=(16, 6))
    g2.create_dataset("ts", data=np.linspace(100.0, 103.0, 6), chunks=(16,))
    rep = verify_episode(ep, hw=hw)
    by = {s["name"]: s for s in rep["streams"]}
    assert any("single row" in w for w in by["arm_ft"]["warns"])
    assert any("covers only" in w for w in by["arm_ft"]["warns"])
    assert any("rate low" in w for w in by["arm_qd"]["warns"])
    assert rep["ok"]                        # warns are not hard failures


def test_verify_demotes_missing_on_config_hash_mismatch(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="full",
                        drop=("tactile_left_keyframes",))
    meta_path = ep / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["config_hash"] = "someone_elses_rig"
    meta_path.write_text(json.dumps(meta))
    rep = verify_episode(ep, hw=hw)
    assert "tactile_left_keyframes" in rep["missing"]
    assert rep["ok"]                        # advisory, not a hard failure
    assert any("DIFFERENT hardware config" in w for w in rep["warns"])


def test_expect_hz_uses_tactile_ring_rate():
    """Review fix: fields_ds/wrench/area ride the main tactile ring at
    tactile.rate_hz — recording.field_ds_rate_hz is not their disk rate."""
    hw2 = make_small_hw(tactile={"rate_hz": 60.0})
    assert _expect_hz(hw2, "tactile_left_wrench") == 60.0
    assert _expect_hz(hw2, "tactile_left_fields_ds") == 60.0
    assert _expect_hz(hw2, "tactile_left_keyframes") == \
        hw2.recording.keyframe_rate_hz


def test_expected_streams_modes(hw):
    full = set(expected_streams(hw, "full"))
    lite = set(expected_streams(hw, "lite"))
    assert lite < full
    assert "tactile_left_fields_ds" in full - lite
    assert "actions_abs" in lite and "camera_scene_color" in lite
    assert "camera_wrist_color" not in full     # wrist cam disabled in yaml


def test_split_tactile():
    assert _split_tactile("tactile_left_fields_ds") == ("left", "fields_ds")
    assert _split_tactile("tactile_a_b_wrench") == ("a_b", "wrench")
    assert _split_tactile("arm_q") is None
    assert _split_tactile("tactile_left_unknown") is None


# ---------------------------------------------------------------------------
# player (stub view — no rerun-sdk needed)
# ---------------------------------------------------------------------------

class StubView:
    def __init__(self):
        self.times, self.scalars, self.images, self.events = [], [], [], []

    def _set_time(self, t):
        self.times.append(float(t))

    def _scalar(self, path, v):
        self.scalars.append((path, float(v)))

    def _image_u8(self, path, arr):
        self.images.append((path, tuple(arr.shape)))

    def _image_f(self, path, arr):
        self.images.append((path, tuple(arr.shape)))

    def log_event(self, text, level="INFO"):
        self.events.append(text)


def _wait_done(player, timeout=10.0):
    t0 = time.time()
    while player.active() and time.time() - t0 < timeout:
        time.sleep(0.01)
    assert not player.active(), "player did not finish in time"


def test_player_streams_everything(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="full")
    view = StubView()
    player = EpisodePlayer(view)
    player.play(ep, speed=16.0, hw=hw)
    _wait_done(player)
    st = player.status()
    assert st["state"] == "done" and st["pos_s"] == pytest.approx(st["dur_s"])
    spaths = {p for p, _ in view.scalars}
    ipaths = {p for p, _ in view.images}
    assert "/arm/q/j0" in spaths and "/arm/ft/Fz" in spaths
    assert "/gripper/pos" in spaths and "/actions/gripper" in spaths
    assert "/tactile/left/wrench/Fx" in spaths and "/tactile/left/area" in spaths
    assert ("/camera/scene", (60, 80, 3)) in view.images
    assert "/tactile/left/depth" in ipaths      # fields_ds split into channels
    assert "/tactile/left/infer" in ipaths
    assert view.times == sorted(view.times) or len(set(view.times)) > 1


def _wait_state(player, want, timeout=15.0):
    t0 = time.time()
    while player.status()["state"] not in want and time.time() - t0 < timeout:
        time.sleep(0.02)
    assert player.status()["state"] in want, player.status()


def test_player_pause_and_stop(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="lite", dur=8.0)
    player = EpisodePlayer(StubView())
    player.play(ep, speed=0.5, hw=hw)       # 16 s wall — never finishes here
    _wait_state(player, ("playing",))
    player.pause()
    _wait_state(player, ("paused",))
    p1 = player.status()["pos_s"]
    time.sleep(0.4)
    p2 = player.status()["pos_s"]
    assert p2 == pytest.approx(p1, abs=0.05)
    player.resume()
    t0 = time.time()
    while player.status()["pos_s"] <= p2 and time.time() - t0 < 5.0:
        time.sleep(0.02)
    assert player.status()["pos_s"] > p2
    player.stop()
    _wait_done(player, timeout=5.0)
    assert player.status()["state"] == "stopped"


def test_player_rejects_double_play(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="lite", dur=4.0)
    player = EpisodePlayer(StubView())
    player.play(ep, speed=0.5, hw=hw)
    with pytest.raises(RuntimeError, match="already running"):
        player.play(ep, speed=0.5, hw=hw)
    player.stop()


def test_player_terminates_on_nonmonotonic_ts(tmp_path, hw):
    """Review fix: an interior ts maximum must not leave undrainable cursors
    (the loop would never satisfy its termination condition)."""
    ep = _write_episode(tmp_path, hw, mode="lite", drop=("arm_ft",))
    g = zarr.open_group(str(ep / "arm_ft.zarr"), mode="a")
    ts = np.linspace(100.0, 102.0, 10)
    ts[5] = 103.5                           # max in the MIDDLE
    g.create_dataset("data", data=np.zeros((10, 6), dtype="f4"), chunks=(16, 6))
    g.create_dataset("ts", data=ts, chunks=(16,))
    player = EpisodePlayer(StubView())
    player.play(ep, speed=16.0, hw=hw)
    _wait_done(player, timeout=20.0)
    assert player.status()["state"] == "done"


def test_player_skips_nonfinite_ts_stream(tmp_path, hw):
    ep = _write_episode(tmp_path, hw, mode="lite", drop=("arm_ft",))
    g = zarr.open_group(str(ep / "arm_ft.zarr"), mode="a")
    ts = np.linspace(100.0, 102.0, 10)
    ts[3] = np.nan                          # would never drain in searchsorted
    g.create_dataset("data", data=np.zeros((10, 6), dtype="f4"), chunks=(16, 6))
    g.create_dataset("ts", data=ts, chunks=(16,))
    view = StubView()
    player = EpisodePlayer(view)
    player.play(ep, speed=16.0, hw=hw)
    _wait_done(player, timeout=20.0)
    assert player.status()["state"] == "done"
    assert not any(p.startswith("/arm/ft") for p, _ in view.scalars)
    assert any("skipped" in e for e in view.events)


def test_player_speed_clamped_and_bad_path(tmp_path, hw):
    player = EpisodePlayer(StubView())
    player.play(tmp_path / "nope", speed=999.0)
    _wait_done(player)
    assert player.status()["state"] == "error"
    assert player.status()["speed"] == 16.0


# ---------------------------------------------------------------------------
# controller: scanning, path confinement, mutual exclusion
# ---------------------------------------------------------------------------

def _cc(drive, staging):
    return types.SimpleNamespace(
        storage=types.SimpleNamespace(external_drive=str(drive),
                                      staging_root=str(staging)),
        rerun=types.SimpleNamespace(web_port=0, grpc_port=0))


def test_scan_finds_drive_and_staging(tmp_path, hw):
    drive, staging = tmp_path / "drive", tmp_path / "staging"
    a = _write_episode(drive / "20260722_x_demo", hw, name="ep_demo_1_000")
    time.sleep(0.02)
    b = _write_episode(staging / "20260722_y_demo", hw, name="ep_demo_2_000")
    pc = PlaybackController(_cc(drive, staging), hw, None)
    eps = pc.scan()
    assert [e["name"] for e in eps] == ["ep_demo_2_000", "ep_demo_1_000"]
    assert {e["source"] for e in eps} == {"drive", "staging"}
    assert eps[0]["session"] == "20260722_y_demo"
    assert all(e["mode"] == "lite" and e["size_mb"] >= 0 for e in eps)
    assert pc._resolve(str(a)) is not None and pc._resolve(str(b)) is not None


def test_resolve_confined_to_roots(tmp_path, hw):
    drive, staging = tmp_path / "drive", tmp_path / "staging"
    outside = _write_episode(tmp_path / "elsewhere", hw)
    pc = PlaybackController(_cc(drive, staging), hw, None)
    assert pc._resolve(str(outside)) is None
    assert pc._resolve("") is None
    err = pc.play(str(outside))
    assert err is not None and "roots" in err


def test_play_blocked_while_session_busy(tmp_path, hw):
    drive, staging = tmp_path / "drive", tmp_path / "staging"
    ep = _write_episode(drive / "s", hw)
    pc = PlaybackController(_cc(drive, staging), hw, None,
                            session_busy=lambda: True)
    assert "session/offload" in pc.play(str(ep))
    assert "session/offload" in pc.verify(str(ep))


def test_runner_blocked_while_playback_active():
    runner = CollectRunner(app=None, panel=None)
    runner.playback = types.SimpleNamespace(blocking=lambda: True)
    err = runner.start_session({"task": "demo", "mode": "full"})
    assert err is not None and "playback" in err
    err = runner.start_offload()
    assert err is not None and "playback" in err


def test_controller_verify_updates_panel_state(tmp_path, hw):
    drive, staging = tmp_path / "drive", tmp_path / "staging"
    ep = _write_episode(drive / "s", hw)
    panel = types.SimpleNamespace(state=CollectPanelState())
    pc = PlaybackController(_cc(drive, staging), hw, panel)
    assert pc.verify(str(ep)) is None
    t0 = time.time()
    while pc.blocking() and time.time() - t0 < 10:
        time.sleep(0.02)
    snap = panel.state.snapshot()
    assert snap["verify_running"] is False
    assert snap["verify_report"]["ok"] is True


# ---------------------------------------------------------------------------
# panel endpoints
# ---------------------------------------------------------------------------

class StubPlayback:
    def __init__(self):
        self.calls = []

    def scan(self):
        return [{"path": "p", "name": "ep_x", "session": "s",
                 "source": "drive", "task": "t", "status": "finalized",
                 "success": True, "mode": "lite", "size_mb": 1.0, "mtime": 0.0}]

    def play(self, path, speed=1.0):
        self.calls.append(("play", path, speed))
        return None if path else "no episode selected"

    def verify(self, path):
        self.calls.append(("verify", path))
        return None

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def stop(self):
        self.calls.append(("stop",))

    def set_speed(self, v):
        self.calls.append(("speed", v))

    def blocking(self):
        return False


@pytest.fixture
def pb_panel():
    pb = StubPlayback()
    p = CollectPanel(CollectPanelState(), host="127.0.0.1", port=0, playback=pb)
    p.start()
    yield p, pb
    p.stop()


def _get(p, path):
    try:
        with urllib.request.urlopen(f"{p.url}{path}", timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(p, path, body):
    req = urllib.request.Request(
        f"{p.url}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_panel_episode_endpoints(pb_panel):
    p, pb = pb_panel
    code, r = _get(p, "/api/episodes")
    assert code == 200 and r["episodes"][0]["name"] == "ep_x"
    code, r = _post(p, "/api/playback/play", {"path": "p", "speed": 2})
    assert code == 200 and ("play", "p", 2.0) in pb.calls
    code, r = _post(p, "/api/playback/play", {"path": ""})
    assert code == 409 and "error" in r
    code, r = _post(p, "/api/playback/verify", {"path": "p"})
    assert code == 200 and ("verify", "p") in pb.calls
    for op in ("pause", "resume", "stop"):
        code, _ = _post(p, f"/api/playback/{op}", {})
        assert code == 200 and (op,) in pb.calls
    code, _ = _post(p, "/api/playback/speed", {"speed": 4})
    assert code == 200 and ("speed", 4.0) in pb.calls
    code, r = _post(p, "/api/playback/speed", {"speed": "abc"})
    assert code == 400
    code, r = _post(p, "/api/playback/nope", {})
    assert code == 404


def test_panel_nondict_json_body_no_crash(pb_panel):
    """Review fix: a valid-JSON non-object body must not AttributeError the
    handler thread on .get()."""
    p, pb = pb_panel
    req = urllib.request.Request(
        f"{p.url}/api/playback/pause", data=b"[1,2]",
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
    assert ("pause",) in pb.calls


def test_reset_session_clears_playback_fields():
    st = CollectPanelState()
    st.update(playback_pos_s=5.0, playback_dur_s=9.0, playback_state="done",
              playback_episode="ep_x", verify_report={"ok": True})
    st.reset_session()
    s = st.snapshot()
    assert s["playback_pos_s"] == 0.0 and s["playback_dur_s"] == 0.0
    assert s["playback_state"] == "idle" and s["playback_episode"] == ""
    assert s["verify_report"] is None


def test_panel_without_playback_409():
    p = CollectPanel(CollectPanelState(), host="127.0.0.1", port=0)
    p.start()
    try:
        code, r = _get(p, "/api/episodes")
        assert code == 409
        code, r = _post(p, "/api/playback/play", {"path": "x"})
        assert code == 409
    finally:
        p.stop()


def test_page_has_playback_card(pb_panel):
    p, _ = pb_panel
    with urllib.request.urlopen(f"{p.url}/", timeout=5) as r:
        html = r.read().decode()
    assert "Episodes" in html and "Verify recording" in html
    assert "/api/playback/" in html and "vTable" in html
