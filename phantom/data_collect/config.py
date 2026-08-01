"""data_collect configuration — configs/data_collect.yaml is THE user-facing file.

It holds everything an operator has to touch between rigs and sessions:

 - ports/serials of the two DM-Tac W2L sensors, the RealSense camera and the
   Echo leader (USB VID/PID);
 - the external hard drive episodes are offloaded to at session end;
 - teleop feel tuning (tracking bounds, filter, gripper mapping);
 - the DM-Tac safeguard defaults (enabled / force threshold);
 - the default collection mode (full | lite).

`load_collect()` loads this file plus the hardware yaml it names, and PATCHES
the hardware config with the ports section (sensor dev_ids, camera serial,
Echo VID/PID, gripper tick calibration, teleop filter tuning) BEFORE
validation — so the rest of the stack keeps a single source of truth
(HardwareConfig) while the operator edits only data_collect.yaml.

An optional gitignored configs/data_collect.local.yaml is deep-merged on top
for per-machine values (drive letters, serials) that must not be committed.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from phantom.config.hardware import HardwareConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLLECT_YAML = REPO_ROOT / "configs" / "data_collect.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PortsConfig(_Frozen):
    """Physical port / serial identities of every device on the rig."""
    # DM-Tac dev_id: str serial from the yellow cable label (stable across
    # replug) or int SDK index (single-sensor bench tests only).
    dmtac_left: int | str
    dmtac_right: int | str
    realsense_serial: str = ""      # librealsense serial; "" = first device
    echo_vid: int = 1603            # Echo leader USB VID/PID (port auto-discovered)
    echo_pid: int = 1868


class StorageConfig(_Frozen):
    staging_root: str = "runs/collect"   # local fast-disk staging during a session
    # Offload target (external USB drive mount / drive letter). Episodes are
    # MOVED here when the session ends — nothing stays on the local disk
    # after a verified offload.
    external_drive: str
    min_free_gb: float = Field(default=20.0, ge=0)
    verify: Literal["size", "sha256"] = "size"
    # Safety valve: keep the local copy even after a verified offload.
    keep_local: bool = False
    # Refuse to offload when external_drive resolves to the SAME filesystem
    # as the staging dir — on Linux an unmounted /media/... mountpoint is a
    # plain local directory and the "offload" would silently land on the
    # local disk. Enable on the rig; off by default so dev/test paths work.
    require_separate_device: bool = False


class TeleopTuning(_Frozen):
    """Feel parameters for the device-rate command path (collect/teleop.py).

    Defaults reproduce the vendored main11.py feel: no meaningful software
    lag, servoJ lookahead does the smoothing. The tracker bounds are a safety
    net sized ABOVE human motion (main11 has none at all)."""
    v_max_rad_s: float = Field(default=3.0, gt=0)
    a_max_rad_s2: float = Field(default=10.0, gt=0)          # ENGAGE glide accel
    # Acceleration bound of the TRACK-path sqrt-braking tracker. It sets two
    # things at once:
    #  - steady tracking lag = v^2/(2*a_max) (at 1.5 rad/s: 40 -> ~1.3 deg /
    #    15 ms; lower a_max = more lag, the old "over-smoothed" feel);
    #  - the hard cap on commanded acceleration (servoJ feasibility): 40 keeps
    #    sustained accel well under the ~88 rad/s^2 that tripped C153A3, and the
    #    sqrt law's one-tick <=2*a_max landing transient (80) still clears it.
    # 40 balances low lag against jerk/C153A3 safety. It replaced a deadbeat
    # clamp at 20 that tracked with zero lag but OVERSHOT and rang (~1-2 Hz
    # "wiggle" on the rig when the dropout extrapolation snapped the target).
    track_a_max_rad_s2: float = Field(default=40.0, gt=0)
    # Slow glide used ONLY to engage (arm meets the leader pose on session
    # start and on safeguard resume) — never during tracking.
    engage_v_max_rad_s: float = Field(default=0.5, gt=0)
    engage_eps_rad: float = Field(default=0.05, gt=0)
    # Leader drop-out handling. The Echo firmware stalls ~100 ms about twice a
    # second (request/response times out); without bridging, the target freezes
    # then jumps -> "ведёт ровно, потом дёргается". extrap_cap_s: how long to
    # keep gliding the target at its last velocity before holding (covers the
    # ~100 ms stalls with margin). stale_reengage_s: a gap longer than this is
    # treated as a re-connect -> glide back via ENGAGE instead of chasing a big
    # accumulated delta.
    extrap_cap_s: float = Field(default=0.12, gt=0)
    stale_reengage_s: float = Field(default=0.4, gt=0)
    # One-euro (Casiez, CHI 2012) is the ONLY signal filter in the path; tune
    # it the way the paper intends — near-transparent while the operator is
    # moving, heavy only at rest:
    #   min_cutoff  = rest floor, kills exo tremor + encoder quantization
    #                 (0.073 deg/tick). Applies only near zero speed.
    #   beta        = how fast the cutoff opens with motion. Too LOW (the old
    #                 0.5) leaves it a fixed low-pass -> laggy. Too HIGH (10)
    #                 lets the exo's ~0.5 deg rest jitter — sampled at the
    #                 device's true ~360 Hz, so each tick step is a big velocity
    #                 spike — fully open the cutoff, passing high-frequency
    #                 noise straight through as command BUZZ (measured 88
    #                 rad/s^2, the C153A3 / "дёргано" trigger). 3.0 opens enough
    #                 for responsive motion without amplifying rest noise.
    #   d_cutoff    = smooths the SPEED estimate that drives beta. High values
    #                 (5) let noise velocity-spikes through; 1.5 rejects them
    #                 (keeps onset ~30 ms, still far snappier than the 49 ms
    #                 that felt over-smoothed).
    # Measured on real Echo data: rest command jitter 0.52 -> 0.36 deg, motion
    # lag ~8 ms at 1.5 rad/s, onset ~30 ms. The acceleration cap
    # (track_a_max_rad_s2) is the hard jerk/C153A3 net; the filter sets feel.
    filter_min_cutoff_hz: float = Field(default=1.5, gt=0)
    filter_beta: float = Field(default=3.0, ge=0)
    filter_d_cutoff_hz: float = Field(default=1.5, gt=0)


class GripperTuning(_Frozen):
    """Continuous (proportional) gripper teleop — NOT the lab's binary mode."""
    open_tick: int = 5              # leader ticks at full open  (panel: Calibrate)
    closed_tick: int = 155          # leader ticks at full squeeze
    ema_alpha: float = Field(default=0.5, gt=0, le=1)
    deadband: float = Field(default=0.008, ge=0, le=1)   # ~2/255 counts
    speed: float = Field(default=1.0, gt=0, le=1)        # snappy (main11: 255/255)
    rate_hz: float = Field(default=50.0, gt=0)           # URCap socket command cap
    # Force-limited grasp: stop closing FURTHER once the tactile resultant
    # force (max over both DM-Tac pads) reaches grasp_stop_n, so the operator
    # can squeeze the Echo leader fully without overgripping. Opening is always
    # obeyed instantly (and clears the ceiling). A grabbed hold resumes closing
    # only after force relaxes below grasp_resume_frac*grasp_stop_n AND the
    # leader is still pressing (anti-chatter hysteresis). Stale tactile (freshest
    # sample older than grasp_stale_s — sensor/worker lag or death) degrades to
    # plain proportional control: the ceiling NEVER blocks the gripper on dead
    # data. This is the WORKING grip limit and sits below the emergency layers:
    # pad safeguard force_limit_n (15 N) and the driver pad ceiling (30 N).
    grasp_stop_n: float = Field(default=10.0, gt=0)
    grasp_resume_frac: float = Field(default=0.7, gt=0, le=1)
    grasp_stale_s: float = Field(default=0.3, gt=0)


class SafeguardConfig(_Frozen):
    """DM-Tac pad protection during teleop (collect/safeguard.py).

    Trips when any sensor's resultant force ‖F‖ (getForce, calibrated N)
    exceeds force_limit_n, or its indentation depth exceeds depth_limit
    (uncalibrated SDK units — the always-on fallback layer). Latched: teleop
    freezes, the gripper opens fully, the episode is saved as a failure, and
    collection resumes only from the panel's Resume button. Enabled state and
    threshold are runtime-editable from the panel."""
    enabled: bool = True
    # EMERGENCY overgrip/impact limit — NOT the working grip limit. Normal
    # grasping force is now bounded by the gripper's force-limited grasp
    # (gripper.grasp_stop_n, 10 N): it stops closing at 10 N, so a good grasp
    # never reaches this guard. 15 N therefore trips only on a real overgrip or
    # collision (leader jammed, object slips into a hard mount), well under the
    # DM-Tac pad ceiling (30 N). Was 4 N, which fired during ordinary grasps and
    # fought the grasp-stop — the two limits now nest: 10 N grasp < 15 N guard <
    # 30 N pad. See gripper.grasp_stop_n.
    force_limit_n: float = Field(default=15.0, gt=0)
    depth_limit: float = Field(default=0.5, gt=0)
    release_gripper: bool = True


class PanelConfig(_Frozen):
    host: str = "127.0.0.1"         # 0.0.0.0 for a rig tablet (trusted LAN only)
    port: int = 8899


class RerunConfig(_Frozen):
    # "web": embed the rerun web (WASM) viewer as the panel iframe — the browser
    #   decodes/renders every frame, measured ~3.5 cores on the rig NUC, which
    #   starves the 125 Hz control loop when the operator views locally on the
    #   rig. "native": launch the native wgpu viewer in its own window (measured
    #   ~0.3 cores, full 640x480) and DON'T embed the WASM viewer (the panel
    #   shows a note). Use "native" whenever the viewer runs on the rig itself.
    #   "off": don't launch any viewer window at all (saves the native
    #   viewer's ~0.3 core too) while the mode-aware rerun logger keeps running
    #   in its isolated child, so the operator can still attach a viewer
    #   mid-session (rerun+http://<rig>:<grpc_port>/proxy) if something needs a
    #   look. Use "native"/"web" for setup + verification, "off" while actually
    #   recording a long collection campaign (pipeline.md throughput notes).
    viewer: Literal["web", "native", "off"] = "web"
    web_port: int = 9091
    grpc_port: int = 9878
    # 10 Hz (down from 15): full-mode logs ~10 images + ~30 scalars per tick;
    # at 15 Hz that saturated the rerun 0.34 gRPC channel (backpressure ->
    # libarrow segfault). 10 Hz + the tick back-off keeps the channel drained.
    rate_hz: float = Field(default=10.0, gt=0)
    # The RealSense delivers a rock-steady 30 fps (measured: 0 drops, 33.4 ms
    # median interframe, full USB3), but the ring was sampled at only rate_hz
    # (10) — so the viewer showed choppy 10 fps video and looked "laggy". Now
    # that the scene is JPEG'd (~50 KB/frame -> ~1.5 MB/s at 30 Hz) the camera
    # can be logged at its native rate for smooth playback. Only the camera
    # runs this fast; the heavier tactile field maps stay at their own arrival
    # rate (self-decimated), so this does not reheat the old backpressure path.
    # 15, not 30: JPEG *encode* is cheap (~1.4 ms measured on the NUC), but the
    # browser WASM viewer has to DECODE every frame, and on the operator's
    # (weaker) machine 30/s overran that and stalled the stream. 15 Hz halves
    # the client decode load while still looking smooth (vs the old choppy 10).
    camera_rate_hz: float = Field(default=15.0, gt=0)
    # Downscale the scene for the viewer by this integer factor before JPEG
    # (2 -> 640x480 becomes 320x240). NOTE: field-tested with NO effect — the
    # lag/spinner is identical at full res and 320x240, and identical across
    # 10/15/30 Hz, so per-frame cost is NOT the bottleneck (it's the viewer's
    # render environment / data cadence, not pixel throughput). Kept as a knob
    # but defaulted to 1 (full res) since downscaling only hurt image quality.
    camera_viz_downscale: int = Field(default=1, ge=1)
    # Cap the gRPC server's in-memory history. Without this the server buffers
    # everything logged since start (0.34's default is 1GiB); over a long
    # session the buffer — and the WASM web viewer that mirrors the whole
    # stream — grow until the viewer lags behind realtime, then hangs. Oldest
    # data is dropped once the cap is hit; static data is never dropped.
    # re_memory format: "512MB", "2GB", or a percentage like "25%".
    server_memory_limit: str = "512MB"
    # JPEG-compress the RealSense scene frame before logging. Raw 640x480x3 at
    # 10 Hz is ~9 MB/s — on its own that saturates the gRPC channel and buries
    # the viewer's decode/store loop (the dominant cause of the lag/hang). q=75
    # JPEG is ~0.4 MB/s (~20x less) with no visible loss for monitoring. 0 =
    # log raw (only the tactile field maps stay raw floats — they're tiny).
    camera_jpeg_quality: int = Field(default=75, ge=0, le=100)
    # Operator-saved viewer layout ("как сейчас"): a .rbl exported from the
    # rerun viewer's Menu -> "Save blueprint". If this file exists it is
    # loaded and made the active blueprint on every session start (so it
    # persists across panel restarts), overriding the code-defined layout.
    # Path is relative to the repo root; missing file -> code-defined default.
    blueprint_path: str = "configs/rerun_blueprint.rbl"


class CollectConfig(_Frozen):
    # Hardware yaml this rig runs on (relative to the repo root).
    hardware: str = "configs/hardware.yaml"
    ports: PortsConfig
    storage: StorageConfig
    teleop: TeleopTuning = TeleopTuning()
    gripper: GripperTuning = GripperTuning()
    safeguard: SafeguardConfig = SafeguardConfig()
    # full = everything PHANTOM needs (fields, keyframes, gel, wrench, ...);
    # lite = UR3 + RealSense RGB + per-sensor 6-axis wrench only.
    default_mode: Literal["full", "lite"] = "full"
    panel: PanelConfig = PanelConfig()
    rerun: RerunConfig = RerunConfig()


# ---------------------------------------------------------------------------


def _deep_merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _patch_hardware_raw(raw_hw: dict, cc: CollectConfig) -> dict:
    """Inject the ports/tuning sections into the hardware dict BEFORE
    validation, so HardwareConfig stays the single source of truth."""
    raw = copy.deepcopy(raw_hw)
    raw.setdefault("tactile", {})["sensors"] = [
        {"name": "left", "dev_id": cc.ports.dmtac_left},
        {"name": "right", "dev_id": cc.ports.dmtac_right},
    ]
    raw.setdefault("cameras", {}).setdefault("scene", {})["serial"] = \
        cc.ports.realsense_serial
    teleop = raw.get("teleop")
    if not teleop or not teleop.get("echo"):
        raise ValueError(
            f"hardware yaml {cc.hardware!r} has no teleop.echo section — collect "
            "needs the Echo base_pose from the hardware config (see "
            "configs/hardware.nuc.yaml for the rig values)")
    echo = teleop["echo"]
    echo["vid"] = cc.ports.echo_vid
    echo["pid"] = cc.ports.echo_pid
    echo["gripper_open_tick"] = cc.gripper.open_tick
    echo["gripper_closed_tick"] = cc.gripper.closed_tick
    echo["gripper_ema_alpha"] = cc.gripper.ema_alpha
    echo["filter_min_cutoff"] = cc.teleop.filter_min_cutoff_hz
    echo["filter_beta"] = cc.teleop.filter_beta
    echo["filter_d_cutoff"] = cc.teleop.filter_d_cutoff_hz
    return raw


def load_collect(path: str | Path | None = None,
                 *, local_path: str | Path | None = None,
                 ) -> tuple[CollectConfig, HardwareConfig]:
    """Load data_collect.yaml (+ optional data_collect.local.yaml) and the hardware
    yaml it names, returning (collect_cfg, ports-patched hardware_cfg)."""
    path = Path(path) if path is not None else DEFAULT_COLLECT_YAML
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    local = Path(local_path) if local_path is not None else \
        path.with_name(path.stem + ".local.yaml")
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            _deep_merge(raw, yaml.safe_load(f) or {})
    cc = CollectConfig.model_validate(raw)

    hw_path = Path(cc.hardware)
    if not hw_path.is_absolute():
        hw_path = path.parent.parent / hw_path if not hw_path.exists() else hw_path
    with open(hw_path, "r", encoding="utf-8") as f:
        raw_hw = yaml.safe_load(f)
    hw = HardwareConfig.model_validate(_patch_hardware_raw(raw_hw, cc))
    return cc, hw
