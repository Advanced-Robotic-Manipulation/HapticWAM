"""Mock DM-Tac W2L sensor: renders physically co-varying tactile fields from
the shared ContactScenario. Shapes come from the hardware config only.

The rendering (`render_fields`) is a pure function of (scenario state, config,
grid) — it is also used directly by SyntheticEpisodeGenerator, guaranteeing
that mock recordings and synthetic training episodes share one signal model.
"""

from __future__ import annotations

import time

import numpy as np

from phantom.config.hardware import TactileConfig, TactileSensorEntry
from phantom.drivers.base import TactileFrame, TactileSensor
from phantom.drivers.mock.scenario import ContactScenario, ScenarioState


def _grid(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    return np.meshgrid(xs, ys)  # xx (H,W), yy (H,W)


def render_fields(state: ScenarioState, cfg: TactileConfig,
                  rng: np.random.Generator, finger_phase: float = 0.0) -> dict[str, np.ndarray]:
    """Render one canonical tactile frame from a scenario state.

    finger_phase de-correlates the two fingertips slightly (mirrored x).
    Returns dict with keys deformation (H,W,2), depth (H,W), shear (H,W,2),
    dist_force (H,W,3), wrench (6,), area_mm2 (float).
    """
    H, W = cfg.field.hw
    xx, yy = _grid(H, W)
    cx, cy = state.blob_center_uv
    if finger_phase > 0:
        cx = -cx  # mirror on the second finger
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sigma = max(state.blob_radius, 1e-6)
    blob = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32) if state.press_depth > 0 \
        else np.zeros((H, W), dtype=np.float32)

    depth = state.press_depth * blob
    # in-plane displacement: radial squeeze-out around the blob + drift advection
    dx_r = (xx - cx) * blob * 0.1 * state.press_depth
    dy_r = (yy - cy) * blob * 0.1 * state.press_depth
    drift_x, drift_y = state.tangential_drift_uv
    deformation = np.stack([dx_r + drift_x * blob, dy_r + drift_y * blob], axis=-1)
    # shear tracks the tangential loading; exceeds the friction cone during slip
    shear = np.stack([drift_x * blob, drift_y * blob], axis=-1).astype(np.float32)
    # distributed force, PHYSICAL newtons: peak per-pixel fz = scenario fz_total;
    # emitted in SDK units (divided by the configured unit scale) exactly like
    # the real sensor, so calibrated thresholds work against mock data too
    fz_N = state.fz_total * blob
    fxy_N = shear * 2.0
    sdk_scale = (1.0 / cfg.dist_force_unit_to_N) if cfg.force_calibrated else 1.0
    dist_force = np.stack([fxy_N[..., 0], fxy_N[..., 1], fz_N], axis=-1) * sdk_scale

    noise = 0.003
    depth = depth + rng.normal(0, noise, depth.shape).astype(np.float32)
    deformation = (deformation + rng.normal(0, noise, deformation.shape)).astype(np.float32)
    shear = (shear + rng.normal(0, noise, shear.shape)).astype(np.float32)
    dist_force = (dist_force + rng.normal(0, noise, dist_force.shape)).astype(np.float32)

    # resultant wrench from the PHYSICAL (N) fields — the TactileFrame contract
    # is calibrated wrench (the real driver multiplies by force_unit_to_N)
    cell = 4.0 / (H * W)  # normalized cell area
    force_N = np.stack([fxy_N[..., 0], fxy_N[..., 1], fz_N], axis=-1)
    fsum = force_N.sum(axis=(0, 1)) * cell * 25.0
    torque = np.array([
        float((yy * force_N[..., 2]).sum() * cell),
        float((-xx * force_N[..., 2]).sum() * cell),
        float((xx * force_N[..., 1] - yy * force_N[..., 0]).sum() * cell),
    ], dtype=np.float32)
    wrench = np.concatenate([fsum.astype(np.float32), torque])
    area_mm2 = float((blob > 0.5).sum()) / (H * W) * 36.0 * 27.0

    return {"deformation": deformation.astype(np.float32), "depth": depth.astype(np.float32),
            "shear": shear, "dist_force": dist_force.astype(np.float32),
            "wrench": wrench, "area_mm2": area_mm2}


class MockTactileSensor(TactileSensor):
    def __init__(self, cfg: TactileConfig, sensor: TactileSensorEntry,
                 scenario: ContactScenario, finger_index: int = 0):
        super().__init__(cfg, sensor)
        self.scenario = scenario
        self.finger_index = finger_index
        dev_seed = sensor.dev_id if isinstance(sensor.dev_id, int) \
            else int.from_bytes(str(sensor.dev_id).encode(), "little") % 100_000
        self._rng = np.random.default_rng(1000 + dev_seed)
        self._connected = False
        self._t0 = 0.0
        self._seq = 0

    def connect(self) -> None:
        self._t0 = time.perf_counter()
        self._seq = 0
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def reset_reference(self) -> None:
        pass  # mock is always referenced

    def render(self, t_scenario: float, t_host: float) -> TactileFrame:
        st = self.scenario.state(t_scenario)
        f = render_fields(st, self.cfg, self._rng, finger_phase=float(self.finger_index))
        img = None
        if self.cfg.infer_img.c == 1:
            img_shape = (self.cfg.infer_img.h, self.cfg.infer_img.w)
        else:
            img_shape = (self.cfg.infer_img.h, self.cfg.infer_img.w, self.cfg.infer_img.c)
        img = (self._rng.random(img_shape) * 30 + st.press_depth * 200).astype(np.uint8)
        frame = TactileFrame(
            t_host=t_host, seq=self._seq,
            deformation=f["deformation"], depth=f["depth"], shear=f["shear"],
            dist_force=f["dist_force"], wrench=f["wrench"],
            contact_area_mm2=f["area_mm2"], infer_img=img, raw_img=None,
        )
        self._seq += 1
        self._check_first_frame(frame)
        return frame

    def read(self) -> TactileFrame:
        if not self._connected:
            raise RuntimeError("read() before connect()")
        period = 1.0 / self.cfg.rate_hz
        now = time.perf_counter()
        target = self._t0 + self._seq * period
        if target < now - period:
            # epoch in the past (shared session_t0) or render too slow: skip ahead
            self._seq = int((now - self._t0) / period) + 1
            target = self._t0 + self._seq * period
        if target > now:
            time.sleep(target - now)
            now = target
        return self.render(now - self._t0, now)
