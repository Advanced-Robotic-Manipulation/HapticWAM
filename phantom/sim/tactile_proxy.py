"""Static measured sensor baseline plus newly simulated gel contacts.

This is a diagnostic sensor model, not a validated optical simulator. It never
reads a recorded sequence or advances a recorded sample. The distributed-force
units/sign are grounded in the Sep4 recordings: mean(field_force)*110592 equals
the SDK wrench force. Depth slopes are estimates from seven/ten loaded samples;
their residuals are about0.08/0.067mm. Optical deformation and contact area remain
uncertain. See artifacts/isaac_waffles/teacher_debug_v1/sensor_mapping_proposal.json.

The caller MUST supply gel-only contact forces. Whole-pad net force also includes
backing and linkage impacts and is not an admissible substitute. SDK image axes
must be mapped explicitly by the contact extractor, independently for each pad.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NATIVE_HW = (288, 384)
NATIVE_PIXELS = 288 * 384


@dataclass(frozen=True)
class TactileProxyParameters:
    """Explicit estimates; none of these settings relax deployment safety."""

    active_size_mm: tuple[float, float] = (36.0, 27.0)  # SDK image width/height
    patch_sigma_mm: tuple[float, float] = (3.0, 3.0)
    depth_mm_per_n: tuple[float, float] = (0.04032048, 0.04958851)
    normal_radial_displacement_gain: float = 0.25
    area_mm2_per_n_two_thirds: float = 1.0
    torque_nm_to_sdk: float = 100.0

    def __post_init__(self):
        values = np.r_[
            self.active_size_mm,
            self.patch_sigma_mm,
            self.depth_mm_per_n,
            self.normal_radial_displacement_gain,
            self.area_mm2_per_n_two_thirds,
            self.torque_nm_to_sdk,
        ]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("proxy parameters must be finite and nonnegative")
        if min(*self.active_size_mm, *self.patch_sigma_mm) <= 0:
            raise ValueError("sensor dimensions and contact sigma must be positive")


@dataclass
class TactileProxyFrame:
    gel: np.ndarray
    fields_ds: np.ndarray
    keyframes: np.ndarray
    wrench: np.ndarray
    area: np.ndarray
    depth_peaks_mm: np.ndarray
    contact_force_sdk_n: np.ndarray
    contact_uv: np.ndarray

    def as_sensor_dict(self, t: float, names=("left", "right")) -> dict:
        if not np.isfinite(t) or len(names) != 2:
            raise ValueError("finite timestamp and two sensor names required")
        return {
            name: {
                "t": float(t),
                "fields_ds": self.fields_ds[i],
                "keyframe": self.keyframes[i],
                "infer_img": self.gel[i],
                "wrench": self.wrench[i],
                "area": float(self.area[i]),
            }
            for i, name in enumerate(names)
        }


def _array(value, shape, name):
    x = np.asarray(value, dtype=np.float64)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite with shape{shape}")
    return x


class MeasuredBaselineTactileProxy:
    """Stateless force-to-sensor map using one immutable observed baseline.

    ``normal_force_n`` is compressive positive for BOTH pads. Tangential forces
    and displacements are already in SDK image X/Y axes, not world or pad axes.
    ``contact_uv`` is normalized SDK image(column,row) coordinates in[-1,1].
    Actual contact area may be supplied inmm²; otherwise an explicitly uncertain
    Hertz-style F^(2/3) area estimate is used. No force/depth output is clipped.
    """

    def __init__(self, baseline_npz: str | Path, parameters=None):
        self.parameters = parameters or TactileProxyParameters()
        with np.load(baseline_npz, allow_pickle=False) as data:
            self._gel = np.stack(
                [np.asarray(data[f"{s}_infer_img"]) for s in ("left", "right")]
            )
            if self._gel.shape != (2, *NATIVE_HW) or self._gel.dtype != np.uint8:
                raise ValueError("baseline gel must be uint8[2,288,384]")
            self._fields = _array(
                np.stack([data[f"{s}_fields_ds"] for s in ("left", "right")]),
                (2, 72, 96, 8),
                "baseline fields_ds",
            ).astype(np.float32)
            self._keyframes = _array(
                np.stack([data[f"{s}_keyframes"] for s in ("left", "right")]),
                (2, 144, 192, 8),
                "baseline keyframes",
            ).astype(np.float32)
            self._wrench = _array(
                np.stack([data[f"{s}_wrench"] for s in ("left", "right")]),
                (2, 6),
                "baseline wrench",
            ).astype(np.float32)
            self._area = _array(
                np.asarray([data[f"{s}_area"] for s in ("left", "right")]),
                (2,),
                "baseline area",
            ).astype(np.float32)
        for a in (self._gel, self._fields, self._keyframes, self._wrench, self._area):
            a.setflags(write=False)

    def _patch(self, hw, uv):
        h, w = hw
        width, height = self.parameters.active_size_mm
        # Pixel centers: one native pixel is36/384=27/288=0.09375mm.
        x = ((np.arange(w) + 0.5) / w - 0.5) * width - uv[0] * width / 2
        y = ((np.arange(h) + 0.5) / h - 0.5) * height - uv[1] * height / 2
        xx, yy = np.meshgrid(x, y)
        sx, sy = self.parameters.patch_sigma_mm
        weight = np.exp(-0.5 * ((xx / sx) ** 2 + (yy / sy) ** 2))
        peak_weight = weight / weight.max()
        return xx, yy, weight, peak_weight

    def _displacement(self, xx, yy, patch, depth, tangent):
        radius = np.hypot(xx, yy)
        radial = self.parameters.normal_radial_displacement_gain * depth
        dx = (tangent[0] + radial * xx / np.maximum(radius, 1e-9)) * patch
        dy = (tangent[1] + radial * yy / np.maximum(radius, 1e-9)) * patch
        return dx, dy

    def synthesize(
        self,
        normal_force_n,
        contact_uv=None,
        *,
        tangential_force_sdk_n=None,
        tangential_displacement_sdk_mm=None,
        contact_area_mm2=None,
    ) -> TactileProxyFrame:
        normal = _array(normal_force_n, (2,), "gel normal force")
        if np.any(normal < 0):
            raise ValueError("normal force is compressive positive")
        uv = _array(
            np.zeros((2, 2)) if contact_uv is None else contact_uv, (2, 2), "contact_uv"
        )
        if np.any(abs(uv) > 1):
            raise ValueError("contact_uv must lie within the active sensor")
        tangent = _array(
            np.zeros((2, 2))
            if tangential_force_sdk_n is None
            else tangential_force_sdk_n,
            (2, 2),
            "tangential force",
        )
        displacement = _array(
            np.zeros((2, 2))
            if tangential_displacement_sdk_mm is None
            else tangential_displacement_sdk_mm,
            (2, 2),
            "tangential displacement",
        )
        if np.any(np.linalg.norm(tangent[normal == 0], axis=1) > 0):
            raise ValueError("tangential contact force requires positive normal load")
        area_input = None
        if contact_area_mm2 is not None:
            area_input = _array(contact_area_mm2, (2,), "contact area")
            if np.any(area_input < 0):
                raise ValueError("contact area cannot be negative")
        fields, keys = self._fields.copy(), self._keyframes.copy()
        gel, wrench, area = self._gel.copy(), self._wrench.copy(), self._area.copy()
        force = np.c_[tangent, -normal]
        depth = normal * self.parameters.depth_mm_per_n
        for side in range(2):
            if normal[side] == 0:
                continue  # Stale displacement cannot paint a released gel.
            for array in (fields, keys):
                xx, yy, weights, patch = self._patch(array.shape[1:3], uv[side])
                dx, dy = self._displacement(
                    xx, yy, patch, depth[side], displacement[side]
                )
                array[side, ..., 0] += dx
                array[side, ..., 1] += dy
                array[side, ..., 2] += depth[side] * patch
                array[side, ..., 3] += displacement[side, 0] * patch
                array[side, ..., 4] += displacement[side, 1] * patch
                # Conserve TOTAL force even for patches truncated at an edge.
                array[side, ..., 5:8] += (
                    weights[..., None] * force[side] / (weights.mean() * NATIVE_PIXELS)
                )
            wrench[side, :3] += force[side]
            contact_mm = np.r_[
                uv[side] * np.asarray(self.parameters.active_size_mm) / 2, 0.0
            ]
            wrench[side, 3:] += (
                np.cross(contact_mm / 1000, force[side])
                * self.parameters.torque_nm_to_sdk
            )
            area[side] += (
                area_input[side]
                if area_input is not None
                else self.parameters.area_mm2_per_n_two_thirds * normal[side] ** (2 / 3)
            )
            import cv2

            xx, yy, _, patch = self._patch(NATIVE_HW, uv[side])
            dx, dy = self._displacement(xx, yy, patch, depth[side], displacement[side])
            grid_y, grid_x = np.indices(NATIVE_HW, dtype=np.float32)
            width, height = self.parameters.active_size_mm
            gel[side] = cv2.remap(
                self._gel[side],
                (grid_x - dx * NATIVE_HW[1] / width).astype(np.float32),
                (grid_y - dy * NATIVE_HW[0] / height).astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return TactileProxyFrame(
            gel,
            fields,
            keys,
            wrench,
            area,
            fields[..., 2].max(axis=(1, 2)),
            force.astype(np.float32),
            uv.astype(np.float32),
        )
