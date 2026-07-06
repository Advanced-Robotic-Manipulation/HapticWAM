"""Test helpers (uniquely named to avoid site-packages shadowing)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from phantom.config.hardware import HardwareConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
HW_YAML = REPO_ROOT / "configs" / "hardware.yaml"

SMALL_OVERRIDES = dict(
    tactile={"field": {"h": 48, "w": 64},
             "raw_img": {"h": 60, "w": 80, "c": 1},
             "infer_img": {"h": 60, "w": 80, "c": 1},
             "rate_hz": 30.0},
    cameras={"scene": {"color": {"h": 60, "w": 80, "c": 3}, "fps": 10.0}},
    recording={"field_ds": {"h": 24, "w": 32}, "field_ds_rate_hz": 30.0,
               "keyframe_rate_hz": 5.0, "infer_img_rate_hz": 10.0,
               "zarr_chunk_frames": 16},
    derived={"cpk_downsample": 4},
    wrist_ft={"window_s": 0.1},
)


def load_raw() -> dict:
    with open(HW_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_hw(**overrides) -> HardwareConfig:
    """Build a HardwareConfig from the repo yaml with nested-dict overrides."""
    raw = load_raw()

    def merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(raw, copy.deepcopy(overrides))
    return HardwareConfig.model_validate(raw)


def make_small_hw(**extra) -> HardwareConfig:
    o = copy.deepcopy(SMALL_OVERRIDES)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(o.get(k), dict):
            o[k].update(v)
        else:
            o[k] = v
    return make_hw(**o)
