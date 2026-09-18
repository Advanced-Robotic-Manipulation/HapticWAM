"""Shared fixtures. Tests are cosmos-free unless explicitly marked; the
`small_hw` fixture mutates hardware VALUES (never ranks) to keep test tensors
tiny AND to double as the config-robustness check the user required."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phantom_test_utils import HW_YAML, make_small_hw  # noqa: E402

from phantom.config.hardware import HardwareConfig, load_hardware  # noqa: E402


@pytest.fixture
def default_hw() -> HardwareConfig:
    return load_hardware(HW_YAML, quiet=True)


@pytest.fixture
def small_hw() -> HardwareConfig:
    """Small VALUES, same ranks — fast tests + robustness-to-value-change."""
    return make_small_hw()


# ---------------------------------------------------------------------------
# Environment capability markers.
#
# A plain CPU-only CI runner (GitHub Actions ubuntu-latest) has no GPU, no rig
# hardware (UR arm, Robotiq gripper, DM-Tac pads, RealSense), no Isaac Sim /
# OpenUSD, no OpenCV, and neither the cosmos-predict2.5 source checkout (a
# submodule) nor the ~4.8 GB Cosmos-Predict2.5-2B weights (gitignored).  Tests
# needing any of those carry one of the markers below and are SKIPPED with a
# clear reason instead of failing.  On a box that does have the dependency the
# very same tests run normally — nothing is masked.
#
# Usage in a test module:
#     pytestmark = pytest.mark.requires_usd          # whole module
#     @pytest.mark.requires_cv2                      # single test
# Module-level `import cv2` still needs `pytest.importorskip` instead, because
# markers cannot rescue an import that happens at collection time.
# ---------------------------------------------------------------------------


def _module_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def _paths():
    from phantom.config.paths import load_paths

    return load_paths()


def _cosmos_repo_missing() -> bool:
    """True when configs/paths.yaml does not point at a real cosmos checkout."""
    try:
        return not (_paths().cosmos_repo / "cosmos_predict2" / "__about__.py").is_file()
    except Exception:
        return True


def _cosmos_weights_missing() -> bool:
    try:
        return not _paths().cosmos_checkpoint.is_file()
    except Exception:
        return True


def _data_root_missing() -> bool:
    """True when paths.data_root's parent is absent (box-only data tree)."""
    try:
        return not _paths().data_root.parent.exists()
    except Exception:
        return True


# marker name -> (unavailable?, reason).  Probed once per session.
_CAPABILITIES: dict[str, tuple[bool, str]] = {
    "requires_cv2": (
        _module_missing("cv2"),
        "needs OpenCV (pip install '.[sim]'); not installed on a plain CI runner",
    ),
    "requires_usd": (
        _module_missing("pxr"),
        "needs OpenUSD / Isaac Sim (pxr); not available on a plain CI runner",
    ),
    "requires_torch": (
        _module_missing("torch"),
        "needs torch (pip install '.[train]')",
    ),
    "requires_cosmos_repo": (
        _cosmos_repo_missing(),
        "needs the cosmos-predict2.5 source checkout (submodule + "
        "configs/paths.local.yaml); absent on a plain CI runner",
    ),
    "requires_cosmos_weights": (
        _cosmos_weights_missing(),
        "needs the ~4.8 GB Cosmos-Predict2.5-2B weights; absent on a plain CI runner",
    ),
    "requires_data_root": (
        _data_root_missing(),
        "needs the recording box's paths.data_root tree; absent on a plain CI runner",
    ),
    "requires_linux": (
        sys.platform != "linux",
        "needs Linux-only kernel interfaces (/proc)",
    ),
}


def pytest_configure(config: pytest.Config) -> None:
    for name in _CAPABILITIES:
        config.addinivalue_line("markers", f"{name}: skipped when unavailable")


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    for item in items:
        for name, (unavailable, reason) in _CAPABILITIES.items():
            if unavailable and item.get_closest_marker(name) is not None:
                item.add_marker(pytest.mark.skip(reason=reason))
