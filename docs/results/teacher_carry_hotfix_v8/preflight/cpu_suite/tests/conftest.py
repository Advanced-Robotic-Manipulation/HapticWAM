"""Shared fixtures. Tests are cosmos-free unless explicitly marked; the
`small_hw` fixture mutates hardware VALUES (never ranks) to keep test tensors
tiny AND to double as the config-robustness check the user required."""

from __future__ import annotations

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
