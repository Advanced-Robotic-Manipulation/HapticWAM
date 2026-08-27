"""terminal_eval close-step rule: the prediction is judged against the GT's absolute close aperture."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terminal_eval as TE  # noqa: E402


def _ramp(a, b, n):
    return np.linspace(a, b, n)


def test_pred_partial_rise_is_not_a_close():
    gt = np.concatenate([np.full(8, 0.10), _ramp(0.10, 0.62, 4), np.full(4, 0.62)])   # closes at 8..11
    gi, pi = TE.close_steps(gt, np.concatenate([np.full(6, 0.10), _ramp(0.10, 0.40, 4), np.full(6, 0.40)]))
    assert gi is not None and pi == 16, "a 0.40 plateau must not count as closing to 0.62"
    gi2, pi2 = TE.close_steps(gt, np.concatenate([np.full(9, 0.10), _ramp(0.10, 0.60, 4), np.full(3, 0.60)]))
    assert pi2 in (11, 12) and gi2 in (10, 11)


def test_wide_carton_close_uses_fallback_for_gt_only():
    gt = np.concatenate([np.full(6, 0.28), _ramp(0.28, 0.42, 6), np.full(4, 0.42)])   # never > 0.45
    gi, pi = TE.close_steps(gt, gt.copy())
    assert gi is not None and abs(pi - gi) <= 1
    _, pi_none = TE.close_steps(gt, np.full(16, 0.28))
    assert pi_none == 16


def test_gt_without_close_yields_none():
    gi, pi = TE.close_steps(np.full(16, 0.2), np.full(16, 0.6))
    assert gi is None and pi == 16
