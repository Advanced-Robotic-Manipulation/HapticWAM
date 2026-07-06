"""Speed governor (pipeline.md §4): the contact-group predictive uncertainty
sigma scales chunk PLAYBACK SPEED (time reparametrization, path-preserving) —
the student slows down exactly where its own hallucination is unreliable."""

from __future__ import annotations

import numpy as np

from phantom.config.hardware import GovernorConfig


class SpeedGovernor:
    def __init__(self, cfg: GovernorConfig):
        self.cfg = cfg

    def scale(self, sigma: float) -> float:
        """sigma -> playback-speed factor in [min_scale, 1]."""
        c = self.cfg
        u = (sigma - c.sigma_lo) / (c.sigma_hi - c.sigma_lo)
        u = float(np.clip(u, 0.0, 1.0))
        return 1.0 - u * (1.0 - c.min_scale)

    def scale_profile(self, sigma_per_step: np.ndarray) -> float:
        """Conservative: govern the whole chunk by the worst upcoming step."""
        return self.scale(float(np.max(sigma_per_step))) if sigma_per_step.size \
            else 1.0
