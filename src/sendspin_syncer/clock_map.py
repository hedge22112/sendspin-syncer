"""Map microphone sample indices to the Sendspin server clock.

During capture, every audio callback gives us an observation: "the sample at
index ``n`` was captured at about server time ``t``". These observations are
noisy, and the error only goes one way: a callback runs *after* the samples it
delivers were captured, so its estimate is never early, only late by some
scheduling jitter. We fit the slope (the mic's real sample rate measured in
server microseconds) by least squares. The intercept comes from a low
percentile of the residuals, i.e. the least-delayed callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ClockMap:
    """Linear map ``server_us = origin_us + sample_index * us_per_sample``."""

    origin_us: float
    us_per_sample: float
    nominal_rate: int
    jitter_us: float = 0.0
    observations: int = 0
    fitted: bool = False
    """Whether the rate was measured; if not, the nominal rate is assumed."""
    raw_drift_ppm: float | None = None
    """The drift the timing data suggested, even if it was rejected."""

    @property
    def effective_rate(self) -> float:
        """Mic sample rate measured against the server clock."""
        return 1e6 / self.us_per_sample

    @property
    def drift_ppm(self) -> float:
        return (self.effective_rate / self.nominal_rate - 1.0) * 1e6

    def to_server_us(self, sample_index: float | np.ndarray) -> float | np.ndarray:
        return self.origin_us + np.asarray(sample_index) * self.us_per_sample

    def to_sample(self, server_us: float | np.ndarray) -> float | np.ndarray:
        return (np.asarray(server_us) - self.origin_us) / self.us_per_sample

    @classmethod
    def nominal(cls, origin_us: float, sample_rate: int) -> ClockMap:
        return cls(origin_us, 1e6 / sample_rate, sample_rate)

    @classmethod
    def fit(
        cls,
        sample_indices: np.ndarray,
        server_us: np.ndarray,
        nominal_rate: int,
        *,
        envelope_percentile: float = 5.0,
        max_drift_ppm: float = 2000.0,
    ) -> ClockMap:
        """Fit a map from (sample index, observed server time) pairs."""
        n = np.asarray(sample_indices, dtype=np.float64)
        t = np.asarray(server_us, dtype=np.float64)
        if n.shape != t.shape or n.size == 0:
            raise ValueError("need matching, non-empty observation arrays")
        nominal_us = 1e6 / nominal_rate

        slope = nominal_us
        fitted = False
        raw_ppm = None
        if n.size >= 10 and np.ptp(n) > nominal_rate:  # at least ~1 s of data
            raw, _ = np.polyfit(n, t, 1)
            raw_ppm = float((nominal_us / raw - 1.0) * 1e6)
            # Outside a sane range means the timing data is bad; trust nominal.
            if abs(raw_ppm) <= max_drift_ppm:
                slope, fitted = float(raw), True

        residual = t - n * slope
        origin = float(np.percentile(residual, envelope_percentile))
        jitter = float(np.percentile(residual, 95) - origin)
        return cls(origin, float(slope), nominal_rate, jitter, int(n.size), fitted, raw_ppm)
