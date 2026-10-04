"""Helpers that fake an acoustic recording of a schedule."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from sendspin_syncer.signals import Schedule


def room_impulse(sample_rate: int, rng: np.random.Generator, rt60_s: float = 0.25) -> np.ndarray:
    """A simple room: direct sound, a couple of early reflections, a noisy tail."""
    n = int(rt60_s * sample_rate)
    ir = np.zeros(n)
    ir[0] = 1.0
    for delay_ms, gain in ((3.1, 0.6), (7.4, -0.45), (12.0, 0.35)):
        ir[int(delay_ms * sample_rate / 1000)] += gain
    t = np.arange(n) / sample_rate
    ir += 0.05 * rng.standard_normal(n) * np.exp(-6.9 * t / rt60_s)
    return ir


def render_recording(
    schedule: Schedule,
    delays_ms: dict[str, float],
    *,
    mic_rate: int = 48_000,
    lead_s: float = 0.3,
    noise: float = 0.01,
    gains: dict[str, float] | None = None,
    reverb: bool = True,
    drift_ppm: float = 0.0,
    seed: int = 1,
) -> tuple[np.ndarray, int]:
    """Return (recording, sample index of stream start)."""
    rng = np.random.default_rng(seed)
    ratio = 1 + drift_ppm / 1e6  # mic samples per nominal sample
    total_s = lead_s + schedule.duration_us / 1e6 + 1.5
    rec = np.zeros(int(total_s * mic_rate * ratio))
    start = int(lead_s * mic_rate * ratio)
    for pid in schedule.player_ids:
        g = (gains or {}).get(pid, 0.3)
        ir = room_impulse(mic_rate, rng) if reverb else np.array([1.0])
        for em in schedule.for_player(pid):
            chirp = em.chirp.render(mic_rate).astype(np.float64)
            if drift_ppm:
                chirp = sps.resample(chirp, round(len(chirp) * ratio))
            sound = np.convolve(chirp, ir) * g
            at = start + (em.offset_us / 1e6 + delays_ms[pid] / 1000) * mic_rate * ratio
            i = int(np.floor(at))
            frac = at - i
            # Fractional delay through linear interpolation of the shift.
            shifted = np.interp(
                np.arange(len(sound) + 1) - frac, np.arange(len(sound)), sound, left=0, right=0
            )
            rec[i : i + len(shifted)] += shifted[: len(rec) - i]
    rec += noise * rng.standard_normal(len(rec))
    return rec.astype(np.float32), start
