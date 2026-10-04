"""Test signal generation and the playback schedule.

Every player gets a short logarithmic sweep ("chirp"). A chirp has a sharp,
unambiguous cross-correlation peak, unlike a steady sine wave whose correlation
repeats every period. In *sequential* mode the players take turns and all use
the same full-band chirp. In *simultaneous* mode every player plays at once,
each in its own frequency band, so the players can be told apart in one
recording.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import numpy as np

DEFAULT_SAMPLE_RATE = 48_000


@dataclass(frozen=True, slots=True)
class ChirpSpec:
    """A logarithmic sine sweep from ``f0`` to ``f1`` Hz."""

    f0: float
    f1: float
    duration_s: float = 0.2
    fade_s: float = 0.005
    amplitude: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.f0 < self.f1:
            raise ValueError(f"need 0 < f0 < f1, got f0={self.f0} f1={self.f1}")
        if self.duration_s <= 2 * self.fade_s:
            raise ValueError("chirp duration must be longer than its two fades")

    @property
    def centre_hz(self) -> float:
        return float(np.sqrt(self.f0 * self.f1))

    def render(self, sample_rate: int) -> np.ndarray:
        """Render the chirp as float32 samples in [-amplitude, amplitude]."""
        if self.f1 >= sample_rate / 2:
            raise ValueError(f"f1={self.f1} Hz is above Nyquist for {sample_rate} Hz")
        n = round(self.duration_s * sample_rate)
        t = np.arange(n) / sample_rate
        k = np.log(self.f1 / self.f0)
        phase = 2 * np.pi * self.f0 * self.duration_s / k * (np.exp(t * k / self.duration_s) - 1)
        sig = np.sin(phase)
        fade_n = max(1, round(self.fade_s * sample_rate))
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_n))
        sig[:fade_n] *= ramp
        sig[-fade_n:] *= ramp[::-1]
        return (self.amplitude * sig).astype(np.float32)


def allocate_bands(
    count: int,
    f_lo: float = 400.0,
    f_hi: float = 12_000.0,
    guard: float = 0.08,
) -> list[tuple[float, float]]:
    """Split ``[f_lo, f_hi]`` into ``count`` log-spaced bands with guard gaps.

    ``guard`` is the fraction of each band (in log-frequency) kept free on each
    side, so neighbouring players' chirps don't overlap after filtering.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    edges = np.geomspace(f_lo, f_hi, count + 1)
    bands = []
    for lo, hi in itertools.pairwise(edges):
        span = np.log(hi / lo)
        bands.append((float(lo * np.exp(span * guard)), float(hi * np.exp(-span * guard))))
    return bands


@dataclass(frozen=True, slots=True)
class Emission:
    """One chirp scheduled on one player.

    ``offset_us`` is relative to the start of the test stream (the server-clock
    timestamp that is passed as ``play_start_us`` for the first chunk).
    """

    player_id: str
    repeat: int
    offset_us: int
    chirp: ChirpSpec


@dataclass(slots=True)
class Schedule:
    """Everything that will be played, per player."""

    mode: str
    sample_rate: int
    duration_us: int
    emissions: list[Emission] = field(default_factory=list)
    player_ids: list[str] = field(default_factory=list)

    def for_player(self, player_id: str) -> list[Emission]:
        return [e for e in self.emissions if e.player_id == player_id]

    def render_player(self, player_id: str) -> np.ndarray:
        """Render the whole test track (mono float32) for one player."""
        total = round(self.duration_us * self.sample_rate / 1_000_000)
        out = np.zeros(total, dtype=np.float32)
        for em in self.for_player(player_id):
            sig = em.chirp.render(self.sample_rate)
            start = round(em.offset_us * self.sample_rate / 1_000_000)
            end = min(total, start + len(sig))
            out[start:end] += sig[: end - start]
        return out


@dataclass(frozen=True, slots=True)
class ScheduleOptions:
    """Knobs that shape the schedule."""

    repeats: int = 5
    simultaneous: bool = False
    chirp_s: float = 0.2
    f_lo: float = 400.0
    f_hi: float = 12_000.0
    # Silence before the first chirp: players often fine-tune their sync early
    # in a stream, so the first second or two is not representative.
    lead_in_s: float = 2.0
    # Window the analysis searches around each expected arrival.
    max_early_s: float = 0.2
    max_late_s: float = 1.0
    # Silence after the window so the room's reverb dies away.
    tail_s: float = 0.3
    # Random extra gap so slots don't line up with periodic background noise.
    jitter_s: float = 0.15
    amplitude: float = 0.5
    seed: int | None = None

    @property
    def slot_s(self) -> float:
        return self.max_early_s + self.chirp_s + self.max_late_s + self.tail_s


def build_schedule(
    player_ids: list[str],
    options: ScheduleOptions | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Schedule:
    """Build the per-player emission schedule."""
    opts = options or ScheduleOptions()
    if not player_ids:
        raise ValueError("no players to schedule")
    if opts.repeats < 1:
        raise ValueError("repeats must be >= 1")
    rng = random.Random(opts.seed)
    sched = Schedule(
        mode="simultaneous" if opts.simultaneous else "sequential",
        sample_rate=sample_rate,
        duration_us=0,
        player_ids=list(player_ids),
    )

    if opts.simultaneous:
        bands = allocate_bands(len(player_ids), opts.f_lo, opts.f_hi)
        chirps = {
            pid: ChirpSpec(lo, hi, opts.chirp_s, amplitude=opts.amplitude)
            for pid, (lo, hi) in zip(player_ids, bands, strict=True)
        }
    else:
        shared = ChirpSpec(opts.f_lo, opts.f_hi, opts.chirp_s, amplitude=opts.amplitude)
        chirps = dict.fromkeys(player_ids, shared)

    # The first slot starts after the lead-in plus the "early" margin, so even
    # a player that plays early still lands inside the recording.
    t = opts.lead_in_s + opts.max_early_s
    for rep in range(opts.repeats):
        order = player_ids if opts.simultaneous else _rotated(player_ids, rep)
        if opts.simultaneous:
            for pid in player_ids:
                sched.emissions.append(Emission(pid, rep, round(t * 1e6), chirps[pid]))
            t += opts.slot_s + rng.uniform(0, opts.jitter_s)
        else:
            for pid in order:
                sched.emissions.append(Emission(pid, rep, round(t * 1e6), chirps[pid]))
                t += opts.slot_s + rng.uniform(0, opts.jitter_s)
    sched.duration_us = round((t + opts.tail_s) * 1e6)
    return sched


def _rotated(items: list[str], n: int) -> list[str]:
    """Rotate the play order each repeat so no player is always first."""
    if not items:
        return items
    n %= len(items)
    return items[n:] + items[:n]


def to_pcm16_stereo(mono: np.ndarray) -> bytes:
    """Convert mono float samples to interleaved 16-bit stereo PCM bytes."""
    clipped = np.clip(mono, -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    return np.repeat(ints[:, None], 2, axis=1).tobytes()
