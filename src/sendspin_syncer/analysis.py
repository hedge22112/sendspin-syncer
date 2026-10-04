"""Find each chirp in the microphone recording and turn it into a delay."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .clock_map import ClockMap
from .signals import ChirpSpec, Schedule

# A local peak of the correlation envelope this close to the strongest peak,
# and at most this far before it, counts as the direct sound. Room
# reflections arrive after the direct sound and can be louder than it.
FIRST_ARRIVAL_RATIO = 0.5
FIRST_ARRIVAL_LOOKBACK_S = 0.02


@dataclass(frozen=True, slots=True)
class Detection:
    """One chirp found in the recording."""

    arrival_sample: float
    snr_db: float


@dataclass(frozen=True, slots=True)
class EmissionResult:
    player_id: str
    repeat: int
    expected_us: int
    delay_ms: float | None
    snr_db: float
    accepted: bool
    reason: str = ""


@dataclass(slots=True)
class PlayerResult:
    """Aggregated measurement for one player."""

    player_id: str
    emissions: list[EmissionResult] = field(default_factory=list)
    median_ms: float | None = None
    spread_ms: float | None = None
    snr_db: float | None = None
    confidence: str = "none"

    @property
    def accepted(self) -> list[EmissionResult]:
        return [e for e in self.emissions if e.accepted]

    @property
    def detected_count(self) -> int:
        return len(self.accepted)


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    max_early_s: float = 0.2
    max_late_s: float = 1.0
    # Pure noise peaks at roughly 11-12 dB above the median of a 1 s window.
    min_snr_db: float = 14.0
    # Chirps within this window of each other are taken to be the same arrival.
    # Real players wander by a few ms between chirps (their sync corrections);
    # a wrong peak (a reflection, or noise) is usually much further off.
    cluster_ms: float = 15.0
    # Detections are kept within max(this, 4 x the robust spread) of the median.
    min_outlier_ms: float = 5.0


def bandpass(x: np.ndarray, sample_rate: int, f_lo: float, f_hi: float) -> np.ndarray:
    """Zero-phase band-pass filter, so filtering adds no delay of its own."""
    nyq = sample_rate / 2
    lo = max(20.0, f_lo * 0.8)
    hi = min(f_hi * 1.25, nyq * 0.95)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, x)


def detect_chirp(
    recording: np.ndarray,
    sample_rate: int,
    chirp: ChirpSpec,
    window_start: int,
    window_end: int,
) -> Detection | None:
    """Find where ``chirp`` starts within ``[window_start, window_end]``.

    Returns the sub-sample index (in ``recording``) where the chirp begins, or
    ``None`` if the window is outside the recording.
    """
    template = chirp.render_at(sample_rate).astype(np.float64)
    pad = len(template)
    seg_start = max(0, window_start)
    seg_end = min(len(recording), window_end + pad)
    if seg_end - seg_start < pad + 8:
        return None
    seg = bandpass(
        np.asarray(recording[seg_start:seg_end], dtype=np.float64),
        sample_rate,
        chirp.f0,
        chirp.f1,
    )
    corr = signal.correlate(seg, template, mode="valid", method="fft")
    # The envelope ignores the carrier phase, so polarity-inverted speakers and
    # phase shifts in the speaker or room don't move the peak.
    env = np.abs(signal.hilbert(corr))
    peak = int(np.argmax(env))
    peak_val = env[peak]
    noise = float(np.median(env)) + 1e-12
    snr_db = float(20 * np.log10(peak_val / noise + 1e-12))

    # Prefer the earliest strong arrival (direct path) over a louder reflection.
    lookback = round(FIRST_ARRIVAL_LOOKBACK_S * sample_rate)
    lo = max(1, peak - lookback)
    best = peak
    if lo < peak:
        region = env[lo:peak]
        is_local_max = (region > env[lo - 1 : peak - 1]) & (region >= env[lo + 1 : peak + 1])
        candidates = np.nonzero(is_local_max & (region >= FIRST_ARRIVAL_RATIO * peak_val))[0]
        if candidates.size:
            best = lo + int(candidates[0])

    frac = 0.0
    if 0 < best < len(env) - 1:
        a, b, c = env[best - 1], env[best], env[best + 1]
        denom = a - 2 * b + c
        if denom != 0:
            frac = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return Detection(arrival_sample=seg_start + best + frac, snr_db=snr_db)


def analyse(
    recording: np.ndarray,
    sample_rate: int,
    clock: ClockMap,
    schedule: Schedule,
    stream_start_us: int,
    options: AnalysisOptions | None = None,
) -> dict[str, PlayerResult]:
    """Measure every scheduled emission and aggregate the results per player."""
    opts = options or AnalysisOptions()
    results = {pid: PlayerResult(pid) for pid in schedule.player_ids}
    early = round(opts.max_early_s * sample_rate)
    late = round(opts.max_late_s * sample_rate)

    for em in schedule.emissions:
        expected_us = stream_start_us + em.offset_us
        expected = float(clock.to_sample(expected_us))
        det = detect_chirp(
            recording,
            sample_rate,
            em.chirp,
            round(expected) - early,
            round(expected) + late,
        )
        if det is None:
            res = EmissionResult(
                em.player_id, em.repeat, expected_us, None, 0.0, False, "outside recording"
            )
        else:
            delay_ms = (float(clock.to_server_us(det.arrival_sample)) - expected_us) / 1000
            ok = det.snr_db >= opts.min_snr_db
            res = EmissionResult(
                em.player_id,
                em.repeat,
                expected_us,
                delay_ms,
                det.snr_db,
                ok,
                "" if ok else "weak signal",
            )
        results[em.player_id].emissions.append(res)

    for pr in results.values():
        _aggregate(pr, opts)
    return results


def _aggregate(pr: PlayerResult, opts: AnalysisOptions) -> None:
    good = [e for e in pr.emissions if e.accepted and e.delay_ms is not None]
    if not good:
        pr.confidence = "none"
        return
    centre = _densest_delay(good, opts.cluster_ms)
    group = np.array([e.delay_ms for e in good if abs(e.delay_ms - centre) <= opts.cluster_ms])
    mad = float(np.median(np.abs(group - np.median(group))))
    tolerance = max(opts.min_outlier_ms, 4 * 1.4826 * mad)
    # Drop detections that latched onto the wrong peak (or onto noise).
    for i, e in enumerate(pr.emissions):
        if e.accepted and e.delay_ms is not None and abs(e.delay_ms - centre) > tolerance:
            pr.emissions[i] = EmissionResult(
                e.player_id, e.repeat, e.expected_us, e.delay_ms, e.snr_db, False, "outlier"
            )
    kept = pr.accepted
    delays = np.array([e.delay_ms for e in kept])
    snrs = np.array([e.snr_db for e in kept])
    pr.median_ms = float(np.median(delays))
    pr.spread_ms = float(np.std(delays)) if len(delays) > 1 else None
    pr.snr_db = float(np.median(snrs))
    pr.confidence = _confidence(pr.snr_db, pr.spread_ms, len(kept), len(pr.emissions))


def _densest_delay(detections: list[EmissionResult], tolerance_ms: float) -> float:
    """Median of the largest group of detections that agree within ``tolerance_ms``.

    Ties go to the group with the stronger signal.
    """
    best: tuple[int, float, float] | None = None
    for e in detections:
        assert e.delay_ms is not None
        group = [d for d in detections if abs(d.delay_ms - e.delay_ms) <= tolerance_ms]  # type: ignore[operator]
        key = (len(group), sum(d.snr_db for d in group))
        if best is None or key > best[:2]:
            best = (*key, float(np.median([d.delay_ms for d in group])))
    assert best is not None
    return best[2]


def _confidence(snr_db: float, spread_ms: float | None, kept: int, total: int) -> str:
    """How much to trust a player's median delay.

    Mostly about detection: were the chirps heard clearly and consistently?
    A player whose own timing wanders by a few ms between chirps still gets a
    trustworthy median, but drops a grade.
    """
    fraction = kept / total if total else 0.0
    spread = spread_ms if spread_ms is not None else float("inf")
    if snr_db >= 20 and spread <= 3.0 and fraction >= 0.8 and kept >= 3:
        return "high"
    if snr_db >= 14 and spread <= 10.0 and fraction >= 0.5 and kept >= 2:
        return "medium"
    return "low"
