import numpy as np
import pytest

from sendspin_syncer.analysis import AnalysisOptions, analyse
from sendspin_syncer.clock_map import ClockMap
from sendspin_syncer.signals import ScheduleOptions, build_schedule, top_frequency_for

from .synth import render_recording

STREAM_START_US = 5_000_000


def run(schedule, delays, mic_rate=48_000, **kw):
    rec, start = render_recording(schedule, delays, mic_rate=mic_rate, **kw)
    clock = ClockMap.nominal(STREAM_START_US - start * 1e6 / mic_rate, mic_rate)
    return analyse(rec, mic_rate, clock, schedule, STREAM_START_US)


@pytest.mark.parametrize("simultaneous", [False, True])
def test_recovers_known_delays_with_reverb(simultaneous):
    delays = {"kitchen": 3.0, "lounge": 87.4, "bedroom": 12.13, "early": -15.5}
    sched = build_schedule(
        list(delays), ScheduleOptions(repeats=4, simultaneous=simultaneous, seed=3)
    )
    res = run(sched, delays)
    for pid, d in delays.items():
        r = res[pid]
        # Narrow simultaneous bands blur the direct sound with early reflections.
        assert r.median_ms == pytest.approx(d, abs=0.3 if simultaneous else 0.1), pid
        assert r.detected_count == 4
        assert r.confidence == "high"


def test_mic_at_44k1():
    delays = {"a": 10.0, "b": 250.0}
    sched = build_schedule(list(delays), ScheduleOptions(repeats=3, seed=5))
    res = run(sched, delays, mic_rate=44_100)
    for pid, d in delays.items():
        assert res[pid].median_ms == pytest.approx(d, abs=0.1)


@pytest.mark.parametrize("simultaneous", [False, True])
def test_low_rate_webcam_mic(simultaneous):
    """A 16 kHz mic hears only the bottom of a full-band sweep."""
    delays = {"a": 10.0, "b": 47.3, "c": 3.2}
    for f_hi in (12_000, top_frequency_for(16_000)):
        if simultaneous and f_hi == 12_000:
            continue  # bands above 8 kHz are inaudible to this mic; the CLI avoids that
        opts = ScheduleOptions(repeats=3, seed=5, simultaneous=simultaneous, f_hi=f_hi)
        sched = build_schedule(list(delays), opts)
        res = run(sched, delays, mic_rate=16_000)
        for pid, d in delays.items():
            assert res[pid].median_ms == pytest.approx(d, abs=0.3), (pid, f_hi)


def test_silent_player_reports_no_signal():
    delays = {"a": 5.0, "silent": 0.0}
    sched = build_schedule(list(delays), ScheduleOptions(repeats=3, seed=2))
    res = run(sched, delays, gains={"silent": 0.0})
    assert res["a"].median_ms == pytest.approx(5.0, abs=0.1)
    assert res["silent"].median_ms is None
    assert res["silent"].confidence == "none"


def test_noisy_room_still_measures():
    delays = {"a": 40.0, "b": 41.5}
    sched = build_schedule(list(delays), ScheduleOptions(repeats=5, seed=9))
    res = run(sched, delays, noise=0.1, gains={"a": 0.05, "b": 0.05})
    for pid, d in delays.items():
        assert res[pid].median_ms == pytest.approx(d, abs=0.25)


def test_delay_beyond_window_is_not_found():
    delays = {"a": 1500.0}
    opts = ScheduleOptions(repeats=2, seed=1, max_late_s=1.0)
    sched = build_schedule(list(delays), opts)
    res = run(sched, delays, reverb=False)
    # Whatever it latches onto, it must not claim a confident 1.5 s result.
    assert res["a"].median_ms is None or abs(res["a"].median_ms - 1500) > 100


def test_clock_drift_is_corrected_by_fitted_map():
    delays = {"a": 20.0, "b": 60.0}
    sched = build_schedule(list(delays), ScheduleOptions(repeats=6, seed=4))
    drift = 300.0  # ppm, deliberately large
    rec, start = render_recording(sched, delays, drift_ppm=drift)
    rate = 48_000
    true_us_per_sample = 1e6 / (rate * (1 + drift / 1e6))
    origin = STREAM_START_US - start * true_us_per_sample
    # Simulate callback observations: late by 0-3 ms of jitter.
    rng = np.random.default_rng(0)
    idx = np.arange(0, len(rec), 480)
    obs = origin + idx * true_us_per_sample + rng.uniform(0, 3000, idx.size)
    clock = ClockMap.fit(idx, obs, rate)
    assert clock.drift_ppm == pytest.approx(drift, abs=30)
    res = analyse(rec, rate, clock, sched, STREAM_START_US, AnalysisOptions())
    relative = res["b"].median_ms - res["a"].median_ms
    assert relative == pytest.approx(40.0, abs=0.2)


def _emissions(pid, delays, snr=30.0):
    from sendspin_syncer.analysis import EmissionResult

    return [EmissionResult(pid, i, 0, d, snr, True) for i, d in enumerate(delays)]


def test_real_player_wander_is_kept_but_wrong_peaks_are_not():
    """From a real run: chirps a few ms apart are all genuine."""
    from sendspin_syncer.analysis import PlayerResult, _aggregate

    desktop = PlayerResult(
        "desktop", _emissions("desktop", [250.26, 252.0, 254.06, 247.58, 245.51])
    )
    living = PlayerResult("living", _emissions("living", [-26.62, -26.7, -27.81, -28.02, -30.46]))
    for pr in (desktop, living):
        _aggregate(pr, AnalysisOptions())
        assert pr.detected_count == 5
    assert desktop.median_ms == pytest.approx(250.26)
    assert desktop.confidence == "medium"  # its own timing wanders ±3 ms
    assert living.confidence == "high"

    # A reflection or noise pick far from the rest is still rejected.
    noisy = PlayerResult("n", _emissions("n", [40.1, 40.3, 39.8, 87.0, 40.0]))
    _aggregate(noisy, AnalysisOptions())
    assert noisy.detected_count == 4
    assert noisy.median_ms == pytest.approx(40.05)
