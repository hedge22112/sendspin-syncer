"""The recorder's timeline must survive audio dropouts from the driver."""

from types import SimpleNamespace

import numpy as np
import pytest

from sendspin_syncer.analysis import analyse
from sendspin_syncer.capture import InputDevice, SoundDeviceRecorder
from sendspin_syncer.signals import ScheduleOptions, build_schedule

from .synth import render_recording

RATE = 48_000
BLOCK = 1024
STREAM_START_US = 50_000_000
DEVICE = InputDevice(7, "USB PnP Sound Device", "ALSA", 1, 48000.0, False)


def feed(rec, recorder, origin_us, *, adc_usable, drop_every=0, seed=0):
    """Push ``rec`` through the real callback in blocks, losing some on the way.

    Sample ``i`` of ``rec`` was captured at server time ``origin_us + i / RATE``.
    Like PortAudio, a lost block is reported as an overflow on the next one.
    Returns the number of blocks dropped.
    """
    rng = np.random.default_rng(seed)
    clock = {"now": 0.0}
    recorder._now_us = lambda: clock["now"]
    dropped = 0
    pending_overflow = False
    for b, i in enumerate(range(0, len(rec) - BLOCK, BLOCK)):
        if drop_every and b % drop_every == drop_every - 1:
            dropped += 1
            pending_overflow = True
            continue
        capture_s = (origin_us + i * 1e6 / RATE) / 1e6
        late_s = BLOCK / RATE + rng.uniform(0, 0.004)  # callback jitter
        clock["now"] = (capture_s + late_s) * 1e6
        t = SimpleNamespace(
            inputBufferAdcTime=capture_s if adc_usable else 0.0,
            currentTime=capture_s + late_s,
        )
        status = SimpleNamespace(input_overflow=pending_overflow)
        pending_overflow = False
        recorder._callback(rec[i : i + BLOCK, None], BLOCK, t, status)
    return dropped


@pytest.mark.parametrize("adc_usable", [True, False])
@pytest.mark.parametrize("drop_every", [0, 37])
def test_dropouts_do_not_shift_the_timeline(adc_usable, drop_every):
    delays = {"desktop": 52.0, "living-room": 0.0}
    sched = build_schedule(list(delays), ScheduleOptions(repeats=5, seed=2))
    rec, start = render_recording(sched, delays, mic_rate=RATE)
    recorder = SoundDeviceRecorder(DEVICE, lambda: 0, sample_rate=RATE)
    dropped = feed(
        rec,
        recorder,
        STREAM_START_US - start * 1e6 / RATE,
        adc_usable=adc_usable,
        drop_every=drop_every,
    )
    recording = recorder.stop()
    assert recording.overflows == dropped
    assert recording.gaps_ms == pytest.approx(dropped * BLOCK / RATE * 1000, rel=0.1, abs=1)
    if drop_every:
        assert dropped >= 20

    res = analyse(recording.samples, RATE, recording.clock, sched, STREAM_START_US)
    # Without driver timestamps, a short segment between two dropouts can only
    # be placed to within the callback jitter, so allow a little more there.
    tol = 0.3 if adc_usable else 1.0
    for pid, d in delays.items():
        r = res[pid]
        # A chirp that falls into a gap can be lost; the rest must be right.
        assert r.detected_count >= 3, pid
        assert r.median_ms == pytest.approx(d, abs=tol), pid
