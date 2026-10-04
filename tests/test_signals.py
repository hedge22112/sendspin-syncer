import itertools

import numpy as np
import pytest

from sendspin_syncer.signals import (
    ChirpSpec,
    ScheduleOptions,
    allocate_bands,
    build_schedule,
    to_pcm16_stereo,
)


def test_chirp_shape_and_level():
    sig = ChirpSpec(500, 8000, 0.2, amplitude=0.5).render(48_000)
    assert sig.dtype == np.float32
    assert len(sig) == 9600
    assert np.max(np.abs(sig)) <= 0.5 + 1e-6
    assert abs(sig[0]) < 1e-3 and abs(sig[-1]) < 1e-3  # faded


def test_chirp_rejects_bad_specs():
    with pytest.raises(ValueError):
        ChirpSpec(8000, 500)
    with pytest.raises(ValueError):
        ChirpSpec(500, 30_000).render(48_000)


def test_bands_do_not_overlap():
    bands = allocate_bands(6, 400, 12_000)
    assert len(bands) == 6
    for (lo1, hi1), (lo2, hi2) in itertools.pairwise(bands):
        assert lo1 < hi1 < lo2 < hi2


def test_sequential_schedule_spacing():
    opts = ScheduleOptions(repeats=3, seed=1)
    sched = build_schedule(["a", "b"], opts)
    assert len(sched.emissions) == 6
    offsets = [e.offset_us for e in sched.emissions]
    assert offsets == sorted(offsets)
    gaps = np.diff(offsets) / 1e6
    assert np.all(gaps >= opts.slot_s)
    assert sched.duration_us > offsets[-1]
    # Rotation: player order changes between repeats.
    assert sched.emissions[0].player_id == "a"
    assert sched.emissions[2].player_id == "b"


def test_simultaneous_schedule_uses_distinct_bands():
    sched = build_schedule(["a", "b", "c"], ScheduleOptions(repeats=2, simultaneous=True))
    first = [e for e in sched.emissions if e.repeat == 0]
    assert len({e.offset_us for e in first}) == 1
    assert len({e.chirp for e in first}) == 3


def test_render_player_contains_only_its_chirps():
    sched = build_schedule(["a", "b"], ScheduleOptions(repeats=1, seed=0))
    a = sched.render_player("a")
    b = sched.render_player("b")
    assert len(a) == len(b) == round(sched.duration_us * 48_000 / 1e6)
    ea = sched.for_player("a")[0]
    i = round(ea.offset_us * 48_000 / 1e6)
    assert np.abs(a[i : i + 9600]).max() > 0.4
    assert np.abs(b[i : i + 9600]).max() == 0


def test_pcm16_stereo():
    pcm = to_pcm16_stereo(np.array([0.0, 1.0, -1.0], dtype=np.float32))
    assert len(pcm) == 3 * 2 * 2
    vals = np.frombuffer(pcm, dtype="<i2")
    assert list(vals) == [0, 0, 32767, 32767, -32767, -32767]
