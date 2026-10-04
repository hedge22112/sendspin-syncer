"""Audio input devices: listing, selection and timestamped recording."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .clock_map import ClockMap

PREFERRED_RATES = (48_000, 44_100, 96_000)


class DeviceError(RuntimeError):
    """Raised when an input device can't be found or opened."""


def _sd() -> Any:
    """Import sounddevice lazily, so a missing PortAudio only breaks audio commands."""
    try:
        import sounddevice
    except OSError as err:  # PortAudio library not installed
        raise DeviceError(
            "PortAudio is not available. On Debian/Ubuntu/Raspberry Pi OS install it with "
            "`sudo apt install libportaudio2`."
        ) from err
    return sounddevice


@dataclass(frozen=True, slots=True)
class InputDevice:
    index: int
    name: str
    host_api: str
    channels: int
    default_rate: float
    is_default: bool

    def describe(self) -> str:
        return f"[{self.index}] {self.name} ({self.host_api})"


def list_input_devices() -> list[InputDevice]:
    """Return every device that can record audio."""
    sd = _sd()
    try:
        devices = sd.query_devices()
        apis = sd.query_hostapis()
        default_in = sd.default.device[0]
    except sd.PortAudioError as err:
        raise DeviceError(f"could not query audio devices: {err}") from err
    if default_in is None or default_in < 0:
        # Fall back to the host APIs' own default input.
        default_in = next(
            (a["default_input_device"] for a in apis if a["default_input_device"] >= 0), -1
        )
    return [
        InputDevice(
            index=i,
            name=d["name"],
            host_api=apis[d["hostapi"]]["name"],
            channels=int(d["max_input_channels"]),
            default_rate=float(d["default_samplerate"]),
            is_default=i == default_in,
        )
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def resolve_device(spec: str | int | None, devices: list[InputDevice] | None = None) -> InputDevice:
    """Pick an input device by index, (partial) name, or the system default."""
    devices = list_input_devices() if devices is None else devices
    if not devices:
        raise DeviceError("no audio input devices found")
    if spec is None or spec == "":
        for d in devices:
            if d.is_default:
                return d
        return devices[0]
    if isinstance(spec, int) or str(spec).strip().isdigit():
        idx = int(spec)
        for d in devices:
            if d.index == idx:
                return d
        raise DeviceError(f"no input device with index {idx}; run `sendspin-syncer devices`")
    needle = str(spec).strip().lower()
    exact = [d for d in devices if d.name.lower() == needle]
    if len(exact) == 1:
        return exact[0]
    matches = exact or [d for d in devices if needle in d.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DeviceError(f"no input device matches {spec!r}; run `sendspin-syncer devices`")
    listing = "\n".join(f"  {d.describe()}" for d in matches)
    raise DeviceError(
        f"{spec!r} matches more than one input device; use the index instead:\n{listing}"
    )


def pick_sample_rate(device: InputDevice, channels: int) -> int:
    """Use 48 kHz when the device supports it, else its native rate."""
    sd = _sd()
    candidates = [*PREFERRED_RATES, int(device.default_rate)]
    for rate in dict.fromkeys(candidates):
        try:
            sd.check_input_settings(
                device=device.index, channels=channels, samplerate=rate, dtype="float32"
            )
            return rate
        except Exception:
            continue
    raise DeviceError(f"{device.describe()} rejects every sample rate we tried")


@dataclass(slots=True)
class Recording:
    samples: np.ndarray
    sample_rate: int
    clock: ClockMap
    device: str
    overflows: int = 0
    gaps_ms: float = 0.0
    """Audio lost by the driver, replaced with silence to keep the timeline."""


class Recorder(Protocol):
    """Something that records the room while the test plays."""

    sample_rate: int

    def start(self) -> None: ...

    def stop(self) -> Recording: ...


class SoundDeviceRecorder:
    """Record one channel of an input device, timestamping every block.

    ``now_us`` must be the Sendspin server clock, so that mic samples can be
    lined up with the timestamps the players were given.
    """

    def __init__(
        self,
        device: InputDevice,
        now_us: Callable[[], int],
        *,
        channel: int = 1,
        sample_rate: int | None = None,
        blocksize: int = 0,
    ) -> None:
        if not 1 <= channel <= device.channels:
            raise DeviceError(
                f"{device.describe()} has {device.channels} input channel(s); "
                f"--channel {channel} is out of range"
            )
        self.device = device
        self._now_us = now_us
        self._channel = channel
        self.sample_rate = sample_rate or pick_sample_rate(device, channel)
        self._blocksize = blocksize
        self._us_per_sample = 1e6 / self.sample_rate
        self._blocks: list[np.ndarray] = []
        self._obs_us: list[float] = []
        self._breaks: list[bool] = []
        self._overflows = 0
        self._last_capture_us: float | None = None
        self._last_frames = 0
        self._lock = threading.Lock()
        self._stream: Any = None
        self._adc_usable: bool | None = None

    def _callback(self, indata: np.ndarray, frames: int, t: Any, status: Any) -> None:
        now = self._now_us()
        overflow = bool(status and status.input_overflow)
        # How long ago the first sample of this block hit the ADC. PortAudio
        # gives this on most host APIs; when it doesn't (zero or nonsense),
        # assume the block was handed over the moment its last sample arrived.
        age_s = t.currentTime - t.inputBufferAdcTime
        if self._adc_usable is None:
            self._adc_usable = t.inputBufferAdcTime > 0 and 0 <= age_s < 1.0
        if not self._adc_usable or not 0 <= age_s < 1.0:
            age_s = frames / self.sample_rate
        capture_us = now - age_s * 1e6
        # Some drivers hand over NaN/inf garbage; never let it reach the maths.
        block = np.nan_to_num(indata[:, self._channel - 1], nan=0.0, posinf=0.0, neginf=0.0)
        with self._lock:
            if overflow:
                self._overflows += 1
            self._breaks.append(self._is_dropout(capture_us, frames, overflow))
            self._blocks.append(block)
            self._obs_us.append(capture_us)
            self._last_capture_us = capture_us
            self._last_frames = frames

    def _is_dropout(self, capture_us: float, frames: int, overflow: bool) -> bool:
        """Whether audio was lost just before this block."""
        if self._last_capture_us is None:
            return False
        expected_us = self._last_capture_us + self._last_frames * self._us_per_sample
        gap_us = capture_us - expected_us
        block_us = max(frames, self._last_frames) * self._us_per_sample
        # A flagged overflow is trusted from half a block up. Without a flag,
        # only a jump far beyond normal callback jitter counts as a loss.
        unflagged_limit = max(4 * block_us, 2 * block_us if self._adc_usable else 50_000)
        return (overflow and gap_us > 0.5 * block_us) or gap_us > unflagged_limit

    def start(self) -> None:
        sd = _sd()
        try:
            self._stream = sd.InputStream(
                device=self.device.index,
                channels=self._channel,
                samplerate=self.sample_rate,
                blocksize=self._blocksize,
                dtype="float32",
                # Timing comes from the per-block timestamps, so there's no
                # need for low latency; big buffers make dropouts much rarer.
                latency="high",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as err:
            raise DeviceError(f"could not open {self.device.describe()}: {err}") from err

    def stop(self) -> Recording:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            blocks, obs, breaks = list(self._blocks), list(self._obs_us), list(self._breaks)
        if not blocks:
            raise DeviceError(f"{self.device.describe()} delivered no audio")
        samples, clock, gap_samples = assemble(blocks, obs, breaks, self.sample_rate)
        return Recording(
            samples,
            self.sample_rate,
            clock,
            self.device.describe(),
            self._overflows,
            gap_samples / self.sample_rate * 1000,
        )


def assemble(
    blocks: list[np.ndarray],
    obs_us: list[float],
    breaks: list[bool],
    sample_rate: int,
) -> tuple[np.ndarray, ClockMap, int]:
    """Stitch recorded blocks into one timeline, filling dropouts with silence.

    The blocks between two dropouts form a segment with no missing samples.
    Each segment is placed using all of its own timestamps (the least-delayed
    ones, as in :class:`ClockMap`), so a mistake in one gap's length can't
    shift the segments after it. Without this, every dropout would move the
    rest of the recording earlier, and later chirps would be searched for in
    the wrong place.
    """
    segments: list[list[int]] = [[]]
    for i, brk in enumerate(breaks):
        if brk and segments[-1]:
            segments.append([])
        segments[-1].append(i)

    seg_idx: list[np.ndarray] = []
    seg_obs: list[np.ndarray] = []
    for seg in segments:
        lengths = np.array([len(blocks[i]) for i in seg])
        seg_idx.append(np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.float64))
        seg_obs.append(np.array([obs_us[i] for i in seg], dtype=np.float64))

    slope = _pooled_slope(seg_idx, seg_obs, sample_rate) or 1e6 / sample_rate
    origins = [
        float(np.percentile(o - n * slope, 5.0)) for n, o in zip(seg_idx, seg_obs, strict=True)
    ]
    first = origins[0]
    parts: list[np.ndarray] = []
    starts: list[int] = []
    pos = 0
    gap_total = 0
    for seg, origin in zip(segments, origins, strict=True):
        start = round((origin - first) / slope)
        if start > pos:
            parts.append(np.zeros(start - pos, dtype=np.float32))
            gap_total += start - pos
            pos = start
        # A segment estimated to overlap the previous one is butted up to it.
        starts.append(pos)
        for i in seg:
            parts.append(np.asarray(blocks[i], dtype=np.float32))
            pos += len(blocks[i])
    samples = np.concatenate(parts)

    all_idx = np.concatenate([n + st for n, st in zip(seg_idx, starts, strict=True)])
    all_obs = np.concatenate(seg_obs)
    residual = all_obs - all_idx * slope
    jitter = float(np.percentile(residual, 95) - np.percentile(residual, 5))
    clock = ClockMap(first, slope, sample_rate, jitter, int(all_obs.size))
    return samples, clock, gap_total


def _pooled_slope(
    seg_idx: list[np.ndarray], seg_obs: list[np.ndarray], sample_rate: int
) -> float | None:
    """Microseconds per sample, fitted within segments (each with its own offset)."""
    num = den = span = 0.0
    for n, o in zip(seg_idx, seg_obs, strict=True):
        if n.size < 3:
            continue
        dn = n - n.mean()
        num += float(np.dot(dn, o - o.mean()))
        den += float(np.dot(dn, dn))
        span += float(np.ptp(n))
    if den == 0 or span < sample_rate:  # need at least ~1 s of data
        return None
    slope = num / den
    # Outside a sane range means the timing data is bad; trust nominal.
    return slope if abs(slope * sample_rate / 1e6 - 1.0) * 1e6 <= 2000 else None


@dataclass(frozen=True, slots=True)
class LevelReport:
    rms_dbfs: float
    peak_dbfs: float
    seconds: float


def check_level(device: InputDevice, seconds: float = 2.0, channel: int = 1) -> LevelReport:
    """Record a few seconds and report how loud the input is."""
    rec = SoundDeviceRecorder(device, lambda: time.monotonic_ns() // 1000, channel=channel)
    rec.start()
    time.sleep(seconds)
    recording = rec.stop()
    x = recording.samples.astype(np.float64)
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return LevelReport(_dbfs(rms), _dbfs(peak), x.size / recording.sample_rate)


def _dbfs(v: float) -> float:
    return float(20 * np.log10(max(v, 1e-10)))
