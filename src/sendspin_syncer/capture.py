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
        blocksize: int = 480,
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
        self._blocks: list[np.ndarray] = []
        self._obs_index: list[int] = []
        self._obs_us: list[float] = []
        self._frames = 0
        self._overflows = 0
        self._lock = threading.Lock()
        self._stream: Any = None
        self._adc_usable: bool | None = None

    def _callback(self, indata: np.ndarray, frames: int, t: Any, status: Any) -> None:
        now = self._now_us()
        if status and status.input_overflow:
            self._overflows += 1
        # How long ago the first sample of this block hit the ADC. PortAudio
        # gives this on most host APIs; when it doesn't (zero or nonsense),
        # assume the block was handed over the moment its last sample arrived.
        age_s = t.currentTime - t.inputBufferAdcTime
        if self._adc_usable is None:
            self._adc_usable = t.inputBufferAdcTime > 0 and 0 <= age_s < 1.0
        if not self._adc_usable or not 0 <= age_s < 1.0:
            age_s = frames / self.sample_rate
        with self._lock:
            # Some drivers hand over NaN/inf garbage; never let it reach the maths.
            self._blocks.append(
                np.nan_to_num(indata[:, self._channel - 1], nan=0.0, posinf=0.0, neginf=0.0)
            )
            self._obs_index.append(self._frames)
            self._obs_us.append(now - age_s * 1e6)
            self._frames += frames

    def start(self) -> None:
        sd = _sd()
        try:
            self._stream = sd.InputStream(
                device=self.device.index,
                channels=self._channel,
                samplerate=self.sample_rate,
                blocksize=self._blocksize,
                dtype="float32",
                latency="low",
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
            samples = (
                np.concatenate(self._blocks) if self._blocks else np.zeros(0, dtype=np.float32)
            )
            idx = np.array(self._obs_index)
            obs = np.array(self._obs_us)
        if idx.size == 0:
            raise DeviceError(f"{self.device.describe()} delivered no audio")
        clock = ClockMap.fit(idx, obs, self.sample_rate)
        return Recording(samples, self.sample_rate, clock, self.device.describe(), self._overflows)


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
