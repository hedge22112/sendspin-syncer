"""Simulated players and a virtual room, for self-tests without hardware.

Each simulated player is a real Sendspin client (from ``aiosendspin``)
listening on localhost. It "plays" audio by mixing it into a shared virtual
room at the time the protocol says, plus a deliberate extra latency. A
virtual microphone records the room. If the whole pipeline is correct, the
measured delays match the latencies that were configured.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from aiohttp import web
from aiosendspin.client import ClientListener, SendspinClient
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryClientPairingStore

from .capture import Recording
from .clock_map import ClockMap
from .discovery import DiscoveredPlayer


class VirtualRoom:
    """A mono mixing buffer on the server clock, recorded by a virtual mic."""

    def __init__(self, now_us: Callable[[], int], sample_rate: int = 48_000, noise: float = 0.002):
        self.sample_rate = sample_rate
        self._now_us = now_us
        self._noise = noise
        self._origin_us: int | None = None
        self._buf = np.zeros(sample_rate * 120, dtype=np.float64)
        self._lock = threading.Lock()

    def add(self, play_at_us: int, samples: np.ndarray, gain: float = 0.3) -> None:
        if self._origin_us is None:
            return
        pos = (play_at_us - self._origin_us) * self.sample_rate / 1e6
        if pos < 0:
            return
        i = int(np.floor(pos))
        frac = pos - i
        x = np.asarray(samples, dtype=np.float64) * gain
        shifted = np.interp(np.arange(len(x) + 1) - frac, np.arange(len(x)), x, left=0, right=0)
        with self._lock:
            end = min(len(self._buf), i + len(shifted))
            if i < end:
                self._buf[i:end] += shifted[: end - i]

    # Recorder protocol
    def start(self) -> None:
        self._origin_us = self._now_us()

    def stop(self) -> Recording:
        assert self._origin_us is not None, "room was never started"
        n = int((self._now_us() - self._origin_us) * self.sample_rate / 1e6)
        rng = np.random.default_rng(0)
        with self._lock:
            rec = self._buf[:n] + self._noise * rng.standard_normal(n)
        return Recording(
            rec.astype(np.float32),
            self.sample_rate,
            ClockMap.nominal(self._origin_us, self.sample_rate),
            "virtual microphone",
        )


@dataclass(slots=True)
class SimulatedPlayer:
    name: str
    latency_ms: float
    output_delay_ms: float = 0.0
    port: int = 0
    gain: float = 0.3
    identity: Identity = field(default_factory=Identity.generate)
    _listener: ClientListener | None = None
    _client: SendspinClient | None = None

    @property
    def target(self) -> DiscoveredPlayer:
        return DiscoveredPlayer(
            instance=self.identity.peer_id,
            name=self.name,
            host="127.0.0.1",
            port=self.port,
            path="/sendspin",
        )

    async def start(self, room: VirtualRoom) -> None:
        if not self.port:
            self.port = _free_port()
        pairing_store = InMemoryClientPairingStore()
        config = await pairing_store.get_pairing_config()
        await pairing_store.store_pairing_config(replace(config, unpaired_access_enabled=True))

        # One client object serves every incoming connection, like a real player,
        # so competing servers are arbitrated by the protocol's admission rules.
        client = SendspinClient(
            self.identity,
            self.name,
            [Roles.PLAYER],
            pairing_store=pairing_store,
            player_support=ClientHelloPlayerSupport(
                supported_formats=[
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16
                    )
                ],
                buffer_capacity=2_000_000,
                supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
            ),
            static_delay_ms=self.output_delay_ms,
        )
        latency_us = round(self.latency_ms * 1000)

        def on_chunk(server_ts: int, data: bytes, fmt: object) -> None:
            pcm = np.frombuffer(data, dtype="<i2").reshape(-1, 2)[:, 0] / 32768.0
            room.add(client.compute_play_time(server_ts) + latency_us, pcm, self.gain)

        client.add_audio_chunk_listener(on_chunk)
        self._client = client

        async def on_connection(ws: web.WebSocketResponse) -> None:
            await client.attach_websocket(ws)

        self._listener = ClientListener(
            self.identity.peer_id,
            on_connection,
            port=self.port,
            host="127.0.0.1",
            advertise_mdns=False,
            client_name=self.name,
        )
        await self._listener.start()

    async def stop(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.disconnect()
        if self._listener is not None:
            await self._listener.stop()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


DEFAULT_SIMULATION = (
    ("Kitchen (simulated)", 4.0, 0.0),
    ("Living room AVR (simulated)", 92.5, 0.0),
    ("Bedroom (simulated)", 31.2, 20.0),
)


async def start_simulation(
    room: VirtualRoom,
    players: tuple[tuple[str, float, float], ...] = DEFAULT_SIMULATION,
) -> list[SimulatedPlayer]:
    sims = [SimulatedPlayer(name, lat, delay) for name, lat, delay in players]
    await asyncio.gather(*(s.start(room) for s in sims))
    return sims
