"""Connect to players, play the test schedule, and hand the players back.

This is the only module that talks to players. It is deliberately read-only
towards their settings: it never sends volume, mute or output-delay commands.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from aiosendspin.models.types import ConnectionReason, GoodbyeReason
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import (
    InMemoryServerPairingStore,
    PskCategory,
    TrustedUnpairedClient,
)
from aiosendspin.server import (
    ClientDisconnectedEvent,
    SendspinClient,
    SendspinEvent,
    SendspinGroup,
    SendspinServer,
)
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.push_stream import PushStream

from .capture import Recorder, Recording
from .discovery import DiscoveredPlayer
from .signals import Schedule, to_pcm16_stereo

logger = logging.getLogger(__name__)

SERVER_NAME = "Sendspin Syncer"
CHUNK_MS = 100
# Audio is pushed at most this far ahead of real time, so players with small
# buffers aren't flooded.
MAX_SEND_AHEAD_US = 1_500_000
# Extra margin on top of the slowest player's required lead time.
START_MARGIN_US = 700_000
CHANNEL_NS = uuid.UUID("0b6f5c3e-7d1c-4a0e-9a3f-5e3c2f1d9b11")


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "sendspin-syncer"


def load_identity(path: Path | None = None) -> Identity:
    """Load the tool's long-term server identity, creating it on first use.

    A stable identity means players see one consistent server across runs
    instead of a new stranger each time.
    """
    path = path or config_dir() / "identity.json"
    if path.exists():
        data = json.loads(path.read_text())
        return Identity.from_private_bytes(_b64u_decode(data["private_key"]))
    identity = Identity.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"private_key": identity.private_b64u}))
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return identity


def _b64u_decode(value: str) -> bytes:
    from aiosendspin.noise.keys import b64url_decode

    return b64url_decode(value)


@dataclass(slots=True)
class PlayerInfo:
    """What we learned about a player, plus how its connection went."""

    target: DiscoveredPlayer
    client_id: str | None = None
    name: str | None = None
    product: str | None = None
    security: str | None = None
    output_delay_ms: int | None = None
    output_delay_settable: bool = False
    required_lead_time_ms: int | None = None
    volume: int | None = None
    muted: bool | None = None
    audio_format: str | None = None
    status: str = "pending"
    detail: str = ""
    channel: uuid.UUID = field(default_factory=uuid.uuid4)

    @property
    def label(self) -> str:
        return self.name or self.target.name

    @property
    def ready(self) -> bool:
        return self.status == "ready"


class MeasurementSession:
    """A temporary Sendspin server that holds the players for one test run."""

    def __init__(
        self,
        identity: Identity | None = None,
        *,
        connect_timeout_s: float = 15.0,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._identity = identity or load_identity()
        self._connect_timeout_s = connect_timeout_s
        self._loop = loop
        self._server: SendspinServer | None = None
        self._players: list[PlayerInfo] = []
        self._group: SendspinGroup | None = None
        self._stream: PushStream | None = None
        self._goodbyes: dict[str, GoodbyeReason | None] = {}
        self._playing = False
        self._unsub: Callable[[], None] | None = None

    @property
    def server(self) -> SendspinServer:
        assert self._server is not None, "session not started"
        return self._server

    @property
    def now_us(self) -> Callable[[], int]:
        return self.server.clock.now_us

    async def __aenter__(self) -> MeasurementSession:
        loop = self._loop or asyncio.get_running_loop()
        # No listener and no mDNS advert: we only dial out to the players we
        # chose, and players can't find us and connect on their own.
        self._server = SendspinServer(
            loop,
            self._identity,
            SERVER_NAME,
            None,
            pairing_store=InMemoryServerPairingStore(),
            allow_unencrypted=True,
            allow_noncompliant_clients=True,
        )
        self._unsub = self._server.add_event_listener(self._on_event)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.release()

    def _on_event(self, _server: SendspinServer, event: SendspinEvent) -> None:
        if isinstance(event, ClientDisconnectedEvent):
            self._goodbyes[event.client_id] = event.goodbye_reason
            if self._playing:
                for p in self._players:
                    if p.client_id == event.client_id and p.status == "ready":
                        p.status = "interrupted"
                        reason = event.goodbye_reason
                        p.detail = (
                            "taken back by another server (e.g. Music Assistant)"
                            if reason is GoodbyeReason.ANOTHER_SERVER
                            else f"disconnected ({reason.value if reason else 'connection lost'})"
                        )

    # ------------------------------------------------------------------ connect

    async def connect(self, targets: list[DiscoveredPlayer]) -> list[PlayerInfo]:
        """Connect to every target for playback. Failures are recorded, not raised."""
        server = self.server
        self._players = [PlayerInfo(target=t) for t in targets]
        for p in self._players:
            if p.target.looks_encrypted:
                # Approve unpaired playback up front so the very first
                # activation already asks for playback.
                await server.pairing_store.add_trusted_unpaired(
                    TrustedUnpairedClient(client_id=p.target.client_id_hint)
                )
        await asyncio.gather(*(self._await_ready(p) for p in self._players))
        return self._players

    async def _await_ready(self, p: PlayerInfo) -> None:
        server = self.server
        deadline = asyncio.get_running_loop().time() + self._connect_timeout_s
        try:
            await asyncio.wait_for(
                server.connect_to_client_and_wait(
                    p.target.url, connection_reason=ConnectionReason.PLAYBACK
                ),
                self._connect_timeout_s,
            )
        except Exception as err:
            p.status = "failed"
            p.detail = f"could not connect ({_describe_error(err)})"
            server.disconnect_from_client(p.target.url)
            return
        trusted_late = False
        while asyncio.get_running_loop().time() < deadline:
            cid = server.get_client_id_for_url(p.target.url)
            client = server.get_client(cid) if cid else None
            if client is not None and client.is_connected:
                if client.roles_by_family("player"):
                    self._fill_info(p, client)
                    p.status = "ready"
                    return
                sec = client.connection_security
                if (
                    sec is not None
                    and sec.psk_category is PskCategory.SENTINEL
                    and not trusted_late
                ):
                    # Its mDNS name wasn't its client id; approve it now. The
                    # player retries the connection with playback allowed.
                    trusted_late = True
                    await server.trust_unpaired(client.client_id)
            await asyncio.sleep(0.1)

        p.status = "failed"
        cid = server.get_client_id_for_url(p.target.url)
        client = server.get_client(cid) if cid else None
        goodbye = self._goodbyes.get(cid) if cid else None
        if client is None and cid is None:
            p.detail = "could not connect (offline, firewalled, or wrong address)"
        elif goodbye is GoodbyeReason.CONCURRENT_ATTEMPT:
            p.detail = "player refused: busy with another server"
        elif goodbye in (GoodbyeReason.PAIRING_REQUIRED, GoodbyeReason.UNPAIRED):
            p.detail = "player requires pairing; unpaired access is disabled on it"
        elif client is not None and client.is_connected:
            p.detail = "connected, but the player role was not activated"
        else:
            p.detail = "connection did not complete"
        if cid:
            p.client_id = cid
        server.disconnect_from_client(p.target.url)

    @staticmethod
    def _fill_info(p: PlayerInfo, client: SendspinClient) -> None:
        p.client_id = client.client_id
        p.name = client.name
        info = client.info_or_none
        if info is not None and info.device_info is not None:
            parts = [info.device_info.manufacturer, info.device_info.product_name]
            p.product = " ".join(x for x in parts if x) or None
        sec = client.connection_security
        if sec is None:
            p.security = "unencrypted (legacy)"
        elif sec.psk_category is PskCategory.LONG_TERM:
            p.security = "paired"
        else:
            p.security = "encrypted, unpaired"
        role = client.roles_by_family("player")[0]
        p.output_delay_ms = _safe(lambda: int(role.static_delay_ms))  # type: ignore[attr-defined]
        p.required_lead_time_ms = _safe(lambda: int(role.required_lead_time_ms))  # type: ignore[attr-defined]
        p.volume = _safe(lambda: role.volume)  # type: ignore[attr-defined]
        p.muted = _safe(lambda: role.muted)  # type: ignore[attr-defined]
        cmds = _safe(lambda: role.state_supported_commands) or []  # type: ignore[attr-defined]
        p.output_delay_settable = any("delay" in str(getattr(c, "value", c)) for c in cmds)

    # --------------------------------------------------------------------- play

    async def play(
        self,
        schedule: Schedule,
        recorder: Recorder,
        *,
        progress: Callable[[float], None] | None = None,
        settle_s: float = 3.0,
    ) -> tuple[int, Recording]:
        """Play the schedule on all ready players while recording.

        ``settle_s`` is a pause before playing, so the players' clock
        synchronisation has converged.

        Returns the server-clock timestamp of the start of the test track and
        the recording.
        """
        ready = [p for p in self._players if p.ready]
        if not ready:
            raise RuntimeError("no players are ready")
        clients = [self.server.get_client(p.client_id or "") for p in ready]
        if any(c is None for c in clients):
            raise RuntimeError("a player vanished before playback")

        # One temporary group (in this tool only) holds every player, and each
        # player gets its own channel with its own test track.
        group = clients[0].group  # type: ignore[union-attr]
        for c in clients[1:]:
            if c.group is not group:  # type: ignore[union-attr]
                await group.add_client(c)  # type: ignore[arg-type]
        self._group = group
        channels = {p.client_id: uuid.uuid5(CHANNEL_NS, p.client_id or "") for p in ready}
        for p in ready:
            p.channel = channels[p.client_id]

        def resolver(player_id: str) -> uuid.UUID:
            return channels.get(player_id, uuid.uuid5(CHANNEL_NS, "unused"))

        # Give every player's clock sync time to converge after connecting.
        await asyncio.sleep(settle_s)
        stream = group.start_stream(channel_resolver=resolver)
        self._stream = stream

        fmt = AudioFormat(sample_rate=schedule.sample_rate, bit_depth=16, channels=2)
        tracks = {p.client_id: schedule.render_player(p.target.client_id_hint) for p in ready}
        chunk = schedule.sample_rate * CHUNK_MS // 1000
        n_chunks = int(np.ceil(len(next(iter(tracks.values()))) / chunk))

        lead_us = max((p.required_lead_time_ms or 250) for p in ready) * 1000 + START_MARGIN_US
        recorder.start()
        self._playing = True
        try:
            start_us = self.server.clock.now_us() + lead_us
            for i in range(n_chunks):
                if not any(p.ready for p in ready):
                    break
                for cid, track in tracks.items():
                    part = track[i * chunk : (i + 1) * chunk]
                    if len(part) < chunk:
                        part = np.pad(part, (0, chunk - len(part)))
                    stream.prepare_audio(to_pcm16_stereo(part), fmt, channel_id=channels[cid])
                await stream.commit_audio(play_start_us=start_us + i * CHUNK_MS * 1000)
                await stream.sleep_to_limit_buffer(MAX_SEND_AHEAD_US)
                if progress:
                    progress((i + 1) / n_chunks * 0.8)
            for p in ready:
                p.audio_format = _describe_format(self.server.get_client(p.client_id or ""))
            # Keep recording until the last sound (and its echo) has arrived.
            end_us = start_us + schedule.duration_us + 1_200_000
            while (remaining := end_us - self.server.clock.now_us()) > 0:
                await asyncio.sleep(min(remaining / 1e6, 0.1))
                if progress:
                    progress(1 - 0.2 * remaining / (end_us - start_us + 1))
        finally:
            self._playing = False
            recording = recorder.stop()
            with contextlib.suppress(Exception):
                stream.stop()
            self._stream = None
        if progress:
            progress(1.0)
        return start_us, recording

    # ------------------------------------------------------------------ release

    async def release(self) -> None:
        """Stop playing and disconnect, so Music Assistant can take the players back."""
        if self._server is None:
            return
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
            self._stream = None
        for p in self._players:
            self._server.disconnect_from_client(p.target.url)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._server.close(), timeout=10)
        if self._unsub:
            self._unsub()
        self._server = None


def _describe_error(err: BaseException) -> str:
    if isinstance(err, TimeoutError):
        return "timed out"
    text = str(err) or type(err).__name__
    return text.split(" [")[0]


def _safe[T](fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except Exception:
        return None


def _describe_format(client: SendspinClient | None) -> str | None:
    if client is None:
        return None
    roles = client.roles_by_family("player")
    if not roles:
        return None
    role = roles[0]
    codec = _safe(lambda: role.preferred_codec)  # type: ignore[attr-defined]
    fmt = _safe(lambda: role.preferred_format)  # type: ignore[attr-defined]
    parts = []
    if codec is not None:
        parts.append(str(getattr(codec, "value", codec)))
    if fmt is not None:
        parts.append(f"{fmt.sample_rate / 1000:g}kHz/{fmt.bit_depth}bit")
    return " ".join(parts) or None
