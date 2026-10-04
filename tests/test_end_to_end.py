"""Full pipeline: real server code, simulated players, virtual microphone."""

from __future__ import annotations

import asyncio
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sendspin_syncer.analysis import analyse
from sendspin_syncer.capture import Recording
from sendspin_syncer.discovery import DiscoveredPlayer
from sendspin_syncer.report import build_report
from sendspin_syncer.session import MeasurementSession, load_identity
from sendspin_syncer.signals import ScheduleOptions, build_schedule
from sendspin_syncer.simulate import VirtualRoom, _free_port, start_simulation

SIM = (("Kitchen", 4.0, 0.0), ("Lounge AVR", 92.5, 0.0), ("Bedroom", 31.2, 20.0))


@pytest.fixture
def identity(tmp_path: Path):
    return load_identity(tmp_path / "identity.json")


async def _run(identity, targets_extra=(), simultaneous=False, sim=SIM, room_wrapper=None):
    async with MeasurementSession(identity, connect_timeout_s=8) as session:
        room = VirtualRoom(session.now_us)
        sims = await start_simulation(room, sim)
        try:
            players = await session.connect([s.target for s in sims] + list(targets_extra))
            ready = [p for p in players if p.ready]
            schedule = build_schedule(
                [p.target.client_id_hint for p in ready],
                ScheduleOptions(repeats=3, simultaneous=simultaneous, seed=7),
            )
            recorder = room_wrapper(room) if room_wrapper else room
            start_us, rec = await session.play(schedule, recorder, settle_s=2.0)
        finally:
            await session.release()
            for s in sims:
                await s.stop()
    results = analyse(rec.samples, rec.sample_rate, rec.clock, schedule, start_us)
    return players, build_report(players, results, schedule, rec)


def _expected_relative(sim):
    eff = {name: lat - delay for name, lat, delay in sim}
    base = min(eff.values())
    return {k: v - base for k, v in eff.items()}


@pytest.mark.parametrize("simultaneous", [False, True])
async def test_measures_simulated_players(identity, simultaneous):
    players, report = await _run(identity, simultaneous=simultaneous)
    assert all(p.ready for p in players)
    by_name = {r.name: r for r in report.rows}
    for name, want in _expected_relative(SIM).items():
        row = by_name[name]
        assert row.status == "ok"
        assert row.relative_ms == pytest.approx(want, abs=0.4), name
    # Player state is read, never written.
    assert by_name["Bedroom"].output_delay_ms == 20
    assert by_name["Bedroom"].security == "encrypted, unpaired"
    # Late players get more output delay; the earliest keeps its own.
    assert by_name["Kitchen"].suggested_output_delay_ms == 0
    assert by_name["Lounge AVR"].suggested_output_delay_ms == pytest.approx(88.5, abs=1)
    assert by_name["Bedroom"].suggested_output_delay_ms == pytest.approx(27.2, abs=1)


async def test_unreachable_player_is_reported_not_fatal(identity):
    dead = DiscoveredPlayer.from_url(f"ws://127.0.0.1:{_free_port()}/sendspin")
    players, report = await _run(identity, targets_extra=[dead], sim=SIM[:1])
    failed = [p for p in players if not p.ready]
    assert len(failed) == 1
    assert "could not connect" in failed[0].detail
    assert {r.name: r.status for r in report.rows}["Kitchen"] == "ok"


# ------------------------------------------------------------- legacy players

LEGACY_PYTHON = os.environ.get("SENDSPIN_LEGACY_PYTHON")


class _MixFileIntoRoom:
    """Recorder that adds a legacy player's played audio (from its file) at stop."""

    def __init__(self, room: VirtualRoom, path: Path) -> None:
        self._room = room
        self._path = path
        self.sample_rate = room.sample_rate

    def start(self) -> None:
        self._room.start()

    def stop(self) -> Recording:
        data = self._path.read_bytes()
        i = 0
        while i + 12 <= len(data):
            t, n = struct.unpack_from("<qi", data, i)
            i += 12
            pcm = np.frombuffer(data, dtype="<f4", count=n, offset=i)
            i += 4 * n
            self._room.add(t, pcm, 0.3)
        return self._room.stop()


@pytest.mark.skipif(
    not LEGACY_PYTHON, reason="set SENDSPIN_LEGACY_PYTHON to a python with aiosendspin~=6.0"
)
async def test_legacy_unencrypted_player(identity, tmp_path):
    port = _free_port()
    out = tmp_path / "legacy.bin"
    proc = subprocess.Popen(
        [
            LEGACY_PYTHON,
            str(Path(__file__).with_name("legacy_player.py")),
            str(port),
            "Old Pi",
            "45.0",
            "10",
            str(out),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        line = await asyncio.wait_for(asyncio.to_thread(proc.stdout.readline), 30)
        assert line.strip() == "READY"
        legacy = DiscoveredPlayer.from_url(f"ws://127.0.0.1:{port}/sendspin")
        _players, report = await _run(
            identity,
            targets_extra=[legacy],
            sim=SIM[:1],
            room_wrapper=lambda room: _MixFileIntoRoom(room, out),
        )
    finally:
        proc.terminate()
        proc.wait(10)
    by_name = {r.name: r for r in report.rows}
    old = by_name["Old Pi"]
    assert old.status == "ok", old.detail
    assert old.security == "unencrypted (legacy)"
    assert old.output_delay_ms == 10
    # Kitchen 4.0 ms; Old Pi 45 - 10 = 35 ms -> 31 ms later.
    assert old.relative_ms == pytest.approx(31.0, abs=0.5)


# ------------------------------------------------------- Music Assistant hand-off


async def _wait_ready(server, url, timeout=10.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        cid = server.get_client_id_for_url(url)
        client = server.get_client(cid) if cid else None
        if client is not None and client.is_connected and client.roles_by_family("player"):
            return client
        await asyncio.sleep(0.1)
    return None


async def test_takes_player_from_other_server_and_hands_it_back(identity):
    """Another server ("Music Assistant") holds the player before and after the test."""
    from aiosendspin.models.types import ConnectionReason
    from aiosendspin.noise.keys import Identity
    from aiosendspin.noise.trust_store import InMemoryServerPairingStore, TrustedUnpairedClient
    from aiosendspin.server import SendspinServer

    loop = asyncio.get_running_loop()
    ma = SendspinServer(
        loop, Identity.generate(), "Music Assistant", pairing_store=InMemoryServerPairingStore()
    )
    room = VirtualRoom(lambda: 0)
    (sim,) = await start_simulation(room, SIM[:1])
    url = sim.target.url
    try:
        await ma.pairing_store.add_trusted_unpaired(TrustedUnpairedClient(sim.target.instance))
        ma.connect_to_client(url, connection_reason=ConnectionReason.PLAYBACK)
        ma_client = await _wait_ready(ma, url)
        assert ma_client is not None, "stand-in MA could not connect"

        async with MeasurementSession(identity, connect_timeout_s=8) as session:
            room._now_us = session.now_us
            players = await session.connect([sim.target])
            assert players[0].ready, players[0].detail
            await asyncio.sleep(0.5)
            assert not ma_client.is_connected  # displaced with 'another_server'
            schedule = build_schedule([sim.target.instance], ScheduleOptions(repeats=1, seed=1))
            await session.play(schedule, room, settle_s=1.0)
        # Released: MA reclaims the player the way it does when you press play.
        assert ma.reclaim_client_for_playback(ma_client.client_id)
        assert await _wait_ready(ma, url) is not None, "MA could not take the player back"
    finally:
        await ma.close()
        await sim.stop()
