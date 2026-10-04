"""A pre-1.0 (unencrypted) Sendspin player, for interoperability tests.

Run it with an interpreter that has ``aiosendspin~=6.0`` installed (the
protocol generation that predates encryption). It listens on localhost and
appends everything it "plays" to ``out`` as records of
``<int64 play_time_us><int32 n><n float32 samples>``. Both library versions
time-stamp with CLOCK_MONOTONIC_RAW, so the times are on the server's clock.

    python legacy_player.py PORT NAME LATENCY_MS OUTPUT_DELAY_MS OUT_FILE
"""

import asyncio
import signal
import struct
import sys

import numpy as np
from aiohttp import web
from aiosendspin.client import ClientListener, SendspinClient
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles


async def main() -> None:
    port, name, latency_ms, delay_ms, out = (
        int(sys.argv[1]),
        sys.argv[2],
        float(sys.argv[3]),
        float(sys.argv[4]),
        sys.argv[5],
    )
    client_id = f"legacy-{name.lower().replace(' ', '-')}"
    fh = open(out, "wb")  # noqa: SIM115

    async def on_connection(ws: web.WebSocketResponse) -> None:
        client = SendspinClient(
            client_id,
            name,
            [Roles.PLAYER],
            player_support=ClientHelloPlayerSupport(
                supported_formats=[
                    SupportedAudioFormat(
                        codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16
                    )
                ],
                buffer_capacity=2_000_000,
                supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
            ),
            static_delay_ms=delay_ms,
        )

        def on_chunk(server_ts: int, data: bytes, fmt: object) -> None:
            pcm = (np.frombuffer(data, dtype="<i2").reshape(-1, 2)[:, 0] / 32768.0).astype("<f4")
            t = client.compute_play_time(server_ts) + round(latency_ms * 1000)
            fh.write(struct.pack("<qi", t, len(pcm)) + pcm.tobytes())
            fh.flush()

        client.add_audio_chunk_listener(on_chunk)
        done = asyncio.Event()
        client.add_disconnect_listener(done.set)
        await client.attach_websocket(ws)
        await done.wait()

    listener = ClientListener(
        client_id,
        on_connection,
        port=port,
        host="127.0.0.1",
        advertise_mdns=False,
        client_name=name,
    )
    await listener.start()
    print("READY", flush=True)
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    await stop.wait()
    await listener.stop()


if __name__ == "__main__":
    asyncio.run(main())
