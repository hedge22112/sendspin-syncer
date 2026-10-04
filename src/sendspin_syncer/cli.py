"""Command-line interface: ``devices``, ``list``, ``measure`` and ``selftest``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from . import __version__
from .analysis import AnalysisOptions, analyse
from .capture import (
    DeviceError,
    SoundDeviceRecorder,
    check_level,
    list_input_devices,
    pick_sample_rate,
    resolve_device,
)
from .discovery import DiscoveredPlayer, discover_players, select_players
from .report import build_report, print_report, write_csv, write_json
from .session import MeasurementSession, PlayerInfo
from .signals import ScheduleOptions, build_schedule, top_frequency_for

console = Console()
err = Console(stderr=True)

QUIET_MIC_DBFS = -55.0


# ------------------------------------------------------------------ devices


def cmd_devices(args: argparse.Namespace) -> int:
    devices = list_input_devices()
    if args.check is not None:
        dev = resolve_device(args.check, devices)
        console.print(f"Recording {args.seconds:g} s from {dev.describe()} … make some noise.")
        level = check_level(dev, args.seconds, args.channel)
        bar_len = max(0, min(40, round((level.peak_dbfs + 80) / 2)))
        console.print(
            f"Peak {level.peak_dbfs:6.1f} dBFS  RMS {level.rms_dbfs:6.1f} dBFS  "
            f"[green]{'█' * bar_len}[/][dim]{'░' * (40 - bar_len)}[/]"
        )
        if level.peak_dbfs < QUIET_MIC_DBFS:
            console.print(
                "[yellow]That is very quiet. Is this the right device, and is it unmuted?[/]"
            )
        return 0

    if args.json:
        print(json.dumps([asdict(d) for d in devices], indent=2))
        return 0
    if not devices:
        console.print("[yellow]No audio input devices found.[/]")
        return 1
    table = Table(title="Audio input devices")
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("Audio system")
    table.add_column("Inputs", justify="right")
    table.add_column("Default rate", justify="right")
    table.add_column("")
    for d in devices:
        table.add_row(
            str(d.index),
            d.name,
            d.host_api,
            str(d.channels),
            f"{d.default_rate:g} Hz",
            "[green]default[/]" if d.is_default else "",
        )
    console.print(table)
    console.print(
        "[dim]Use one with `sendspin-syncer measure --device <# or part of the name>`. "
        "Check it picks up sound with `sendspin-syncer devices --check <#>`.[/]"
    )
    return 0


# --------------------------------------------------------------------- list


def cmd_list(args: argparse.Namespace) -> int:
    players = asyncio.run(discover_players(args.timeout))
    if args.json:
        print(
            json.dumps(
                [
                    {"name": p.name, "id": p.instance, "url": p.url, "properties": p.properties}
                    for p in players
                ],
                indent=2,
            )
        )
        return 0
    if not players:
        console.print(
            "[yellow]No Sendspin players found.[/] They advertise as _sendspin._tcp over mDNS; "
            "make sure this machine is on the same network/VLAN. You can also pass players "
            "directly to `measure` with --url ws://HOST:PORT/sendspin."
        )
        return 1
    table = Table(title=f"Sendspin players ({len(players)})")
    table.add_column("Name", style="bold")
    table.add_column("Address")
    table.add_column("Protocol")
    table.add_column("Id", overflow="fold")
    for p in players:
        table.add_row(
            p.name,
            p.url,
            "1.0 (encrypted)" if p.looks_encrypted else "legacy",
            p.instance,
        )
    console.print(table)
    console.print(
        "[dim]Listing only listens for adverts; it doesn't connect, so Music Assistant is not "
        "disturbed.[/]"
    )
    return 0


# ------------------------------------------------------------------ measure


def _parse_distances(values: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for v in values or []:
        name, sep, metres = v.rpartition("=")
        if not sep or not name:
            raise SystemExit(f"--distance expects NAME=METRES, got {v!r}")
        try:
            out[name] = float(metres.lower().removesuffix("m"))
        except ValueError:
            raise SystemExit(f"--distance: {metres!r} is not a number of metres") from None
    return out


def _confirm(players: list[DiscoveredPlayer], assume_yes: bool) -> bool:
    console.print(f"\n[bold]Players to measure ({len(players)}):[/]")
    for p in players:
        console.print(f"  • {p.name}  [dim]{p.url}[/]")
    console.print(
        "\n[yellow]Heads up:[/] while the test runs (about "
        "half a minute) this tool takes the players over from Music Assistant. Anything they are "
        "playing stops. Afterwards they are released; Music Assistant picks them up again "
        "the next time you press play there.\n"
        "Nothing on the players is changed (no volume, mute or delay settings). The test "
        "plays at their current volume, so set a comfortable, audible level first."
    )
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        err.print("Not a terminal; pass --yes to confirm.")
        return False
    answer = console.input("Continue? [y/N] ").strip().lower()
    return answer in ("y", "yes")


async def _measure(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    console.print(f"Microphone: [bold]{device.describe()}[/]")

    targets: list[DiscoveredPlayer] = [DiscoveredPlayer.from_url(u) for u in args.url or []]
    if args.player or not targets:
        with console.status(f"Looking for players ({args.discover_time:g} s)…"):
            found = await discover_players(args.discover_time)
        try:
            chosen = select_players(found, args.player)
        except LookupError as e:
            err.print(f"[red]{e}[/]. Run `sendspin-syncer list` to see what was found.")
            return 2
        targets += [p for p in chosen if p.url not in {t.url for t in targets}]
    if not targets:
        err.print("[red]No players to measure.[/] Try `sendspin-syncer list`, or pass --url.")
        return 1
    if not _confirm(targets, args.yes):
        console.print("Cancelled.")
        return 1

    # Open the mic settings before taking any player over, so a bad device
    # fails early. A low-rate mic can't hear high frequencies, so the chirps
    # are kept within what it can record.
    recorder_rate = pick_sample_rate(device, args.channel)
    f_hi = top_frequency_for(recorder_rate)
    if f_hi < 12_000:
        console.print(
            f"[yellow]Note:[/] the microphone records at {recorder_rate} Hz, so the test sweeps "
            f"stop at {f_hi / 1000:.1f} kHz. That's fine, just slightly less precise."
        )
    sched_opts = ScheduleOptions(
        f_hi=f_hi,
        repeats=args.repeats,
        simultaneous=args.simultaneous,
        max_early_s=args.max_early_ms / 1000,
        max_late_s=args.max_delay_ms / 1000,
    )
    analysis_opts = AnalysisOptions(
        max_early_s=sched_opts.max_early_s,
        max_late_s=sched_opts.max_late_s,
        min_snr_db=args.min_snr,
    )

    async with MeasurementSession(connect_timeout_s=args.connect_timeout) as session:
        with console.status("Connecting to players…"):
            players = await session.connect(targets)
        _print_connections(players)
        ready = [p for p in players if p.ready]
        if not ready:
            err.print("[red]Could not use any player.[/]")
            return 1
        schedule = build_schedule([p.target.client_id_hint for p in ready], sched_opts)
        recorder = SoundDeviceRecorder(
            device, session.now_us, channel=args.channel, sample_rate=recorder_rate
        )
        with Progress(
            TextColumn("Playing test signals"), BarColumn(), console=console, transient=True
        ) as prog:
            task = prog.add_task("play", total=1.0)
            start_us, recording = await session.play(
                schedule, recorder, progress=lambda f: prog.update(task, completed=f)
            )
        await session.release()
    console.print("Players released.")

    if args.save_recording:
        from scipy.io import wavfile

        wavfile.write(args.save_recording, recording.sample_rate, recording.samples)
        console.print(f"Recording saved to {args.save_recording}")

    peak = float(abs(recording.samples).max()) if recording.samples.size else 0.0
    if peak < 10 ** (QUIET_MIC_DBFS / 20):
        err.print(
            "[yellow]The microphone recorded almost nothing. Check --device, and that the mic "
            "isn't muted.[/]"
        )

    results = analyse(
        recording.samples,
        recording.sample_rate,
        recording.clock,
        schedule,
        start_us,
        analysis_opts,
    )
    report = build_report(
        players,
        results,
        schedule,
        recording,
        distances_m=_parse_distances(args.distance),
        reference=args.reference,
    )
    console.print()
    print_report(report, console, hints=not args.no_hints, details=args.verbose > 0)
    if args.json:
        write_json(report, Path(args.json))
        console.print(f"JSON report written to {args.json}")
    if args.csv:
        write_csv(report, Path(args.csv))
        console.print(f"CSV report written to {args.csv}")
    return 0


def _print_connections(players: list[PlayerInfo]) -> None:
    for p in players:
        if p.ready:
            extra = ", ".join(
                x
                for x in (
                    p.security,
                    f"output delay {p.output_delay_ms} ms" if p.output_delay_ms is not None else "",
                    f"volume {p.volume}%" if p.volume is not None else "",
                    "[red]muted![/]" if p.muted else "",
                )
                if x
            )
            console.print(f"  [green]✓[/] {p.label} [dim]({extra})[/]")
        else:
            console.print(f"  [red]✗[/] {p.target.name}: {p.detail}")


def cmd_measure(args: argparse.Namespace) -> int:
    return asyncio.run(_measure(args))


# ----------------------------------------------------------------- selftest


async def _selftest(args: argparse.Namespace) -> int:
    from .simulate import DEFAULT_SIMULATION, VirtualRoom, start_simulation

    console.print(
        "Self-test: simulated players with known latencies, a virtual room and a virtual "
        "microphone. No network or audio hardware is used."
    )
    async with MeasurementSession(connect_timeout_s=10) as session:
        room = VirtualRoom(session.now_us)
        sims = await start_simulation(room)
        try:
            players = await session.connect([s.target for s in sims])
            _print_connections(players)
            ready = [p for p in players if p.ready]
            schedule = build_schedule(
                [p.target.client_id_hint for p in ready],
                ScheduleOptions(repeats=args.repeats, simultaneous=args.simultaneous),
            )
            with console.status("Playing into the virtual room…"):
                start_us, recording = await session.play(schedule, room)
        finally:
            await session.release()
            for s in sims:
                await s.stop()
    results = analyse(recording.samples, recording.sample_rate, recording.clock, schedule, start_us)
    report = build_report(players, results, schedule, recording)
    print_report(report, console)

    expected = {name: lat - delay for name, lat, delay in DEFAULT_SIMULATION}
    base = min(expected.values())
    ok = True
    for row in report.rows:
        want = expected[row.name] - base
        got = row.relative_ms
        good = got is not None and abs(got - want) < 0.5
        ok &= good
        console.print(
            f"  {'[green]✓' if good else '[red]✗'}[/] {row.name}: expected {want:+.1f} ms, "
            f"measured {'–' if got is None else f'{got:+.1f} ms'}"
        )
    console.print("[green]Self-test passed.[/]" if ok else "[red]Self-test FAILED.[/]")
    return 0 if ok else 1


def cmd_selftest(args: argparse.Namespace) -> int:
    return asyncio.run(_selftest(args))


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sendspin-syncer",
        description="Measure the playback delay of Sendspin players with a microphone.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("devices", help="list audio input devices (microphones)")
    p.add_argument(
        "--check", metavar="DEVICE", help="record briefly from DEVICE and show its level"
    )
    p.add_argument("--seconds", type=float, default=2.0, help="length of --check (default 2)")
    p.add_argument("--channel", type=int, default=1, help="input channel for --check (default 1)")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("list", help="list Sendspin players on the network (doesn't connect)")
    p.add_argument("--timeout", type=float, default=4.0, help="seconds to listen (default 4)")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("measure", help="play test signals and report each player's delay")
    g = p.add_argument_group("players")
    g.add_argument(
        "-p",
        "--player",
        action="append",
        metavar="NAME",
        help="only players whose name/id/host contains NAME (repeatable; default: all found)",
    )
    g.add_argument(
        "--url", action="append", metavar="WS_URL", help="add a player by URL (repeatable)"
    )
    g.add_argument("--discover-time", type=float, default=4.0, help="mDNS listen time (s)")
    g.add_argument("--connect-timeout", type=float, default=15.0, help="per-player timeout (s)")
    g.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    g = p.add_argument_group("microphone")
    g.add_argument("-d", "--device", help="input device index or (part of) its name")
    g.add_argument("--channel", type=int, default=1, help="input channel to use (default 1)")
    g.add_argument("--save-recording", metavar="WAV", help="save the raw recording")
    g = p.add_argument_group("test signal")
    g.add_argument("-r", "--repeats", type=int, default=5, help="chirps per player (default 5)")
    g.add_argument(
        "--simultaneous",
        action="store_true",
        help="all players at once, each in its own band (faster, a bit less precise)",
    )
    g.add_argument(
        "--max-delay-ms", type=float, default=1000, help="latest arrival searched (default 1000)"
    )
    g.add_argument(
        "--max-early-ms", type=float, default=200, help="earliest arrival searched (default 200)"
    )
    g.add_argument("--min-snr", type=float, default=14.0, help="reject weaker detections (dB)")
    g = p.add_argument_group("report")
    g.add_argument(
        "--distance",
        action="append",
        metavar="NAME=METRES",
        help="player's distance from the mic, to remove sound travel time (repeatable)",
    )
    g.add_argument("--reference", metavar="NAME", help="report delays relative to this player")
    g.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    g.add_argument("--csv", metavar="PATH", help="also write the report as CSV")
    g.add_argument("--no-hints", action="store_true", help="don't print alignment suggestions")
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="show every chirp's result (and more logging with -vv)",
    )
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("selftest", help="check the pipeline with simulated players (no hardware)")
    p.add_argument("-r", "--repeats", type=int, default=3)
    p.add_argument("--simultaneous", action="store_true")
    p.set_defaults(func=cmd_selftest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # -v shows every chirp's result; -vv adds debug logging.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.verbose < 2:
        # The protocol library is chatty about expected things, such as older
        # players speaking a pre-1.0 protocol; only show that with -vv.
        logging.getLogger("aiosendspin").setLevel(logging.ERROR)
    try:
        return int(args.func(args) or 0)
    except DeviceError as e:
        err.print(f"[red]Audio device error:[/] {e}")
        return 2
    except KeyboardInterrupt:
        err.print("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
