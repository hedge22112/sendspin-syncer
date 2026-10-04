"""Turn measurements into a report: terminal table, JSON and CSV."""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import __version__
from .analysis import PlayerResult
from .capture import Recording
from .session import PlayerInfo
from .signals import Schedule

SPEED_OF_SOUND_M_S = 343.0
MAX_OUTPUT_DELAY_MS = 5000


@dataclass(slots=True)
class PlayerRow:
    name: str
    client_id: str | None
    url: str
    product: str | None
    security: str | None
    audio_format: str | None
    output_delay_ms: int | None
    status: str
    detail: str = ""
    raw_ms: float | None = None
    distance_m: float | None = None
    relative_ms: float | None = None
    spread_ms: float | None = None
    snr_db: float | None = None
    detections: str = ""
    confidence: str = "none"
    suggested_output_delay_ms: int | None = None
    is_reference: bool = False


@dataclass(slots=True)
class Report:
    created: str
    tool_version: str
    mode: str
    repeats: int
    microphone: dict[str, Any]
    reference: str | None
    rows: list[PlayerRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "tool_version": self.tool_version,
            "mode": self.mode,
            "repeats": self.repeats,
            "microphone": self.microphone,
            "reference": self.reference,
            "players": [_row_dict(r) for r in self.rows],
            "notes": self.notes,
        }


def _row_dict(row: PlayerRow) -> dict[str, Any]:
    return asdict(row)


def build_report(
    players: list[PlayerInfo],
    results: dict[str, PlayerResult],
    schedule: Schedule,
    recording: Recording | None,
    *,
    distances_m: dict[str, float] | None = None,
    reference: str | None = None,
) -> Report:
    """Combine player info and analysis into report rows.

    ``results`` is keyed by the schedule's player ids (the player's mDNS
    instance name). Raw delays still contain the microphone's own latency, so
    the meaningful number is the delay *relative* to the reference player.
    """
    rows: list[PlayerRow] = []
    for p in players:
        key = p.target.client_id_hint
        res = results.get(key)
        row = PlayerRow(
            name=p.label,
            client_id=p.client_id,
            url=p.target.url,
            product=p.product,
            security=p.security,
            audio_format=p.audio_format,
            output_delay_ms=p.output_delay_ms,
            status=p.status,
            detail=p.detail,
        )
        if p.status == "ready":
            row.status = "ok"
        if res is not None:
            row.detections = f"{res.detected_count}/{len(res.emissions)}"
            row.confidence = res.confidence
            row.spread_ms = res.spread_ms
            row.snr_db = res.snr_db
            if res.median_ms is not None:
                dist = _distance_for(p, distances_m)
                row.distance_m = dist
                travel = (dist or 0.0) / SPEED_OF_SOUND_M_S * 1000
                row.raw_ms = res.median_ms - travel
            elif row.status == "ok":
                row.status = "no signal"
                row.detail = row.detail or "not heard by the microphone"
        rows.append(row)

    measured = [r for r in rows if r.raw_ms is not None]
    ref_row: PlayerRow | None = None
    if reference:
        matches = [r for r in measured if reference.lower() in r.name.lower()]
        ref_row = matches[0] if matches else None
    if ref_row is None and measured:
        ref_row = min(measured, key=lambda r: r.raw_ms)  # type: ignore[arg-type,return-value]
    if ref_row is not None:
        ref_row.is_reference = True
        assert ref_row.raw_ms is not None
        for r in measured:
            assert r.raw_ms is not None
            r.relative_ms = r.raw_ms - ref_row.raw_ms

    _suggest(measured)

    mic: dict[str, Any] = {}
    if recording is not None:
        mic = {
            "device": recording.device,
            "sample_rate": recording.sample_rate,
            "clock_drift_ppm": round(recording.clock.drift_ppm, 1),
            "callback_jitter_ms": round(recording.clock.jitter_us / 1000, 2),
            "overflows": recording.overflows,
            "seconds": round(len(recording.samples) / recording.sample_rate, 1),
        }
    report = Report(
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        tool_version=__version__,
        mode=schedule.mode,
        repeats=max((e.repeat for e in schedule.emissions), default=-1) + 1,
        microphone=mic,
        reference=ref_row.name if ref_row else None,
        rows=rows,
    )
    if reference and (ref_row is None or reference.lower() not in ref_row.name.lower()):
        report.notes.append(
            f"Reference {reference!r} was not measured; using the earliest player instead."
        )
    if recording is not None and recording.overflows:
        report.notes.append(
            f"The microphone dropped audio {recording.overflows} time(s); results may be off."
        )
    return report


def _distance_for(p: PlayerInfo, distances_m: dict[str, float] | None) -> float | None:
    if not distances_m:
        return None
    for key, metres in distances_m.items():
        k = key.lower()
        if k in p.label.lower() or k == p.target.client_id_hint.lower():
            return metres
    return None


def _suggest(measured: list[PlayerRow]) -> None:
    """Work out output delays that would line everyone up with the earliest player.

    A player's output delay makes it play *earlier* by that amount (it is
    meant to cancel delay after the audio leaves the device). So the earliest
    player keeps its setting, and each later player needs its current value
    plus how late it is.
    """
    if not measured:
        return
    earliest = min(r.raw_ms for r in measured if r.raw_ms is not None)
    for r in measured:
        assert r.raw_ms is not None
        late = r.raw_ms - earliest
        current = r.output_delay_ms or 0
        r.suggested_output_delay_ms = int(min(MAX_OUTPUT_DELAY_MS, round(current + late)))


# ------------------------------------------------------------------- output


def _fmt_ms(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v:+.1f} ms" if signed else f"{v:.1f} ms"


def print_report(report: Report, console: Console | None = None, *, hints: bool = True) -> None:
    console = console or Console()
    table = Table(title="Sendspin player delays", title_style="bold", show_lines=False)
    table.add_column("Player", style="bold", ratio=3)
    table.add_column("Delay", justify="right", no_wrap=True)
    table.add_column("±", justify="right", no_wrap=True)
    table.add_column("Out. delay", justify="right", no_wrap=True)
    table.add_column("Heard", justify="right", no_wrap=True)
    table.add_column("SNR", justify="right", no_wrap=True)
    table.add_column("Conf.", no_wrap=True)
    table.add_column("Status", ratio=2)

    conf_style = {"high": "green", "medium": "yellow", "low": "red", "none": "dim"}
    for r in report.rows:
        name = r.name + (" [dim](ref)[/]" if r.is_reference else "")
        if r.product:
            name += f"\n[dim]{r.product}[/]"
        status = r.status if not r.detail else f"{r.status}: {r.detail}"
        status_style = "green" if r.status == "ok" else "red"
        table.add_row(
            name,
            _fmt_ms(r.relative_ms, signed=True),
            "–" if r.spread_ms is None else f"{r.spread_ms:.1f}",
            "–" if r.output_delay_ms is None else f"{r.output_delay_ms} ms",
            r.detections or "–",
            "–" if r.snr_db is None else f"{r.snr_db:.0f} dB",
            f"[{conf_style.get(r.confidence, '')}]{r.confidence}[/]",
            f"[{status_style}]{status}[/]",
        )
    console.print(table)

    mic = report.microphone
    if mic:
        console.print(
            f"[dim]Microphone: {mic['device']} @ {mic['sample_rate']} Hz, "
            f"clock drift {mic['clock_drift_ppm']} ppm, "
            f"{report.mode} mode × {report.repeats} repeats.[/]"
        )
    console.print(
        "[dim]Delay: how much later each player is heard than the reference (ref). "
        "±: spread across repeats (ms). Out. delay: the player's current output delay "
        "setting, already included in its measured delay.[/]"
    )
    for note in report.notes:
        console.print(f"[yellow]Note:[/] {note}")

    measured = [r for r in report.rows if r.suggested_output_delay_ms is not None]
    if hints and len(measured) > 1:
        console.print()
        console.print(
            "[bold]To line players up[/] (information only, nothing was changed). "
            "A player's output delay makes it play earlier, so the earliest player stays as it "
            "is and later ones need more:"
        )
        for r in sorted(measured, key=lambda r: r.relative_ms or 0):
            current = r.output_delay_ms or 0
            if r.suggested_output_delay_ms == current:
                console.print(f"  • {r.name}: keep {current} ms")
            else:
                console.print(
                    f"  • {r.name}: {current} ms → [bold]{r.suggested_output_delay_ms} ms[/]"
                )


def write_json(report: Report, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")


CSV_FIELDS = [
    "name",
    "status",
    "relative_ms",
    "raw_ms",
    "spread_ms",
    "snr_db",
    "detections",
    "confidence",
    "output_delay_ms",
    "suggested_output_delay_ms",
    "distance_m",
    "product",
    "audio_format",
    "security",
    "client_id",
    "url",
    "detail",
]


def write_csv(report: Report, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["created", *CSV_FIELDS])
        w.writeheader()
        for r in report.rows:
            row = _row_dict(r)
            w.writerow({"created": report.created, **{k: row[k] for k in CSV_FIELDS}})
