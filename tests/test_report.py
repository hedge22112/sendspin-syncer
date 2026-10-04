import csv
import json

import numpy as np
import pytest
from rich.console import Console

from sendspin_syncer.analysis import EmissionResult, PlayerResult
from sendspin_syncer.capture import Recording
from sendspin_syncer.clock_map import ClockMap
from sendspin_syncer.discovery import DiscoveredPlayer
from sendspin_syncer.report import build_report, print_report, write_csv, write_json
from sendspin_syncer.session import PlayerInfo
from sendspin_syncer.signals import ScheduleOptions, build_schedule


def _player(name, delay_ms, status="ready", detail=""):
    target = DiscoveredPlayer(name.lower(), name, "10.0.0.2", 8928, "/sendspin")
    return PlayerInfo(
        target,
        client_id=f"id-{name}",
        name=name,
        output_delay_ms=delay_ms,
        status=status,
        detail=detail,
    )


def _result(pid, median, n=5):
    r = PlayerResult(pid)
    r.emissions = [EmissionResult(pid, i, 0, median, 30.0, True) for i in range(n)]
    r.median_ms, r.spread_ms, r.snr_db, r.confidence = median, 0.2, 30.0, "high"
    return r


@pytest.fixture
def report():
    players = [
        _player("Kitchen", 0),
        _player("Lounge", 10),
        _player("Garage", 0),
        _player("Shed", None, status="failed", detail="could not connect"),
    ]
    sched = build_schedule(["kitchen", "lounge", "garage"], ScheduleOptions(repeats=5))
    results = {
        "kitchen": _result("kitchen", 12.0),
        "lounge": _result("lounge", 52.0),
        "garage": PlayerResult(
            "garage", [EmissionResult("garage", 0, 0, None, 3.0, False, "weak signal")]
        ),
    }
    rec = Recording(np.zeros(48_000, np.float32), 48_000, ClockMap.nominal(0, 48_000), "Mic")
    return build_report(players, results, sched, rec, distances_m={"Lounge": 3.43})


def test_relative_delays_and_reference(report):
    rows = {r.name: r for r in report.rows}
    assert report.reference == "Kitchen"
    assert rows["Kitchen"].relative_ms == 0
    # 3.43 m of air is 10 ms, removed from the raw 52 ms.
    assert rows["Lounge"].relative_ms == pytest.approx(30.0)
    assert rows["Garage"].status == "no signal"
    assert rows["Shed"].status == "failed"


def test_suggestions_add_to_late_players_only(report):
    rows = {r.name: r for r in report.rows}
    assert rows["Kitchen"].suggested_output_delay_ms == 0
    assert rows["Lounge"].suggested_output_delay_ms == 40  # current 10 + 30 late
    assert rows["Garage"].suggested_output_delay_ms is None


def test_explicit_reference():
    players = [_player("A", 0), _player("B", 0)]
    sched = build_schedule(["a", "b"], ScheduleOptions(repeats=1))
    res = {"a": _result("a", 10.0), "b": _result("b", 25.0)}
    rep = build_report(players, res, sched, None, reference="b")
    rows = {r.name: r for r in rep.rows}
    assert rows["A"].relative_ms == -15.0 and rows["B"].relative_ms == 0
    assert rows["B"].is_reference


def test_outputs(report, tmp_path):
    write_json(report, tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text())
    assert [p["name"] for p in data["players"]] == ["Kitchen", "Lounge", "Garage", "Shed"]
    assert data["microphone"]["device"] == "Mic"
    write_csv(report, tmp_path / "r.csv")
    rows = list(csv.DictReader((tmp_path / "r.csv").open()))
    assert rows[1]["name"] == "Lounge" and float(rows[1]["relative_ms"]) == pytest.approx(30)

    console = Console(record=True, width=120)
    print_report(report, console)
    text = console.export_text()
    assert "Lounge" in text and "+30.0 ms" in text and "10 ms → 40 ms" in text
    assert "nothing was changed" in text
