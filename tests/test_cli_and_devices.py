import pytest

from sendspin_syncer.capture import DeviceError, InputDevice, resolve_device
from sendspin_syncer.cli import _parse_distances, build_parser
from sendspin_syncer.discovery import DiscoveredPlayer, select_players

DEVICES = [
    InputDevice(0, "Built-in Microphone", "ALSA", 2, 48000.0, True),
    InputDevice(3, "USB Audio Device: MiniDSP UMIK-1", "ALSA", 1, 48000.0, False),
    InputDevice(4, "USB Audio Device: Webcam", "ALSA", 1, 16000.0, False),
]


def test_resolve_device_default_index_and_name():
    assert resolve_device(None, DEVICES).index == 0
    assert resolve_device("3", DEVICES).index == 3
    assert resolve_device(4, DEVICES).index == 4
    assert resolve_device("umik", DEVICES).index == 3


def test_resolve_device_ambiguous_and_missing():
    with pytest.raises(DeviceError, match="more than one"):
        resolve_device("usb audio", DEVICES)
    with pytest.raises(DeviceError, match="no input device matches"):
        resolve_device("focusrite", DEVICES)
    with pytest.raises(DeviceError, match="index 9"):
        resolve_device("9", DEVICES)
    with pytest.raises(DeviceError, match="no audio input"):
        resolve_device(None, [])


def test_parse_distances():
    assert _parse_distances(["Kitchen=2.5", "Living room=4m"]) == {
        "Kitchen": 2.5,
        "Living room": 4.0,
    }
    with pytest.raises(SystemExit):
        _parse_distances(["Kitchen"])


def test_measure_arguments():
    args = build_parser().parse_args(
        [
            "measure",
            "-p",
            "kitchen",
            "-p",
            "lounge",
            "-d",
            "umik",
            "--repeats",
            "7",
            "--simultaneous",
            "--json",
            "out.json",
            "-y",
        ]
    )
    assert args.player == ["kitchen", "lounge"]
    assert args.device == "umik" and args.repeats == 7 and args.simultaneous and args.yes
    assert args.verbose == 0
    assert build_parser().parse_args(["measure", "-v"]).verbose == 1
    assert build_parser().parse_args(["measure", "-vv"]).verbose == 2
    assert build_parser().parse_args(["-v", "measure"]).verbose == 1


def test_select_players_and_urls():
    found = [
        DiscoveredPlayer("abc", "Kitchen", "10.0.0.5", 8928, "/sendspin"),
        DiscoveredPlayer("def", "Living Room", "10.0.0.6", 8928, "/sendspin"),
    ]
    assert select_players(found, None) == found
    assert [p.name for p in select_players(found, ["living"])] == ["Living Room"]
    with pytest.raises(LookupError):
        select_players(found, ["garage"])
    p = DiscoveredPlayer.from_url("ws://10.0.0.9:8927/sendspin")
    assert p.url == "ws://10.0.0.9:8927/sendspin" and not p.looks_encrypted
    with pytest.raises(ValueError):
        DiscoveredPlayer.from_url("http://nope")
    assert DiscoveredPlayer("A" * 43, "x", "h", 1, "/").looks_encrypted
