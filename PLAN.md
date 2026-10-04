# sendspin-syncer — Plan

Goal: a Python CLI that finds Sendspin players on the network, plays a test
signal on each one, records them with a microphone, and **reports** each
player's current delay.

**This tool is read-only towards players.** It never changes a player's
settings. It never sends `set_output_delay` (or the older `set_static_delay`),
volume or mute. Players are controlled by Music Assistant (MA), and any
correction is left for the user to make in MA.

## 1. Key facts from the `aiosendspin` library (v9.x) and the spec

- **Requires Python ≥ 3.12.** We'll use `aiosendspin[server]`.
- `SendspinServer` handles mDNS discovery (`_sendspin._tcp`), connections,
  groups and clock sync. Players advertise themselves, and the server connects
  to them.
- `SendspinGroup.start_stream(channel_resolver=...)` returns a `PushStream`.
  `channel_resolver(player_id) -> UUID` lets **each player in one group get
  different audio**: we call `prepare_audio(pcm, fmt, channel_id=...)` per
  channel, then one `commit_audio(play_start_us=...)`. Every player is told to
  play at the same server-clock timestamp, so if all players were perfectly in
  sync, every test signal would arrive at the mic at the same time.
- Players report `output_delay_ms` (0–5000; called `static_delay_ms` inside the
  library). The player applies it itself, so **what we measure already
  includes the player's current output delay setting.** We read and report
  that setting; we never change it.
- Security: players here are **unpaired / unencrypted**. Use
  `allow_unencrypted=True` for pre-encryption (legacy) players and accept
  unpaired (sentinel-PSK) Noise sessions for newer ones. No pairing flow is
  needed.

## 2. Coexisting with Music Assistant

Per the spec ("Multiple servers, server-initiated"):

- A player holds **one** admitted connection at a time. A new server that
  declares the `playback` activity outranks or ties the current holder and
  **displaces** it. MA gets `client/goodbye` reason `another_server`.
- The player stores the `server_id` of the last server it played from (the
  "last-playback server").

How the tool handles this:

1. **Don't run during playback.** Before measuring, warn and ask for
   confirmation. Players that are currently playing are skipped unless
   `--force` is given.
2. Use a **stable server identity**, stored in
   `~/.config/sendspin-syncer/identity`, so every run is one consistent server
   and not a new unknown server each time.
3. Take the players only for the measurement (~10–30 s). Then stop the stream,
   clear our `playback` activity, and close cleanly.
4. Afterwards MA gets players back the normal way: when MA reconnects or the
   user presses play in MA (a `playback` connection always wins). M3 includes
   a check that this hand-back works for each player type you own, and the
   README will document how it behaves.
5. If MA takes a player back mid-measurement (someone pressed play), mark that
   player as "interrupted" in the report instead of failing the whole run.

## 3. What "delay" we measure

Every player is told to play at time **T** on the server clock. The mic hears
player *i* at **T + dᵢ + m**, where:

- `dᵢ` = the player's real output error *with its current output delay already
  applied* (DAC/amp latency it doesn't compensate for, AVR/soundbar
  processing, Bluetooth sinks, etc.). **This is what we report.**
- `m` = the host's mic capture latency plus sound travel time
  (about 2.9 ms per metre). This is the same for every player except the
  distance part.

Because `m` cancels out, **relative delay** (`dᵢ − d_ref`) is robust and is the
main number reported. Per-player mic distance (`--distance name=3.5m`) can be
given so the tool subtracts sound travel time. An optional loopback
calibration (later milestone) would make absolute numbers meaningful.

## 4. Test signal design

Pure, constant sine waves are a bad timing reference: they repeat forever, so
cross-correlation gives many equal peaks, and room reflections make it worse.
We still keep the idea of a "unique tone per player", but shape it:

- **Default: sequential mode.** Players take turns. Each plays a short
  **log-sweep chirp** (e.g. 500 Hz → 8 kHz, 200 ms, faded in and out), each one
  scheduled at a known offset such as `T + k·1.5 s`. A matched filter
  (cross-correlation with the reference chirp) gives a sharp, unambiguous peak.
  This is the most robust option and has no crosstalk.
- **Simultaneous mode** (`--simultaneous`; faster, closest to the original
  idea). All players play at the same time, each in its **own frequency band**
  (chirps over separate non-overlapping bands). Each player is detected with a
  band-pass filter plus its own matched filter.
- Repeat N times (default 5) with a random gap between repeats, and report the
  median and spread. Reject a detection when its peak-to-noise ratio falls
  below a threshold.
- Sample rate 48 kHz, 16-bit PCM, mono duplicated to stereo. The test runs at
  the player's current volume (we don't touch volume). If the signal is too
  weak, the tool tells you to raise it in MA.

## 5. Architecture

```
src/sendspin_syncer/
  cli.py          # entrypoint: `list`, `devices` and `measure`
  server.py       # SendspinServer lifecycle, identity, discovery, connect, clean hand-back
  signals.py      # chirp generation, band allocation, schedule
  playback.py     # PushStream per-player channels, timed commit at play_start_us
  capture.py      # input device listing/selection; sounddevice InputStream → buffer with timestamps
  clock_map.py    # map mic sample index ↔ server clock (monotonic) time
  analysis.py     # band-pass + matched filter, peak picking, SNR, stats
  report.py       # table / JSON / CSV output
tests/
  test_signals.py
  test_analysis.py  # synthetic: mix known-delayed signals + noise + reverb → recover delays
  test_e2e_sim.py   # fake players + fake mic, so the pipeline runs without hardware
```

**`list`**: discovers players and shows each one's name, id, product/type,
connection state, current `output_delay_ms`, and whether it is playing. It
connects without the `playback` activity, so it doesn't displace MA.

**`devices`**: lists the audio input devices on this machine, with index,
name, host API (ALSA/PulseAudio/CoreAudio/WASAPI…), input channel count,
default sample rate, and which one is the system default. A quick
`--check` option records 2 s from a device and shows its level, so you can
make sure you picked the right mic.

**Choosing the mic:** `measure --device <index|name>`. Partial,
case-insensitive name matches are allowed; if a name matches more than one
device, the tool stops and lists the matches. `--channel N` picks one channel
of a multi-channel interface. Without `--device`, the system default input is
used and its name is printed at the start of the run. The device and its
sample rate are saved in the JSON report. If the device can't do 48 kHz, the
tool records at the device's native rate and resamples the reference signal
to match.

**`measure`** flow:
1. Start the server, discover players (or filter with `--player`), confirm
   with the user, and connect with the `playback` activity.
2. Put the chosen players into one temporary group (in our server only; MA's
   groups aren't touched).
3. Open the selected input device (see above), start mic capture, and record mic stream time against `server.clock` (sample
   index ↔ µs) using the sounddevice callback's `inputBufferAdcTime`.
4. Build the schedule and push per-channel PCM (silence plus the test signal)
   with `play_start_us = now + lead_time` (lead time above each player's
   `required_lead_time_ms`).
5. Record until the last signal + tail, then stop the stream and release the
   players (§2.3).
6. Analysis: for each player and repeat, find the detected time minus the
   expected time to get a raw delay. Then compute median, stddev and
   confidence.
7. Report.

**Report** (terminal table, plus `--json` / `--csv` for saving and comparing
runs):

| Player | Current output delay | Measured delay vs. earliest | Spread (±) | Confidence |
|---|---|---|---|---|
| Kitchen (ESPHome) | 0 ms | +0.0 ms (ref) | 0.3 ms | high |
| Living room (AVR) | 0 ms | +87.4 ms | 0.6 ms | high |
| Bedroom (Pi) | 20 ms | +12.1 ms | 0.4 ms | medium |

The report also adds an informational line: "to align, add X ms of output
delay in MA to every player except the latest". This text is for the user
only and is never sent to players. It can be turned off with `--no-hints`.

## 6. Milestones

- **M1 – Offline core.** `signals.py` + `analysis.py`, with synthetic tests
  (known delays ±0.1 ms, noise, reverb, multiple players at once). No network
  or hardware.
- **M2 – Discovery.** `list` command against the real network. Connect to
  unencrypted and unpaired players without displacing MA.
- **M3 – Playback + hand-back.** Play a scheduled test signal on one player,
  then on several with per-player channels. Check that each player type goes
  back to MA afterwards.
- **M4 – Capture and measure.** `devices` command and `--device` /
  `--channel` selection, mic capture, clock mapping, the full `measure`
  pipeline, and the report with JSON/CSV output.
- **M5 – Polish.** Mic distance correction,
  optional loopback calibration for absolute latency, packaging
  (`pyproject.toml`, `uv`), README, CI (ruff + pytest).

## 7. Dependencies

`aiosendspin[server]`, `numpy`, `scipy` (filters, correlation), `sounddevice`
(PortAudio mic capture), `rich` (tables), `pytest`, `ruff`.
Tooling: `uv` + `pyproject.toml`, Python 3.12+.

## 8. Remaining risks

1. **Hand-back to MA** differs by player implementation. The spec says MA
   "SHOULD NOT auto-reconnect" after `another_server`, so MA may show a player
   as unavailable until its next play or discovery cycle. This will be checked
   per player type in M3.
2. **Mixed player firmware.** Older players may speak a pre-1.0 protocol
   draft. `allow_noncompliant_clients=True` covers most of these. Any player
   that can't be driven will be listed as "unsupported" in the report.
3. **Mic placement.** Players in different rooms need either one central mic
   with good SNR, or several runs that share one reference player, which the
   tool then stitches together.
4. **Host mic clock drift** over a ~30 s run is small (<1.5 ms at 50 ppm), but
   it can be estimated from the repeated signals and corrected.
