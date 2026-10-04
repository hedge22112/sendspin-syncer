# sendspin-syncer — Plan

Goal: a Python CLI that finds Sendspin players on the network, plays a test
signal on each one, records them with a microphone, and works out each
player's delay. It can then optionally write a correcting delay back to each
player.

## 1. Key facts from the `aiosendspin` library (v9.x)

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
- Players report `static_delay_ms` and may support the `set_static_delay`
  command (0–5000 ms) through `PlayerV1Role.set_static_delay()`. That's how we
  can apply the fix.
- The library uses Noise encryption and pairing (`ServerPairingStore`,
  `allow_unencrypted`). Our tool needs either a pairing flow or "trusted
  unpaired" playback. See the open questions.

## 2. What "delay" we measure

Every player is told to play at time **T** on the server clock. The mic hears
player *i* at **T + dᵢ + m**, where:

- `dᵢ` = the player's real output error (DAC/amp latency, bad static delay,
  Bluetooth/HDMI sink, etc.). **This is what we want.**
- `m` = the host's mic capture latency plus sound travel time
  (about 2.9 ms per metre). This is the same for every player except the
  distance part.

Because `m` cancels out, **relative delay** (`dᵢ − d_ref`) is robust and is all
that's needed to sync players to each other. **Absolute delay** needs `m`,
which takes an optional loopback calibration (§5, M5). The tool will also
accept per-player mic distances so it can subtract travel time.

## 3. Test signal design

Pure, constant sine waves are a bad timing reference: they repeat forever, so
cross-correlation gives many equal peaks, and room reflections make it worse.
We still keep the idea of a "unique tone per player", but shape it:

- **Default: sequential mode.** Players take turns. Each plays a short
  **log-sweep chirp** (e.g. 500 Hz → 8 kHz, 200 ms, faded in and out), each one
  scheduled at a known offset such as `T + k·1.5 s`. A matched filter
  (cross-correlation with the reference chirp) gives a sharp, unambiguous peak.
  This is the most robust option and has no crosstalk.
- **Simultaneous mode** (faster, closest to the original idea). All players
  play at the same time, each in its **own frequency band** (e.g. chirps over
  separate non-overlapping bands, or tone bursts at distinct frequencies).
  Each player is detected with a band-pass filter plus its own matched filter.
- Repeat N times (default 5) with a random gap between repeats, and report the
  median and spread. Reject a detection when its peak-to-noise ratio falls
  below a threshold.
- Sample rate 48 kHz, 16-bit PCM, mono duplicated to stereo.

## 4. Architecture

```
src/sendspin_syncer/
  cli.py          # argparse/typer entrypoint: list | measure | apply
  server.py       # SendspinServer lifecycle, discovery, wait for players, group setup
  signals.py      # chirp / tone-burst generation, band allocation, schedule
  playback.py     # PushStream per-player channels, timed commit at play_start_us
  capture.py      # sounddevice InputStream → ring buffer with timestamps
  clock_map.py    # map mic sample index ↔ server clock (monotonic) time
  analysis.py     # band-pass + matched filter, peak picking, SNR, stats
  report.py       # table / JSON output, suggested static delays
tests/
  test_signals.py
  test_analysis.py  # synthetic: mix known-delayed signals + noise + reverb → recover delays
  test_e2e_sim.py   # fake players + fake mic, so the pipeline runs without hardware
```

**Measurement flow (`measure`):**
1. Start the `SendspinServer`, discover players for X seconds (or use `--player`
   filters), and connect to them.
2. Put the chosen players into one temporary group. Remember their previous
   group, volume and static delay so we can put them back afterwards.
3. Start mic capture and record mic stream time against `server.clock` (sample
   index ↔ µs) using the sounddevice callback's `inputBufferAdcTime`.
4. Build the schedule and push per-channel PCM (silence plus the test signal)
   with `play_start_us = now + lead_time` (lead time well above the players'
   `required_lead_time_ms`).
5. Record until the last signal + tail, then stop the stream.
6. Analysis: for each player and repeat, find the detected time minus the
   expected time to get a raw delay. Then compute median, stddev and
   confidence.
7. Report a table of each player's delay relative to the earliest player, its
   current static delay, the suggested static delay, and its confidence.
8. Put players back into their original state.

**Apply (`apply` or `measure --apply`):** the player that plays latest gets
extra delay 0. Every other player gets `static_delayᵢ += (d_max − dᵢ)`. Only
players listing `set_static_delay` in `state_supported_commands` are changed,
values are clamped to 0–5000 ms, and an optional verification pass runs
afterwards.

## 5. Milestones

- **M1 – Offline core.** `signals.py` + `analysis.py`, with synthetic tests
  (known delays ±0.1 ms, noise, reverb, multiple players at once). No network.
- **M2 – Discovery.** A `list` command: start the server, show the players that
  were found (id, name, connected, static delay, supported commands). Sort out
  pairing and trust here.
- **M3 – Playback.** Play a scheduled test signal on one player, then on
  several with per-player channels. Check it by ear.
- **M4 – Capture and measure.** Mic capture, clock mapping, and the full
  `measure` pipeline with a report and JSON output.
- **M5 – Apply and calibrate.** Write static delays and run a verify pass.
  Optional absolute-latency calibration using a loopback cable or the host's
  own speaker as a reference.
- **M6 – Polish.** Logging, `--device` for mic selection, packaging
  (`pyproject.toml`, `uv`), README, CI (ruff + pytest).

## 6. Dependencies

`aiosendspin[server]`, `numpy`, `scipy` (filters, correlation), `sounddevice`
(PortAudio mic capture), `rich` (tables, optional), `pytest`, `ruff`.
Tooling: `uv` + `pyproject.toml`, Python 3.12+.

## 7. Open questions / risks

1. **Existing server conflict.** Are your players currently attached to Music
   Assistant (or another Sendspin server)? A player usually follows one server
   at a time. We may need to (a) take over temporarily and then hand the player
   back, or (b) run while MA is stopped. This needs testing with real players.
2. **Pairing/encryption.** Are your players paired with MA? We may need to pair
   our tool too (it shows a PIN), or rely on `allow_unencrypted` or trusted
   unpaired playback if the players allow it.
3. **Which players/hardware?** (sendspin CLI on a Pi, ESPHome, MA web player,
   …) This affects whether `set_static_delay` is supported.
4. **Mic placement.** Players in different rooms need either one central mic
   with good SNR or a run per room against a shared reference player.
5. **Relative vs. absolute.** Is relative sync between players enough (the
   likely answer), or do you also want absolute latency (e.g. for A/V lip-sync)?
6. **Host mic clock drift** over a ~10 s run is small (<1 ms at 50 ppm), but we
   can estimate and correct it from repeated signals if needed.
