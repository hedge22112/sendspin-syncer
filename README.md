# sendspin-syncer

Measure how far out of sync your [Sendspin](https://github.com/Sendspin/spec)
players are, using a microphone.

The tool finds the players on your network and tells every one of them to
play a short test sound at a precise moment. It records the room and reports
how much later (or earlier) each player was actually heard than the others.
Example report:

```
                         Sendspin player delays
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ Player               ┃    Delay ┃   ± ┃ Out. delay ┃ Heard ┃   SNR ┃ Conf. ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ Kitchen (ref)        │  +0.0 ms │ 0.2 │       0 ms │   5/5 │ 38 dB │ high  │ ok     │
│ Living room AVR      │ +88.6 ms │ 0.3 │       0 ms │   5/5 │ 41 dB │ high  │ ok     │
│ Bedroom              │  +7.2 ms │ 0.4 │      20 ms │   5/5 │ 29 dB │ high  │ ok     │
└──────────────────────┴──────────┴─────┴────────────┴───────┴───────┴───────┴────────┘
To line players up (information only, nothing was changed). …
  • Kitchen: keep 0 ms
  • Bedroom: 20 ms → 27 ms
  • Living room AVR: 0 ms → 89 ms
```

**It only reports.** It never changes anything on a player: no volume, mute
or output-delay commands are sent. If you want to act on the results, change
the delay in Music Assistant yourself.

## Install

You need **Python 3.12 or newer** and the PortAudio library for microphone
access.

```bash
# Debian / Ubuntu / Raspberry Pi OS
sudo apt install libportaudio2
# macOS: PortAudio is bundled with the sounddevice wheel, nothing to do.

pipx install git+https://github.com/hedge22112/sendspin-syncer
# or: uv tool install git+https://github.com/hedge22112/sendspin-syncer
```

Check the install without any hardware:

```bash
sendspin-syncer selftest
```

The self-test runs simulated players against a virtual microphone and checks
that the measured delays match the ones that were configured.

## Use

### 1. Pick a microphone

```bash
sendspin-syncer devices                # list audio inputs
sendspin-syncer devices --check 3      # record 2 s from device 3 and show the level
```

Any microphone works, as long as it hears every player clearly. Put it
roughly in the middle of the players, or where you normally listen. A
measurement mic (e.g. a miniDSP UMIK-1) is nice to have, but a laptop or
webcam mic is fine. Only the *differences* between players are reported, so
the mic's own latency doesn't matter.

### 2. See which players are around

```bash
sendspin-syncer list
```

This only listens for mDNS adverts (`_sendspin._tcp`). It doesn't connect,
so Music Assistant isn't disturbed.

### 3. Measure

```bash
sendspin-syncer measure --device 3
sendspin-syncer measure --device umik -p kitchen -p "living room"   # some players only
sendspin-syncer measure --device 3 --json delays.json --csv delays.csv
```

Useful options:

| Option | What it does |
|---|---|
| `-d, --device` | Input device, by index or part of its name (default: system default) |
| `--channel N` | Input channel of a multi-channel interface (default 1) |
| `-p, --player NAME` | Only players whose name, id or host contains NAME (repeatable) |
| `--url ws://HOST:PORT/sendspin` | Add a player that mDNS doesn't find |
| `-r, --repeats N` | Test sounds per player (default 5; the median is reported) |
| `--simultaneous` | All players at once, each in its own frequency band. Faster, slightly less precise |
| `--max-delay-ms` | Latest arrival to search for (default 1000; raise it for very laggy players) |
| `--distance NAME=METRES` | Remove sound travel time (2.9 ms per metre) for a player far from the mic |
| `--reference NAME` | Report delays relative to this player instead of the earliest |
| `--json PATH`, `--csv PATH` | Save the report |
| `--save-recording WAV` | Keep the raw recording for inspection |
| `--no-hints` | Hide the alignment suggestions |
| `-y` | Don't ask for confirmation |

A run takes about `players × repeats × 1.8 s` (30 s for 3 players), or
`repeats × 1.8 s` with `--simultaneous`.

### Reading the report

- **Delay**: how much later this player is heard than the reference (by
  default the earliest player, marked *ref*). This is the number that matters
  for sync.
- **±**: the spread (standard deviation) across the repeats, in ms. A few
  tenths of a millisecond is normal.
- **Out. delay**: the player's current *output delay* setting. Players apply
  it themselves, so it's **already included** in the measured delay.
- **Heard / SNR / Conf.**: how many test sounds were detected, how clearly,
  and an overall confidence. *no signal* means the mic didn't hear that
  player. Check its volume and that it's in range of the mic.
- **To line players up**: in Sendspin, a player's output delay makes it play
  *earlier*, to cancel out delay after the audio leaves the device (an AVR,
  a soundbar, …). So the earliest player keeps its setting, and each later
  player would need its current output delay plus how late it is. This is
  shown for information only.

## Music Assistant and your players

A Sendspin player is connected to one server at a time. To play the test
sounds, this tool connects as a (temporary) server with the *playback*
activity, which takes the player over from Music Assistant. The player tells
MA it switched to `another_server`.

- **Don't run it while you're listening to something.** Playback on the
  measured players stops.
- When the test finishes (about half a minute), the tool disconnects.
  Music Assistant gets the players back the normal way, i.e. the next time
  you press play in MA. Until then MA may show them as unavailable for a
  moment. This hand-back is covered by the test suite.
- The tool keeps a fixed server identity in
  `~/.config/sendspin-syncer/identity.json`, so players see the same server
  every time instead of a new one on every run.
- Volume, mute and output delay are never touched. The test plays at each
  player's current volume, so set a comfortable, audible level in MA first.

Both kinds of player are supported:

- **Older, unencrypted players** (pre-1.0 protocol).
- **Spec 1.0 players** that allow *unpaired* access. The tool approves them
  for unpaired playback for the duration of the run.

Players that require pairing are reported as such and skipped.

## How it works

1. **Test signal.** Each player gets short logarithmic sine sweeps
   ("chirps", 400 Hz → 12 kHz, 0.2 s). A chirp has one sharp
   cross-correlation peak, unlike a steady tone, so its arrival time can be
   found to a fraction of a millisecond even in an echoey room. By default
   the players take turns. With `--simultaneous`, every player gets its own
   frequency band and they all play at once.
2. **Scheduling.** All players are put in one temporary group (inside this
   tool only; MA's groups aren't touched). Each player gets its own audio
   channel carrying only its own chirps, and every chunk is timestamped on
   the Sendspin server clock. If the players were perfectly in sync, each
   chirp would sound exactly at its scheduled time.
3. **Recording.** The microphone is recorded the whole time. Each audio
   block is stamped with the server clock, and a line is fitted through
   those stamps. That gives sample-accurate server time for every recorded
   sample and corrects for mic clock drift.
4. **Detection.** Around each scheduled time, the recording is band-pass
   filtered (zero-phase) and cross-correlated with the chirp. The earliest
   strong peak is taken as the direct sound (reflections arrive later), and
   it is refined to sub-sample precision. Weak detections and outliers are
   discarded, and the median of the repeats is reported.
5. **Report.** The mic's own latency is the same for every player, so it
   cancels out when delays are shown relative to a reference player.

Before the first chirp there are a few seconds of silence. This gives
players time to finish synchronising their clocks and to settle any start-up
corrections, which would otherwise make the first measurement unreliable.

## Development

```bash
uv venv -p 3.12 && uv pip install -e ".[test]"
pytest                     # unit + end-to-end tests (simulated players, no hardware)
ruff check src tests && ruff format --check src tests
```

The end-to-end tests run the real server code against Sendspin players built
on `aiosendspin`'s own client, with a virtual room and microphone. That
includes taking a player over from a stand-in Music Assistant and handing it
back. To also test against an old, unencrypted player, point
`SENDSPIN_LEGACY_PYTHON` at an interpreter that has `aiosendspin~=6.0`:

```bash
uv venv -p 3.12 .venv-legacy && VIRTUAL_ENV=.venv-legacy uv pip install "aiosendspin~=6.0.1"
SENDSPIN_LEGACY_PYTHON=.venv-legacy/bin/python pytest
```

See [PLAN.md](PLAN.md) for the design notes.
