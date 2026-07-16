# The Logicians

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)
![Logic Pro](https://img.shields.io/badge/Logic%20Pro-FF4F00?logo=apple&logoColor=white)
![MIDI](https://img.shields.io/badge/MIDI-Live%20I%2FO-009688?logo=musicbrainz&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![AI](https://img.shields.io/badge/AI-Melody%20Improviser-8B5CF6?logo=openai&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Version](https://img.shields.io/badge/Version-0.1.0-blue)

AI-assisted melody improviser for Logic Pro. Capture a MIDI loop, analyze its harmony and rhythm, then generate human-like melodies — exported as MIDI or sent live back into Logic.

Built by Patrick Yao, Oriol Garrobé Guilera, and Alfredo Zermini.

## Features

- Parse MIDI loops from file or live capture
- Quantize notes to symbolic musical positions
- Infer key, chord progression, bass roots, and rhythm density
- Generate melodies with detailed musical rules
- Export as MIDI or send live to Logic Pro
- Deterministic output with a fixed random seed
- Live performance UI for session control during rehearsal or performance

## Requirements

- Python 3.11+
- macOS with Logic Pro (for live MIDI I/O)
- IAC Driver enabled for routing MIDI between this tool and Logic

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

Generate an example input loop, then create a melody:

```bash
python scripts/create_example_loop.py
python -m logicians.cli generate \
  --input examples/input_loops/loop.mid \
  --output examples/generated/melody.mid \
  --seed 1
```

Analyze a loop:

```bash
python -m logicians.cli analyze --input examples/input_loops/loop.mid
```

List MIDI ports:

```bash
python -m logicians.cli ports
```

## Connecting Logic Pro

1. Open **Audio MIDI Setup** → **Window** → **Show MIDI Studio**
2. Double-click **IAC Driver** and enable **Device is online**
3. Create a bus (e.g. "IAC Driver Bus 1") if needed
4. In Logic, set a software instrument track's MIDI input to the IAC bus to receive output from this tool
5. Route Logic's loop output to the IAC bus for live capture

### Live playback

```bash
python -m logicians.cli play \
  --input examples/input_loops/loop.mid \
  --midi-output "IAC Driver Bus 1" \
  --tempo 120 \
  --count-in 1
```

### Live capture

The capture clock does **not** start when you run the command. Switch to Logic first, then press Play — capture begins on the first MIDI note (or when you press Enter with `--sync enter`). It records exactly `--bars` of musical time.

```bash
python -m logicians.cli live \
  --midi-input "IAC Driver Bus 1" \
  --midi-output "IAC Driver Bus 2" \
  --tempo 120 \
  --bars 4
```

Workflow:

1. Run the command — it waits (no timer yet)
2. Switch to Logic and press Play
3. **Loop 1** — first MIDI note syncs to bar 1; loop is captured
4. **Loop 2** — buffer while the melody is generated
5. **Loop 3+** — generated melody enters; each cycle is a fresh improvisation that connects smoothly to where the last loop ended, until you press Ctrl+C

Use `--once` to play only the first melody. Use `--sync enter` to sync manually.

Pass `--key F` (or your project key) so analysis and melody generation match Logic. After capture, the CLI prints the detected chord progression — verify it before improvising.

Live capture splits bass (notes below E3) from harmony automatically. Keep bass in a low register on a separate MIDI stream if possible.

### Live performance UI

For performance use, start the local app — it opens in a native window (no browser needed):

```bash
logicians serve
```

Use `logicians serve --browser` if you prefer the system browser at http://127.0.0.1:8765.

**Important:** Tempo, bars, time signature, and key in the UI must match your Logic project.

Workflow:

1. Build your loop in Logic (do not start improvising yet).
2. Run `logicians serve` — a compact app window opens (works well on a second screen, or on an iPad via browser with `--browser`).
3. Enter settings (MIDI ports, tempo, bars, key, etc.) and press **Start Session**.
4. Switch to Logic and press Play — the first MIDI note syncs the loop clock.
5. Keep looping as long as you need while building or rehearsing.
6. When ready, press **Improvise** — the system waits for the next loop boundary, captures one full loop, analyzes the chord progression, and starts continuous melody/drum/bass improvisation aligned to Logic.
7. Press **Stop** when finished.

The UI shows live loop position (loop number, bar, beat) while synced, and displays detected key and chords after capture.

## Architecture

```
MIDI File / Live Input → MidiInputAdapter → LoopCapture
  → Quantizer → LoopAnalyzer → LoopContext
  → MelodyGenerator (RuleBasedMelodyGenerator)
  → MelodyClip → MidiScheduler / Export → Logic Pro
```

## CLI Commands

| Command    | Description                            |
| ---------- | -------------------------------------- |
| `analyze`  | Print tempo, key, chords, density      |
| `generate` | Create melody MIDI file                |
| `ports`    | List MIDI I/O ports                    |
| `play`     | Generate and send melody live          |
| `live`     | Capture loop and play generated melody |
| `serve`    | Start the live performance web UI      |

Use `--key` to match Logic's project key (e.g. `--key Am`, `--key D`). When set, the melody is strictly diatonic to that key — no chromatic passing tones.

## Testing

```bash
pytest
```

## Project Structure

```
the-logicians/
  src/logicians/
    models.py         # Core dataclasses
    quantize.py       # Symbolic quantization
    midi_io.py        # MIDI parsing and ports
    analysis.py       # Key/chord/density inference
    generator.py      # Rule-based melody generator
    export.py         # MIDI file export
    scheduler.py      # Live playback scheduling
    live_session.py   # Live session state machine
    server.py         # FastAPI web UI backend
    static/           # Live performance frontend
    cli.py            # Command-line interface
  tests/
  examples/
```
