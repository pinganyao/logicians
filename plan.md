# Implementation Plan: First Working Prototype

## Project Goal

Build the first working prototype of an AI-assisted melody improviser for Logic Pro. This first version does **not** use a trained neural model yet. Instead, it implements the full architecture with a detailed rule-based melody generator as a placeholder for the future MMM-inspired model.

The prototype should prove that the system can:

1. Capture or ingest a completed 4-bar or 8-bar MIDI loop from Logic Pro.
2. Parse the loop into a structured symbolic representation.
3. Analyze key, chord progression, bass roots, rhythm density, and loop length.
4. Generate a natural, human-like melody using detailed musical rules.
5. Queue the generated melody and send it back to Logic Pro through MIDI.
6. Start playback at a musically clean entry point, preferably the next loop boundary.

The rule-based generator must be implemented behind a stable `MelodyGenerator` interface so it can later be replaced by a learned MMM-inspired Transformer without rewriting the Logic/MIDI/scheduling pipeline.

## Scope

### In Scope

- Python prototype application.
- MIDI input/output routing.
- MIDI loop capture from Logic or MIDI file ingestion for testing.
- Symbolic loop representation.
- Quantization to musical positions.
- Basic chord and key inference.
- Detailed rule-based melody generation.
- Melody scheduling and MIDI output.
- Export generated melody as MIDI file for debugging.
- Clear internal interfaces for future model replacement.

### Out Of Scope For This Prototype

- Training a neural model.
- Implementing MMM or Transformer inference.
- Audio analysis.
- Real-time note-by-note improvisation.
- Polyphonic melody generation.
- Full Logic Pro plugin development.
- Advanced GUI. A CLI or minimal local control interface is enough.

## Recommended Tech Stack

Use Python.

Suggested libraries:

- `mido` for MIDI parsing/writing.
- `python-rtmidi` as the MIDI backend for live ports.
- `pretty_midi` may be useful for MIDI file parsing, but avoid depending on it if `mido` is sufficient.
- `music21` may help with chord/key inference, but if dependency friction is high, implement a lightweight internal heuristic.
- `typer` or `argparse` for CLI.
- `dataclasses` or `pydantic` for structured loop objects. Prefer `dataclasses` for lower dependency burden.

The implementation should run in a local Python environment and not require a web server.

## Prototype Workflow

The intended user workflow:

1. Performer builds a 4-bar or 8-bar loop in Logic Pro.
2. Performer presses an "improvise" trigger. For the first prototype this can be:
   - a CLI command,
   - a MIDI CC message,
   - a MIDI note trigger,
   - or loading a saved/exported MIDI file.
3. Python snapshots or ingests the loop.
4. Python analyzes the loop.
5. Rule-based generator creates a melody phrase for the full loop length.
6. Python queues the melody.
7. Logic continues playing.
8. Generated melody starts at the next full loop boundary or next configured entry point.

For initial development, support MIDI file ingestion first. Live Logic routing can then use the same parser once the data arrives through MIDI ports.

## Architecture

Use this modular architecture:

```text
Logic Pro / MIDI File
        |
        v
MidiInputAdapter
        |
        v
LoopCapture
        |
        v
LoopParser / Quantizer
        |
        v
LoopAnalyzer
        |
        v
LoopContext
        |
        v
MelodyGenerator interface
        |
        v
RuleBasedMelodyGenerator
        |
        v
MelodyClip
        |
        v
MidiScheduler / MidiOutputAdapter
        |
        v
Logic Pro Melody VST Track
```

## Suggested Project Structure

```text
the-logicians/
  plan.md
  pyproject.toml
  README.md
  src/
    logicians/
      __init__.py
      cli.py
      midi_io.py
      models.py
      quantize.py
      analysis.py
      generator.py
      scheduler.py
      export.py
  tests/
    test_quantize.py
    test_analysis.py
    test_generator.py
  examples/
    input_loops/
    generated/
```

## Core Data Models

Implement these as dataclasses.

### NoteEvent

```python
@dataclass
class NoteEvent:
    track: str
    pitch: int
    velocity: int
    start: float
    duration: float
    bar: int
    position: Fraction
    duration_beats: Fraction
```

`start` and `duration` are in beats or seconds, but be consistent. Internally prefer beats for symbolic logic.

### ChordEvent

```python
@dataclass
class ChordEvent:
    bar: int
    beat: Fraction
    root: int
    quality: str
    pitch_classes: set[int]
    name: str
```

Pitch classes use `0=C`, `1=C#`, ..., `11=B`.

### LoopContext

```python
@dataclass
class LoopContext:
    tempo_bpm: float
    time_signature: tuple[int, int]
    bars: int
    ppq: int
    key_root: int
    key_mode: str
    scale_pitch_classes: set[int]
    chords: list[ChordEvent]
    tracks: dict[str, list[NoteEvent]]
    bass_roots_by_bar: dict[int, int]
    rhythm_density: float
```

### MelodyNote

```python
@dataclass
class MelodyNote:
    pitch: int
    velocity: int
    bar: int
    position: Fraction
    duration: Fraction
```

### MelodyClip

```python
@dataclass
class MelodyClip:
    bars: int
    notes: list[MelodyNote]
```

## MIDI Input Strategy

Implement two input paths.

### Path 1: MIDI File Ingestion

This should be the first working path.

CLI example:

```bash
python -m logicians.cli generate --input examples/input_loops/loop.mid --output examples/generated/melody.mid
```

Requirements:

- Parse a MIDI file exported from Logic.
- Extract tempo and time signature where available.
- Read note events.
- Assign tracks based on MIDI track names if available.
- If track names are missing, infer rough roles:
  - channel 10 or percussion notes: `drums`
  - low-register monophonic material: `bass`
  - sustained/polyphonic mid-register material: `chords`
  - high-register lead-like material: `melody_reference` if present
- For the prototype, allow manual track mapping through CLI options.

### Path 2: Live MIDI Capture

Implement after file ingestion works.

Requirements:

- Open configured MIDI input port.
- Listen to note-on/note-off events from Logic.
- Capture events into a buffer for a configured loop length.
- On trigger, freeze the current loop buffer and pass it to the parser.
- Keep Logic playing; do not block the transport.

For live capture, it is acceptable for the first version to require user-provided:

- tempo,
- loop length in bars,
- time signature,
- MIDI input port,
- MIDI output port.

## Quantization

Quantize incoming notes to symbolic positions.

Support at least:

- whole note,
- minim / half note,
- crotchet / quarter note,
- quaver / eighth note,
- semiquaver / sixteenth note,
- demisemiquaver / 32nd note,
- eighth-note triplets,
- quarter-note triplets,
- sixteenth-note triplets.

Use `fractions.Fraction` internally.

Suggested duration grid in beats:

```python
ALLOWED_DURATIONS = [
    Fraction(4, 1),    # whole note
    Fraction(3, 1),    # dotted half
    Fraction(2, 1),    # half
    Fraction(3, 2),    # dotted quarter
    Fraction(1, 1),    # quarter
    Fraction(3, 4),    # dotted eighth
    Fraction(1, 2),    # eighth
    Fraction(1, 3),    # quarter triplet subdivision
    Fraction(3, 8),    # dotted sixteenth
    Fraction(1, 4),    # sixteenth
    Fraction(1, 6),    # eighth-note triplet subdivision
    Fraction(1, 8),    # 32nd
    Fraction(1, 12),   # sixteenth triplet subdivision
]
```

Suggested position grid:

```python
POSITION_GRID = sorted(set([
    Fraction(n, 8) for n in range(0, 32)    # 32nd grid over 4 beats
] + [
    Fraction(n, 6) for n in range(0, 24)    # triplet grid over 4 beats
]))
```

Quantization rules:

- Snap note starts to the nearest allowed position.
- Snap durations to the nearest allowed duration.
- Preserve very short notes only if they are musically intentional; otherwise merge or discard notes shorter than `1/12` beat.
- Avoid overlapping generated melody notes unless legato behavior is explicitly enabled.

## Loop Analysis

### Key Inference

Implement a lightweight key inference heuristic:

1. Collect pitch-class histogram from chord and bass tracks.
2. Weight bass notes strongly.
3. Weight long chord/pad notes strongly.
4. Weight melody/reference notes lightly if present.
5. Compare against major/minor scale templates.
6. Choose the highest-scoring key.

Major template:

```text
0, 2, 4, 5, 7, 9, 11
```

Natural minor template:

```text
0, 2, 3, 5, 7, 8, 10
```

Also support harmonic minor as an optional flavor:

```text
0, 2, 3, 5, 7, 8, 11
```

### Chord Inference

Infer one chord per bar initially. Later support two chords per bar.

Rules:

1. For each bar, collect simultaneous chord/pad notes and bass notes.
2. Use bass pitch class as preferred root candidate.
3. Score candidate chord qualities against observed pitch classes.
4. Prefer simple triads/sevenths unless evidence supports extensions.

Supported chord qualities:

```text
major:      0, 4, 7
minor:      0, 3, 7
diminished: 0, 3, 6
sus2:       0, 2, 7
sus4:       0, 5, 7
dom7:       0, 4, 7, 10
maj7:       0, 4, 7, 11
min7:       0, 3, 7, 10
minMaj7:    0, 3, 7, 11
```

If chord inference is uncertain, fall back to:

- bass root + key-compatible triad,
- or key tonic chord.

### Rhythm Density

Compute density from drums, bass, and chords:

- notes per bar,
- average inter-onset interval,
- kick/snare density if drums are present,
- syncopation score based on offbeat events.

Use density to control melody busyness.

## Melody Generator Interface

Create an abstract base class or protocol:

```python
class MelodyGenerator(Protocol):
    def generate(self, context: LoopContext, options: GenerationOptions) -> MelodyClip:
        ...
```

### GenerationOptions

```python
@dataclass
class GenerationOptions:
    seed: int | None
    register_low: int = 60
    register_high: int = 84
    density: str = "medium"  # sparse, medium, busy
    variation: float = 0.5
    chord_tone_weight: float = 0.65
    passing_tone_weight: float = 0.25
    chromatic_weight: float = 0.10
    rhythmic_complexity: float = 0.5
    phrase_length_bars: int | None = None
    start_on_loop_boundary: bool = True
```

## Rule-Based Melody Generator

The rule-based generator should be musically detailed enough to produce plausible melody phrases over electronic loops.

### High-Level Generation Steps

1. Choose phrase structure.
2. Choose rhythmic skeleton.
3. Choose target notes at important metrical positions.
4. Fill connecting notes using melodic motion rules.
5. Add rests and syncopation.
6. Apply contour shaping.
7. Apply humanization.
8. Validate and repair the melody.

## Musical Rules

### 1. Phrase Structure

For 4-bar loops:

- Default phrase shape: `A A' B resolution`.
- Bar 1: introduce motif.
- Bar 2: repeat motif with small variation.
- Bar 3: contrast, higher register or altered rhythm.
- Bar 4: resolution or pickup into next loop.

For 8-bar loops:

- Default phrase shape: `A A' B B' C C' D resolution`.
- Bars 1-2: establish motif.
- Bars 3-4: develop motif.
- Bars 5-6: increase contrast or intensity.
- Bars 7-8: resolve or create turnaround.

Represent phrase sections internally so future model conditioning can reuse this.

### 2. Entry Rules

Generated melody should not always start immediately.

Options:

- start on bar 1 beat 1 for strong lead lines,
- start on bar 1 beat 2 for more relaxed entrance,
- start with pickup on final eighth or sixteenth before bar 1,
- start after a rest of half a bar for suspense.

Default:

- medium density: enter between beat 1 and beat 2 of bar 1.
- sparse density: enter after a rest.
- busy density: allow pickup.

### 3. Chord And Key Coherence

On strong beats, prefer chord tones.

Strong positions:

- beat 1,
- beat 3,
- long note starts,
- phrase endings,
- notes with high velocity.

Weak positions:

- offbeat eighths,
- sixteenths,
- triplet passing positions.

Pitch selection probabilities:

Strong beat default:

```text
chord tone: 75%
scale non-chord tone: 20%
chromatic approach tone: 5%
```

Weak beat default:

```text
chord tone: 45%
scale non-chord tone: 40%
chromatic passing/neighbor tone: 15%
```

Phrase ending:

```text
chord tone: 90%
scale non-chord tone: 10%
chromatic: 0%
```

Avoid landing phrase endings on avoid tones unless intentionally configured.

For major keys, avoid overemphasizing scale degree 4 over major chords unless resolving.

For minor keys, allow scale degree 6 or raised 7 depending on harmonic context.

### 4. Melodic Motion

Balance conjunct and disjunct motion.

Definitions:

- conjunct: stepwise movement, 1-2 semitones.
- small skip: 3-5 semitones.
- large leap: 6-12 semitones.
- very large leap: more than 12 semitones.

Default interval probabilities:

```text
repeat same pitch: 8%
step: 45%
small skip: 30%
large leap: 14%
very large leap: 3%
```

Rules:

- After a large leap, prefer stepwise motion in the opposite direction.
- Avoid more than two large leaps in a row.
- Avoid repeated notes more than 3 times unless the rhythm is intentionally hook-like.
- Prefer mostly stepwise motion inside fast passages.
- Allow larger leaps at phrase starts, phrase peaks, and bar transitions.
- Keep melody within register bounds.
- If a candidate pitch exceeds register, fold it by octave if musically valid.

### 5. Contour Rules

Each phrase should have a contour type:

```text
arch: rises then falls
inverted_arch: falls then rises
ascending: gradual rise
descending: gradual fall
static_hook: narrow range with rhythmic focus
call_response: bar 1 call, bar 2 response
```

Default:

- choose `arch` or `call_response` most often.

Rules:

- Select one phrase peak.
- Put phrase peak between 60% and 80% through the phrase unless generating a pickup/build.
- Do not make every bar peak at the same beat.
- Last bar should usually settle lower or on a stable chord tone unless `variation` is high.

### 6. Rhythmic Rules

Support note durations:

- whole notes,
- dotted half notes,
- half notes,
- dotted quarter notes,
- quarter notes / crotchets,
- dotted eighth notes,
- eighth notes / quavers,
- quarter-note triplets,
- sixteenth notes / semiquavers,
- eighth-note triplets,
- 32nd notes,
- sixteenth-note triplets.

Use density presets.

Sparse:

```text
rests: high
common durations: quarter, dotted quarter, half
fast notes: rare
notes per bar: 2-5
```

Medium:

```text
rests: medium
common durations: eighth, quarter, dotted eighth, sixteenth
fast notes: occasional
notes per bar: 5-9
```

Busy:

```text
rests: low
common durations: eighth, sixteenth, triplet eighth
fast notes: allowed
notes per bar: 8-16
```

Rules:

- Avoid filling every subdivision unless density is busy.
- Use rests as phrasing, not only gaps.
- Prefer longer notes on phrase endings.
- Use short notes as pickups, passing tones, ornaments, or anticipation.
- Avoid long strings of 32nd notes; max 4 consecutive 32nd notes by default.
- Triplets should appear in short groups, not mixed randomly with straight 16ths unless configured.
- Syncopation should relate to the drum/bass groove.

### 7. Groove Interaction

The melody should react to the loop.

Rules:

- If drums are dense, melody can be sparser.
- If drums are sparse, melody can be more rhythmically active.
- If bass has a strong repeated rhythm, melody may either:
  - answer in gaps,
  - double selected rhythmic accents,
  - or contrast with sustained notes.
- Avoid melody attacks that constantly collide with kick/snare accents unless the style is intentionally aggressive.
- Prefer phrase starts near important groove points.

Implement a simple accent map over the loop:

```python
accent_score[bar][position] = weighted sum of drum, bass, chord onsets
```

Use this to decide note starts.

### 8. Motif Rules

Generate a short motif in bar 1.

A motif includes:

- rhythm pattern,
- interval pattern,
- approximate contour,
- anchor pitch.

Reuse motif with transformations:

- transpose to current chord,
- shift rhythm by eighth or sixteenth,
- invert small intervals,
- truncate ending,
- extend with pickup,
- change final note to fit chord resolution.

Rules:

- Bar 2 should usually relate clearly to bar 1.
- Later bars should contain recognizable variation, not totally unrelated material.
- Keep one or two signature rhythmic cells across the loop.

### 9. Passing Tones, Neighbor Tones, And Chromaticism

Allowed non-chord tones:

- diatonic passing tone between chord tones,
- upper/lower neighbor tone returning to chord tone,
- anticipation of next chord tone,
- suspension from previous chord,
- chromatic approach from semitone below or above.

Rules:

- Chromatic notes should generally be short.
- Chromatic notes should resolve by step.
- Do not place chromatic notes on strong phrase endings.
- Do not use many chromatic notes in sparse mode.
- Passing tones are most appropriate on weak subdivisions.

### 10. Note Duration And Articulation

Rules:

- Long notes should usually be chord tones.
- Fast notes may be scale or chromatic passing tones.
- Phrase-ending notes should be longer than nearby passing notes.
- Avoid all notes having the same duration.
- Avoid overly fragmented melodies unless busy mode is selected.
- Insert small rests between some notes to avoid mechanical legato.

Articulation:

- Default generated MIDI note length should be 85-95% of notated duration.
- For staccato sections, use 45-65%.
- For legato sections, use 95-105%, but prevent overlapping note-off bugs.

### 11. Velocity And Humanization

Velocity rules:

- Strong beats get slightly higher velocity.
- Phrase peaks get higher velocity.
- Passing tones get lower velocity.
- Repeated notes should have small velocity variation.
- Phrase endings may decrescendo unless build-up is configured.

Suggested range:

```text
soft: 55-75
medium: 70-100
strong: 90-115
```

Humanization:

- Apply small timing offsets only at MIDI output, not in symbolic representation.
- Default timing jitter: 0-15 ms.
- Keep downbeats and loop boundaries tight.
- Velocity jitter: +/- 3 to 8.
- Do not humanize so much that sync with Logic feels broken.

### 12. Validation And Repair

After generating, run validation:

- All notes fit inside loop length.
- No negative durations.
- No overlapping monophonic melody notes unless legato enabled.
- Pitches are inside register.
- Phrase endings land on stable tones.
- Maximum consecutive chromatic notes is not exceeded.
- Maximum consecutive large leaps is not exceeded.
- 32nd-note runs are limited.
- Triplet groups are coherent.

If validation fails, repair locally rather than regenerate everything.

## MIDI Output

The generated `MelodyClip` should be convertible to:

1. MIDI file.
2. Live MIDI output to Logic.

For MIDI file output:

- Create one melody track.
- Preserve tempo and time signature.
- Use configured MIDI channel.
- Write note-on/note-off events.

For live output:

- Wait until selected entry point.
- Send note-on/note-off events according to the generated phrase schedule.
- Continue looping the generated melody if configured.
- Provide an option to regenerate between loop cycles.

## Scheduler

The scheduler must support:

- `start_at_next_loop`: default.
- `start_at_next_bar`: useful for testing.
- `start_immediately`: debugging only.

For first prototype, if reliable Logic transport position is not available, allow manual synchronization:

```bash
python -m logicians.cli play --midi-output "IAC Driver Bus 1" --tempo 128 --bars 4 --count-in 1
```

Live transport support can be improved later.

## CLI Commands

Implement at least:

```bash
python -m logicians.cli analyze --input loop.mid
```

Print:

- tempo,
- time signature,
- estimated loop length,
- detected tracks,
- estimated key,
- chord progression,
- density.

```bash
python -m logicians.cli generate --input loop.mid --output melody.mid --seed 1
```

Generates melody and saves MIDI.

```bash
python -m logicians.cli ports
```

Lists available MIDI input/output ports.

```bash
python -m logicians.cli play --input loop.mid --midi-output "PORT NAME"
```

Generates and sends melody to MIDI output.

Optional:

```bash
python -m logicians.cli live --midi-input "PORT IN" --midi-output "PORT OUT" --tempo 128 --bars 4
```

Captures live loop and outputs generated melody.

## Testing Requirements

Add focused tests for:

- duration quantization,
- position quantization,
- key inference on simple examples,
- chord inference on triads/sevenths,
- melody notes stay within key/chord constraints,
- generated phrase length equals loop length,
- no overlapping monophonic notes,
- deterministic output when seed is fixed,
- MIDI export creates a valid file.

## Development Milestones

### Milestone 1: MIDI File Prototype

- Parse MIDI file.
- Quantize notes.
- Build `LoopContext`.
- Print analysis.
- Generate rule-based melody.
- Export melody MIDI file.

Acceptance:

- Running `generate` on an example MIDI loop creates `melody.mid`.
- The generated melody has correct loop length and no invalid notes.

### Milestone 2: Better Musical Rules

- Add motif generation.
- Add phrase structure.
- Add contour rules.
- Add rhythmic density presets.
- Add chord-tone/passing-tone/chromatic logic.
- Add validation/repair.

Acceptance:

- Same input and seed produce deterministic melody.
- Different seeds produce musically varied melodies.
- Output sounds coherent over a simple chord progression.

### Milestone 3: Live MIDI Output

- List MIDI ports.
- Send generated melody to Logic via virtual MIDI port.
- Support count-in or next-loop scheduling.

Acceptance:

- Logic receives generated MIDI on a software instrument track.
- Melody starts at a predictable musical boundary.

### Milestone 4: Live Capture

- Capture MIDI events from Logic.
- Freeze loop on trigger.
- Analyze captured loop.
- Generate and queue melody.

Acceptance:

- Performer can build loop in Logic, trigger generation, and hear generated melody enter while Logic keeps looping.

## Future Model Replacement

The rule-based generator should be easy to replace with:

```python
class TransformerMelodyGenerator:
    def generate(self, context: LoopContext, options: GenerationOptions) -> MelodyClip:
        tokens = tokenizer.encode_context(context, options)
        generated = model.generate(tokens)
        return tokenizer.decode_melody(generated)
```

Therefore:

- Keep `LoopContext` stable.
- Keep `MelodyClip` stable.
- Avoid leaking rule-based-specific structures into the rest of the pipeline.
- Make the generator swappable through configuration.

## Implementation Priorities

Prioritize working end-to-end behavior over perfect music theory.

Order of importance:

1. Correct timing and loop length.
2. Stable MIDI parsing and exporting.
3. Melody fits key/chords.
4. Melody has phrase/motif coherence.
5. Humanization and expressive detail.
6. Live MIDI capture.
7. Advanced harmonic inference.

## Notes For The Implementing Agent

- Do not start by building a GUI.
- Start with MIDI file ingestion because it is deterministic and testable.
- Keep all symbolic timing in beats using `Fraction`.
- Keep MIDI tick conversion isolated in one module.
- Use a seeded random generator object, not global randomness.
- Make outputs reproducible for tests.
- Add example MIDI generation if no external loop files exist.
- Include clear README instructions for connecting Logic via IAC Bus on macOS.
- The codebase may currently contain only planning artifacts; scaffold the project if needed.
