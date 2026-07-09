"""Root–fifth bassline generation.

A deliberately simple, tight bass: for each chord, the root on beat 1 and the
fifth on beat 3 -- two notes a bar. This is the "start easy" version; the shape
is pattern-driven so it is cheap to build up from here (add onsets, bring in
approach tones via ``approach_tone``, lock to the kick, add probability, ...).

Separated concerns, each swappable on its own:

- ``bass_pattern``        -- RHYTHM + SHAPE: the per-bar ``(position, degree)``
                             slots. This is the main seam -- edit it to change
                             the line.
- ``degree_pitch_class``  -- PITCH: a chord degree (root / fifth) -> pitch class.
- ``clamp_to_register`` / ``nearest_in_register`` -- REGISTER.
- ``approach_tone``       -- a leading-tone building block, kept for the next
                             step (not used by this version).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .drums import DrumClip
from .models import ChordEvent, LoopContext, MelodyClip
from .cellular_automaton import CellularAutomatonBassGenerator

# --- Tunables ---------------------------------------------------------------

# Bass register, inclusive, as MIDI note numbers: E2 (40) .. E4 (64).
BASS_LOW = 40
BASS_HIGH = 64

# How far short of the next onset each note is cut (in beats) so the line reads
# tight/detached instead of legato. Larger = more space between notes.
DETACH = Fraction(1, 8)

# Floor on note length (beats).
MIN_DURATION = Fraction(1, 16)

# Chord degrees this version uses.
ROOT = "root"
FIFTH = "fifth"

# Per-degree velocity: the downbeat root is accented over the mid-bar fifth.
VELOCITY = {ROOT: 100, FIFTH: 86}
DEFAULT_VELOCITY = 90


# --- Data models (output format shared with the scheduler/exporter) ----------


@dataclass
class BassNote:
    pitch: int
    velocity: int
    bar: int
    position: Fraction
    duration: Fraction = field(default_factory=lambda: Fraction(1))


@dataclass
class BassClip:
    bars: int
    notes: list[BassNote] = field(default_factory=list)


@dataclass
class BassOptions:
    """Generation knobs.

    ``seed`` and ``variation`` are seams for the upcoming per-loop variation and
    probabilistic pitch work; this first version is deterministic and ignores
    both."""

    seed: int | None = None
    variation: float = 0.0


# --- RHYTHM + SHAPE ---------------------------------------------------------


def bass_pattern(beats_per_bar: int) -> list[tuple[Fraction, str]]:
    """The per-bar bass slots as ``(position_in_beats, chord degree)``.

    Root on beat 1 (position 0) and the fifth on beat 3 (mid-bar). This is the
    main seam: add slots/degrees here to build a busier line."""
    mid = Fraction(beats_per_bar, 2)
    return [(Fraction(0), ROOT), (mid, FIFTH)]


# --- REGISTER ---------------------------------------------------------------


def clamp_to_register(pitch: int, low: int = BASS_LOW, high: int = BASS_HIGH) -> int:
    """Fold ``pitch`` into ``[low, high]`` by whole octaves, then hard-clamp."""
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return max(low, min(high, pitch))


def nearest_in_register(
    pitch_class: int, near: int, low: int = BASS_LOW, high: int = BASS_HIGH
) -> int:
    """The octave of ``pitch_class`` within ``[low, high]`` closest to ``near``,
    for smooth voice leading (the line moves as little as possible)."""
    best: int | None = None
    for octave in range((low // 12) - 1, (high // 12) + 2):
        candidate = octave * 12 + (pitch_class % 12)
        if low <= candidate <= high and (best is None or abs(candidate - near) < abs(best - near)):
            best = candidate
    return best if best is not None else clamp_to_register(pitch_class, low, high)


# --- Harmony lookup ---------------------------------------------------------


def chord_at(context: LoopContext, bar: int, position: Fraction) -> ChordEvent | None:
    """The chord sounding at ``(bar, position)``.

    Chords are currently one-per-bar (onset at beat 0); this picks the latest
    chord whose onset is at or before the given time, so it keeps working if
    sub-bar chords are added later."""
    active: ChordEvent | None = None
    for chord in context.chords:
        if (chord.bar, chord.beat) <= (bar, position):
            if active is None or (chord.bar, chord.beat) >= (active.bar, active.beat):
                active = chord
    if active is not None:
        return active
    return context.chords[0] if context.chords else None


# --- PITCH ------------------------------------------------------------------


def degree_pitch_class(degree: str, chord: ChordEvent | None, context: LoopContext) -> int:
    """The pitch class for a chord degree. Root by default; a perfect fifth
    above the root for ``FIFTH`` (fine for the diatonic pop chords here)."""
    root = (chord.root if chord else context.key_root) % 12
    if degree == FIFTH:
        return (root + 7) % 12
    return root


def approach_tone(
    next_root_pc: int,
    scale: set[int],
    toward: int,
    low: int = BASS_LOW,
    high: int = BASS_HIGH,
) -> int:
    """A leading tone into ``next_root_pc`` as an in-register pitch. Kept as a
    building block for the next iteration; not used by the root–fifth version.

    Candidates are a diatonic step above and below the incoming root plus a
    chromatic half-step on each side; picks the pitch nearest the incoming root
    (diatonic wins ties), then voice-leads toward ``toward``."""
    scale = scale or set(range(12))

    candidate_pcs = {(next_root_pc + 1) % 12, (next_root_pc - 1) % 12}
    for direction in (1, -1):
        for step in (1, 2):
            pc = (next_root_pc + direction * step) % 12
            if pc in scale:
                candidate_pcs.add(pc)
                break

    def rank(pc: int) -> tuple[int, int, int, int]:
        semitones_to_root = min((pc - next_root_pc) % 12, (next_root_pc - pc) % 12)
        pitch = nearest_in_register(pc, toward, low, high)
        return (semitones_to_root, 0 if pc in scale else 1, abs(pitch - toward), pitch)

    best_pc = min(candidate_pcs, key=rank)
    return nearest_in_register(best_pc, toward, low, high)


# --- Generator --------------------------------------------------------------


class BassGenerator:

    _low = BASS_LOW
    _high = BASS_HIGH

    def generate(
        self,
        context,
        melody=None,
        drums=None,
        options=None,
    ):

        options = options or BassOptions()

        beats_per_bar = context.time_signature[0]
        pattern = bass_pattern(beats_per_bar)

        slots = [
            (bar, pos)
            for bar in range(1, context.bars + 1)
            for pos, _ in pattern
        ]

        ca = CellularAutomatonBassGenerator(
            len(slots),
            seed=options.seed,
        )

        notes = []
        prev_pitch = self._mid()

        for i, (bar, position) in enumerate(slots):

            chord = chord_at(context, bar, position)

            root_pc = degree_pitch_class(ROOT, chord, context)
            fifth_pc = degree_pitch_class(FIFTH, chord, context)

            state = int(ca.state[i])

            velocity = 90
            duration = Fraction(2)

            if state == 0:
                pitch_pc = root_pc

            elif state == 1:
                pitch_pc = root_pc
                velocity = 115

            elif state == 2:
                pitch_pc = root_pc

            elif state == 3:
                pitch = approach_tone(
                    root_pc,
                    set(range(12)),
                    prev_pitch,
                    self._low,
                    self._high,
                )
                notes.append(
                    BassNote(
                        pitch=pitch,
                        velocity=90,
                        bar=bar,
                        position=position,
                        duration=Fraction(1),
                    )
                )
                prev_pitch = pitch
                continue

            elif state == 4:
                pitch_pc = fifth_pc

            elif state == 5:
                pitch_pc = (root_pc - 1) % 12

            elif state == 6:
                pitch_pc = random.choice(
                    [
                        root_pc,
                        fifth_pc,
                        (root_pc + 9) % 12,
                    ]
                )

            elif state == 7:
                pitch_pc = root_pc
                duration = Fraction(1)

            pitch = nearest_in_register(
                pitch_pc,
                prev_pitch,
                self._low,
                self._high,
            )

            if state == 2:
                if pitch + 12 <= self._high:
                    pitch += 12
                elif pitch - 12 >= self._low:
                    pitch -= 12

            notes.append(
                BassNote(
                    pitch=pitch,
                    velocity=velocity,
                    bar=bar,
                    position=position,
                    duration=duration,
                )
            )

            prev_pitch = pitch

        ca.step()

        return BassClip(
            bars=context.bars,
            notes=notes,
        )