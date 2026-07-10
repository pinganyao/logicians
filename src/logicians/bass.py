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

import random
from dataclasses import dataclass, field, replace
from fractions import Fraction

from .cellular_automaton import CellularAutomatonBassGenerator
from .drums import DrumClip
from .models import ChordEvent, LoopContext, MelodyClip

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

# How many cellular-automaton generations to evolve before reading its state to
# drive the per-note variations. More generations = the local root/fifth seed
# spreads further, giving busier, less predictable lines.
CA_GENERATIONS = 10


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

    ``seed`` makes a run reproducible: it seeds the cellular automaton (and the
    per-note random choices it drives), so the same seed yields the same
    variation and different seeds yield different ones. ``variation`` is kept as
    a reserved seam and is currently unused."""

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


# --- Variation (cellular automaton) -----------------------------------------


class BasslineEnhancer:
    """Turns the seed line into a variation, one automaton cell per note.

    Each note is transformed by the automaton's state at its index (see
    ``CellularAutomatonBassGenerator`` for the state meanings):

    - 0 keep, 1 accent (louder root)
    - 2 octave, 3 neighbour, 4 fifth, 5 chromatic, 6 melodic leap -- pitch moves
    - 7 rhythmic break -- split the note in two, rest out its first half, or
      repeat it (second hit softer)

    Pitch moves are folded back into the bass register so the line stays in
    range. Rhythmic breaks keep the note's onset and split only *within* its own
    duration, so bars never overflow.
    """

    def __init__(
        self,
        notes: list[BassNote],
        automaton: CellularAutomatonBassGenerator,
        low: int = BASS_LOW,
        high: int = BASS_HIGH,
    ):
        self.notes = notes
        self.automaton = automaton
        self.low = low
        self.high = high

    def _shift(self, pitch: int, amount: int) -> int:
        return clamp_to_register(pitch + amount, self.low, self.high)

    def enhance(self) -> list[BassNote]:
        out: list[BassNote] = []
        for index, base in enumerate(self.notes):
            state = int(self.automaton.state[index % self.automaton.length])
            note = replace(base)

            if state == 0:  # keep
                out.append(note)
            elif state == 1:  # accent
                note.velocity = 120
                out.append(note)
            elif state == 2:  # octave jump
                note.pitch = self._shift(note.pitch, random.choice([-12, 12]))
                out.append(note)
            elif state == 3:  # neighbour tone
                note.pitch = self._shift(note.pitch, random.choice([-2, -1, 1, 2]))
                out.append(note)
            elif state == 4:  # fifth movement
                note.pitch = self._shift(note.pitch, random.choice([-7, 7]))
                out.append(note)
            elif state == 5:  # chromatic approach
                note.pitch = self._shift(note.pitch, random.choice([-1, 1]))
                out.append(note)
            elif state == 6:  # larger melodic leap
                note.pitch = self._shift(note.pitch, random.choice([-5, -4, -3, 3, 4, 5]))
                out.append(note)
            elif state == 7:  # rhythmic variation
                out.extend(self._rhythmic(note))
        return out

    def _rhythmic(self, note: BassNote) -> list[BassNote]:
        variation = random.choice(["split", "rest", "repeat"])
        half = note.duration / 2

        if variation == "split":
            first = replace(note, duration=half)
            second = replace(
                note,
                position=note.position + half,
                duration=half,
                pitch=self._shift(note.pitch, random.choice([-3, 2, 4, 7])),
            )
            return [first, second]

        if variation == "rest":
            # First half is silence (a gap); only the second half sounds.
            return [replace(note, position=note.position + half, duration=half)]

        # repeat: two hits, the second one softer.
        first = replace(note, duration=half)
        second = replace(
            note,
            position=note.position + half,
            duration=half,
            velocity=max(30, note.velocity - 20),
        )
        return [first, second]


# --- Generator --------------------------------------------------------------


class BassGenerator:
    """Cellular-automaton bassline.

    A tight root–fifth line is used as the *seed* shape; a cellular automaton is
    then evolved over it and its per-cell state drives a per-note variation
    (accent, octave/fifth/neighbour/chromatic/leap moves, and split/rest/repeat
    rhythmic breaks). ``generate(context, melody, drums, options)`` returns a
    single ``BassClip`` variation -- reproducible when ``options.seed`` is set,
    fresh each call otherwise.

    The ``melody`` and ``drums`` arguments are accepted for pipeline
    compatibility (and as future seams) but are unused by this version.
    """

    _low: int = BASS_LOW
    _high: int = BASS_HIGH

    def generate(
        self,
        context: LoopContext,
        melody: MelodyClip | None = None,
        drums: DrumClip | None = None,
        options: BassOptions | None = None,
    ) -> BassClip:
        options = options or BassOptions()

        base = self._base_notes(context)
        automaton = CellularAutomatonBassGenerator(max(len(base), 1), seed=options.seed)
        for _ in range(CA_GENERATIONS):
            automaton.step()

        notes = BasslineEnhancer(base, automaton, self._low, self._high).enhance()
        return BassClip(bars=context.bars, notes=notes)

    # -- base line ------------------------------------------------------------

    def _base_notes(self, context: LoopContext) -> list[BassNote]:
        """The seed root–fifth line the automaton varies: root on beat 1, fifth
        on beat 3, per chord, voice-led and kept in the bass register."""
        beats_per_bar = context.time_signature[0]
        pattern = bass_pattern(beats_per_bar)  # RHYTHM + SHAPE
        loop_beats = context.bars * beats_per_bar

        # Every slot across the loop, in time order.
        slots = [
            (bar, position, degree)
            for bar in range(1, context.bars + 1)
            for position, degree in pattern
        ]
        slots.sort(key=lambda s: (s[0], float(s[1])))

        notes: list[BassNote] = []
        prev_pitch = self._mid()
        for index, (bar, position, degree) in enumerate(slots):
            chord = chord_at(context, bar, position)
            pitch_class = degree_pitch_class(degree, chord, context)  # PITCH
            pitch = clamp_to_register(  # REGISTER
                nearest_in_register(pitch_class, prev_pitch, self._low, self._high),
                self._low,
                self._high,
            )
            duration = self._duration(index, slots, beats_per_bar, loop_beats)
            velocity = VELOCITY.get(degree, DEFAULT_VELOCITY)
            notes.append(BassNote(pitch, velocity, bar, position, duration))
            prev_pitch = pitch
        return notes


    # -- helpers --------------------------------------------------------------

    def _mid(self) -> int:
        return (self._low + self._high) // 2

    def _duration(
        self,
        index: int,
        slots: list[tuple[int, Fraction, str]],
        beats_per_bar: int,
        loop_beats: int,
    ) -> Fraction:
        """Extend to just before the next onset (or the loop end for the last
        note), detached by ``DETACH`` and never overlapping the next onset."""
        bar, position = slots[index][0], slots[index][1]
        this_beat = (bar - 1) * beats_per_bar + position
        if index + 1 < len(slots):
            next_bar, next_pos = slots[index + 1][0], slots[index + 1][1]
            next_beat = (next_bar - 1) * beats_per_bar + next_pos
        else:
            next_beat = loop_beats
        gap = next_beat - this_beat
        return min(gap, max(MIN_DURATION, gap - DETACH))
