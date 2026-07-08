"""Reactive bass generation.

Generates a bass line from a bass part the player routes in on its own channel
(Logic channel 3 -> mido channel 2, the "bass" role), the harmonic context, and
the generated melody. Self-contained and deterministic for a fixed seed.

The line is built to *carry the rhythmic weight* -- it is meant to hold a song up
together with the melody, drums or not. Two ideas drive it:

1. GROOVE TEMPLATES give the rhythm. A template is a per-bar pattern of onsets
   (with accents) that REPEATS every bar -- repetition is what makes a groove.
   `--bass-variation` selects a busier template as it rises; the rhythm itself is
   deterministic, so it locks in instead of wandering.

2. A MARKOV CHAIN gives the pitches. States are chord-relative degrees
   (root / third / fifth / seventh / octave). Strong "anchor" slots always play
   the root; other slots sample the next degree from an idiom transition table,
   restricted to the current chord's tones, then map to the octave nearest the
   previous note (voice leading). The rhythm stays fixed while the pitch contour
   evolves loop to loop.

Harmonic anchoring, register (kept in the octave the player used), kick-velocity
accenting, monophony, and melody deference are all still enforced.

`--bass-variation` map:
- 0.0  -> the captured bass verbatim.
- >0   -> pick a groove template by intensity; Markov-sampled pitches with root
          on every anchor; chromatic/scale approach into chord changes.
Templates, sparse -> busy: whole, half, root-five, four-on-floor, syncopated,
eighths.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction

from .drums import KICK_PITCHES, DrumClip
from .models import ChordEvent, LoopContext, MelodyClip

# Bass register: E1 (28) .. E3 (52). Used only when no bass is routed in.
BASS_LOW = 28
BASS_HIGH = 52

# `variation` at/above which a walking approach leads into a chord change.
APPROACH_THRESHOLD = 0.4

MIN_DURATION = Fraction(1, 8)

# --- Markov pitch model -----------------------------------------------------

# Chord-relative degree tokens used as Markov states.
ROOT, THIRD, FIFTH, SEVENTH, OCTAVE = "R", "3", "5", "7", "8"

# Hand-authored first-order transition weights encoding common bass idioms
# (root -> fifth/octave, fifth -> root, walk back to root, etc.). A corpus
# trainer could replace these counts via MarkovBassModel.from_sequences().
BASS_TRANSITIONS: dict[str, dict[str, int]] = {
    ROOT:    {ROOT: 2, FIFTH: 4, OCTAVE: 3, THIRD: 2, SEVENTH: 1},
    THIRD:   {ROOT: 4, FIFTH: 3, THIRD: 1, SEVENTH: 1},
    FIFTH:   {ROOT: 4, OCTAVE: 2, THIRD: 2, FIFTH: 1, SEVENTH: 1},
    SEVENTH: {ROOT: 2, FIFTH: 2, THIRD: 2, OCTAVE: 1},
    OCTAVE:  {FIFTH: 4, ROOT: 3, THIRD: 1},
}


class MarkovBassModel:
    """First-order Markov model over chord-relative degree tokens."""

    def __init__(self, transitions: dict[str, dict[str, int]]):
        self.transitions = transitions

    @classmethod
    def default(cls) -> "MarkovBassModel":
        return cls(BASS_TRANSITIONS)

    @classmethod
    def from_sequences(cls, sequences) -> "MarkovBassModel":
        """Train transition counts from an iterable of degree-token sequences
        (e.g. extracted from a MIDI bass corpus)."""
        counts: dict[str, dict[str, int]] = {}
        for seq in sequences:
            for current, nxt in zip(seq, seq[1:]):
                counts.setdefault(current, {}).setdefault(nxt, 0)
                counts[current][nxt] += 1
        return cls(counts)

    def next_degree(self, current: str, allowed: set[str], rng: random.Random) -> str:
        """Sample the next degree, restricted to degrees the chord offers."""
        weights = self.transitions.get(current, {})
        pool = [(deg, w) for deg, w in weights.items() if deg in allowed]
        if not pool:
            pool = [(deg, 1) for deg in sorted(allowed)] or [(ROOT, 1)]
        total = sum(w for _, w in pool)
        threshold = rng.random() * total
        acc = 0
        for deg, w in pool:
            acc += w
            if threshold <= acc:
                return deg
        return pool[-1][0]


# --- Groove templates -------------------------------------------------------


@dataclass
class GrooveSlot:
    position: Fraction
    accent: float      # 0..1, drives velocity (downbeats loudest = weight)
    anchor: bool       # anchors always play the chord root


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
    seed: int | None = None
    # 0.0 = the input bass untouched; higher = busier groove template.
    variation: float = 0.3
    channel: int = 2  # Logic channel 3 (0-indexed)
    # Optional GMD-trained style whose kick rhythm becomes the bass groove.
    style: str | None = None


def _clamp_register(pitch: int, low: int = BASS_LOW, high: int = BASS_HIGH) -> int:
    """Fold a pitch into the [low, high] register by octaves, then clamp."""
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return max(low, min(high, pitch))


def _nearest_pitch(pitch_class: int, near: int, low: int = BASS_LOW, high: int = BASS_HIGH) -> int:
    """The octave of `pitch_class` closest to `near`, within [low, high]."""
    best: int | None = None
    for octave in range((low // 12) - 1, (high // 12) + 2):
        candidate = octave * 12 + (pitch_class % 12)
        if low <= candidate <= high:
            if best is None or abs(candidate - near) < abs(best - near):
                best = candidate
    return best if best is not None else _clamp_register(pitch_class % 12, low, high)


class RuleBasedBassGenerator:
    """Groove-template + Markov bass that follows the harmony and melody."""

    _low: int = BASS_LOW
    _high: int = BASS_HIGH

    def __init__(self, model: MarkovBassModel | None = None):
        self.model = model or MarkovBassModel.default()

    def generate(
        self,
        context: LoopContext,
        melody: MelodyClip,
        drums: DrumClip | None = None,
        options: BassOptions | None = None,
    ) -> BassClip:
        options = options or BassOptions()
        rng = random.Random(options.seed)
        beats_per_bar = context.time_signature[0]
        bars = context.bars
        v = max(0.0, min(1.0, options.variation))

        original = context.tracks.get("bass", [])
        self._low, self._high = self._register(original)

        # variation 0 -> the captured bass verbatim.
        if v <= 0:
            return BassClip(bars=bars, notes=self._sorted(self._base_notes(original, bars, beats_per_bar, context)))

        template = None
        if options.style:
            from .groove import style_bass_template
            template = style_bass_template(options.style, beats_per_bar)
        if template is None:
            template = self._select_groove(v, beats_per_bar)
        kick_velocity = self._kick_velocity_map(drums, context)

        notes: list[BassNote] = []
        prev_pitch = self._mid()
        prev_degree = ROOT
        for bar in range(1, bars + 1):
            chord = self._chord_for_bar(context, bar)
            tones = self._chord_tones(chord) if chord else None  # once per bar
            allowed = self._allowed_degrees(tones)
            next_bar = bar + 1 if bar < bars else 1
            upcoming = self._chord_for_bar(context, next_bar)
            changes = bool(chord and upcoming and chord.root != upcoming.root)

            for i, slot in enumerate(template):
                is_last = i == len(template) - 1
                if changes and is_last and v >= APPROACH_THRESHOLD:
                    # Walking approach into the next chord's root.
                    target = _nearest_pitch(upcoming.root, prev_pitch, self._low, self._high)
                    step = target - 1 if rng.random() < 0.5 else self._scale_step_below(target, context)
                    pitch, degree = _clamp_register(step, self._low, self._high), ROOT
                elif slot.anchor or chord is None:
                    degree = ROOT
                    pitch = _nearest_pitch(self._degree_pc(ROOT, tones, context), prev_pitch, self._low, self._high)
                else:
                    degree = self.model.next_degree(prev_degree, allowed, rng)
                    near = prev_pitch + 12 if degree == OCTAVE else prev_pitch
                    pitch = _nearest_pitch(self._degree_pc(degree, tones, context), near, self._low, self._high)

                velocity = self._slot_velocity(slot, bar, kick_velocity)
                notes.append(BassNote(pitch, velocity, bar, slot.position, MIN_DURATION))
                prev_pitch, prev_degree = pitch, degree

        notes = self._defer_to_melody(notes, melody, context)
        notes = self._finalize(notes, bars, beats_per_bar)
        notes = self._articulate(notes, beats_per_bar, v)
        return BassClip(bars=bars, notes=notes)

    # -- groove selection -----------------------------------------------------

    def _grooves(self, beats_per_bar: int) -> list[list[GrooveSlot]]:
        """Templates ordered sparse -> busy. Each repeats every bar."""
        def s(pos, accent, anchor) -> GrooveSlot:
            return GrooveSlot(Fraction(pos), accent, anchor)

        mid = beats_per_bar // 2
        whole = [s(0, 1.0, True)]
        half = [s(0, 1.0, True), s(mid, 0.82, True)]
        root_five = [s(0, 1.0, True), s(mid, 0.82, True), s(Fraction(beats_per_bar) - Fraction(1, 2), 0.7, False)]
        four = [s(i, 1.0 if i == 0 else 0.74, i == 0) for i in range(beats_per_bar)]
        syncopated = [s(0, 1.0, True), s(Fraction(1, 2), 0.64, False), s(mid, 0.82, True), s(Fraction(beats_per_bar) - Fraction(1, 2), 0.7, False)]
        eighths = [s(Fraction(i, 2), 1.0 if i == 0 else 0.66, i == 0) for i in range(beats_per_bar * 2)]
        return [whole, half, root_five, four, syncopated, eighths]

    def _select_groove(self, variation: float, beats_per_bar: int) -> list[GrooveSlot]:
        grooves = self._grooves(beats_per_bar)
        index = min(len(grooves) - 1, int(variation * len(grooves)))
        return grooves[index]

    def _slot_velocity(self, slot: GrooveSlot, bar: int, kick_velocity: dict[int, dict[Fraction, int]]) -> int:
        velocity = int(60 + 55 * slot.accent)
        aligned = kick_velocity.get(bar, {}).get(slot.position)
        if aligned is not None:  # punch with the kick where they coincide
            velocity = max(velocity, aligned)
        return max(1, min(127, velocity))

    # -- Markov degree helpers ------------------------------------------------

    def _allowed_degrees(self, tones: tuple[int, int, int | None, int | None] | None) -> set[str]:
        allowed = {ROOT, FIFTH, OCTAVE}
        if tones is None:
            return allowed
        _, _, third, seventh = tones
        if third is not None:
            allowed.add(THIRD)
        if seventh is not None:
            allowed.add(SEVENTH)
        return allowed

    def _degree_pc(self, degree: str, tones: tuple[int, int, int | None, int | None] | None, context: LoopContext) -> int:
        if tones is None:
            return context.key_root
        root, fifth, third, seventh = tones
        return {
            ROOT: root,
            OCTAVE: root,
            FIFTH: fifth,
            THIRD: third if third is not None else root,
            SEVENTH: seventh if seventh is not None else fifth,
        }[degree]

    # -- harmony helpers ------------------------------------------------------

    def _chord_for_bar(self, context: LoopContext, bar: int) -> ChordEvent | None:
        for chord in context.chords:
            if chord.bar == bar:
                return chord
        if context.chords:
            idx = min(max(bar, 1), len(context.chords)) - 1
            return context.chords[idx]
        return None

    def _chord_tones(self, chord: ChordEvent) -> tuple[int, int, int | None, int | None]:
        """Return (root, fifth, third, seventh) as pitch classes; third and
        seventh are None when the chord doesn't contain them."""
        root = chord.root % 12
        pcs = chord.pitch_classes
        fifth = (root + 7) % 12
        if fifth not in pcs and (root + 6) % 12 in pcs:
            fifth = (root + 6) % 12  # diminished fifth
        third = next((t for t in ((root + 4) % 12, (root + 3) % 12) if t in pcs), None)
        seventh = next((t for t in ((root + 10) % 12, (root + 11) % 12) if t in pcs), None)
        return root, fifth, third, seventh

    def _scale_step_below(self, pitch: int, context: LoopContext) -> int:
        scale = context.scale_pitch_classes or set(range(12))
        candidate = pitch - 1
        for _ in range(12):
            if candidate % 12 in scale:
                return candidate
            candidate -= 1
        return pitch - 2

    # -- register -------------------------------------------------------------

    def _register(self, original) -> tuple[int, int]:
        """Keep the variation in the register the player used (with at least an
        octave of room); fall back to the fixed E1-E3 range when nothing was
        routed in on the bass channel."""
        pitches = [n.pitch for n in original]
        if not pitches:
            return BASS_LOW, BASS_HIGH
        low, high = min(pitches), max(pitches)
        return low, max(high, low + 12)

    def _mid(self) -> int:
        return (self._low + self._high) // 2

    # -- base groove (verbatim / no-input fallback) ---------------------------

    def _base_notes(self, original, bars, beats_per_bar, context) -> list[BassNote]:
        notes: list[BassNote] = []
        for note in original:
            duration = note.duration_beats or Fraction(1)
            notes.append(BassNote(note.pitch, note.velocity, note.bar, note.position, duration))
        if notes:
            return notes
        return self._synth_roots(bars, beats_per_bar, context)

    def _synth_roots(self, bars, beats_per_bar, context) -> list[BassNote]:
        notes: list[BassNote] = []
        prev = self._mid()
        for bar in range(1, bars + 1):
            chord = self._chord_for_bar(context, bar)
            root_pc = chord.root if chord else context.key_root
            pitch = _nearest_pitch(root_pc, prev, self._low, self._high)
            notes.append(BassNote(pitch, 90, bar, Fraction(0), Fraction(beats_per_bar)))
            prev = pitch
        return notes

    # -- drum accenting -------------------------------------------------------

    def _kick_velocity_map(self, drums, context) -> dict[int, dict[Fraction, int]]:
        """bar -> {position: kick velocity}, empty when no drums are present."""
        positions: dict[int, dict[Fraction, int]] = {}

        def add(bar: int, pos: Fraction, velocity: int) -> None:
            slot = positions.setdefault(bar, {})
            slot[pos] = max(slot.get(pos, 0), velocity)

        if drums is not None:
            for hit in drums.hits:
                if hit.pitch in KICK_PITCHES:
                    add(hit.bar, hit.position, hit.velocity)
        if not positions:
            for note in context.tracks.get("drums", []):
                if note.pitch in KICK_PITCHES:
                    add(note.bar, note.position, note.velocity)
        return positions

    # -- melody deference -----------------------------------------------------

    def _defer_to_melody(self, notes, melody, context) -> list[BassNote]:
        """On the melody's peak and loudest onsets, snap the bass to the nearest
        root or fifth so it supports rather than competes."""
        if not melody.notes:
            return notes
        loudest = max(n.velocity for n in melody.notes)
        strong = {(n.bar, n.position) for n in melody.notes if n.velocity >= loudest - 5}

        out: list[BassNote] = []
        for note in notes:
            if (note.bar, note.position) in strong:
                chord = self._chord_for_bar(context, note.bar)
                if chord:
                    root, fifth, _, _ = self._chord_tones(chord)
                    cand_root = _nearest_pitch(root, note.pitch, self._low, self._high)
                    cand_fifth = _nearest_pitch(fifth, note.pitch, self._low, self._high)
                    pitch = cand_root if abs(cand_root - note.pitch) <= abs(cand_fifth - note.pitch) else cand_fifth
                    note = BassNote(pitch, note.velocity, note.bar, note.position, note.duration)
            out.append(note)
        return out

    # -- finalization ---------------------------------------------------------

    def _sorted(self, notes) -> list[BassNote]:
        return sorted(notes, key=lambda n: (n.bar, float(n.position)))

    def _abs_beat(self, note, beats_per_bar) -> float:
        return (note.bar - 1) * beats_per_bar + float(note.position)

    def _finalize(self, notes, bars, beats_per_bar) -> list[BassNote]:
        """Bound to the loop, clamp velocity/register, and keep one (lowest) note
        per onset. Durations are set afterwards by `_articulate`, which is what
        keeps the line monophonic."""
        best: dict[tuple[int, Fraction], BassNote] = {}
        for note in notes:
            if not (1 <= note.bar <= bars):
                continue
            if not (Fraction(0) <= note.position < beats_per_bar):
                continue
            clean = BassNote(
                _clamp_register(note.pitch, self._low, self._high),
                max(1, min(127, note.velocity)),
                note.bar,
                note.position,
                note.duration,
            )
            key = (note.bar, note.position)
            existing = best.get(key)
            if existing is None or clean.pitch < existing.pitch:
                best[key] = clean
        return self._sorted(best.values())

    def _articulate(self, notes, beats_per_bar, variation) -> list[BassNote]:
        """Set note lengths: sustained toward the next onset at low variation,
        pluckier (more space) as it rises."""
        ordered = self._sorted(notes)
        factor = Fraction(85 - int(45 * variation), 100)  # 0.85 -> 0.40
        out: list[BassNote] = []
        for i, note in enumerate(ordered):
            if i < len(ordered) - 1:
                gap = Fraction(self._abs_beat(ordered[i + 1], beats_per_bar) - self._abs_beat(note, beats_per_bar)).limit_denominator(192)
            else:
                gap = Fraction(beats_per_bar) - note.position
            if gap <= 0:
                out.append(note)
                continue
            duration = max(Fraction(1, 16), min(gap, factor * gap))
            out.append(BassNote(note.pitch, note.velocity, note.bar, note.position, duration))
        return out
