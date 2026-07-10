"""Tests for the root–fifth bassline generator."""

from fractions import Fraction

from logicians.bass import (
    BASS_HIGH,
    BASS_LOW,
    FIFTH,
    ROOT,
    VELOCITY,
    BassGenerator,
    BassOptions,
    approach_tone,
    bass_pattern,
    chord_at,
    clamp_to_register,
    degree_pitch_class,
)
from logicians.models import ChordEvent, LoopContext

# D minor / G Dorian: D E F G A Bb C.
D_MINOR_SCALE = {0, 2, 4, 5, 7, 9, 10}

# Gm -> Bb -> Dm -> C, one chord per bar.
_PROGRESSION = [
    (7, "minor", "Gm"),
    (10, "major", "Bb"),
    (2, "minor", "Dm"),
    (0, "major", "C"),
]


def _chord(bar: int) -> ChordEvent:
    root, quality, name = _PROGRESSION[bar - 1]
    return ChordEvent(
        bar=bar,
        beat=Fraction(0),
        root=root,
        quality=quality,
        pitch_classes={(root + i) % 12 for i in ((0, 3, 7) if quality == "minor" else (0, 4, 7))},
        name=name,
    )


def _context(bars: int = 4) -> LoopContext:
    return LoopContext(
        tempo_bpm=125.0,
        time_signature=(4, 4),
        bars=bars,
        ppq=480,
        key_root=2,
        key_mode="minor",
        scale_pitch_classes=D_MINOR_SCALE,
        chords=[_chord(b) for b in range(1, bars + 1)],
        tracks={},
        bass_roots_by_bar={},
        rhythm_density=0.5,
    )


def _abs_beat(note, beats_per_bar=4) -> float:
    return (note.bar - 1) * beats_per_bar + float(note.position)


def _generate(bars: int = 4):
    # Seeded so the cellular-automaton variation is reproducible: the enhancer's
    # pitch/rhythm moves are driven by `random`, which the automaton seeds.
    return BassGenerator().generate(_context(bars), None, None, BassOptions(seed=1234))


def _base_line(bars: int = 4):
    """The pre-enhancement root–fifth seed line the automaton varies.

    Root/fifth placement, two-notes-per-bar shape, close voice leading, and the
    accented downbeat are properties of this seed; the automaton deliberately
    varies them in the enhanced output (octave jumps, neighbour tones, rhythmic
    splits), so those structural checks target the seed line here."""
    return BassGenerator()._base_notes(_context(bars))


# --- rhythm / shape ---------------------------------------------------------


def test_pattern_is_root_on_one_fifth_on_three():
    assert bass_pattern(4) == [(Fraction(0), ROOT), (Fraction(2), FIFTH)]


def test_two_notes_per_bar_on_beats_one_and_three():
    notes = _base_line(bars=4)
    assert len(notes) == 8  # 2 per bar over 4 bars
    for bar in range(1, 5):
        positions = sorted(n.position for n in notes if n.bar == bar)
        assert positions == [Fraction(0), Fraction(2)]


# --- pitch ------------------------------------------------------------------


def test_downbeat_plays_the_root():
    notes = _base_line()
    for bar, (root, _, _) in enumerate(_PROGRESSION, start=1):
        note = next(n for n in notes if n.bar == bar and n.position == Fraction(0))
        assert note.pitch % 12 == root


def test_beat_three_plays_the_fifth():
    notes = _base_line()
    for bar, (root, _, _) in enumerate(_PROGRESSION, start=1):
        note = next(n for n in notes if n.bar == bar and n.position == Fraction(2))
        assert note.pitch % 12 == (root + 7) % 12  # perfect fifth


def test_degree_pitch_class_helper():
    ctx = _context()
    gm = chord_at(ctx, 1, Fraction(0))
    assert degree_pitch_class(ROOT, gm, ctx) == 7      # G
    assert degree_pitch_class(FIFTH, gm, ctx) == 2      # D, the fifth of G


# --- register / voice leading -----------------------------------------------


def test_all_pitches_within_bass_register():
    bass = _generate()
    assert all(BASS_LOW <= n.pitch <= BASS_HIGH for n in bass.notes)


def test_clamp_folds_by_octave():
    assert clamp_to_register(88) == 64  # E6 folds by octaves down onto E4
    assert clamp_to_register(24) == 48  # C1 folds up to C3 (pitch class preserved)
    assert clamp_to_register(52) == 52  # already in range


def test_voice_leading_keeps_steps_small():
    # The seed line should not leap more than an octave (nearest-octave pick);
    # the automaton's enhanced output may leap further by design.
    notes = sorted(_base_line(), key=lambda n: _abs_beat(n))
    assert all(abs(b.pitch - a.pitch) <= 12 for a, b in zip(notes, notes[1:]))


# --- articulation / velocity ------------------------------------------------


def test_notes_are_detached_and_never_overlap():
    bass = sorted(_generate().notes, key=lambda n: _abs_beat(n))
    for cur, nxt in zip(bass, bass[1:]):
        assert _abs_beat(cur) + float(cur.duration) <= _abs_beat(nxt) + 1e-9
    last = bass[-1]
    assert _abs_beat(last) + float(last.duration) <= 4 * 4 + 1e-9


def test_root_is_accented_over_the_fifth():
    notes = _base_line()
    root_note = next(n for n in notes if n.bar == 1 and n.position == Fraction(0))
    fifth_note = next(n for n in notes if n.bar == 1 and n.position == Fraction(2))
    assert root_note.velocity == VELOCITY[ROOT]
    assert fifth_note.velocity == VELOCITY[FIFTH]
    assert root_note.velocity > fifth_note.velocity


# --- determinism ------------------------------------------------------------


def test_deterministic():
    a = _generate()
    b = _generate()
    assert [(n.pitch, n.bar, n.position, n.duration) for n in a.notes] == [
        (n.pitch, n.bar, n.position, n.duration) for n in b.notes
    ]


# --- approach-tone building block (kept for the next iteration) --------------


def test_approach_tone_prefers_nearest_and_diatonic():
    # Into Bb (10): A(9) is a diatonic half-step, chosen over the chromatic B(11).
    assert approach_tone(10, D_MINOR_SCALE, toward=40) % 12 == 9
