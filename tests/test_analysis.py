"""Tests for loop analysis."""

from fractions import Fraction

from logicians.analysis import (
    build_loop_context,
    format_key,
    infer_chord_for_bar,
    infer_key,
    parse_key,
)
from logicians.models import NoteEvent


def test_parse_key_major():
    root, mode, scale = parse_key("C")
    assert root == 0
    assert mode == "major"
    assert scale == {0, 2, 4, 5, 7, 9, 11}


def test_parse_key_minor():
    root, mode, scale = parse_key("Am")
    assert root == 9
    assert mode == "minor"
    assert 9 in scale and 11 in scale


def test_parse_key_flats_and_harmonic_minor():
    root, mode, _ = parse_key("Bb harmonic minor")
    assert root == 10
    assert mode == "harmonic_minor"
    assert format_key(root, mode) == "A# harmonic minor"


def test_build_loop_context_key_override():
    notes = [
        NoteEvent("chords", 60, 80, 0, 4.0, 1, Fraction(0), Fraction(4)),
        NoteEvent("bass", 48, 90, 0, 4.0, 1, Fraction(0), Fraction(4)),
    ]
    inferred = build_loop_context(notes, 120.0, (4, 4), 480, bars=4)
    overridden = build_loop_context(notes, 120.0, (4, 4), 480, bars=4, key="G")
    assert overridden.key_root == 7
    assert overridden.key_mode == "major"
    assert overridden.key_root != inferred.key_root or overridden.key_mode != inferred.key_mode


def _make_c_major_triad(bar: int) -> list[NoteEvent]:
    return [
        NoteEvent("chords", 60, 80, (bar - 1) * 4, 4.0, bar, Fraction(0), Fraction(4)),
        NoteEvent("chords", 64, 80, (bar - 1) * 4, 4.0, bar, Fraction(0), Fraction(4)),
        NoteEvent("chords", 67, 80, (bar - 1) * 4, 4.0, bar, Fraction(0), Fraction(4)),
    ]


def test_infer_key_c_major():
    tracks = {
        "chords": _make_c_major_triad(1) + _make_c_major_triad(2),
        "bass": [NoteEvent("bass", 48, 90, 0, 4.0, 1, Fraction(0), Fraction(4))],
    }
    root, mode, scale = infer_key(tracks)
    assert root == 0  # C
    assert mode in ("major", "minor", "harmonic_minor")


def _make_seventh_chord(bar: int, pitches: list[int], bass: int, track: str = "chords") -> list[NoteEvent]:
    start = (bar - 1) * 4
    notes = [
        NoteEvent(track, p, 80, start, 4.0, bar, Fraction(0), Fraction(4))
        for p in pitches
    ]
    notes.append(NoteEvent("bass", bass, 95, start, 4.0, bar, Fraction(0), Fraction(4)))
    return notes


def test_infer_d_major_progression():
    """Em7 -> F#m7 -> Gmaj7 -> A in D major."""
    notes = (
        _make_seventh_chord(1, [64, 67, 71, 74], 40)   # Em7
        + _make_seventh_chord(2, [61, 66, 69, 64], 42)  # F#m7
        + _make_seventh_chord(3, [67, 71, 74, 78], 43)  # Gmaj7
        + _make_seventh_chord(4, [69, 73, 76], 45)      # A major
    )
    context = build_loop_context(notes, 120.0, (4, 4), 480, bars=4, key="D")
    names = [c.name for c in context.chords]
    assert names[0] in ("Em7", "Em")
    assert names[1] in ("F#m7", "F#m")
    assert names[2] in ("Gmaj7", "G")
    assert names[3] in ("A", "Amaj7")


def test_infer_chord_gmaj7():
    chord_notes = [
        NoteEvent("chords", 67, 80, 0, 4.0, 1, Fraction(0), Fraction(4)),
        NoteEvent("chords", 71, 80, 0, 4.0, 1, Fraction(0), Fraction(4)),
        NoteEvent("chords", 74, 80, 0, 4.0, 1, Fraction(0), Fraction(4)),
        NoteEvent("chords", 78, 80, 0, 4.0, 1, Fraction(0), Fraction(4)),
    ]
    bass = [NoteEvent("bass", 43, 95, 0, 4.0, 1, Fraction(0), Fraction(4))]
    d_major = {2, 4, 6, 7, 9, 11, 1}
    chord = infer_chord_for_bar(1, chord_notes, bass, key_root=2, scale=d_major, key_mode="major")
    assert chord.root == 7
    assert chord.quality in ("maj7", "major")
    assert chord.name in ("Gmaj7", "G")


def test_infer_chord_c_major():
    chord_notes = _make_c_major_triad(1)
    bass = [NoteEvent("bass", 48, 90, 0, 4.0, 1, Fraction(0), Fraction(4))]
    chord = infer_chord_for_bar(1, chord_notes, bass, key_root=0, scale={0, 2, 4, 5, 7, 9, 11}, key_mode="major")
    assert chord.root == 0
    assert chord.quality in ("major", "maj7")


def test_build_loop_context():
    notes = _make_c_major_triad(1) + [
        NoteEvent("bass", 48, 90, 0, 4.0, 1, Fraction(0), Fraction(4)),
        NoteEvent("drums", 36, 100, 0, 0.5, 1, Fraction(0), Fraction(1, 2)),
    ]
    context = build_loop_context(notes, 120.0, (4, 4), 480, bars=4)
    assert context.bars == 4
    assert context.tempo_bpm == 120.0
    assert len(context.chords) == 4
