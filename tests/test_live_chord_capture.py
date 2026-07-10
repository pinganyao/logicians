"""Verify live-capture chord detection matches the-logicians register split."""

from fractions import Fraction

from logicians.analysis import build_loop_context
from logicians.midi_io import role_for_live_note
from logicians.models import NoteEvent


def _live_bar(bar_num: int, bass_pitch: int, harmony: list[int]) -> list[NoteEvent]:
    """Simulate notes as tagged during live capture on Logic channel 1 (mido 0)."""
    start = (bar_num - 1) * 4
    notes = [
        NoteEvent(
            track=role_for_live_note(bass_pitch, 0),
            pitch=bass_pitch,
            velocity=95,
            start=start,
            duration=4.0,
            bar=bar_num,
            position=Fraction(0),
            duration_beats=Fraction(4),
        )
    ]
    for pitch in harmony:
        notes.append(
            NoteEvent(
                track=role_for_live_note(pitch, 0),
                pitch=pitch,
                velocity=80,
                start=start,
                duration=4.0,
                bar=bar_num,
                position=Fraction(0),
                duration_beats=Fraction(4),
            )
        )
    return notes


def test_live_capture_f_minor_progression():
    """Gm -> Bb -> Dm -> C in F minor, all on one MIDI channel like Logic."""
    notes = (
        _live_bar(1, 43, [58, 62, 67])
        + _live_bar(2, 46, [58, 62, 65])
        + _live_bar(3, 38, [62, 65, 69])
        + _live_bar(4, 36, [60, 64, 67])
    )
    context = build_loop_context(notes, 120.0, (4, 4), 480, bars=4, key="F minor")
    names = [c.name for c in context.chords]
    roots = [c.root for c in context.chords]
    assert roots == [7, 10, 2, 0], f"got roots {roots} names {names}"
    assert names[0] in ("Gm", "Gm7")
    assert names[1] in ("Bb", "Bbmaj7", "A#", "A#maj7")
    assert names[2] in ("Dm", "Dm7")
    assert names[3] in ("C", "Cmaj7")


def test_role_for_live_note_splits_register_on_channel_zero():
    assert role_for_live_note(36, 0) == "bass"
    assert role_for_live_note(60, 0) == "chords"
    assert role_for_live_note(36, 9) == "drums"
