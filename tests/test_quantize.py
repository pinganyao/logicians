"""Tests for quantization."""

from fractions import Fraction

from logicians.models import NoteEvent
from logicians.quantize import (
    ALLOWED_DURATIONS,
    POSITION_GRID,
    quantize_duration,
    quantize_note,
    quantize_position,
)


def test_quantize_duration_snaps_to_grid():
    assert quantize_duration(Fraction(1, 3)) in ALLOWED_DURATIONS
    assert quantize_duration(Fraction(7, 16)) in ALLOWED_DURATIONS
    assert quantize_duration(Fraction(1, 20)) == Fraction(0)


def test_quantize_position_snaps_to_grid():
    pos = quantize_position(Fraction(1, 3))
    assert pos in POSITION_GRID or pos == Fraction(4)


def test_quantize_note_discards_tiny_notes():
    note = NoteEvent(track="bass", pitch=36, velocity=80, start=0.0, duration=0.01)
    result = quantize_note(note)
    assert result is None


def test_quantize_note_assigns_bar_and_position():
    note = NoteEvent(track="bass", pitch=36, velocity=80, start=4.5, duration=1.0)
    result = quantize_note(note)
    assert result is not None
    assert result.bar == 2
    assert result.position in POSITION_GRID
    assert result.duration_beats in ALLOWED_DURATIONS
