"""Quantize note events to symbolic musical positions and durations."""

from __future__ import annotations

from fractions import Fraction

from .models import NoteEvent

ALLOWED_DURATIONS: list[Fraction] = [
    Fraction(4, 1),
    Fraction(3, 1),
    Fraction(2, 1),
    Fraction(3, 2),
    Fraction(1, 1),
    Fraction(3, 4),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(3, 8),
    Fraction(1, 4),
    Fraction(1, 6),
    Fraction(1, 8),
    Fraction(1, 12),
]

POSITION_GRID: list[Fraction] = sorted(
    set(
        [Fraction(n, 8) for n in range(0, 32)]
        + [Fraction(n, 6) for n in range(0, 24)]
    )
)

MIN_NOTE_DURATION = Fraction(1, 12)
BEATS_PER_BAR = 4


def _nearest(value: Fraction, candidates: list[Fraction]) -> Fraction:
    return min(candidates, key=lambda c: abs(c - value))


def quantize_position(position_in_bar: Fraction, beats_per_bar: int = BEATS_PER_BAR) -> Fraction:
    """Snap a position within a bar to the nearest grid point."""
    bar_positions = [p for p in POSITION_GRID if p < beats_per_bar]
    bar_positions.append(Fraction(beats_per_bar))
    wrapped = position_in_bar % beats_per_bar
    return _nearest(wrapped, bar_positions)


def quantize_duration(duration: Fraction) -> Fraction:
    """Snap a duration to the nearest allowed value."""
    if duration < MIN_NOTE_DURATION:
        return Fraction(0)
    return _nearest(duration, ALLOWED_DURATIONS)


def beats_to_bar_position(start_beats: float, beats_per_bar: int = BEATS_PER_BAR) -> tuple[int, Fraction]:
    """Convert absolute beat offset to (bar, position). Bars are 1-indexed."""
    bar = int(start_beats // beats_per_bar) + 1
    position = Fraction(start_beats % beats_per_bar).limit_denominator(24)
    return bar, position


def quantize_note(
    note: NoteEvent,
    beats_per_bar: int = BEATS_PER_BAR,
) -> NoteEvent | None:
    """Quantize a note's timing. Returns None if note should be discarded."""
    bar, position = beats_to_bar_position(note.start, beats_per_bar)
    duration_frac = Fraction(note.duration).limit_denominator(24)
    q_duration = quantize_duration(duration_frac)

    if q_duration == Fraction(0):
        return None

    q_position = quantize_position(position, beats_per_bar)

    return NoteEvent(
        track=note.track,
        pitch=note.pitch,
        velocity=note.velocity,
        start=note.start,
        duration=note.duration,
        bar=bar,
        position=q_position,
        duration_beats=q_duration,
    )


def quantize_notes(
    notes: list[NoteEvent],
    beats_per_bar: int = BEATS_PER_BAR,
) -> list[NoteEvent]:
    """Quantize a list of notes, merging overlaps on same track/pitch where needed."""
    quantized: list[NoteEvent] = []
    for note in notes:
        q = quantize_note(note, beats_per_bar)
        if q is not None:
            quantized.append(q)
    return quantized
