"""Tests for live MIDI capture timing."""

from unittest.mock import patch

from logicians.midi_io import MidiInputAdapter, loop_duration_seconds
from logicians.models import NoteEvent


def test_loop_duration_seconds_4_bars_120_bpm():
    assert loop_duration_seconds(4, (4, 4), 120.0) == 8.0


def test_playback_start_loop_3():
    bars = 4
    tempo = 120.0
    time_sig = (4, 4)
    duration = loop_duration_seconds(bars, time_sig, tempo)
    playback_loop = 3
    sync_time = 1000.0
    playback_start = sync_time + (playback_loop - 1) * duration
    assert playback_start == sync_time + 2 * duration
    assert playback_start == 1016.0  # 16 seconds after sync at 120 BPM


def test_loop_duration_seconds_8_bars_140_bpm():
    expected = 8 * 4 * 60.0 / 140.0
    assert abs(loop_duration_seconds(8, (4, 4), 140.0) - expected) < 0.001


def test_capture_bars_from_current_position_rebases_notes():
    adapter = MidiInputAdapter("dummy")
    adapter._start_time = 1000.0
    adapter._buffer = [
        NoteEvent("chords", 60, 80, 16.0, 4.0),
        NoteEvent("chords", 64, 80, 16.0, 4.0),
        NoteEvent("bass", 48, 90, 20.0, 4.0),
        NoteEvent("chords", 67, 80, 32.0, 4.0),
    ]

    beat_times = [16.0, 16.5, 31.9, 32.1]

    def fake_elapsed(_tempo):
        return beat_times.pop(0) if beat_times else 32.1

    with patch.object(adapter, "_elapsed_beats", side_effect=fake_elapsed):
        with patch.object(adapter, "poll"):
            with patch.object(adapter, "_flush_active_notes"):
                notes = adapter.capture_bars(4, (4, 4), 120.0, from_current_position=True)

    assert len(notes) == 3
    assert all(note.bar == 1 for note in notes[:2])
    assert notes[2].bar == 2
