"""Tests for live MIDI capture timing."""

from logicians.midi_io import loop_duration_seconds


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
