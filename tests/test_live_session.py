"""Tests for live session timing and state machine."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from logicians.live_session import (
    LiveSession,
    LiveSessionConfig,
    SessionState,
    compute_first_playback_loop,
)
from logicians.midi_io import loop_position, wait_until_loop_boundary


class TestLoopTiming:
    def test_loop_position_at_sync(self):
        sync = 1000.0
        duration = 8.0  # 4 bars at 120 BPM
        with patch("logicians.midi_io.time.time", return_value=1000.0):
            pos = loop_position(sync, duration, 120.0, (4, 4))
        assert pos["loop"] == 1
        assert pos["bar"] == 1
        assert pos["beat"] == 1.0

    def test_loop_position_mid_loop(self):
        sync = 1000.0
        duration = 8.0
        # 10 seconds in = loop 2, 2 seconds into loop 2 = 4 beats = bar 2 beat 1
        with patch("logicians.midi_io.time.time", return_value=1010.0):
            pos = loop_position(sync, duration, 120.0, (4, 4))
        assert pos["loop"] == 2
        assert pos["bar"] == 2
        assert pos["beat"] == 1.0

    def test_wait_until_loop_boundary_already_at_boundary(self):
        sync = 1000.0
        duration = 8.0
        with patch("logicians.midi_io.time.time", return_value=1000.0):
            boundary = wait_until_loop_boundary(sync, duration)
        assert boundary == 1000.0

    def test_wait_until_loop_boundary_waits(self):
        sync = 1000.0
        duration = 8.0
        times = [1003.0, 1008.0]

        def fake_time():
            return times.pop(0) if times else 1008.0

        with patch("logicians.midi_io.time.time", side_effect=fake_time):
            with patch("logicians.midi_io.time.sleep") as sleep_mock:
                boundary = wait_until_loop_boundary(sync, duration)
        sleep_mock.assert_called_once()
        assert boundary == 1008.0


class TestPlaybackLoopRebasing:
    def test_first_playback_at_session_start(self):
        config = LiveSessionConfig(playback_loop=3, count_in=0)
        duration = 8.0
        sync = 1000.0
        capture_start = sync
        result = compute_first_playback_loop(capture_start, sync, duration, config)
        assert result == 3

    def test_first_playback_after_later_capture(self):
        config = LiveSessionConfig(playback_loop=3, count_in=1)
        duration = 8.0
        sync = 1000.0
        # Capture starts at loop 5 boundary (32s in)
        capture_start = sync + 4 * duration
        result = compute_first_playback_loop(capture_start, sync, duration, config)
        assert result == 5 + 2 + 1  # capture loop 5 + (3-1) + count_in


class TestLiveSessionState:
    def test_initial_state(self):
        session = LiveSession()
        assert session.state == SessionState.IDLE

    def test_start_session_transitions_to_arming(self):
        session = LiveSession()
        config = LiveSessionConfig()

        mock_in = MagicMock()
        mock_out = MagicMock()

        with patch("logicians.live_session.MidiInputAdapter", return_value=mock_in):
            with patch("logicians.live_session.MidiOutputAdapter", return_value=mock_out):
                mock_in.open = MagicMock()
                mock_out.open = MagicMock()
                mock_in.wait_for_sync = MagicMock(side_effect=lambda *a, **k: time.sleep(0.2) or 1000.0)
                session.start_session(config)

        assert session.state == SessionState.ARMING
        session.stop()

    def test_improvise_requires_synced(self):
        session = LiveSession()
        with pytest.raises(RuntimeError, match="Cannot improvise"):
            session.improvise()

    def test_improvise_from_synced(self):
        session = LiveSession()
        session._state = SessionState.SYNCED
        session._sync_time = 1000.0
        session._config = LiveSessionConfig()
        session._midi_in = MagicMock()
        session._midi_out = MagicMock()
        session._midi_in.reset_capture_buffer = MagicMock()
        session._midi_in.capture_bars = MagicMock(return_value=[])
        session._midi_out.open = MagicMock()

        with patch.object(session, "_improvise_worker"):
            session.improvise()
        assert session.state == SessionState.CAPTURING

    def test_reset_returns_to_idle(self):
        session = LiveSession()
        session._state = SessionState.SYNCED
        session._sync_time = 1000.0
        session._context = MagicMock()
        session._error = "old error"
        session._message = "old message"
        session._midi_in = MagicMock()
        session.reset()
        assert session.state == SessionState.IDLE
        assert session._sync_time is None
        assert session._context is None
        assert session._error is None
        assert session._message is None
