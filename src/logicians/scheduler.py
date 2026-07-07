"""MIDI scheduling for live melody playback."""

from __future__ import annotations

import time
from enum import Enum
from fractions import Fraction

from .midi_io import MidiOutputAdapter
from .models import LoopContext, MelodyClip, MelodyNote
from .export import melody_note_to_beats


class StartMode(str, Enum):
    IMMEDIATELY = "immediately"
    NEXT_BAR = "next_bar"
    NEXT_LOOP = "next_loop"


class MidiScheduler:
    """Schedule and play a MelodyClip over MIDI output."""

    def __init__(
        self,
        output: MidiOutputAdapter,
        context: LoopContext,
        humanize_timing_ms: float = 10.0,
        humanize_velocity: int = 5,
    ):
        self.output = output
        self.context = context
        self.humanize_timing_ms = humanize_timing_ms
        self.humanize_velocity = humanize_velocity
        self._rng = None

    def _beat_duration_sec(self) -> float:
        return 60.0 / self.context.tempo_bpm

    def _wait_for_start(
        self,
        start_mode: StartMode,
        count_in_bars: int,
        reference_time: float | None = None,
    ) -> float:
        beat_sec = self._beat_duration_sec()
        beats_per_bar = self.context.time_signature[0]
        loop_beats = self.context.bars * beats_per_bar

        if start_mode == StartMode.IMMEDIATELY:
            return time.time()

        if reference_time is None:
            reference_time = time.time()

        elapsed = time.time() - reference_time
        elapsed_beats = elapsed / beat_sec

        if start_mode == StartMode.NEXT_BAR:
            target_beat = ((int(elapsed_beats // beats_per_bar) + 1 + count_in_bars) * beats_per_bar)
        else:
            loops_elapsed = int(elapsed_beats // loop_beats)
            target_beat = (loops_elapsed + 1 + count_in_bars) * loop_beats

        wait_beats = target_beat - elapsed_beats
        if wait_beats > 0:
            time.sleep(wait_beats * beat_sec)

        return time.time()

    def _schedule_note(
        self,
        note: MelodyNote,
        start_time: float,
        rng,
        legato: bool = False,
    ) -> None:
        beats_per_bar = self.context.time_signature[0]
        start_beats, dur_beats = melody_note_to_beats(note, beats_per_bar)
        beat_sec = self._beat_duration_sec()

        jitter_sec = 0.0
        pos = float(note.position)
        if pos % 1 != 0:
            jitter_sec = rng.uniform(-self.humanize_timing_ms, self.humanize_timing_ms) / 1000.0

        note_start = start_time + start_beats * beat_sec + jitter_sec
        articulation = rng.uniform(0.85, 0.95)
        note_dur = dur_beats * beat_sec * articulation

        vel_jitter = rng.randint(-self.humanize_velocity, self.humanize_velocity)
        velocity = max(1, min(127, note.velocity + vel_jitter))

        now = time.time()
        if note_start > now:
            time.sleep(note_start - now)

        self.output.send_note_on(note.pitch, velocity)
        time.sleep(note_dur)
        if not legato:
            self.output.send_note_off(note.pitch)

    def play(
        self,
        clip: MelodyClip,
        start_mode: StartMode = StartMode.NEXT_LOOP,
        count_in_bars: int = 0,
        seed: int | None = None,
        loop: bool = False,
        legato: bool = False,
        reference_time: float | None = None,
        start_at: float | None = None,
    ) -> None:
        import random
        rng = random.Random(seed)

        if start_at is not None:
            now = time.time()
            if now < start_at:
                time.sleep(start_at - now)
            start_time = start_at
        else:
            start_time = self._wait_for_start(start_mode, count_in_bars, reference_time)

        while True:
            sorted_notes = sorted(
                clip.notes,
                key=lambda n: (n.bar, float(n.position)),
            )
            for note in sorted_notes:
                self._schedule_note(note, start_time, rng, legato=legato)

            if not loop:
                break

            beat_sec = self._beat_duration_sec()
            beats_per_bar = self.context.time_signature[0]
            loop_duration = clip.bars * beats_per_bar * beat_sec
            elapsed = time.time() - start_time
            remaining = loop_duration - (elapsed % loop_duration)
            if remaining > 0:
                time.sleep(remaining)
            start_time = time.time()
