"""MIDI scheduling for live melody playback."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import Enum

from .midi_io import MidiOutputAdapter
from .models import LoopContext, MelodyClip, MelodyNote
from .export import melody_note_to_beats


def _play_timed_notes(output, events_source, start_at: float, beat_sec: float, beats_per_bar: int) -> None:
    """Dispatch note events (each with pitch/velocity/bar/position/duration) at
    absolute wall-clock times relative to `start_at`. Polyphony-safe: several
    notes may sound at once, and a note_off sharing an instant with a note_on is
    sent first."""
    events: list[tuple[float, int, int, int]] = []  # (time, order, pitch, velocity); order 0=off, 1=on
    for note in events_source:
        start_beats, dur_beats = melody_note_to_beats(note, beats_per_bar)
        on_time = start_at + start_beats * beat_sec
        off_time = on_time + dur_beats * beat_sec
        events.append((on_time, 1, note.pitch, note.velocity))
        events.append((off_time, 0, note.pitch, 0))

    events.sort(key=lambda e: (e[0], e[1]))

    now = time.time()
    if now < start_at:
        time.sleep(start_at - now)

    for event_time, order, pitch, velocity in events:
        now = time.time()
        if event_time > now:
            time.sleep(event_time - now)
        if order == 1:
            output.send_note_on(pitch, velocity)
        else:
            output.send_note_off(pitch)


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

    def play_improvisation(
        self,
        initial_clip: MelodyClip,
        clip_factory: Callable[[int, MelodyClip], MelodyClip],
        sync_time: float,
        loop_duration_sec: float,
        first_playback_loop: int,
        seed: int | None = None,
        on_loop_start: Callable[[int, MelodyClip], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Play an initial clip then keep improvising new clips each loop until interrupted."""
        iteration = 0
        clip = initial_clip

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break

                loop_start = sync_time + (first_playback_loop - 1 + iteration) * loop_duration_sec
                self._wait_until(loop_start)

                if stop_event and stop_event.is_set():
                    break

                if on_loop_start:
                    on_loop_start(iteration + first_playback_loop, clip)

                next_clip_box: list[MelodyClip | None] = [None]

                def _generate_next(idx: int, previous: MelodyClip) -> None:
                    next_clip_box[0] = clip_factory(idx + 1, previous)

                gen_thread = threading.Thread(
                    target=_generate_next,
                    args=(iteration, clip),
                    daemon=True,
                )
                gen_thread.start()

                play_seed = None if seed is None else seed + iteration
                self.play(
                    clip,
                    start_mode=StartMode.IMMEDIATELY,
                    seed=play_seed,
                    start_at=loop_start,
                )

                gen_thread.join()
                if next_clip_box[0] is None:
                    break
                clip = next_clip_box[0]
                iteration += 1
        except KeyboardInterrupt:
            pass

    def _wait_until(self, target_time: float) -> None:
        now = time.time()
        if now < target_time:
            time.sleep(target_time - now)


class ClipScheduler:
    """Schedule and play a set of timed notes (drum hits or bass notes) over MIDI
    output, using polyphony-safe absolute-time dispatch. Distinct from
    MidiScheduler, which does humanized, sequential, monophonic melody playback.
    """

    def __init__(self, output: MidiOutputAdapter, context: LoopContext):
        self.output = output
        self.context = context

    def play(self, notes, start_at: float) -> None:
        """Play `notes` (any iterable of objects with pitch/velocity/bar/position/
        duration), timed relative to the absolute wall-clock `start_at`."""
        _play_timed_notes(
            self.output,
            notes,
            start_at,
            60.0 / self.context.tempo_bpm,
            self.context.time_signature[0],
        )
