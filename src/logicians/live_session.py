"""Two-phase live improvisation session for performance use."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .analysis import _resolve_harmony_tracks, build_loop_context
from .bass import BassGenerator, BassOptions
from .drums import DrumOptions, RuleBasedDrumGenerator
from .generator import RuleBasedMelodyGenerator
from .groove import MarkovDrumModel
from .midi_io import (
    NOTE_NAMES,
    MidiInputAdapter,
    MidiOutputAdapter,
    group_notes_by_track,
    loop_duration_seconds,
    loop_position,
    wait_until_loop_boundary,
)
from .models import GenerationOptions, LoopContext, MelodyClip
from .scheduler import ClipScheduler, MidiScheduler, StartMode


class SessionState(str, Enum):
    IDLE = "idle"
    ARMING = "arming"
    SYNCED = "synced"
    CAPTURING = "capturing"
    IMPROVISING = "improvising"
    STOPPED = "stopped"


@dataclass
class LiveSessionConfig:
    midi_input: str = "IAC Driver Bus 1"
    midi_output: str = "IAC Driver Bus 2"
    drum_output: str | None = "IAC Driver Bus 2"
    drum_channel: int = 1
    drum_variation: float = 0.3
    drum_style: str | None = None
    bass_output: str | None = "IAC Driver Bus 2"
    bass_channel: int = 2
    bass_variation: float = 0.0
    tempo: float = 120.0
    bars: int = 4
    time_signature: str = "4/4"
    sync: str = "first-note"
    seed: int | None = None
    density: str = "medium"
    count_in: int = 0
    playback_loop: int = 3
    key: str | None = None
    continuous: bool = True

    @property
    def time_sig(self) -> tuple[int, int]:
        num, denom = self.time_signature.split("/")
        return int(num), int(denom)

    @property
    def duration_sec(self) -> float:
        return loop_duration_seconds(self.bars, self.time_sig, self.tempo)


@dataclass
class SessionStatus:
    state: SessionState
    loop: int = 0
    bar: int = 0
    beat: float = 0.0
    key_label: str | None = None
    chord_progression: list[str] = field(default_factory=list)
    message: str | None = None
    error: str | None = None
    captured_notes: int = 0
    harmony_notes: int = 0
    capture_warning: str | None = None


def compute_first_playback_loop(capture_start: float, sync_time: float, duration_sec: float, config: LiveSessionConfig) -> int:
    capture_loop_num = int((capture_start - sync_time) / duration_sec) + 1
    return capture_loop_num + (config.playback_loop - 1) + config.count_in


class LiveSession:
    """Manages a two-phase live session: sync first, improvise later."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = SessionState.IDLE
        self._config: LiveSessionConfig | None = None
        self._sync_time: float | None = None
        self._context: LoopContext | None = None
        self._error: str | None = None
        self._message: str | None = None

        self._midi_in: MidiInputAdapter | None = None
        self._midi_out: MidiOutputAdapter | None = None
        self._drum_out: MidiOutputAdapter | None = None
        self._bass_out: MidiOutputAdapter | None = None
        self._drum_markov: MarkovDrumModel | None = None

        self._poll_thread: threading.Thread | None = None
        self._improvise_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._on_loop_callback: Callable[[int, MelodyClip], None] | None = None
        self._captured_notes: int = 0
        self._harmony_notes: int = 0
        self._capture_warning: str | None = None

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def detected_context(self) -> LoopContext | None:
        with self._lock:
            return self._context

    def get_status(self) -> SessionStatus:
        with self._lock:
            status = SessionStatus(
                state=self._state,
                message=self._message,
                error=self._error,
            )
            if self._sync_time is not None and self._config is not None:
                pos = loop_position(
                    self._sync_time,
                    self._config.duration_sec,
                    self._config.tempo,
                    self._config.time_sig,
                )
                status.loop = int(pos["loop"])
                status.bar = int(pos["bar"])
                status.beat = float(pos["beat"])
            if self._context is not None and self._state in (SessionState.IMPROVISING, SessionState.SYNCED):
                status.key_label = f"{NOTE_NAMES[self._context.key_root]} {self._context.key_mode}"
                status.chord_progression = [chord.name for chord in self._context.chords]
            status.captured_notes = self._captured_notes
            status.harmony_notes = self._harmony_notes
            status.capture_warning = self._capture_warning
            return status

    def start_session(self, config: LiveSessionConfig) -> None:
        with self._lock:
            if self._state not in (SessionState.IDLE, SessionState.STOPPED):
                raise RuntimeError(f"Cannot start session from state {self._state.value}")
            self._stop_event.clear()
            self._config = config
            self._sync_time = None
            self._context = None
            self._error = None
            self._message = "Opening MIDI ports..."
            self._state = SessionState.ARMING
            if config.drum_style:
                self._drum_markov = MarkovDrumModel.load()
                if self._drum_markov is None:
                    raise RuntimeError("No trained groove tables found. Run scripts/train_groove.py first.")
                available = self._drum_markov.styles()
                if config.drum_style not in available:
                    raise RuntimeError(f"Unknown drum style '{config.drum_style}'. Choose from: {', '.join(available)}")

        self._cleanup_ports()

        midi_in = MidiInputAdapter(config.midi_input)
        midi_out = MidiOutputAdapter(config.midi_output)
        drum_out = (
            MidiOutputAdapter(config.drum_output, channel=config.drum_channel)
            if config.drum_output else None
        )
        bass_out = (
            MidiOutputAdapter(config.bass_output, channel=config.bass_channel)
            if config.bass_output else None
        )
        try:
            midi_in.open()
            midi_out.open()
            if drum_out:
                drum_out.open()
            if bass_out:
                bass_out.open()
        except Exception:
            midi_in.close()
            midi_out.close()
            if drum_out:
                drum_out.close()
            if bass_out:
                bass_out.close()
            with self._lock:
                self._state = SessionState.STOPPED
                self._message = None
            raise

        with self._lock:
            self._midi_in = midi_in
            self._midi_out = midi_out
            self._drum_out = drum_out
            self._bass_out = bass_out

        threading.Thread(target=self._arm_worker, daemon=True).start()

    def _arm_worker(self) -> None:
        try:
            assert self._midi_in is not None and self._config is not None
            self._message = "Waiting for first MIDI note — press Play in Logic."
            sync_time = self._midi_in.wait_for_sync(self._config.sync, self._config.tempo)
            with self._lock:
                if self._state != SessionState.ARMING:
                    return
                self._sync_time = sync_time
                self._state = SessionState.SYNCED
                self._message = "Synced. Press Improvise when ready."
            self._start_poll_thread()
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._state = SessionState.STOPPED
                self._message = None
            self._cleanup_ports()

    def _start_poll_thread(self) -> None:
        if self._poll_thread and self._poll_thread.is_alive():
            return

        def _poll() -> None:
            while not self._stop_event.is_set():
                with self._lock:
                    midi_in = self._midi_in
                    config = self._config
                    state = self._state
                if midi_in is None or config is None or state not in (SessionState.SYNCED, SessionState.IMPROVISING):
                    break
                midi_in.poll(config.tempo, record=False)
                time.sleep(0.001)

        self._poll_thread = threading.Thread(target=_poll, daemon=True)
        self._poll_thread.start()

    def improvise(self, *, wait_for_boundary: bool = True) -> None:
        with self._lock:
            if self._state != SessionState.SYNCED:
                raise RuntimeError(f"Cannot improvise from state {self._state.value}")
            if self._sync_time is None or self._config is None:
                raise RuntimeError("Session is not synced")
            self._state = SessionState.CAPTURING
            self._context = None
            self._captured_notes = 0
            self._harmony_notes = 0
            self._capture_warning = None
            self._message = (
                "Waiting for next loop boundary to capture..."
                if wait_for_boundary
                else f"Capturing {self._config.bars} bars..."
            )

        self._improvise_thread = threading.Thread(
            target=self._improvise_worker,
            kwargs={"wait_for_boundary": wait_for_boundary},
            daemon=True,
        )
        self._improvise_thread.start()

    def _improvise_worker(self, *, wait_for_boundary: bool = True) -> None:
        try:
            assert self._midi_in is not None and self._config is not None and self._sync_time is not None
            config = self._config
            sync_time = self._sync_time

            if wait_for_boundary:
                capture_start = wait_until_loop_boundary(sync_time, config.duration_sec)
            else:
                capture_start = sync_time
            self._midi_in.reset_capture_buffer()
            self._message = f"Capturing {config.bars} bars..."
            notes = self._midi_in.capture_bars(
                config.bars,
                config.time_sig,
                config.tempo,
                from_current_position=wait_for_boundary,
            )

            chord_notes, bass_notes = _resolve_harmony_tracks(group_notes_by_track(notes))
            harmony_count = len(chord_notes) + len(bass_notes)
            self._captured_notes = len(notes)
            self._harmony_notes = harmony_count
            if harmony_count == 0:
                self._capture_warning = (
                    "No chord or bass notes captured — check Logic is routing harmony to the MIDI input port."
                )

            context = build_loop_context(
                notes, config.tempo, config.time_sig, 480, bars=config.bars, key=config.key
            )
            with self._lock:
                self._context = context
                progression = " → ".join(chord.name for chord in context.chords)
                self._message = (
                    f"Captured {len(notes)} notes ({harmony_count} harmony). "
                    f"Chords: {progression}"
                )

            generator = RuleBasedMelodyGenerator()
            base_options = GenerationOptions(seed=config.seed, density=config.density)
            clip = generator.generate(context, base_options)

            assert self._midi_out is not None
            scheduler = MidiScheduler(self._midi_out, context)
            drum_generator = RuleBasedDrumGenerator() if self._drum_out else None
            drum_scheduler = ClipScheduler(self._drum_out, context) if self._drum_out else None
            bass_generator = BassGenerator() if self._bass_out else None
            bass_scheduler = ClipScheduler(self._bass_out, context) if self._bass_out else None

            def start_drums(melody: MelodyClip, start_at: float, iteration: int):
                if not drum_scheduler:
                    return None, None, 0
                loop_seed = None if config.seed is None else config.seed + iteration
                if config.drum_style and self._drum_markov:
                    drum_clip = self._drum_markov.generate(
                        config.drum_style, config.bars, config.time_sig[0], random.Random(loop_seed)
                    )
                else:
                    opts = DrumOptions(seed=loop_seed, variation=config.drum_variation)
                    drum_clip = drum_generator.generate(context, melody, opts)
                thread = threading.Thread(
                    target=drum_scheduler.play, args=(drum_clip.hits, start_at), daemon=True
                )
                thread.start()
                return thread, drum_clip, len(drum_clip.hits)

            def start_bass(melody: MelodyClip, drum_clip, start_at: float, iteration: int):
                if not bass_scheduler:
                    return None, 0
                opts = BassOptions(
                    seed=None if config.seed is None else config.seed + iteration,
                    variation=config.bass_variation,
                )
                bass_clip = bass_generator.generate(context, melody, drum_clip, opts)
                thread = threading.Thread(
                    target=bass_scheduler.play, args=(bass_clip.notes, start_at), daemon=True
                )
                thread.start()
                return thread, len(bass_clip.notes)

            first_playback_loop = compute_first_playback_loop(
                capture_start, sync_time, config.duration_sec, config
            )

            with self._lock:
                self._state = SessionState.IMPROVISING
                self._message = f"Improvising from loop {first_playback_loop}..."

            if config.continuous:

                def make_clip(iteration: int, previous: MelodyClip) -> MelodyClip:
                    opts = GenerationOptions(
                        seed=None if config.seed is None else config.seed + iteration,
                        density=config.density,
                        previous_clip=previous,
                        iteration=iteration,
                    )
                    return generator.generate(context, opts)

                def on_loop(loop_num: int, playing: MelodyClip) -> None:
                    if self._on_loop_callback:
                        self._on_loop_callback(loop_num, playing)
                    loop_start = sync_time + (loop_num - 1) * config.duration_sec
                    drum_clip = None
                    if drum_scheduler:
                        _, drum_clip, _ = start_drums(playing, loop_start, loop_num)
                    if bass_scheduler:
                        start_bass(playing, drum_clip, loop_start, loop_num)

                scheduler.play_improvisation(
                    initial_clip=clip,
                    clip_factory=make_clip,
                    sync_time=sync_time,
                    loop_duration_sec=config.duration_sec,
                    first_playback_loop=first_playback_loop,
                    seed=config.seed,
                    on_loop_start=on_loop,
                    stop_event=self._stop_event,
                )
            else:
                playback_start = sync_time + (first_playback_loop - 1) * config.duration_sec
                drum_thread, drum_clip, _ = start_drums(clip, playback_start, first_playback_loop)
                bass_thread, _ = start_bass(clip, drum_clip, playback_start, first_playback_loop)
                scheduler.play(
                    clip,
                    start_mode=StartMode.IMMEDIATELY,
                    seed=config.seed,
                    start_at=playback_start,
                )
                if drum_thread:
                    drum_thread.join()
                if bass_thread:
                    bass_thread.join()

            with self._lock:
                if self._stop_event.is_set():
                    self._state = SessionState.STOPPED
                    self._message = "Stopped."
                else:
                    self._state = SessionState.SYNCED
                    self._message = "Playback finished. Press Improvise again or Stop."
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._state = SessionState.STOPPED
                self._message = None
            self._cleanup_ports()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            self._state = SessionState.STOPPED
            self._message = "Stopped."
        self._cleanup_ports()

    def reset(self) -> None:
        """Stop any active session and return to idle, ready for a new start."""
        with self._lock:
            self._stop_event.set()
            self._state = SessionState.IDLE
            self._config = None
            self._sync_time = None
            self._context = None
            self._error = None
            self._message = None
            self._drum_markov = None
            self._captured_notes = 0
            self._harmony_notes = 0
            self._capture_warning = None
        self._cleanup_ports()
        with self._lock:
            self._stop_event.clear()

    def _cleanup_ports(self) -> None:
        with self._lock:
            midi_in = self._midi_in
            midi_out = self._midi_out
            drum_out = self._drum_out
            bass_out = self._bass_out
            self._midi_in = None
            self._midi_out = None
            self._drum_out = None
            self._bass_out = None

        if midi_in:
            midi_in.close()
        if drum_out:
            drum_out.panic()
            drum_out.close()
        if bass_out:
            bass_out.panic()
            bass_out.close()
        if midi_out:
            midi_out.panic()
            midi_out.close()

    def set_on_loop_callback(self, callback: Callable[[int, MelodyClip], None] | None) -> None:
        self._on_loop_callback = callback


_session: LiveSession | None = None
_session_lock = threading.Lock()


def get_session() -> LiveSession:
    global _session
    with _session_lock:
        if _session is None:
            _session = LiveSession()
        return _session
