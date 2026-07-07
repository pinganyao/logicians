"""MIDI file parsing and live port I/O."""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import mido

from .models import NoteEvent
from .quantize import quantize_notes

DRUM_CHANNEL = 9
DEFAULT_PPQ = 480
DEFAULT_TEMPO_BPM = 120.0
DEFAULT_TIME_SIGNATURE = (4, 4)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

TRACK_ROLE_KEYWORDS = {
    "drums": ("drum", "drums", "perc", "percussion", "kick"),
    "bass": ("bass", "sub", "808"),
    "chords": ("chord", "chords", "pad", "keys", "piano", "synth"),
    "melody_reference": ("melody", "lead", "vocal", "top"),
}


def ticks_to_beats(ticks: int, ppq: int) -> float:
    return ticks / ppq


def beats_to_ticks(beats: float, ppq: int) -> int:
    return int(round(beats * ppq))


def _parse_tempo_and_signature(mid: mido.MidiFile) -> tuple[float, tuple[int, int], int]:
    tempo_bpm = DEFAULT_TEMPO_BPM
    time_sig = DEFAULT_TIME_SIGNATURE
    ppq = mid.ticks_per_beat or DEFAULT_PPQ

    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_bpm = mido.tempo2bpm(msg.tempo)
            elif msg.type == "time_signature":
                time_sig = (msg.numerator, msg.denominator)

    return tempo_bpm, time_sig, ppq


def _infer_track_role(
    track_name: str,
    channel: int,
    pitches: list[int],
    manual_mapping: dict[str, str] | None = None,
) -> str:
    if manual_mapping and track_name in manual_mapping:
        return manual_mapping[track_name]

    name_lower = track_name.lower()
    for role, keywords in TRACK_ROLE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return role

    if channel == DRUM_CHANNEL or any(p < 36 or p > 81 for p in pitches if channel == DRUM_CHANNEL):
        if channel == DRUM_CHANNEL:
            return "drums"

    if pitches:
        avg_pitch = sum(pitches) / len(pitches)
        if avg_pitch < 48:
            return "bass"
        if avg_pitch > 72:
            return "melody_reference"
        if len(set(pitches)) > 3:
            return "chords"

    return "chords"


def parse_midi_file(
    path: str | Path,
    track_mapping: dict[str, str] | None = None,
    quantize: bool = True,
) -> tuple[list[NoteEvent], float, tuple[int, int], int]:
    """Parse a MIDI file into NoteEvents grouped by inferred track role."""
    mid = mido.MidiFile(str(path))
    tempo_bpm, time_sig, ppq = _parse_tempo_and_signature(mid)
    beats_per_bar = time_sig[0]

    active_notes: dict[tuple[int, int, int], tuple[int, int]] = {}
    track_notes: dict[str, list[NoteEvent]] = {}
    track_meta: dict[int, tuple[str, int, list[int]]] = {}

    for track_idx, track in enumerate(mid.tracks):
        track_name = track.name or f"track_{track_idx}"
        channel = 0
        pitches: list[int] = []
        tick = 0

        for msg in track:
            tick += msg.time

            if msg.type == "program_change":
                channel = msg.channel
            elif msg.type == "note_on" and msg.velocity > 0:
                key = (track_idx, msg.channel, msg.note)
                active_notes[key] = (tick, msg.velocity)
                pitches.append(msg.note)
            elif msg.type in ("note_off", "note_on") and (msg.type == "note_off" or msg.velocity == 0):
                key = (track_idx, msg.channel, msg.note)
                if key in active_notes:
                    start_tick, velocity = active_notes.pop(key)
                    start_beats = ticks_to_beats(start_tick, ppq)
                    duration_beats = ticks_to_beats(tick - start_tick, ppq)
                    role = _infer_track_role(track_name, msg.channel, pitches, track_mapping)
                    note = NoteEvent(
                        track=role,
                        pitch=msg.note,
                        velocity=velocity,
                        start=start_beats,
                        duration=duration_beats,
                    )
                    track_notes.setdefault(role, []).append(note)

        track_meta[track_idx] = (track_name, channel, pitches)

    all_notes: list[NoteEvent] = []
    for notes in track_notes.values():
        all_notes.extend(notes)

    if quantize:
        all_notes = quantize_notes(all_notes, beats_per_bar)

    return all_notes, tempo_bpm, time_sig, ppq


def group_notes_by_track(notes: list[NoteEvent]) -> dict[str, list[NoteEvent]]:
    grouped: dict[str, list[NoteEvent]] = {}
    for note in notes:
        grouped.setdefault(note.track, []).append(note)
    return grouped


BASS_PITCH_SPLIT = 52  # E3 — notes below are treated as bass in mixed/live capture


def role_for_live_note(pitch: int, channel: int) -> str:
    """Assign bass/chords/drums for live MIDI without track names."""
    if channel == DRUM_CHANNEL:
        return "drums"
    if pitch < BASS_PITCH_SPLIT:
        return "bass"
    return "chords"


def split_mixed_harmony_notes(notes: list[NoteEvent]) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Split a single mixed stream into harmony and bass parts by register."""
    chord_notes: list[NoteEvent] = []
    bass_notes: list[NoteEvent] = []
    for note in notes:
        role = role_for_live_note(note.pitch, 0)
        tagged = NoteEvent(
            track=role,
            pitch=note.pitch,
            velocity=note.velocity,
            start=note.start,
            duration=note.duration,
            bar=note.bar,
            position=note.position,
            duration_beats=note.duration_beats,
        )
        if role == "bass":
            bass_notes.append(tagged)
        else:
            chord_notes.append(tagged)
    return chord_notes, bass_notes


def estimate_loop_bars(notes: list[NoteEvent], time_signature: tuple[int, int]) -> int:
    if not notes:
        return 4
    beats_per_bar = time_signature[0]
    max_end = max(n.start + n.duration for n in notes)
    bars = int(max_end / beats_per_bar) + (1 if max_end % beats_per_bar > 0.01 else 0)
    if bars <= 4:
        return 4
    if bars <= 8:
        return 8
    return ((bars + 3) // 4) * 4


def list_midi_ports() -> tuple[list[str], list[str]]:
    """Return (input_ports, output_ports)."""
    try:
        import rtmidi
        midi_in = rtmidi.MidiIn()
        midi_out = rtmidi.MidiOut()
        inputs = midi_in.get_ports()
        outputs = midi_out.get_ports()
        del midi_in, midi_out
        return inputs, outputs
    except Exception:
        return mido.get_input_names(), mido.get_output_names()


class MidiInputAdapter:
    """Capture live MIDI note events into a buffer."""

    def __init__(self, port_name: str, ppq: int = DEFAULT_PPQ):
        self.port_name = port_name
        self.ppq = ppq
        self._port: mido.ports.BaseInput | None = None
        self._buffer: list[NoteEvent] = []
        self._active: dict[tuple[int, int], tuple[float, int, str]] = {}
        self._start_time: float | None = None
        self._track = "capture"

    def open(self) -> None:
        """Open the port without starting the musical clock."""
        self._port = mido.open_input(self.port_name)
        self._start_time = None
        self._buffer = []
        self._active = {}

    def close(self) -> None:
        if self._port:
            self._port.close()
            self._port = None

    def _elapsed_beats(self, tempo_bpm: float) -> float:
        if self._start_time is None:
            return 0.0
        elapsed_sec = time.time() - self._start_time
        return elapsed_sec * tempo_bpm / 60.0

    def _handle_message(self, msg: mido.Message, tempo_bpm: float) -> bool:
        """Process one message. Returns True if this was the first sync note."""
        if self._start_time is None:
            if msg.type == "note_on" and msg.velocity > 0:
                self._start_time = time.time()
                return True
            return False

        if msg.type == "note_on" and msg.velocity > 0:
            beat = self._elapsed_beats(tempo_bpm)
            role = role_for_live_note(msg.note, msg.channel)
            self._active[(msg.channel, msg.note)] = (beat, msg.velocity, role)
        elif msg.type in ("note_off", "note_on"):
            key = (msg.channel, msg.note)
            if key in self._active:
                start, velocity, track = self._active.pop(key)
                end = self._elapsed_beats(tempo_bpm)
                self._buffer.append(
                    NoteEvent(
                        track=track,
                        pitch=msg.note,
                        velocity=velocity,
                        start=start,
                        duration=max(end - start, Fraction(1, 16)),
                    )
                )
        return False

    def poll(self, tempo_bpm: float) -> None:
        if not self._port:
            return
        for msg in self._port.iter_pending():
            self._handle_message(msg, tempo_bpm)

    def _flush_active_notes(self, tempo_bpm: float) -> None:
        beat = self._elapsed_beats(tempo_bpm)
        for (channel, note), (start, velocity, track) in list(self._active.items()):
            self._buffer.append(
                NoteEvent(
                    track=track,
                    pitch=note,
                    velocity=velocity,
                    start=start,
                    duration=max(beat - start, Fraction(1, 16)),
                )
            )
        self._active.clear()

    def wait_for_sync(self, sync: str, tempo_bpm: float, timeout: float = 120.0) -> float:
        """Block until sync point. Returns wall-clock time of sync (loop bar 1)."""
        if sync == "enter":
            input("Press Enter when Logic starts playing at bar 1... ")
            self._start_time = time.time()
            return self._start_time

        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._port:
                break
            for msg in self._port.iter_pending():
                if self._handle_message(msg, tempo_bpm) and self._start_time is not None:
                    return self._start_time
            time.sleep(0.001)

        raise TimeoutError("No MIDI received — check Logic is sending to the input port.")

    def capture_bars(
        self,
        bars: int,
        time_signature: tuple[int, int],
        tempo_bpm: float,
    ) -> list[NoteEvent]:
        """Capture exactly `bars` of musical time after sync."""
        beats_per_bar = time_signature[0]
        loop_beats = bars * beats_per_bar

        while self._elapsed_beats(tempo_bpm) < loop_beats:
            self.poll(tempo_bpm)
            time.sleep(0.001)

        self._flush_active_notes(tempo_bpm)
        notes = [n for n in self._buffer if n.start < loop_beats]
        return quantize_notes(notes, beats_per_bar)

    def freeze(self, loop_bars: int, time_signature: tuple[int, int], tempo_bpm: float) -> list[NoteEvent]:
        beats_per_bar = time_signature[0]
        loop_length = loop_bars * beats_per_bar
        self._flush_active_notes(tempo_bpm)
        notes = [n for n in self._buffer if n.start < loop_length]
        return quantize_notes(notes, beats_per_bar)

    def clear(self) -> None:
        self._buffer = []
        self._active = {}
        self._start_time = None


def loop_duration_seconds(
    bars: int,
    time_signature: tuple[int, int],
    tempo_bpm: float,
) -> float:
    beats_per_bar = time_signature[0]
    total_beats = bars * beats_per_bar
    return total_beats * 60.0 / tempo_bpm


class MidiOutputAdapter:
    """Send MIDI note events to an output port."""

    def __init__(self, port_name: str, channel: int = 0):
        self.port_name = port_name
        self.channel = channel
        self._port: mido.ports.BaseOutput | None = None

    def open(self) -> None:
        self._port = mido.open_output(self.port_name)

    def close(self) -> None:
        if self._port:
            self._port.close()
            self._port = None

    def send_note_on(self, pitch: int, velocity: int) -> None:
        if self._port:
            self._port.send(mido.Message("note_on", note=pitch, velocity=velocity, channel=self.channel))

    def send_note_off(self, pitch: int) -> None:
        if self._port:
            self._port.send(mido.Message("note_off", note=pitch, velocity=0, channel=self.channel))
