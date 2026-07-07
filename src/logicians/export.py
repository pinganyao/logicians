"""MIDI file export for generated melodies."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import mido

from .midi_io import beats_to_ticks
from .models import LoopContext, MelodyClip, MelodyNote


def melody_note_to_beats(note: MelodyNote, beats_per_bar: int) -> tuple[float, float]:
    start = (note.bar - 1) * beats_per_bar + float(note.position)
    duration = float(note.duration)
    return start, duration


def export_melody_midi(
    clip: MelodyClip,
    output_path: str | Path,
    tempo_bpm: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
    ppq: int = 480,
    channel: int = 0,
    velocity_scale: float = 1.0,
    track_name: str = "Generated Melody",
) -> Path:
    """Write a MelodyClip to a MIDI file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    track.name = track_name
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=time_signature[0],
            denominator=time_signature[1],
            time=0,
        )
    )

    beats_per_bar = time_signature[0]
    events: list[tuple[int, mido.Message]] = []

    for note in clip.notes:
        start_beats, dur_beats = melody_note_to_beats(note, beats_per_bar)
        start_tick = beats_to_ticks(start_beats, ppq)
        end_tick = beats_to_ticks(start_beats + dur_beats, ppq)
        vel = max(1, min(127, int(note.velocity * velocity_scale)))
        events.append((start_tick, mido.Message("note_on", note=note.pitch, velocity=vel, channel=channel)))
        events.append((end_tick, mido.Message("note_off", note=note.pitch, velocity=0, channel=channel)))

    events.sort(key=lambda e: (e[0], 0 if e[1].type == "note_off" else 1))

    last_tick = 0
    for tick, msg in events:
        delta = max(0, tick - last_tick)
        msg.time = delta
        track.append(msg)
        last_tick = tick

    mid.save(str(output_path))
    return output_path


def export_melody_from_context(
    clip: MelodyClip,
    context: LoopContext,
    output_path: str | Path,
    channel: int = 0,
) -> Path:
    return export_melody_midi(
        clip=clip,
        output_path=output_path,
        tempo_bpm=context.tempo_bpm,
        time_signature=context.time_signature,
        ppq=context.ppq,
        channel=channel,
    )
