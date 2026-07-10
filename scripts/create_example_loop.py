"""Generate example input loop MIDI files for testing."""

from __future__ import annotations

from pathlib import Path

import mido

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "input_loops"


def create_loop_4bar(output_path: Path, tempo_bpm: float = 120.0) -> None:
    """Create a 4-bar D major loop: Em7 -> F#m7 -> Gmaj7 -> A."""
    ppq = 480
    beats_per_bar = 4
    bar_ticks = beats_per_bar * ppq
    tempo = mido.bpm2tempo(tempo_bpm)

    mid = mido.MidiFile(ticks_per_beat=ppq)

    def add_note(events: list, start: int, end: int, pitch: int, velocity: int, channel: int) -> None:
        events.append((start, mido.Message("note_on", note=pitch, velocity=velocity, channel=channel)))
        events.append((end, mido.Message("note_off", note=pitch, velocity=0, channel=channel)))

    def flush_track(track: mido.MidiTrack, events: list[tuple[int, mido.Message]], is_first: bool) -> None:
        if is_first:
            track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
            track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        events.sort(key=lambda e: (e[0], 0 if e[1].type == "note_off" else 1))
        last = 0
        for tick, msg in events:
            msg.time = max(0, tick - last)
            track.append(msg)
            last = tick
        mid.tracks.append(track)

    drum_events: list[tuple[int, mido.Message]] = []
    for bar in range(4):
        base = bar * bar_ticks
        add_note(drum_events, base, base + ppq // 4, 36, 100, 9)
        add_note(drum_events, base + ppq, base + ppq + ppq // 4, 38, 90, 9)
        add_note(drum_events, base + 2 * ppq, base + 2 * ppq + ppq // 4, 36, 95, 9)
        add_note(drum_events, base + 3 * ppq, base + 3 * ppq + ppq // 4, 38, 85, 9)
    drums = mido.MidiTrack()
    drums.name = "Drums"
    flush_track(drums, drum_events, is_first=True)

    # Em7, F#m7, Gmaj7, A
    bass_roots = [40, 42, 43, 45]  # E2, F#2, G2, A2
    chord_voicings = [
        [64, 67, 71, 74],   # Em7
        [61, 66, 69, 64],   # F#m7 (C#3 bass-ish voicing)
        [67, 71, 74, 78],   # Gmaj7
        [69, 73, 76],       # A major
    ]

    bass_events: list[tuple[int, mido.Message]] = []
    chord_events: list[tuple[int, mido.Message]] = []
    for bar, (root, pitches) in enumerate(zip(bass_roots, chord_voicings)):
        base = bar * bar_ticks
        add_note(bass_events, base, base + bar_ticks - 1, root, 95, 0)
        for pitch in pitches:
            add_note(chord_events, base, base + bar_ticks - 1, pitch, 75, 1)

    bass = mido.MidiTrack()
    bass.name = "Bass"
    flush_track(bass, bass_events, is_first=False)

    chords = mido.MidiTrack()
    chords.name = "Chords"
    flush_track(chords, chord_events, is_first=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))


if __name__ == "__main__":
    create_loop_4bar(OUTPUT_DIR / "loop.mid")
    print(f"Created {OUTPUT_DIR / 'loop.mid'}")
