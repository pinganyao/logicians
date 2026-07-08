"""Command-line interface for The Logicians."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import typer

from .analysis import build_loop_context, parse_key
from .drums import DrumOptions, RuleBasedDrumGenerator
from .export import export_melody_from_context
from .generator import RuleBasedMelodyGenerator
from .midi_io import (
    NOTE_NAMES,
    MidiInputAdapter,
    MidiOutputAdapter,
    loop_duration_seconds,
    parse_midi_file,
)
from .models import GenerationOptions, MelodyClip, LoopContext
from .scheduler import DrumScheduler, MidiScheduler, StartMode

app = typer.Typer(name="logicians", help="AI-assisted melody improviser for Logic Pro")


def _parse_track_mapping(mapping: Optional[str]) -> dict[str, str] | None:
    if not mapping:
        return None
    return json.loads(mapping)


def _build_context(notes, tempo, time_sig, ppq, bars=None, key=None):
    return build_loop_context(notes, tempo, time_sig, ppq, bars=bars, key=key)


def _echo_detected_context(context: LoopContext, *, key_override: str | None = None) -> None:
    """Print key and chord progression inferred from captured material."""
    key_label = f"{NOTE_NAMES[context.key_root]} {context.key_mode}"
    if key_override:
        key_label += " (override)"
    elif context.key_enforced:
        key_label += " (enforced)"
    typer.echo(f"Detected key: {key_label}")
    typer.echo("Detected chords:")
    if not context.chords:
        typer.echo("  (none)")
        return
    for chord in context.chords:
        typer.echo(f"  Bar {chord.bar}: {chord.name}")
    progression = " → ".join(chord.name for chord in context.chords)
    typer.echo(f"  {progression}")


@app.command()
def analyze(
    input: Path = typer.Option(..., "--input", "-i", help="Input MIDI loop file"),
    bars: Optional[int] = typer.Option(None, "--bars", help="Override loop length in bars"),
    track_mapping: Optional[str] = typer.Option(None, "--track-mapping", help="JSON track name to role mapping"),
    key: Optional[str] = typer.Option(None, "--key", help="Override key, e.g. C, Am, F# minor"),
) -> None:
    """Analyze a MIDI loop and print musical context."""
    notes, tempo, time_sig, ppq = parse_midi_file(input, _parse_track_mapping(track_mapping))
    context = _build_context(notes, tempo, time_sig, ppq, bars=bars, key=key)

    typer.echo(f"Tempo: {context.tempo_bpm:.1f} BPM")
    typer.echo(f"Time signature: {context.time_signature[0]}/{context.time_signature[1]}")
    typer.echo(f"Loop length: {context.bars} bars")
    typer.echo(f"PPQ: {context.ppq}")
    typer.echo(f"Key: {NOTE_NAMES[context.key_root]} {context.key_mode}")
    typer.echo(f"Rhythm density: {context.rhythm_density:.2f}")
    typer.echo("\nTracks:")
    for name, track_notes in context.tracks.items():
        typer.echo(f"  {name}: {len(track_notes)} notes")
    typer.echo()
    _echo_detected_context(context, key_override=key)
    typer.echo("\nBass roots by bar:")
    for bar, root in sorted(context.bass_roots_by_bar.items()):
        typer.echo(f"  Bar {bar}: {NOTE_NAMES[root]}")


@app.command()
def generate(
    input: Path = typer.Option(..., "--input", "-i", help="Input MIDI loop file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output melody MIDI file"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    bars: Optional[int] = typer.Option(None, "--bars", help="Override loop length in bars"),
    density: str = typer.Option("medium", "--density", help="sparse, medium, or busy"),
    track_mapping: Optional[str] = typer.Option(None, "--track-mapping", help="JSON track mapping"),
    key: Optional[str] = typer.Option(None, "--key", help="Override key, e.g. C, Am, F# minor"),
) -> None:
    """Generate a melody from a MIDI loop and export to file."""
    notes, tempo, time_sig, ppq = parse_midi_file(input, _parse_track_mapping(track_mapping))
    context = _build_context(notes, tempo, time_sig, ppq, bars=bars, key=key)

    generator = RuleBasedMelodyGenerator()
    options = GenerationOptions(seed=seed, density=density)
    clip = generator.generate(context, options)

    export_melody_from_context(clip, context, output)
    typer.echo(f"Generated {len(clip.notes)} notes over {clip.bars} bars -> {output}")
    if key:
        typer.echo(f"Key: {NOTE_NAMES[context.key_root]} {context.key_mode} (override)")


@app.command()
def ports() -> None:
    """List available MIDI input and output ports."""
    inputs, outputs = __import__("logicians.midi_io", fromlist=["list_midi_ports"]).list_midi_ports()
    typer.echo("MIDI Input Ports:")
    for i, name in enumerate(inputs):
        typer.echo(f"  [{i}] {name}")
    if not inputs:
        typer.echo("  (none)")
    typer.echo("\nMIDI Output Ports:")
    for i, name in enumerate(outputs):
        typer.echo(f"  [{i}] {name}")
    if not outputs:
        typer.echo("  (none)")


@app.command()
def play(
    input: Path = typer.Option(..., "--input", "-i", help="Input MIDI loop file"),
    midi_output: str = typer.Option(..., "--midi-output", help="MIDI output port name"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    bars: Optional[int] = typer.Option(None, "--bars"),
    tempo: Optional[float] = typer.Option(None, "--tempo", help="Override tempo BPM"),
    density: str = typer.Option("medium", "--density"),
    count_in: int = typer.Option(0, "--count-in", help="Count-in bars before melody starts"),
    start_mode: str = typer.Option("next_loop", "--start-mode", help="immediately, next_bar, next_loop"),
    loop: bool = typer.Option(False, "--loop", help="Loop the generated melody"),
    key: Optional[str] = typer.Option(None, "--key", help="Override key, e.g. C, Am, F# minor"),
) -> None:
    """Generate a melody and send it to a MIDI output port."""
    notes, file_tempo, time_sig, ppq = parse_midi_file(input)
    context = _build_context(notes, tempo or file_tempo, time_sig, ppq, bars=bars, key=key)

    generator = RuleBasedMelodyGenerator()
    options = GenerationOptions(seed=seed, density=density)
    clip = generator.generate(context, options)

    output = MidiOutputAdapter(midi_output)
    output.open()
    try:
        scheduler = MidiScheduler(output, context)
        mode = StartMode(start_mode)
        typer.echo(f"Playing {len(clip.notes)} notes on '{midi_output}' (start: {start_mode})...")
        scheduler.play(clip, start_mode=mode, count_in_bars=count_in, seed=seed, loop=loop)
    finally:
        output.close()


@app.command()
def live(
    midi_input: str = typer.Option("IAC Driver Bus 1", "--midi-input", help="MIDI input port name"),
    midi_output: str = typer.Option("IAC Driver Bus 2", "--midi-output", help="MIDI output port name for the melody"),
    drum_output: Optional[str] = typer.Option(
        "IAC Driver Bus 2",
        "--drum-output",
        help="MIDI output port for the reactive drum variation. "
        "Pass an empty string (--drum-output '') to disable drum generation.",
    ),
    drum_channel: int = typer.Option(
        1,
        "--drum-channel",
        help="MIDI channel for drum output, 0-indexed (default 1 = channel 2 in Logic)",
    ),
    drum_variation: float = typer.Option(
        0.3,
        "--drum-variation",
        help="How far the drums depart from the captured groove: "
        "0.0 = identical to the input, 1.0 = a completely reworked beat",
    ),
    tempo: float = typer.Option(120.0, "--tempo"),
    bars: int = typer.Option(4, "--bars"),
    time_signature: str = typer.Option("4/4", "--time-signature"),
    sync: str = typer.Option(
        "first-note",
        "--sync",
        help="When to start the loop clock: first-note (default) or enter",
    ),
    seed: Optional[int] = typer.Option(None, "--seed"),
    density: str = typer.Option("medium", "--density"),
    count_in: int = typer.Option(0, "--count-in", help="Extra loop cycles before melody enters"),
    playback_loop: int = typer.Option(
        3,
        "--playback-loop",
        help="Which loop cycle the melody enters on (default 3: loop 1=capture, loop 2=generate)",
    ),
    key: Optional[str] = typer.Option(None, "--key", help="Override key, e.g. C, Am, F# minor"),
    continuous: bool = typer.Option(
        True,
        "--continuous/--once",
        help="Keep improvising a new melody each loop (default) or play only once",
    ),
) -> None:
    """Capture a live MIDI loop and output generated melody."""
    num, denom = time_signature.split("/")
    time_sig = (int(num), int(denom))
    duration_sec = loop_duration_seconds(bars, time_sig, tempo)

    midi_in = MidiInputAdapter(midi_input)
    midi_out = MidiOutputAdapter(midi_output)
    drum_out = MidiOutputAdapter(drum_output, channel=drum_channel) if drum_output else None

    midi_in.open()
    midi_out.open()
    if drum_out:
        drum_out.open()
    try:
        if sync == "first-note":
            typer.echo(
                f"Ready. Switch to Logic and press Play.\n"
                f"Capture starts on the first MIDI note, records {bars} bars "
                f"({duration_sec:.1f}s at {tempo:.0f} BPM).\n"
                f"Melody will enter on loop {playback_loop}."
                + (" Press Ctrl+C to stop improvising." if continuous else "")
            )
        else:
            typer.echo(
                f"Switch to Logic. When you press Play at bar 1, press Enter here.\n"
                f"Will capture {bars} bars ({duration_sec:.1f}s at {tempo:.0f} BPM)."
            )

        sync_time = midi_in.wait_for_sync(sync, tempo)
        typer.echo("Sync! Capturing loop...")
        notes = midi_in.capture_bars(bars, time_sig, tempo)
        typer.echo(f"Captured {len(notes)} notes")

        context = _build_context(notes, tempo, time_sig, 480, bars=bars, key=key)
        typer.echo()
        _echo_detected_context(context, key_override=key)
        typer.echo()

        generator = RuleBasedMelodyGenerator()
        base_options = GenerationOptions(seed=seed, density=density)
        clip = generator.generate(context, base_options)

        scheduler = MidiScheduler(midi_out, context)

        drum_generator = RuleBasedDrumGenerator() if drum_out else None
        drum_scheduler = DrumScheduler(drum_out, context) if drum_out else None

        def start_drums(melody: MelodyClip, start_at: float, iteration: int) -> tuple[threading.Thread | None, int]:
            """Generate a drum variation reacting to `melody` and play it (in a
            background thread) aligned to the same loop start. Returns the
            thread and the hit count, or (None, 0) if drums are disabled."""
            if not drum_scheduler:
                return None, 0
            opts = DrumOptions(
                seed=None if seed is None else seed + iteration,
                variation=drum_variation,
            )
            drum_clip = drum_generator.generate(context, melody, opts)
            thread = threading.Thread(
                target=drum_scheduler.play_clip,
                args=(drum_clip, start_at),
                daemon=True,
            )
            thread.start()
            return thread, len(drum_clip.hits)

        if drum_out:
            typer.echo(
                f"Reactive drums -> '{drum_output}' (channel {drum_channel + 1}), "
                f"variation {drum_variation:.2f}."
            )

        if continuous:
            typer.echo(
                f"Improvising from loop {playback_loop} — each cycle connects smoothly to the last. "
                f"Press Ctrl+C to stop."
            )

            def make_clip(iteration: int, previous: MelodyClip) -> MelodyClip:
                opts = GenerationOptions(
                    seed=None if seed is None else seed + iteration,
                    density=density,
                    previous_clip=previous,
                    iteration=iteration,
                )
                return generator.generate(context, opts)

            def on_loop(loop_num: int, playing: MelodyClip) -> None:
                message = f"Loop {loop_num}: playing {len(playing.notes)} notes"
                if drum_scheduler:
                    loop_start = sync_time + (loop_num - 1) * duration_sec
                    _, hit_count = start_drums(playing, loop_start, loop_num)
                    message += f" + {hit_count} drum hits"
                typer.echo(message)

            scheduler.play_improvisation(
                initial_clip=clip,
                clip_factory=make_clip,
                sync_time=sync_time,
                loop_duration_sec=duration_sec,
                first_playback_loop=playback_loop + count_in,
                seed=seed,
                on_loop_start=on_loop,
            )
            typer.echo("Stopped.")
        else:
            loops_before_playback = (playback_loop - 1) + count_in
            playback_start = sync_time + loops_before_playback * duration_sec
            typer.echo(
                f"Playing generated melody ({len(clip.notes)} notes) "
                f"aligned to loop {playback_loop}..."
            )
            drum_thread, hit_count = start_drums(clip, playback_start, playback_loop)
            if drum_thread:
                typer.echo(f"Playing {hit_count} drum hits alongside the melody...")
            scheduler.play(
                clip,
                start_mode=StartMode.IMMEDIATELY,
                seed=seed,
                start_at=playback_start,
            )
            if drum_thread:
                drum_thread.join()
    finally:
        midi_in.close()
        if drum_out:
            drum_out.panic()
            drum_out.close()
        midi_out.panic()
        midi_out.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
