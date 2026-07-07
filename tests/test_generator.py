"""Tests for melody generation."""

from fractions import Fraction
from pathlib import Path

import mido

from logicians.analysis import build_loop_context
from logicians.export import export_melody_midi
from logicians.generator import RuleBasedMelodyGenerator
from logicians.models import GenerationOptions, LoopContext, ChordEvent


def _minimal_context(bars: int = 4) -> LoopContext:
    chords = [
        ChordEvent(bar=i, beat=Fraction(0), root=0, quality="major",
                   pitch_classes={0, 4, 7}, name="C")
        for i in range(1, bars + 1)
    ]
    return LoopContext(
        tempo_bpm=120.0,
        time_signature=(4, 4),
        bars=bars,
        ppq=480,
        key_root=0,
        key_mode="major",
        scale_pitch_classes={0, 2, 4, 5, 7, 9, 11},
        chords=chords,
        tracks={},
        bass_roots_by_bar={i: 0 for i in range(1, bars + 1)},
        rhythm_density=0.5,
    )


def test_phrase_length_equals_loop():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=42))
    assert clip.bars == 4
    max_end = max(
        (n.bar - 1) * 4 + float(n.position) + float(n.duration)
        for n in clip.notes
    )
    assert max_end <= 16.01


def test_no_overlapping_notes():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=42))
    sorted_notes = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))
    for i in range(len(sorted_notes) - 1):
        a = sorted_notes[i]
        b = sorted_notes[i + 1]
        a_end = (a.bar - 1) * 4 + float(a.position) + float(a.duration)
        b_start = (b.bar - 1) * 4 + float(b.position)
        assert a_end <= b_start + 0.001


def test_deterministic_with_seed():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip1 = gen.generate(context, GenerationOptions(seed=99))
    clip2 = gen.generate(context, GenerationOptions(seed=99))
    assert len(clip1.notes) == len(clip2.notes)
    for n1, n2 in zip(clip1.notes, clip2.notes):
        assert n1.pitch == n2.pitch
        assert n1.bar == n2.bar
        assert n1.position == n2.position


def test_different_seeds_vary():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip1 = gen.generate(context, GenerationOptions(seed=1))
    clip2 = gen.generate(context, GenerationOptions(seed=2))
    pitches1 = [n.pitch for n in clip1.notes]
    pitches2 = [n.pitch for n in clip2.notes]
    assert pitches1 != pitches2 or len(clip1.notes) != len(clip2.notes)


def test_pitches_in_register():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    opts = GenerationOptions(seed=7, register_low=60, register_high=84)
    clip = gen.generate(context, opts)
    for n in clip.notes:
        assert 60 <= n.pitch <= 84


def test_enforced_key_stays_in_scale():
    from logicians.midi_io import parse_midi_file

    notes, tempo, ts, ppq = parse_midi_file("examples/input_loops/loop.mid")
    context = build_loop_context(notes, tempo, ts, ppq, bars=4, key="D")
    assert context.key_enforced
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=1))
    for note in clip.notes:
        assert note.pitch % 12 in context.scale_pitch_classes


def test_midi_export_valid(tmp_path: Path):
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=1))
    out = tmp_path / "melody.mid"
    export_melody_midi(clip, out, tempo_bpm=120.0)
    mid = mido.MidiFile(str(out))
    assert len(mid.tracks) >= 1
    note_ons = sum(1 for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0)
    assert note_ons == len(clip.notes)
