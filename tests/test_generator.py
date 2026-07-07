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


def test_connected_loop_stays_within_reach_of_previous_ending():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    first = gen.generate(context, GenerationOptions(seed=7))
    second = gen.generate(
        context,
        GenerationOptions(seed=7, previous_clip=first, iteration=1),
    )
    first_end = sorted(first.notes, key=lambda n: (n.bar, float(n.position)))[-1].pitch
    second_start = sorted(second.notes, key=lambda n: (n.bar, float(n.position)))[0].pitch
    assert abs(first_end - second_start) <= 5


def test_connected_loop_remains_distinct_melody():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    first = gen.generate(context, GenerationOptions(seed=11))
    second = gen.generate(
        context,
        GenerationOptions(seed=99, previous_clip=first, iteration=1),
    )
    first_pitches = [n.pitch for n in first.notes]
    second_pitches = [n.pitch for n in second.notes]
    assert first_pitches != second_pitches
    matching = sum(1 for a, b in zip(first_pitches, second_pitches) if a == b)
    assert matching / max(len(first_pitches), 1) < 0.5


def test_resolution_ends_on_tonic_chord_tone():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=42))
    bar4 = sorted(
        [n for n in clip.notes if n.bar == 4],
        key=lambda n: (float(n.position), -float(n.duration)),
    )
    assert bar4
    last = bar4[-1]
    assert last.pitch % 12 in {0, 4}


def test_strong_beat_notes_favor_chord_tones():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=55))
    strong = [
        n for n in clip.notes
        if float(n.position) % 1 in (0.0, 0.5)
    ]
    assert strong
    chord_tone_ratio = sum(1 for n in strong if n.pitch % 12 in {0, 4, 7}) / len(strong)
    assert chord_tone_ratio >= 0.55


def test_avoid_fourth_rarely_on_strong_beats():
    context = _minimal_context(4)
    context.key_enforced = True
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=88))
    strong = [n for n in clip.notes if float(n.position) % 1 in (0.0, 0.5)]
    fourth_on_strong = sum(1 for n in strong if n.pitch % 12 == 5)
    assert fourth_on_strong / max(len(strong), 1) < 0.15


def test_pop_melody_avoids_large_leaps():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=42))
    ordered = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))
    leaps = [abs(ordered[i].pitch - ordered[i - 1].pitch) for i in range(1, len(ordered))]
    assert leaps
    assert max(leaps) <= 5
    step_ratio = sum(1 for leap in leaps if leap <= 2) / len(leaps)
    assert step_ratio >= 0.65
