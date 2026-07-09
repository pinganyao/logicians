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
    clip = gen.generate(context, GenerationOptions(seed=3))
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
    assert chord_tone_ratio >= 0.50


def test_avoid_fourth_rarely_on_strong_beats():
    context = _minimal_context(4)
    context.key_enforced = True
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=88))
    strong = [n for n in clip.notes if float(n.position) % 1 in (0.0, 0.5)]
    fourth_on_strong = sum(1 for n in strong if n.pitch % 12 == 5)
    assert fourth_on_strong / max(len(strong), 1) < 0.15


def test_loop_ends_on_stable_chord_tone():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    for seed in range(30):
        clip = gen.generate(context, GenerationOptions(seed=seed))
        last = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))[-1]
        assert last.pitch % 12 in {0, 4, 7}


def test_pentatonic_scalic_runs_appear_sometimes():
    context = _minimal_context(8)
    context.key_enforced = True
    gen = RuleBasedMelodyGenerator()
    found_run = False
    for seed in range(80):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="busy"))
        ordered = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))
        pentatonic_pcs = {0, 2, 4, 7, 9}
        for i in range(len(ordered) - 3):
            window = ordered[i : i + 4]
            if not all(float(n.duration) <= 0.34 for n in window):
                continue
            pcs = [n.pitch % 12 for n in window]
            if all(pc in pentatonic_pcs for pc in pcs):
                semis = [abs(window[j + 1].pitch - window[j].pitch) for j in range(3)]
                if all(s in (1, 2, 3, 4, 5) for s in semis):
                    found_run = True
                    break
        if found_run:
            break
    assert found_run


def test_fourth_on_subdominant_chord_strong_beat():
    """Scale degree 4 should appear on downbeats over the IV chord."""
    from fractions import Fraction

    chords = [
        ChordEvent(bar=1, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C"),
        ChordEvent(bar=2, beat=Fraction(0), root=5, quality="major", pitch_classes={5, 9, 0}, name="F"),
        ChordEvent(bar=3, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C"),
        ChordEvent(bar=4, beat=Fraction(0), root=7, quality="major", pitch_classes={7, 11, 2}, name="G"),
    ]
    context = LoopContext(
        tempo_bpm=120.0, time_signature=(4, 4), bars=4, ppq=480,
        key_root=0, key_mode="major", key_enforced=True,
        scale_pitch_classes={0, 2, 4, 5, 7, 9, 11},
        chords=chords, tracks={}, bass_roots_by_bar={1: 0, 2: 5, 3: 0, 4: 7},
        rhythm_density=0.5,
    )
    gen = RuleBasedMelodyGenerator()
    fourth_on_iv = 0
    iv_downbeats = 0
    for seed in range(60):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="medium"))
        for n in clip.notes:
            if n.bar != 2:
                continue
            if float(n.position) % 1 == 0.0:
                iv_downbeats += 1
                if n.pitch % 12 == 5:  # F = scale degree 4 in C major
                    fourth_on_iv += 1
    assert iv_downbeats > 0
    assert fourth_on_iv / iv_downbeats >= 0.02


def test_seventh_can_end_loop_on_dominant():
    """Leading tone (scale degree 7) may end the loop when it is a chord tone of V."""
    from fractions import Fraction

    chords = [
        ChordEvent(bar=i, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C")
        for i in range(1, 4)
    ] + [
        ChordEvent(bar=4, beat=Fraction(0), root=7, quality="major", pitch_classes={7, 11, 2}, name="G"),
    ]
    context = LoopContext(
        tempo_bpm=120.0, time_signature=(4, 4), bars=4, ppq=480,
        key_root=0, key_mode="major", key_enforced=True,
        scale_pitch_classes={0, 2, 4, 5, 7, 9, 11},
        chords=chords, tracks={}, bass_roots_by_bar={1: 0, 2: 0, 3: 0, 4: 7},
        rhythm_density=0.5,
    )
    gen = RuleBasedMelodyGenerator()
    leading_endings = 0
    for seed in range(80):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="medium"))
        last = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))[-1]
        if last.pitch % 12 == 11:  # B = leading tone in C major
            leading_endings += 1
    assert leading_endings >= 3


def test_seventh_only_on_loop_ending_dominant():
    """Leading tone must not appear except as the final note over V."""
    from fractions import Fraction

    chords = [
        ChordEvent(bar=1, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C"),
        ChordEvent(bar=2, beat=Fraction(0), root=5, quality="major", pitch_classes={5, 9, 0}, name="F"),
        ChordEvent(bar=3, beat=Fraction(0), root=9, quality="minor", pitch_classes={9, 0, 4}, name="Am"),
        ChordEvent(bar=4, beat=Fraction(0), root=7, quality="major", pitch_classes={7, 11, 2}, name="G"),
    ]
    context = LoopContext(
        tempo_bpm=120.0, time_signature=(4, 4), bars=4, ppq=480,
        key_root=0, key_mode="major", key_enforced=True,
        scale_pitch_classes={0, 2, 4, 5, 7, 9, 11},
        chords=chords, tracks={}, bass_roots_by_bar={1: 0, 2: 5, 3: 9, 4: 7},
        rhythm_density=0.5,
    )
    gen = RuleBasedMelodyGenerator()
    for seed in range(60):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="medium"))
        ordered = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))
        last = ordered[-1]
        for n in ordered[:-1]:
            assert n.pitch % 12 != 11


def test_fourth_only_on_iv_vi_downbeat():
    """Scale degree 4 may only sound on beat 1 over IV or vi."""
    from fractions import Fraction
    from logicians.melodic_taste import is_predominant_fourth_chord

    chords = [
        ChordEvent(bar=1, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C"),
        ChordEvent(bar=2, beat=Fraction(0), root=5, quality="major", pitch_classes={5, 9, 0}, name="F"),
        ChordEvent(bar=3, beat=Fraction(0), root=9, quality="minor", pitch_classes={9, 0, 4}, name="Am"),
        ChordEvent(bar=4, beat=Fraction(0), root=7, quality="major", pitch_classes={7, 11, 2}, name="G"),
    ]
    context = LoopContext(
        tempo_bpm=120.0, time_signature=(4, 4), bars=4, ppq=480,
        key_root=0, key_mode="major", key_enforced=True,
        scale_pitch_classes={0, 2, 4, 5, 7, 9, 11},
        chords=chords, tracks={}, bass_roots_by_bar={1: 0, 2: 5, 3: 9, 4: 7},
        rhythm_density=0.5,
    )
    gen = RuleBasedMelodyGenerator()
    chord_by_bar = {c.bar: c for c in chords}
    for seed in range(60):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="medium"))
        for n in clip.notes:
            if n.pitch % 12 != 5:
                continue
            chord = chord_by_bar[n.bar]
            assert is_predominant_fourth_chord(chord, context)
            assert float(n.position) % 1 == 0.0


def test_no_excessive_adjacent_scale_alternation():
    """Melodies should not ping-pong between adjacent scale degrees outside fast scalic runs."""
    from logicians.melodic_taste import scale_degree_of_pc, trailing_scale_alternations

    context = _minimal_context(8)
    context.key_enforced = True
    gen = RuleBasedMelodyGenerator()
    for seed in range(80):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="busy"))
        ordered = sorted(clip.notes, key=lambda x: (x.bar, float(x.position)))
        protected = gen._arpeggio_landing_indices(ordered, context)
        degrees: list[int] = []
        for i, n in enumerate(ordered):
            if i in protected:
                continue
            degree = scale_degree_of_pc(context, n.pitch % 12)
            if degree is not None:
                degrees.append(degree)
        for end in range(2, len(degrees) + 1):
            assert trailing_scale_alternations(degrees[:end]) <= 9


def test_stepwise_motion_preferred():
    context = _minimal_context(4)
    gen = RuleBasedMelodyGenerator()
    clip = gen.generate(context, GenerationOptions(seed=42))
    ordered = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))
    leaps = [abs(ordered[i].pitch - ordered[i - 1].pitch) for i in range(1, len(ordered))]
    assert leaps
    assert max(leaps) <= 5
    step_ratio = sum(1 for leap in leaps if leap <= 2) / len(leaps)
    assert step_ratio >= 0.65


def test_loop_ending_can_hold_whole_bar():
    """Last bar may pause on a long note, but only occasionally — not every loop."""
    from fractions import Fraction

    chords = [
        ChordEvent(bar=i, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C")
        for i in range(1, 4)
    ] + [
        ChordEvent(bar=4, beat=Fraction(0), root=7, quality="major", pitch_classes={7, 11, 2}, name="G"),
    ]
    context = LoopContext(
        tempo_bpm=120.0, time_signature=(4, 4), bars=4, ppq=480,
        key_root=0, key_mode="major", key_enforced=True,
        scale_pitch_classes={0, 2, 4, 5, 7, 9, 11},
        chords=chords, tracks={}, bass_roots_by_bar={1: 0, 2: 0, 3: 0, 4: 7},
        rhythm_density=0.5,
    )
    gen = RuleBasedMelodyGenerator()
    long_holds = 0
    for seed in range(80):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="medium"))
        bar4 = [n for n in clip.notes if n.bar == 4]
        if len(bar4) == 1 and float(bar4[0].duration) >= 3.5:
            long_holds += 1
    assert 2 <= long_holds <= 28


def test_loop_breath_alternates_across_iterations():
    """After a long loop ending, the next improvisation should usually stay active."""
    from fractions import Fraction

    chords = [
        ChordEvent(bar=i, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C")
        for i in range(1, 4)
    ] + [
        ChordEvent(bar=4, beat=Fraction(0), root=7, quality="major", pitch_classes={7, 11, 2}, name="G"),
    ]
    context = LoopContext(
        tempo_bpm=120.0, time_signature=(4, 4), bars=4, ppq=480,
        key_root=0, key_mode="major", key_enforced=True,
        scale_pitch_classes={0, 2, 4, 5, 7, 9, 11},
        chords=chords, tracks={}, bass_roots_by_bar={1: 0, 2: 0, 3: 0, 4: 7},
        rhythm_density=0.5,
    )
    gen = RuleBasedMelodyGenerator()
    long_then_active = 0
    for seed in range(40):
        first = gen.generate(context, GenerationOptions(seed=seed, density="medium"))
        bar4 = [n for n in first.notes if n.bar == 4]
        if not (len(bar4) == 1 and float(bar4[0].duration) >= 3.5):
            continue
        second = gen.generate(
            context,
            GenerationOptions(seed=seed + 1000, previous_clip=first, iteration=1, density="medium"),
        )
        bar4_next = [n for n in second.notes if n.bar == 4]
        if len(bar4_next) >= 2 or (
            len(bar4_next) == 1 and float(bar4_next[0].duration) < 3.0
        ):
            long_then_active += 1
    assert long_then_active >= 3


def test_overlapping_pentatonic_pattern():
    """Scalic runs should use overlapping 3-note groups (e.g. C-D-E-D-E-G), not skip-ahead pairs."""
    from logicians.generator import RuleBasedMelodyGenerator

    context = _minimal_context(8)
    context.key_enforced = True
    gen = RuleBasedMelodyGenerator()
    pool = gen._pentatonic_pitches_in_register(context, GenerationOptions(seed=0))

    expected = gen._build_overlapping_pentatonic_pitches(pool, 0, 4, 1)
    assert expected[:6] == [pool[0], pool[1], pool[2], pool[1], pool[2], pool[3]]

    found = False
    for seed in range(80):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="busy"))
        ordered = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))
        for start in range(len(ordered) - 5):
            window = ordered[start : start + 6]
            pitches = [n.pitch for n in window]
            if not all(float(n.duration) <= 0.34 for n in window):
                continue
            for idx in range(len(pool) - 5):
                target = gen._build_overlapping_pentatonic_pitches(pool, idx, 2, 1)
                if pitches[: len(target)] == target:
                    found = True
                    break
            if found:
                break
        if found:
            break
    assert found


def test_triplet_scalic_runs_appear_sometimes():
    """Full-bar or partial-bar pentatonic runs may use triplet quavers."""
    from fractions import Fraction

    context = _minimal_context(8)
    context.key_enforced = True
    gen = RuleBasedMelodyGenerator()
    found = False
    for seed in range(80):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="busy"))
        triplet_runs = [
            n for n in clip.notes
            if abs(float(n.duration) - float(Fraction(1, 3))) < 0.02
        ]
        if len(triplet_runs) >= 6:
            found = True
            break
    assert found


def test_triplets_only_in_allowed_bar_zones():
    """Triplets may only fill a bar, its first half, or its last half."""
    from fractions import Fraction

    context = _minimal_context(8)
    context.key_enforced = True
    gen = RuleBasedMelodyGenerator()
    beats_per_bar = context.time_signature[0]
    bar_end = Fraction(beats_per_bar)
    half = bar_end / 2

    for seed in range(80):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="busy"))
        ordered = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))

        by_bar: dict[int, list[MelodyNote]] = {}
        for note in ordered:
            by_bar.setdefault(note.bar, []).append(note)

        for bar, bar_notes in by_bar.items():
            triplet_runs: list[list[MelodyNote]] = []
            current: list[MelodyNote] = []
            for note in bar_notes:
                if abs(float(note.duration) - float(Fraction(1, 3))) >= 0.02:
                    if current:
                        triplet_runs.append(current)
                        current = []
                    continue
                if current and not gen._notes_are_adjacent(current[-1], note):
                    triplet_runs.append(current)
                    current = [note]
                else:
                    current.append(note)
            if current:
                triplet_runs.append(current)

            for run in triplet_runs:
                assert gen._is_valid_triplet_run(run, bar_end, half), (
                    f"seed {seed}: invalid triplet run in bar {bar} "
                    f"({len(run)} notes from {run[0].position})"
                )


def test_arpeggio_run_landings_favor_chord_tones():
    """Disjunct pentatonic runs should land on chord tones, not random scale degrees."""
    context = _minimal_context(8)
    context.key_enforced = True
    chords = [
        ChordEvent(bar=1, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C"),
        ChordEvent(bar=2, beat=Fraction(0), root=5, quality="major", pitch_classes={5, 9, 0}, name="F"),
        ChordEvent(bar=3, beat=Fraction(0), root=7, quality="major", pitch_classes={7, 11, 2}, name="G"),
        ChordEvent(bar=4, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C"),
        ChordEvent(bar=5, beat=Fraction(0), root=5, quality="major", pitch_classes={5, 9, 0}, name="F"),
        ChordEvent(bar=6, beat=Fraction(0), root=9, quality="minor", pitch_classes={9, 0, 4}, name="Am"),
        ChordEvent(bar=7, beat=Fraction(0), root=7, quality="major", pitch_classes={7, 11, 2}, name="G"),
        ChordEvent(bar=8, beat=Fraction(0), root=0, quality="major", pitch_classes={0, 4, 7}, name="C"),
    ]
    context.chords = chords
    context.bass_roots_by_bar = {c.bar: c.root for c in chords}
    gen = RuleBasedMelodyGenerator()
    chord_by_bar = {c.bar: c for c in chords}
    landing_total = 0
    chord_tone_landings = 0
    for seed in range(60):
        clip = gen.generate(context, GenerationOptions(seed=seed, density="busy"))
        ordered = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))
        protected = gen._arpeggio_landing_indices(ordered, context)
        cluster_ends: list[int] = []
        if protected:
            sorted_protected = sorted(protected)
            group = [sorted_protected[0]]
            for idx in sorted_protected[1:]:
                if idx == group[-1] + 1:
                    group.append(idx)
                else:
                    if len(group) >= 4:
                        cluster_ends.append(group[-1])
                    group = [idx]
            if len(group) >= 4:
                cluster_ends.append(group[-1])
        for idx in cluster_ends:
            landing = ordered[idx]
            chord_pcs = chord_by_bar[landing.bar].pitch_classes
            landing_total += 1
            if landing.pitch % 12 in chord_pcs:
                chord_tone_landings += 1
    assert landing_total > 0
    assert chord_tone_landings / landing_total >= 0.55
