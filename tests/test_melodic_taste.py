"""Tests for idiomatic melody preference scoring."""

from fractions import Fraction

from logicians.melodic_taste import (
    TasteContext,
    avoid_penalty,
    best_pitch_among,
    melodic_suitability,
)
from logicians.models import ChordEvent, LoopContext


def _ctx(chord_root: int = 0, quality: str = "maj7", key_mode: str = "major") -> LoopContext:
    from logicians.analysis import CHORD_QUALITIES

    pcs = {(chord_root + i) % 12 for i in CHORD_QUALITIES[quality]}
    chords = [
        ChordEvent(
            bar=1, beat=Fraction(0), root=chord_root, quality=quality,
            pitch_classes=pcs, name="C",
        )
    ]
    scale = {0, 2, 4, 5, 7, 9, 11} if key_mode == "major" else {0, 2, 3, 5, 7, 8, 10}
    return LoopContext(
        tempo_bpm=120.0,
        time_signature=(4, 4),
        bars=4,
        ppq=480,
        key_root=0,
        key_mode=key_mode,
        scale_pitch_classes=scale,
        chords=chords,
        tracks={},
        bass_roots_by_bar={1: chord_root},
        rhythm_density=0.5,
        key_enforced=True,
    )


def _taste(loop, pitch_role="strong", last_pitch=None):
    return TasteContext(
        chord=loop.chords[0],
        next_chord=loop.chords[0],
        loop=loop,
        pitch_role=pitch_role,
        is_ending=False,
        last_pitch=last_pitch,
        beat_position=Fraction(0),
        bar_cadence="none",
    )


def test_avoid_fourth_over_major_on_strong_beat():
    loop = _ctx(quality="maj7")
    taste = _taste(loop, pitch_role="strong")
    assert avoid_penalty(5, taste) > avoid_penalty(4, taste)  # F worse than E on Cmaj7
    assert avoid_penalty(5, taste) > 0.5


def test_leading_tone_penalized_on_tonic():
    loop = _ctx(chord_root=0, quality="major")
    taste = _taste(loop, pitch_role="strong")
    assert avoid_penalty(11, taste) > 0.5  # B on C major (I) — not a chord tone


def test_tendency_resolution_prefers_tonic_after_leading():
    loop = _ctx(quality="dom7", chord_root=7)  # G7
    taste = _taste(loop, pitch_role="strong", last_pitch=71)  # B
    score_tonic = melodic_suitability(0, taste)  # C
    score_fourth = melodic_suitability(5, taste)  # F
    assert score_tonic > score_fourth


def test_best_pitch_among_prefers_chord_tone():
    import random

    loop = _ctx(quality="maj7")
    taste = _taste(loop)
    rng = random.Random(1)
    candidates = [60, 62, 64, 65, 67, 71]  # C D E F G B
    pick = best_pitch_among(candidates, taste, rng)
    assert pick % 12 in {0, 4, 7, 11}
