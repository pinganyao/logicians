"""Idiomatic melody preferences drawn from tonal voice-leading and jazz pedagogy.

References:
- Open Music Theory: tendency tones and functional dissonances
- Berklee / Nettles: avoid notes (m9 clashes above chord tones)
- Caplin / jazz pedagogy: guide tones (3rd & 7th) for voice leading
- OtoTheory / Macalester: scale-degree stability by harmonic context
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .models import ChordEvent, LoopContext

HarmonicFunction = Literal["tonic", "predominant", "dominant", "other"]
PitchRole = Literal["strong", "weak", "cadence", "anticipation"]
CadenceType = Literal["HC", "PAC", "none"]

_MAJOR_DEGREES = [0, 2, 4, 5, 7, 9, 11]
_MINOR_DEGREES = [0, 2, 3, 5, 7, 8, 10]

# Scale-degree stability when *landing* on a strong beat (1=most stable).
_LANDING_STABILITY: dict[int, float] = {
    1: 1.0,   # do — home
    3: 0.95,  # mi — bright, stable
    5: 0.9,   # sol — grounded
    6: 0.55,  # la — slightly floating
    2: 0.35,  # re — wants to move
    4: 0.25,  # fa — tense over many harmonies
    7: 0.2,   # ti — strong tendency tone
}


@dataclass(frozen=True)
class TasteContext:
    chord: ChordEvent
    next_chord: ChordEvent
    loop: LoopContext
    pitch_role: PitchRole
    is_ending: bool
    last_pitch: int | None
    beat_position: Fraction
    bar_cadence: CadenceType


def scale_degree_pc(loop: LoopContext, degree: int) -> int:
    degrees = _MAJOR_DEGREES if loop.key_mode == "major" else _MINOR_DEGREES
    return (loop.key_root + degrees[(degree - 1) % 7]) % 12


def scale_degree_of_pc(loop: LoopContext, pc: int) -> int | None:
    rel = (pc - loop.key_root) % 12
    degrees = _MAJOR_DEGREES if loop.key_mode == "major" else _MINOR_DEGREES
    for i, interval in enumerate(degrees, start=1):
        if interval == rel:
            return i
    return None


def chord_harmonic_function(chord: ChordEvent, loop: LoopContext) -> HarmonicFunction:
    rel = (chord.root - loop.key_root) % 12
    if rel == 0:
        return "tonic"
    if rel in (5, 2):
        return "predominant"
    if rel in (7, 11):
        return "dominant"
    if rel in (3, 4, 9):
        return "predominant"
    return "other"


def _chord_third_pc(chord: ChordEvent) -> int:
    if "min" in chord.quality and "maj7" not in chord.quality.replace("minMaj7", ""):
        return (chord.root + 3) % 12
    return (chord.root + 4) % 12


def _chord_seventh_pc(chord: ChordEvent) -> int | None:
    if chord.quality in ("dom7", "min7", "maj7", "minMaj7"):
        if chord.quality == "maj7":
            return (chord.root + 11) % 12
        return (chord.root + 10) % 12
    return None


def guide_tone_pcs(chord: ChordEvent) -> set[int]:
    pcs: set[int] = {_chord_third_pc(chord)}
    seventh = _chord_seventh_pc(chord)
    if seventh is not None:
        pcs.add(seventh)
    return pcs & chord.pitch_classes


def _is_minor_ninth_above(pc: int, chord_tone_pc: int) -> bool:
    return (pc - chord_tone_pc) % 12 == 1


def avoid_penalty(pc: int, taste: TasteContext) -> float:
    """0 = fine, 1 = strongly avoid landing here."""
    chord = taste.chord
    loop = taste.loop
    chord_pcs = chord.pitch_classes
    if loop.key_enforced:
        chord_pcs = chord_pcs & loop.scale_pitch_classes

    penalty = 0.0
    degree = scale_degree_of_pc(loop, pc)
    function = chord_harmonic_function(chord, loop)
    is_strong = taste.pitch_role in ("strong", "cadence") or taste.is_ending

    # Berklee avoid note: 4th scale degree over major chord (m9 above the 3rd).
    fourth = scale_degree_pc(loop, 4)
    third = _chord_third_pc(chord)
    is_major_family = chord.quality in ("major", "maj7", "dom7")
    if pc == fourth and is_major_family and _is_minor_ninth_above(pc, third):
        penalty = max(penalty, 0.9 if is_strong else 0.45)

    # 6th scale degree over minor chord (harsh 13th on minor harmonies).
    sixth = scale_degree_pc(loop, 6)
    if pc == sixth and "min" in chord.quality and pc not in chord_pcs:
        penalty = max(penalty, 0.75 if is_strong else 0.35)

    # Leading tone on tonic harmony — functional dissonance if sustained.
    leading = scale_degree_pc(loop, 7)
    if pc == leading and function == "tonic" and pc not in chord_pcs:
        penalty = max(penalty, 0.85 if is_strong else 0.3)

    # Leading tone on predominant — often sounds like wrong chord tone.
    if pc == leading and function == "predominant" and pc not in chord_pcs:
        penalty = max(penalty, 0.55 if is_strong else 0.2)

    # Subdominant scale degree on dominant chord without being chord tone.
    subdominant = scale_degree_pc(loop, 4)
    if pc == subdominant and function == "dominant" and pc not in chord_pcs:
        penalty = max(penalty, 0.4 if is_strong else 0.15)

    # Major 7th of key on V7 — can clash with root of melody in some voicings.
    if degree == 7 and function == "dominant" and pc in chord_pcs:
        pass  # chord tone 7th on V is good

    # Unstable degrees on strong beats (outside chord).
    if is_strong and pc not in chord_pcs and degree is not None:
        instability = 1.0 - _LANDING_STABILITY.get(degree, 0.5)
        penalty = max(penalty, instability * 0.7)

    # 2nd degree sustained on strong beat — floaty.
    if degree == 2 and is_strong and pc not in chord_pcs:
        penalty = max(penalty, 0.5)

    return min(1.0, penalty)


def tendency_bonus(pc: int, taste: TasteContext) -> float:
    """Reward resolutions that honour tendency tones."""
    if taste.last_pitch is None:
        return 0.0

    loop = taste.loop
    last_pc = taste.last_pitch % 12
    bonus = 0.0
    leading = scale_degree_pc(loop, 7)
    tonic = scale_degree_pc(loop, 1)
    subdominant = scale_degree_pc(loop, 4)
    third = _chord_third_pc(taste.chord)

    # ti → do
    if last_pc == leading and pc == tonic:
        bonus += 0.35

    # fa → mi (4th resolving down to 3rd of chord)
    if last_pc == subdominant and pc == third:
        bonus += 0.3

    # 2nd → 1st or 3rd stepwise
    second = scale_degree_pc(loop, 2)
    if last_pc == second and pc in {tonic, third}:
        bonus += 0.2

    # 7th of chord resolving down by step (fa → mi in dominant)
    seventh = _chord_seventh_pc(taste.chord)
    if seventh is not None and last_pc == seventh:
        step_down = (last_pc - 1) % 12
        if pc == step_down:
            bonus += 0.25

    return bonus


def guide_tone_bonus(pc: int, taste: TasteContext) -> float:
    """Prefer 3rds/7ths, especially at chord boundaries."""
    guides = guide_tone_pcs(taste.chord)
    if pc not in guides:
        return 0.0

    bonus = 0.15
    if taste.pitch_role in ("strong", "cadence", "anticipation"):
        bonus += 0.15

    if taste.pitch_role == "anticipation":
        next_guides = guide_tone_pcs(taste.next_chord)
        if pc in next_guides:
            bonus += 0.25

    function = chord_harmonic_function(taste.chord, taste.loop)
    if function == "dominant" and pc == _chord_third_pc(taste.chord):
        bonus += 0.1

    return bonus


def chord_tone_weight(pc: int, taste: TasteContext) -> float:
    """Weighted preference among chord tones — not all are equally idiomatic."""
    chord = taste.chord
    if pc not in chord.pitch_classes:
        return 0.0

    weight = 1.0
    function = chord_harmonic_function(chord, taste.loop)
    third = _chord_third_pc(chord)
    fifth = (chord.root + 7) % 12
    seventh = _chord_seventh_pc(chord)

    if pc == chord.root:
        weight = 1.1 if function == "tonic" else 0.85
    elif pc == third:
        weight = 1.2
    elif pc == fifth:
        weight = 0.95
    elif seventh is not None and pc == seventh:
        weight = 1.15 if function == "dominant" else 0.9

    if taste.pitch_role == "cadence" and function == "tonic":
        if pc in {chord.root, third}:
            weight += 0.3

    return weight


def melodic_suitability(pc: int, taste: TasteContext) -> float:
    """Overall idiomatic score for a pitch class in context (higher = better)."""
    score = 1.0
    score -= avoid_penalty(pc, taste)
    score += tendency_bonus(pc, taste)
    score += guide_tone_bonus(pc, taste)

    if pc in taste.chord.pitch_classes:
        score += 0.1 * chord_tone_weight(pc, taste)

    degree = scale_degree_of_pc(taste.loop, pc)
    if degree is not None and taste.pitch_role in ("strong", "cadence"):
        score += 0.1 * _LANDING_STABILITY.get(degree, 0.5)

    return score


def best_pitch_among(
    candidates: list[int],
    taste: TasteContext,
    rng,
    min_score: float = 0.25,
) -> int:
    """Pick the most idiomatic pitch, with light randomness among good options."""
    if not candidates:
        raise ValueError("No pitch candidates")

    scored = [(p, melodic_suitability(p % 12, taste)) for p in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_score = scored[0][1]
    if top_score < min_score:
        return scored[0][0]

    viable = [p for p, s in scored if s >= top_score - 0.12 and s >= min_score]
    weights = [max(0.05, melodic_suitability(p % 12, taste)) for p in viable]
    return rng.choices(viable, weights=weights, k=1)[0]


def passing_tone_allowed(pc: int, taste: TasteContext) -> bool:
    """Non-chord tones may appear on weak beats only if not harsh avoid notes."""
    if taste.pitch_role in ("strong", "cadence"):
        return False
    return avoid_penalty(pc, taste) < 0.55
