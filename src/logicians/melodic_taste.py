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

_PENTATONIC_DEGREES = frozenset({1, 2, 3, 5, 6})


def is_subdominant_chord(chord: ChordEvent, loop: LoopContext) -> bool:
    """True when the sounding chord is the IV (subdominant) of the key."""
    rel = (chord.root - loop.key_root) % 12
    if loop.key_mode == "major":
        return rel == 5
    return rel == 5 or rel == 3  # iv or IV in minor


def is_submediant_chord(chord: ChordEvent, loop: LoopContext) -> bool:
    """True when the sounding chord is vi (major) or bVI (minor)."""
    rel = (chord.root - loop.key_root) % 12
    if loop.key_mode == "major":
        return rel == 9
    return rel == 8


def is_predominant_fourth_chord(chord: ChordEvent, loop: LoopContext) -> bool:
    """IV or vi — the only harmonies where scale degree 4 may appear."""
    return is_subdominant_chord(chord, loop) or is_submediant_chord(chord, loop)


def is_main_beat(beat_position: Fraction) -> bool:
    """Downbeat of the bar (beat 1), not the half-beat."""
    return beat_position % 1 == Fraction(0)


def fourth_degree_allowed(pc: int, taste: TasteContext) -> bool:
    """Scale degree 4 only on downbeats over IV or vi."""
    fourth_pc = scale_degree_pc(taste.loop, 4)
    if pc != fourth_pc:
        return True
    return (
        is_predominant_fourth_chord(taste.chord, taste.loop)
        and is_main_beat(taste.beat_position)
    )


def seventh_degree_allowed(pc: int, taste: TasteContext) -> bool:
    """Leading tone only as the final note when the loop ends on V."""
    leading_pc = scale_degree_pc(taste.loop, 7)
    if pc != leading_pc:
        return True
    return (
        taste.is_loop_ending
        and chord_harmonic_function(taste.chord, taste.loop) == "dominant"
    )


def is_pentatonic_pc(loop: LoopContext, pc: int) -> bool:
    degree = scale_degree_of_pc(loop, pc)
    return degree is not None and degree in _PENTATONIC_DEGREES


def pentatonic_bonus(pc: int, taste: TasteContext) -> float:
    if not is_pentatonic_pc(taste.loop, pc):
        return 0.0
    bonus = 0.14
    if taste.pitch_role in ("weak", "anticipation"):
        bonus += 0.06
    return bonus


@dataclass(frozen=True)
class TasteContext:
    chord: ChordEvent
    next_chord: ChordEvent
    loop: LoopContext
    pitch_role: PitchRole
    is_ending: bool
    is_loop_ending: bool
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

  # Scale degree 4: only on downbeats over IV or vi.
    fourth_pc = scale_degree_pc(loop, 4)
    if degree == 4 and pc == fourth_pc and not fourth_degree_allowed(pc, taste):
        penalty = max(penalty, 0.95)

    leading = scale_degree_pc(loop, 7)
    if degree == 7 and pc == leading and not seventh_degree_allowed(pc, taste):
        penalty = max(penalty, 0.95)

    # Berklee avoid note: 4th over major chord (m9 above the 3rd).
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

    # Phrase endings: consonant chord tones; 7th only on loop-ending dominant.
    if taste.is_ending or taste.pitch_role == "cadence":
        if pc not in chord_pcs:
            penalty = max(penalty, 0.95)
        elif degree == 4 and not fourth_degree_allowed(pc, taste):
            penalty = max(penalty, 0.95)
        elif degree == 7 and not seventh_degree_allowed(pc, taste):
            penalty = max(penalty, 0.95)

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
        if is_subdominant_chord(chord, taste.loop) and taste.pitch_role in ("strong", "cadence"):
            weight += 0.45
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


def contextual_degree_bonus(pc: int, taste: TasteContext) -> float:
    """Reward 4th/7th only in their tightly restricted contexts."""
    degree = scale_degree_of_pc(taste.loop, pc)
    if degree is None:
        return 0.0

    bonus = 0.0
    fourth_pc = scale_degree_pc(taste.loop, 4)
    leading_pc = scale_degree_pc(taste.loop, 7)

    if degree == 4 and pc == fourth_pc and fourth_degree_allowed(pc, taste):
        bonus += 0.28

    if degree == 7 and pc == leading_pc and seventh_degree_allowed(pc, taste):
        bonus += 0.32

    return bonus


def melodic_suitability(pc: int, taste: TasteContext) -> float:
    """Overall idiomatic score for a pitch class in context (higher = better)."""
    score = 1.0
    score -= avoid_penalty(pc, taste)
    score += tendency_bonus(pc, taste)
    score += guide_tone_bonus(pc, taste)
    score += pentatonic_bonus(pc, taste)
    score += contextual_degree_bonus(pc, taste)

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
    degree = scale_degree_of_pc(taste.loop, pc)
    if degree in (4, 7):
        return False
    return avoid_penalty(pc, taste) < 0.45


def are_adjacent_scale_degrees(degree_a: int, degree_b: int) -> bool:
    """True when two scale degrees are neighbours (including 7↔1)."""
    return (degree_b - degree_a) % 7 in (1, 6)


def trailing_scale_alternations(degrees: list[int]) -> int:
    """Count consecutive alternations between the same adjacent degree pair at the end."""
    if len(degrees) < 2:
        return 0

    streak = 0
    pair: tuple[int, int] | None = None
    for i in range(len(degrees) - 1, 0, -1):
        d_curr, d_prev = degrees[i], degrees[i - 1]
        if d_curr == d_prev or not are_adjacent_scale_degrees(d_prev, d_curr):
            break
        normalized = (min(d_prev, d_curr), max(d_prev, d_curr))
        if pair is None:
            pair = normalized
        elif normalized != pair:
            break
        streak += 1
    return streak


def would_exceed_alternation_limit(
    loop: LoopContext,
    recent_pitches: list[int],
    candidate_pitch: int,
    max_alternations: int = 3,
) -> bool:
    """True when adding candidate would continue an A-B-A-B… streak past the limit."""
    degrees: list[int] = []
    for pitch in recent_pitches:
        degree = scale_degree_of_pc(loop, pitch % 12)
        if degree is not None:
            degrees.append(degree)
    candidate_degree = scale_degree_of_pc(loop, candidate_pitch % 12)
    if candidate_degree is None or not degrees:
        return False

    last_degree = degrees[-1]
    if (
        candidate_degree == last_degree
        or not are_adjacent_scale_degrees(last_degree, candidate_degree)
    ):
        return False

    pair = (min(last_degree, candidate_degree), max(last_degree, candidate_degree))
    streak = trailing_scale_alternations(degrees)
    if streak == 0:
        return False

    # Confirm the trailing streak uses the same adjacent pair.
    d_curr, d_prev = degrees[-1], degrees[-2]
    if (min(d_prev, d_curr), max(d_prev, d_curr)) != pair:
        return False

    return streak >= max_alternations
