"""Rule-based melody generator implementing the MelodyGenerator protocol."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from .models import (
    ChordEvent,
    GenerationOptions,
    LoopContext,
    MelodyClip,
    MelodyNote,
)
from .quantize import ALLOWED_DURATIONS, POSITION_GRID, quantize_duration, quantize_position
from .melodic_taste import (
    TasteContext,
    best_pitch_among,
    guide_tone_pcs,
    melodic_suitability,
    passing_tone_allowed,
    scale_degree_pc,
)

ContourType = Literal["arch", "inverted_arch", "ascending", "descending", "static_hook", "call_response"]
PhraseSection = Literal["A", "A_prime", "B", "B_prime", "C", "C_prime", "D", "resolution"]
FormalRole = Literal["presentation", "antecedent", "continuation", "fragmentation", "cadence"]
CadenceType = Literal["HC", "PAC", "none"]
PitchRole = Literal["strong", "weak", "cadence", "anticipation"]

QUAVER = Fraction(1, 2)       # eighth note
SEMI = Fraction(1, 4)         # sixteenth note
CROTCHET = Fraction(1, 1)     # quarter note
MINIM = Fraction(2, 1)        # half note

# Metric rhythmic figures — durations in beats (no 32nd/64th notes).
RHYTHM_FIGURES: dict[str, list[Fraction]] = {
    "two_quavers": [QUAVER, QUAVER],
    "four_quavers": [QUAVER, QUAVER, QUAVER, QUAVER],
    "crotchet": [CROTCHET],
    "minim": [MINIM],
    "quavers_crotchet": [QUAVER, QUAVER, CROTCHET],
    "crotchet_quavers": [CROTCHET, QUAVER, QUAVER],
    "semi_pair_quavers": [SEMI, SEMI, QUAVER, QUAVER],
    "quaver_semi_pair": [QUAVER, SEMI, SEMI, QUAVER],
    "four_semis": [SEMI, SEMI, SEMI, SEMI],
}

ONE_BEAT_FIGURES = ["two_quavers", "crotchet", "four_semis", "semi_pair_quavers", "quaver_semi_pair"]
TWO_BEAT_FIGURES = ["four_quavers", "quavers_crotchet", "crotchet_quavers"]

DENSITY_PRESETS = {
    "sparse": {
        "rest_prob": 0.35,
        "scalic_prob": 0.0,
        "cells_per_bar": (1, 2),
    },
    "medium": {
        "rest_prob": 0.28,
        "scalic_prob": 0.0,
        "cells_per_bar": (2, 3),
    },
    "busy": {
        "rest_prob": 0.18,
        "scalic_prob": 0.05,
        "cells_per_bar": (2, 4),
    },
}

# Pop-friendly rhythmic cells (singable, not busy).
POP_HOOK_RHYTHMS: list[list[Fraction]] = [
    [CROTCHET, QUAVER, QUAVER, CROTCHET],
    [QUAVER, QUAVER, CROTCHET, CROTCHET],
    [CROTCHET, CROTCHET, QUAVER, QUAVER],
    [QUAVER, QUAVER, QUAVER, QUAVER, CROTCHET],
]
POP_ANSWER_RHYTHMS: list[list[Fraction]] = [
    [QUAVER, QUAVER, CROTCHET, CROTCHET],
    [CROTCHET, QUAVER, QUAVER, CROTCHET],
    [QUAVER, QUAVER, QUAVER, QUAVER],
]
POP_RESOLUTION_RHYTHMS: list[list[Fraction]] = [
    [CROTCHET, CROTCHET, MINIM],
    [QUAVER, QUAVER, CROTCHET, CROTCHET],
    [CROTCHET, MINIM],
]

MAX_MELODIC_LEAP = 5  # minor 6th — pop melodies rarely exceed this
PREFERRED_LEAP = 2

INTERVAL_WEIGHTS = [
    ("repeat", 0.10),
    ("step", 0.55),
    ("small_skip", 0.28),
    ("large_leap", 0.06),
    ("very_large", 0.01),
]

STRONG_BEAT_WEIGHTS = (0.88, 0.10, 0.02)
WEAK_BEAT_WEIGHTS = (0.35, 0.50, 0.15)
ENDING_WEIGHTS = (0.92, 0.08, 0.0)

# Diatonic scale degrees (1-indexed) for major and natural minor.
_MAJOR_DEGREES = [0, 2, 4, 5, 7, 9, 11]
_MINOR_DEGREES = [0, 2, 3, 5, 7, 8, 10]


@dataclass
class Motif:
    intervals: list[int]
    durations: list[Fraction]
    anchor_pitch: int


@dataclass
class PhrasePlan:
    sections: list[tuple[int, PhraseSection]]
    contour: ContourType
    peak_bar: int
    peak_position: Fraction
    entry_position: Fraction
    formal_roles: dict[int, FormalRole] = field(default_factory=dict)
    cadences: dict[int, CadenceType] = field(default_factory=dict)


@dataclass
class LoopHandoff:
    """Boundary state passed from the end of one loop to the start of the next."""

    ending_pitch: int
    ending_velocity: int
    ending_bar: int
    ending_position: Fraction
    ended_with_rest: bool


@dataclass
class _GenState:
    rng: random.Random
    last_pitch: int | None = None
    phrase_anchor: int | None = None
    consecutive_large_leaps: int = 0
    consecutive_repeats: int = 0
    consecutive_chromatic: int = 0
    consecutive_32nds: int = 0
    pending_leap_recovery: bool = False


class RuleBasedMelodyGenerator:
    """Detailed rule-based melody generator as a placeholder for future ML model."""

    def generate(self, context: LoopContext, options: GenerationOptions) -> MelodyClip:
        rng = random.Random(options.seed)
        beats_per_bar = context.time_signature[0]

        density_key = self._resolve_density(options, context)
        preset = DENSITY_PRESETS[density_key]

        handoff = (
            self._extract_handoff(options.previous_clip, context)
            if options.previous_clip is not None
            else None
        )

        phrase_plan = self._plan_phrase(context, options, rng)
        if handoff is not None and not handoff.ended_with_rest:
            phrase_plan = PhrasePlan(
                sections=phrase_plan.sections,
                contour=phrase_plan.contour,
                peak_bar=phrase_plan.peak_bar,
                peak_position=phrase_plan.peak_position,
                entry_position=Fraction(0),
                formal_roles=phrase_plan.formal_roles,
                cadences=phrase_plan.cadences,
            )

        accent_map = self._build_accent_map(context, beats_per_bar)
        motif = self._create_motif(context, options, rng, preset)

        notes: list[MelodyNote] = []
        state = _GenState(rng=rng, phrase_anchor=motif.anchor_pitch)
        if handoff is not None:
            state.last_pitch = handoff.ending_pitch

        if context.bars == 4:
            notes = self._generate_pop_loop(
                context, options, phrase_plan, motif, state, beats_per_bar,
            )
        else:
            for bar, section in phrase_plan.sections:
                chord = self._chord_at_bar(context, bar)
                bar_notes = self._generate_bar(
                    bar=bar,
                    section=section,
                    chord=chord,
                    next_chord=self._chord_at_bar(context, min(bar + 1, context.bars)),
                    context=context,
                    options=options,
                    preset=preset,
                    phrase_plan=phrase_plan,
                    accent_map=accent_map.get(bar, {}),
                    motif=motif,
                    state=state,
                    beats_per_bar=beats_per_bar,
                    is_first_bar=(bar == 1),
                )
                notes.extend(bar_notes)

        notes = self._apply_entry_delay(notes, phrase_plan.entry_position, bar=1)
        if handoff is not None:
            notes = self._connect_loop_transition(notes, handoff, context, options, rng)
        notes = self._validate_and_repair(notes, context, options, beats_per_bar)

        return MelodyClip(bars=context.bars, notes=notes)

    def _resolve_density(self, options: GenerationOptions, context: LoopContext) -> str:
        return options.density

    def _plan_phrase(
        self, context: LoopContext, options: GenerationOptions, rng: random.Random
    ) -> PhrasePlan:
        """Plan phrase structure using period (4-bar) or sentence (8-bar) form.

        Period (Caplin/Open Music Theory): antecedent asks (HC), consequent answers (PAC).
        Sentence: presentation + continuation with fragmentation toward cadence.
        """
        bars = context.bars
        if bars == 4:
            sections: list[tuple[int, PhraseSection]] = [
                (1, "A"),
                (2, "A_prime"),
                (3, "B"),
                (4, "resolution"),
            ]
            formal_roles: dict[int, FormalRole] = {
                1: "presentation",
                2: "antecedent",
                3: "continuation",
                4: "cadence",
            }
            cadences: dict[int, CadenceType] = {2: "HC", 4: "PAC"}
            contour = rng.choice(["arch", "arch", "call_response"])
            peak_bar = 2
            peak_position = Fraction(3, 2)
        else:
            sections = [
                (1, "A"), (2, "A_prime"),
                (3, "B"), (4, "B_prime"),
                (5, "C"), (6, "C_prime"),
                (7, "D"), (8, "resolution"),
            ]
            formal_roles = {
                1: "presentation", 2: "presentation",
                3: "presentation", 4: "presentation",
                5: "continuation", 6: "continuation",
                7: "fragmentation", 8: "cadence",
            }
            cadences = {8: "PAC"}
            contour = rng.choice(["arch", "ascending", "arch"])
            peak_bar = rng.randint(5, 7)
            peak_position = Fraction(rng.choice([1, 3]), 2)

        if options.density == "sparse":
            entry = Fraction(1, 2)
        else:
            entry = Fraction(0)

        return PhrasePlan(
            sections=sections,
            contour=contour,  # type: ignore[arg-type]
            peak_bar=peak_bar,
            peak_position=peak_position,
            entry_position=entry,
            formal_roles=formal_roles,
            cadences=cadences,
        )

    def _extract_handoff(self, clip: MelodyClip, context: LoopContext) -> LoopHandoff:
        sorted_notes = sorted(clip.notes, key=lambda n: (n.bar, float(n.position)))
        if not sorted_notes:
            raise ValueError("Cannot extract handoff from empty clip")

        last = sorted_notes[-1]
        beats_per_bar = context.time_signature[0]
        last_end = float(last.position) + float(last.duration)
        ended_with_rest = last_end < beats_per_bar - float(QUAVER)

        return LoopHandoff(
            ending_pitch=last.pitch,
            ending_velocity=last.velocity,
            ending_bar=last.bar,
            ending_position=last.position,
            ended_with_rest=ended_with_rest,
        )

    def _connected_first_pitch(
        self,
        handoff: LoopHandoff,
        chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        rng: random.Random,
    ) -> int:
        """Pick an opening pitch that links to the previous loop without copying its shape."""
        ending = handoff.ending_pitch
        register_mid = (options.register_low + options.register_high) // 2

        strategy = rng.choices(
            ["step", "unison", "third", "response", "chord_tone"],
            weights=[0.35, 0.15, 0.20, 0.20, 0.10],
            k=1,
        )[0]

        if strategy == "unison":
            return ending
        if strategy == "step":
            return self._scale_step(ending, rng.choice([-1, 1]), context, options)
        if strategy == "third":
            return self._scale_step(ending, rng.choice([-2, 2]), context, options)
        if strategy == "response":
            if ending >= register_mid:
                return self._scale_step(ending, rng.choice([-2, -3, -4]), context, options)
            return self._scale_step(ending, rng.choice([2, 3, 4]), context, options)
        return self._pick_chord_tone(
            chord, context, options, rng, contour_bias=ending,
        )

    def _connect_loop_transition(
        self,
        notes: list[MelodyNote],
        handoff: LoopHandoff,
        context: LoopContext,
        options: GenerationOptions,
        rng: random.Random,
    ) -> list[MelodyNote]:
        """Shape only the loop boundary — the rest of the melody stays independent."""
        if not notes:
            return notes

        sorted_notes = sorted(notes, key=lambda n: (n.bar, float(n.position)))
        first = sorted_notes[0]
        ending = handoff.ending_pitch
        chord = self._chord_at_bar(context, 1)

        connected_pitch = self._connected_first_pitch(handoff, chord, context, options, rng)
        connected_pitch = self._clamp_register(connected_pitch, options)
        connected_pitch = self._snap_to_scale(connected_pitch, context, options)

        if abs(connected_pitch - ending) <= 1:
            velocity = max(1, min(127, (handoff.ending_velocity + first.velocity) // 2))
        else:
            velocity = first.velocity

        result: list[MelodyNote] = []

        if (
            first.position >= QUAVER
            and handoff.ended_with_rest
            and rng.random() < 0.35
        ):
            pickup_dir = -1 if connected_pitch > ending else 1
            pickup_pitch = self._scale_step(ending, pickup_dir, context, options)
            pickup_pitch = self._snap_to_scale(
                self._clamp_register(pickup_pitch, options), context, options,
            )
            result.append(MelodyNote(
                pitch=pickup_pitch,
                velocity=max(1, handoff.ending_velocity - 12),
                bar=first.bar,
                position=first.position - QUAVER,
                duration=SEMI,
            ))

        result.append(MelodyNote(
            pitch=connected_pitch,
            velocity=velocity,
            bar=first.bar,
            position=first.position,
            duration=first.duration,
        ))
        result.extend(sorted_notes[1:])
        return result

    def _build_accent_map(
        self, context: LoopContext, beats_per_bar: int
    ) -> dict[int, dict[Fraction, float]]:
        accent: dict[int, dict[Fraction, float]] = {}
        for role, weight in [("drums", 1.0), ("bass", 0.8), ("chords", 0.4)]:
            for note in context.tracks.get(role, []):
                pos = Fraction(note.position).limit_denominator(24) if note.position else Fraction(note.start % beats_per_bar).limit_denominator(24)
                accent.setdefault(note.bar, {}).setdefault(pos, 0.0)
                accent[note.bar][pos] += weight
        return accent

    def _create_motif(
        self,
        context: LoopContext,
        options: GenerationOptions,
        rng: random.Random,
        preset: dict,
    ) -> Motif:
        """Short singable hook: mostly steps, pop rhythm."""
        chord = self._chord_at_bar(context, 1)
        anchor = self._pick_chord_tone(chord, context, options, rng, register_center=True)
        rhythm = rng.choice(POP_HOOK_RHYTHMS)
        num_notes = len(rhythm)
        intervals = [0]
        pitch = anchor
        for i in range(1, num_notes):
            step_dir = rng.choices([-1, 1, 2, -2], weights=[0.42, 0.42, 0.08, 0.08], k=1)[0]
            if i == num_notes - 1 and rng.random() < 0.35:
                step_dir = rng.choice([-2, 2])
            if context.key_enforced:
                next_pitch = self._scale_step(pitch, 1 if step_dir > 0 else -1, context, options)
                if abs(step_dir) == 2:
                    next_pitch = self._scale_step(
                        next_pitch, 1 if step_dir > 0 else -1, context, options,
                    )
            else:
                next_pitch = pitch + step_dir
            intervals.append(next_pitch - pitch)
            pitch = next_pitch
        return Motif(intervals=intervals, durations=rhythm, anchor_pitch=anchor)

    def _generate_pop_loop(
        self,
        context: LoopContext,
        options: GenerationOptions,
        phrase_plan: PhrasePlan,
        motif: Motif,
        state: _GenState,
        beats_per_bar: int,
    ) -> list[MelodyNote]:
        """4-bar pop: hook → varied hook → contrast → resolution."""
        notes: list[MelodyNote] = []
        notes.extend(self._render_motif_bar(
            bar=1, motif=motif, variation=0, context=context, options=options,
            state=state, phrase_plan=phrase_plan, section="A",
            bar_cadence="none",
        ))
        notes.extend(self._render_motif_bar(
            bar=2, motif=motif, variation=1, context=context, options=options,
            state=state, phrase_plan=phrase_plan, section="A_prime",
            bar_cadence=phrase_plan.cadences.get(2, "none"),
        ))
        notes.extend(self._render_pop_contrast_bar(
            bar=3, context=context, options=options, state=state,
        ))
        notes.extend(self._render_pop_resolution_bar(
            bar=4, context=context, options=options, state=state,
            phrase_plan=phrase_plan,
        ))
        return notes

    def _render_motif_bar(
        self,
        bar: int,
        motif: Motif,
        variation: int,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        phrase_plan: PhrasePlan,
        section: PhraseSection,
        bar_cadence: CadenceType,
    ) -> list[MelodyNote]:
        chord = self._chord_at_bar(context, bar)
        rng = state.rng
        anchor = self._nearest_chord_tone(
            motif.anchor_pitch if bar == 1 else (state.last_pitch or motif.anchor_pitch),
            chord, context, options,
        )
        position = Fraction(0)
        notes: list[MelodyNote] = []
        pitch = anchor

        for i, (interval, dur) in enumerate(zip(motif.intervals, motif.durations)):
            if i == 0:
                pitch = anchor
            elif variation and i >= len(motif.intervals) - 2:
                pitch = self._singable_move(
                    pitch, chord, context, options, state,
                    prefer_step=rng.choice([-1, 1]),
                )
            else:
                pitch = self._apply_interval_singable(pitch, interval, context, options)

            if i == 0:
                pitch = self._nearest_chord_tone(pitch, chord, context, options)

            is_cadence = bar_cadence != "none" and i == len(motif.intervals) - 1
            if is_cadence:
                pitch = self._cadence_pitch(
                    pitch, bar_cadence, chord, context, options,
                )

            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(rng, i == 0, False, is_cadence, section),
                bar=bar,
                position=position,
                duration=dur,
            ))
            position += dur
            state.last_pitch = pitch

        return notes

    def _render_pop_contrast_bar(
        self,
        bar: int,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
    ) -> list[MelodyNote]:
        chord = self._chord_at_bar(context, bar)
        rng = state.rng
        rhythm = rng.choice(POP_ANSWER_RHYTHMS)
        notes: list[MelodyNote] = []
        position = Fraction(0)
        pitch = self._nearest_chord_tone(
            state.last_pitch or self._pick_chord_tone(
                chord, context, options, rng, register_center=True,
            ),
            chord, context, options,
        )
        direction = rng.choice([-1, 1])

        for i, dur in enumerate(rhythm):
            if float(position) >= 3.75:
                break
            if i > 0:
                step_pitch = self._scale_step(pitch, direction, context, options)
                step_pitch = self._clamp_register(step_pitch, options)
                if abs(step_pitch - pitch) <= PREFERRED_LEAP:
                    pitch = step_pitch
                else:
                    pitch = self._scale_step(pitch, -direction, context, options)

            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(rng, i == 0, False, False, "B"),
                bar=bar,
                position=position,
                duration=dur,
            ))
            position += dur
            state.last_pitch = pitch

        return notes

    def _render_pop_resolution_bar(
        self,
        bar: int,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        phrase_plan: PhrasePlan,
    ) -> list[MelodyNote]:
        chord = self._chord_at_bar(context, bar)
        rng = state.rng
        rhythm = rng.choice(POP_RESOLUTION_RHYTHMS)
        notes: list[MelodyNote] = []
        position = Fraction(0)
        pitch = state.last_pitch or self._pick_chord_tone(
            chord, context, options, rng, register_center=True,
        )

        for i, dur in enumerate(rhythm):
            if i > 0 and not (i == len(rhythm) - 1 and dur >= MINIM):
                pitch = self._singable_move(
                    pitch, chord, context, options, state, prefer_step=-1,
                    allow_repeat=False,
                )
            if i == len(rhythm) - 1:
                pitch = self._cadence_pitch(pitch, "PAC", chord, context, options)
            elif i == 0:
                pitch = self._nearest_chord_tone(pitch, chord, context, options)

            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(
                    rng, True, False, i == len(rhythm) - 1, "resolution",
                ),
                bar=bar,
                position=position,
                duration=dur,
            ))
            position += dur
            state.last_pitch = pitch

        return notes

    def _singable_move(
        self,
        from_pitch: int,
        chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        prefer_step: int | None = None,
        allow_repeat: bool = True,
    ) -> int:
        if state.pending_leap_recovery:
            state.pending_leap_recovery = False
            step = prefer_step if prefer_step is not None else -1
            return self._scale_step(from_pitch, step, context, options)

        steps = [0, 1, -1, 2, -2] if prefer_step is None else [prefer_step, 1, -1, 2, -2]
        if not allow_repeat:
            steps = [s for s in steps if s != 0] or [prefer_step or 1, -(prefer_step or 1)]
        candidates: list[int] = []
        for step in steps:
            if step == 0:
                p = from_pitch
            elif context.key_enforced:
                p = from_pitch
                for _ in range(abs(step)):
                    p = self._scale_step(p, 1 if step > 0 else -1, context, options)
            else:
                p = from_pitch + step
            p = self._clamp_register(self._snap_to_scale(p, context, options), options)
            if abs(p - from_pitch) <= MAX_MELODIC_LEAP:
                candidates.append(p)

        chord_pcs = self._effective_chord_pcs(chord, context)
        pool = [p for p in candidates if p % 12 in chord_pcs] or candidates
        if prefer_step is not None:
            directed = [p for p in pool if (p - from_pitch) * prefer_step > 0]
            if directed:
                pool = directed
        return min(pool, key=lambda p: abs(p - from_pitch)) if pool else from_pitch

    def _apply_interval_singable(
        self,
        pitch: int,
        interval: int,
        context: LoopContext,
        options: GenerationOptions,
    ) -> int:
        if abs(interval) <= PREFERRED_LEAP:
            next_pitch = pitch + interval
        elif context.key_enforced:
            direction = 1 if interval > 0 else -1
            next_pitch = pitch
            for _ in range(2):
                next_pitch = self._scale_step(next_pitch, direction, context, options)
        else:
            next_pitch = pitch + max(-PREFERRED_LEAP, min(PREFERRED_LEAP, interval))
        return self._clamp_register(
            self._snap_to_scale(next_pitch, context, options), options,
        )

    def _nearest_chord_tone(
        self,
        pitch: int,
        chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
    ) -> int:
        pcs = self._effective_chord_pcs(chord, context)
        candidates = self._pitches_for_pcs(pcs, options)
        if not candidates:
            return self._clamp_register(pitch, options)
        return min(candidates, key=lambda p: abs(p - pitch))

    def _cadence_pitch(
        self,
        from_pitch: int,
        cadence: CadenceType,
        chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
    ) -> int:
        targets = self._cadence_target_pcs(cadence, context, chord)
        return self._nearest_pitch_to_pcs(
            from_pitch, targets, options, upward=(cadence == "HC"),
        )

    def _make_taste(
        self,
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        state: _GenState,
        pitch_role: PitchRole,
        is_ending: bool,
        beat_position: Fraction,
        bar_cadence: CadenceType,
    ) -> TasteContext:
        return TasteContext(
            chord=chord,
            next_chord=next_chord,
            loop=context,
            pitch_role=pitch_role,
            is_ending=is_ending,
            last_pitch=state.last_pitch,
            beat_position=beat_position,
            bar_cadence=bar_cadence,
        )

    def _refine_with_taste(
        self,
        pitch: int,
        taste: TasteContext,
        candidates: list[int],
        rng: random.Random,
        min_score: float = 0.35,
    ) -> int:
        if melodic_suitability(pitch % 12, taste) >= min_score:
            return pitch
        if not candidates:
            return pitch
        return best_pitch_among(candidates, taste, rng, min_score=0.2)

    def _resolve_tendencies(
        self,
        pitch: int,
        taste: TasteContext,
        context: LoopContext,
        options: GenerationOptions,
        rng: random.Random,
    ) -> int:
        """Nudge pitch when previous note left an unresolved tendency."""
        if taste.last_pitch is None:
            return pitch

        last_pc = taste.last_pitch % 12
        leading = scale_degree_pc(context, 7)
        tonic = scale_degree_pc(context, 1)
        fourth = scale_degree_pc(context, 4)
        third = (taste.chord.root + (3 if "min" in taste.chord.quality else 4)) % 12

        if last_pc == leading and taste.pitch_role in ("strong", "cadence"):
            tonic_pitch = self._nearest_pitch_to_pcs(
                taste.last_pitch, {tonic}, options, upward=True,
            )
            if abs(tonic_pitch - taste.last_pitch) <= 2:
                return tonic_pitch

        if last_pc == fourth and melodic_suitability(pitch % 12, taste) < 0.45:
            third_pitch = self._nearest_pitch_to_pcs(
                taste.last_pitch, {third}, options, upward=False,
            )
            if pitch % 12 == fourth:
                return third_pitch

        return pitch

    def _chord_at_bar(self, context: LoopContext, bar: int) -> ChordEvent:
        for c in context.chords:
            if c.bar == bar:
                return c
        return context.chords[-1] if context.chords else ChordEvent(
            bar=bar, beat=Fraction(0), root=context.key_root,
            quality="major", pitch_classes=context.scale_pitch_classes, name="C",
        )

    def _scale_degree_pc(self, context: LoopContext, degree: int) -> int:
        degrees = _MAJOR_DEGREES if context.key_mode == "major" else _MINOR_DEGREES
        return (context.key_root + degrees[(degree - 1) % 7]) % 12

    def _is_strong_beat(self, position: Fraction) -> bool:
        beat = position % 1
        return beat in (Fraction(0), Fraction(1, 2))

    def _cadence_target_pcs(
        self, cadence: CadenceType, context: LoopContext, chord: ChordEvent,
    ) -> set[int]:
        if cadence == "HC":
            return {self._scale_degree_pc(context, 5), self._scale_degree_pc(context, 7)}
        if cadence == "PAC":
            tonic = self._scale_degree_pc(context, 1)
            third = self._scale_degree_pc(context, 3)
            return {tonic, third} & self._effective_chord_pcs(chord, context) or {tonic}
        return self._effective_chord_pcs(chord, context)

    def _chord_change_imminent(
        self, position: Fraction, beats_per_bar: int, chord: ChordEvent, next_chord: ChordEvent,
    ) -> bool:
        if chord.root == next_chord.root and chord.quality == next_chord.quality:
            return False
        return float(position) >= beats_per_bar - 1.0

    def _generate_bar(
        self,
        bar: int,
        section: PhraseSection,
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        preset: dict,
        phrase_plan: PhrasePlan,
        accent_map: dict[Fraction, float],
        motif: Motif,
        state: _GenState,
        beats_per_bar: int,
        is_first_bar: bool,
    ) -> list[MelodyNote]:
        if section == "A" and is_first_bar:
            motif_notes = self._motif_to_notes(
                motif, bar, chord, context, options, state, phrase_plan, section
            )
            if motif_notes:
                cursor = motif_notes[-1].position + motif_notes[-1].duration
            else:
                cursor = Fraction(0)
            # Leave breathing room after motif; fill second half of bar.
            if cursor < Fraction(beats_per_bar) / 2:
                cursor = max(cursor, Fraction(beats_per_bar) / 2)
            tail = self._fill_bar_from(
                bar, cursor, beats_per_bar, chord, next_chord, context, options,
                preset, phrase_plan, state, section, accent_map,
            )
            return motif_notes + tail

        if section == "A_prime":
            motif_notes = self._vary_motif(
                motif, bar, chord, context, options, state, phrase_plan, section
            )
            if motif_notes:
                cursor = motif_notes[-1].position + motif_notes[-1].duration
            else:
                cursor = Fraction(0)
            tail = self._fill_bar_from(
                bar, cursor, beats_per_bar, chord, next_chord, context, options,
                preset, phrase_plan, state, section, accent_map,
            )
            return motif_notes + tail

        return self._fill_bar_from(
            bar, Fraction(0), beats_per_bar, chord, next_chord, context, options,
            preset, phrase_plan, state, section, accent_map,
        )

    def _fill_bar_from(
        self,
        bar: int,
        start_pos: Fraction,
        beats_per_bar: int,
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        preset: dict,
        phrase_plan: PhrasePlan,
        state: _GenState,
        section: PhraseSection,
        accent_map: dict[Fraction, float],
    ) -> list[MelodyNote]:
        rng = state.rng
        notes: list[MelodyNote] = []
        cursor = start_pos
        bar_end = Fraction(beats_per_bar)
        is_resolution = section == "resolution"
        scalic_used = False
        formal_role = phrase_plan.formal_roles.get(bar, "continuation")
        bar_cadence = phrase_plan.cadences.get(bar, "none")
        use_fragmentation = formal_role == "fragmentation"

        if is_resolution and bar_end - cursor >= MINIM:
            dur = min(MINIM, bar_end - cursor)
            notes.append(self._make_sustained_note(
                bar, cursor, dur, chord, next_chord, context, options,
                state, phrase_plan, section, is_ending=True,
            ))
            cursor += dur

        while cursor < bar_end - SEMI:
            remaining = bar_end - cursor
            gap = self._metric_gap(rng, preset, section, cursor, bar_end)
            if gap > 0 and remaining > gap + QUAVER:
                cursor += gap
                remaining = bar_end - cursor
                if remaining < QUAVER:
                    break

            use_scalic = (
                not scalic_used
                and not is_resolution
                and remaining >= CROTCHET
                and rng.random() < preset["scalic_prob"] * (1.4 if formal_role == "continuation" else 1.0)
                and (cursor % 1) in (Fraction(0), Fraction(1, 2))
            )

            if use_scalic:
                chunk, span = self._render_scalic_passage(
                    bar, cursor, chord, next_chord, context, options,
                    state, phrase_plan, section,
                    target_cadence=bar_cadence if float(cursor) >= float(bar_end) - 2 else "none",
                )
                scalic_used = True
            else:
                figure_name, span = self._pick_figure(
                    remaining, section, rng, is_resolution, fragmented=use_fragmentation,
                )
                if figure_name is None:
                    break
                chunk = self._render_figure(
                    bar, cursor, RHYTHM_FIGURES[figure_name],
                    chord, next_chord, context, options,
                    state, phrase_plan, section,
                    beats_per_bar=beats_per_bar,
                    bar_cadence=bar_cadence,
                )

            notes.extend(chunk)
            cursor += span

        return notes

    def _metric_gap(
        self,
        rng: random.Random,
        preset: dict,
        section: PhraseSection,
        cursor: Fraction,
        bar_end: Fraction,
    ) -> Fraction:
        if section == "resolution":
            return Fraction(0)
        if rng.random() >= preset["rest_prob"]:
            return Fraction(0)
        if cursor % 1 == 0 and rng.random() < 0.6:
            return QUAVER
        return CROTCHET if bar_end - cursor >= CROTCHET + QUAVER else QUAVER

    def _pick_figure(
        self,
        remaining: Fraction,
        section: PhraseSection,
        rng: random.Random,
        is_resolution: bool,
        fragmented: bool = False,
    ) -> tuple[str | None, Fraction]:
        if is_resolution:
            pool = ["crotchet", "two_quavers", "minim", "crotchet_quavers"]
        elif fragmented:
            pool = ["four_semis", "semi_pair_quavers", "quaver_semi_pair", "two_quavers"]
        elif section in ("B", "B_prime", "C", "C_prime"):
            pool = TWO_BEAT_FIGURES + ONE_BEAT_FIGURES
        else:
            pool = ONE_BEAT_FIGURES + TWO_BEAT_FIGURES

        rng.shuffle(pool)
        for name in pool:
            span = sum(RHYTHM_FIGURES[name], Fraction(0))
            if span <= remaining + SEMI:
                return name, min(span, remaining)
        if remaining >= QUAVER:
            return "two_quavers", min(QUAVER * 2, remaining)
        return None, Fraction(0)

    def _render_figure(
        self,
        bar: int,
        start: Fraction,
        durations: list[Fraction],
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        phrase_plan: PhrasePlan,
        section: PhraseSection,
        beats_per_bar: int = 4,
        bar_cadence: CadenceType = "none",
    ) -> list[MelodyNote]:
        rng = state.rng
        notes: list[MelodyNote] = []
        pos = start
        for i, dur in enumerate(durations):
            beat_in_bar = pos % 1
            is_strong = self._is_strong_beat(pos) and i == 0
            is_peak = bar == phrase_plan.peak_bar and pos == phrase_plan.peak_position
            is_last_in_bar = float(pos + dur) >= beats_per_bar - float(SEMI)
            imminent_change = self._chord_change_imminent(pos, beats_per_bar, chord, next_chord)

            if is_last_in_bar and bar_cadence != "none":
                pitch_role: PitchRole = "cadence"
            elif imminent_change:
                pitch_role = "anticipation"
            elif is_strong:
                pitch_role = "strong"
            else:
                pitch_role = "weak"

            pitch = self._choose_pitch(
                chord, next_chord, context, options, state,
                pitch_role=pitch_role,
                is_ending=is_last_in_bar and bar_cadence == "PAC",
                is_peak=is_peak,
                bar=bar, section=section, phrase_plan=phrase_plan,
                bar_cadence=bar_cadence if is_last_in_bar else "none",
                beat_position=pos,
            )
            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(rng, is_strong, is_peak, False, section),
                bar=bar,
                position=pos,
                duration=dur,
            ))
            state.last_pitch = pitch
            pos += dur
        return notes

    def _render_scalic_passage(
        self,
        bar: int,
        start: Fraction,
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        phrase_plan: PhrasePlan,
        section: PhraseSection,
        target_cadence: CadenceType = "none",
    ) -> tuple[list[MelodyNote], Fraction]:
        """Goal-directed scalic run: stepwise motion between chord tones (passing tones)."""
        rng = state.rng
        figure = rng.choice(["four_quavers", "semi_pair_quavers", "quaver_semi_pair"])
        durations = RHYTHM_FIGURES[figure]
        span = sum(durations, Fraction(0))

        start_pitch = state.last_pitch or self._pick_chord_tone(
            chord, context, options, rng, register_center=True,
        )

        if target_cadence != "none":
            targets = self._cadence_target_pcs(target_cadence, context, chord)
            target_pitch = self._nearest_pitch_to_pcs(start_pitch, targets, options, upward=rng.choice([True, False]))
        else:
            target_pitch = self._pick_chord_tone(
                next_chord if chord.root != next_chord.root else chord,
                context, options, rng,
                contour_bias=self._contour_bias(phrase_plan, bar, start_pitch, options),
            )

        direction = 1 if target_pitch >= start_pitch else -1
        if target_pitch == start_pitch:
            direction = rng.choice([-1, 1])

        pitch_pool = context.scale_pitch_classes
        pitch = start_pitch
        notes: list[MelodyNote] = []
        pos = start

        for i, dur in enumerate(durations):
            if i > 0:
                pitch = self._step_in_pool(pitch, direction, pitch_pool, options)
            is_chord_tone = (pitch % 12) in self._effective_chord_pcs(chord, context)
            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(
                    rng, is_chord_tone and i == 0, False, False, section,
                ),
                bar=bar,
                position=pos,
                duration=dur,
            ))
            state.last_pitch = pitch
            pos += dur

        if notes:
            notes[-1].pitch = self._snap_to_scale(
                self._clamp_register(
                    self._nearest_pitch_to_pcs(notes[-1].pitch, self._effective_chord_pcs(
                        next_chord if chord.root != next_chord.root else chord, context,
                    ), options, upward=direction > 0),
                    options,
                ),
                context, options,
            )
            state.last_pitch = notes[-1].pitch

        return notes, span

    def _nearest_pitch_to_pcs(
        self,
        from_pitch: int,
        pcs: set[int],
        options: GenerationOptions,
        upward: bool = True,
    ) -> int:
        candidates = self._pitches_for_pcs(pcs, options)
        if not candidates:
            return from_pitch
        if upward:
            above = [p for p in candidates if p >= from_pitch]
            if above:
                return min(above)
        else:
            below = [p for p in candidates if p <= from_pitch]
            if below:
                return max(below)
        return min(candidates, key=lambda p: abs(p - from_pitch))

    def _step_in_pool(
        self,
        pitch: int,
        direction: int,
        pitch_classes: set[int],
        options: GenerationOptions,
    ) -> int:
        pitches = sorted(
            p for p in range(options.register_low, options.register_high + 1)
            if p % 12 in pitch_classes
        )
        if not pitches:
            return pitch
        idx = min(range(len(pitches)), key=lambda i: abs(pitches[i] - pitch))
        new_idx = max(0, min(len(pitches) - 1, idx + direction))
        return pitches[new_idx]

    def _make_sustained_note(
        self,
        bar: int,
        position: Fraction,
        duration: Fraction,
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        phrase_plan: PhrasePlan,
        section: PhraseSection,
        is_ending: bool = False,
    ) -> MelodyNote:
        rng = state.rng
        pitch = self._choose_pitch(
            chord, next_chord, context, options, state,
            pitch_role="cadence" if is_ending else "strong",
            is_ending=is_ending, is_peak=False,
            bar=bar, section=section, phrase_plan=phrase_plan,
            bar_cadence="PAC" if is_ending else "none",
            beat_position=position,
        )
        state.last_pitch = pitch
        return MelodyNote(
            pitch=pitch,
            velocity=self._choose_velocity(rng, True, False, is_ending, section),
            bar=bar,
            position=position,
            duration=quantize_duration(duration),
        )

    def _scale_step(
        self, pitch: int, direction: int, context: LoopContext, options: GenerationOptions
    ) -> int:
        pitches = sorted(self._pitches_for_pcs(context.scale_pitch_classes, options))
        if not pitches:
            return pitch + direction * 2
        idx = min(range(len(pitches)), key=lambda i: abs(pitches[i] - pitch))
        new_idx = max(0, min(len(pitches) - 1, idx + direction))
        return pitches[new_idx]

    def _motif_to_notes(
        self, motif: Motif, bar: int, chord: ChordEvent,
        context: LoopContext, options: GenerationOptions,
        state: _GenState, phrase_plan: PhrasePlan, section: PhraseSection,
    ) -> list[MelodyNote]:
        notes: list[MelodyNote] = []
        pitch = motif.anchor_pitch
        position = Fraction(0)
        for i, (interval, dur) in enumerate(zip(motif.intervals, motif.durations)):
            if i > 0:
                pitch = self._clamp_register(pitch + interval, options)
                pitch = self._snap_to_scale(pitch, context, options)
            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(state.rng, i == 0, False, False, section),
                bar=bar,
                position=position,
                duration=dur,
            ))
            position += dur
            state.last_pitch = pitch
        return notes

    def _vary_motif(
        self, motif: Motif, bar: int, chord: ChordEvent,
        context: LoopContext, options: GenerationOptions,
        state: _GenState, phrase_plan: PhrasePlan, section: PhraseSection,
    ) -> list[MelodyNote]:
        rng = state.rng
        transposition = rng.choice([-2, -1, 0, 1, 2])
        anchor = self._pick_chord_tone(chord, context, options, rng, register_center=True)
        pitch = anchor
        position = Fraction(0)
        notes: list[MelodyNote] = []
        for i, (interval, dur) in enumerate(zip(motif.intervals, motif.durations)):
            if i == 0:
                pitch = anchor
            else:
                inv = -interval if rng.random() < 0.3 else interval
                if context.key_enforced:
                    pitch = self._scale_step(pitch, 1 if inv >= 0 else -1, context, options)
                else:
                    pitch = self._clamp_register(
                        pitch + inv + (transposition if i == 1 else 0), options
                    )
                if context.key_enforced:
                    pitch = self._snap_to_scale(pitch, context, options)
            if position >= context.time_signature[0]:
                break
            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(rng, i == 0, False, i == len(motif.intervals) - 1, section),
                bar=bar,
                position=quantize_position(position, context.time_signature[0]),
                duration=dur,
            ))
            position += dur
            state.last_pitch = pitch
        return notes

    def _choose_pitch(
        self,
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        pitch_role: PitchRole,
        is_ending: bool,
        is_peak: bool,
        bar: int,
        section: PhraseSection,
        phrase_plan: PhrasePlan,
        bar_cadence: CadenceType = "none",
        beat_position: Fraction = Fraction(0),
    ) -> int:
        rng = state.rng
        taste = self._make_taste(
            chord, next_chord, context, state, pitch_role, is_ending, beat_position, bar_cadence,
        )

        if pitch_role == "cadence" and bar_cadence != "none":
            targets = self._cadence_target_pcs(bar_cadence, context, chord)
            candidates = self._pitches_for_pcs(targets, options)
            if candidates:
                pitch = best_pitch_among(candidates, taste, rng)
                return self._snap_to_scale(self._clamp_register(pitch, options), context, options)

        if pitch_role == "anticipation":
            guides = guide_tone_pcs(next_chord)
            guide_candidates = self._pitches_for_pcs(guides or next_chord.pitch_classes, options)
            if guide_candidates:
                pitch = best_pitch_among(guide_candidates, taste, rng)
                return self._snap_to_scale(self._clamp_register(pitch, options), context, options)

        if pitch_role == "weak" and state.last_pitch is not None:
            target = self._pick_chord_tone(
                chord, context, options, rng, taste=taste,
                contour_bias=self._contour_bias(phrase_plan, bar, state.last_pitch, options),
            )
            embellished = self._embellish_between(
                state.last_pitch, target, chord, context, options, rng, taste=taste,
            )
            if embellished is not None:
                return embellished

        if pitch_role == "strong" or is_ending:
            weights = ENDING_WEIGHTS if is_ending else STRONG_BEAT_WEIGHTS
        else:
            weights = WEAK_BEAT_WEIGHTS

        contour_bias = self._contour_bias(phrase_plan, bar, state.last_pitch, options)

        roll = rng.random()
        if context.key_enforced:
            chord_w, scale_w = weights[0], weights[1]
            threshold = chord_w + scale_w
            if roll * threshold < chord_w:
                pitch = self._pick_chord_tone(
                    chord, context, options, rng, contour_bias=contour_bias, taste=taste,
                )
            else:
                pitch = self._pick_scale_tone(
                    chord, context, options, rng, contour_bias=contour_bias, taste=taste,
                )
        elif roll < weights[0]:
            pitch = self._pick_chord_tone(
                chord, context, options, rng, contour_bias=contour_bias, taste=taste,
            )
        elif roll < weights[0] + weights[1]:
            pitch = self._pick_scale_tone(
                chord, context, options, rng, contour_bias=contour_bias, taste=taste,
            )
        else:
            pitch = self._pick_chromatic_approach(chord, context, options, state, rng, taste=taste)
            state.consecutive_chromatic += 1

        pitch = self._resolve_tendencies(pitch, taste, context, options, rng)

        if state.last_pitch is not None:
            pitch = self._apply_motion_rules(pitch, state, options, rng, is_peak, context)

        pitch = self._clamp_register(pitch, options)
        pitch = self._snap_to_scale(pitch, context, options)

        pool = self._pitches_for_pcs(self._effective_chord_pcs(chord, context), options)
        if pitch_role == "weak":
            non_chord = context.scale_pitch_classes - chord.pitch_classes
            pool = pool + self._pitches_for_pcs(non_chord, options)
        return self._refine_with_taste(pitch, taste, pool, rng)

    def _embellish_between(
        self,
        from_pitch: int,
        to_pitch: int,
        chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        rng: random.Random,
        taste: TasteContext | None = None,
    ) -> int | None:
        """Passing or neighbor tone between chord tones (weak-beat embellishment)."""
        if from_pitch == to_pitch:
            neighbor_dir = rng.choice([-1, 1])
            candidate = self._scale_step(from_pitch, neighbor_dir, context, options)
            if taste and not passing_tone_allowed(candidate % 12, taste):
                return None
            return candidate

        diff = to_pitch - from_pitch
        if abs(diff) == 1:
            return None

        strategy = rng.choice(["passing", "passing", "neighbor"])
        if strategy == "neighbor":
            candidate = self._scale_step(from_pitch, rng.choice([-1, 1]), context, options)
            if taste and not passing_tone_allowed(candidate % 12, taste):
                return None
            return candidate

        step_dir = 1 if diff > 0 else -1
        mid = self._scale_step(from_pitch, step_dir, context, options)
        if mid % 12 in self._effective_chord_pcs(chord, context):
            return None
        if taste and not passing_tone_allowed(mid % 12, taste):
            return None
        return mid

    def _contour_bias(
        self, plan: PhrasePlan, bar: int, last_pitch: int | None, options: GenerationOptions
    ) -> int:
        center = (options.register_low + options.register_high) // 2
        if last_pitch is None:
            return center

        total_bars = plan.sections[-1][0]
        role = plan.formal_roles.get(bar, "continuation")
        progress = bar / max(total_bars, 1)
        span = options.register_high - options.register_low

        if plan.contour == "arch":
            if role == "antecedent":
                offset = int(span * 0.15 * min(progress * 1.5, 1.0))
            elif role == "cadence":
                offset = -int(span * 0.12)
            else:
                offset = int(span * 0.1 * (1 - abs(progress - 0.6) * 2))
        elif plan.contour == "ascending":
            offset = int(span * 0.12 * progress)
        elif plan.contour == "descending":
            offset = -int(span * 0.12 * progress)
        elif plan.contour == "call_response":
            offset = int(span * 0.1) if bar <= total_bars // 2 else -int(span * 0.08)
        else:
            offset = 0

        target = center + offset
        delta = max(-MAX_MELODIC_LEAP, min(MAX_MELODIC_LEAP, target - last_pitch))
        return last_pitch + delta

    def _pitches_for_pcs(
        self, pitch_classes: set[int], options: GenerationOptions
    ) -> list[int]:
        pitches = []
        for p in range(options.register_low, options.register_high + 1):
            if p % 12 in pitch_classes:
                pitches.append(p)
        return pitches

    def _effective_chord_pcs(self, chord: ChordEvent, context: LoopContext) -> set[int]:
        pcs = chord.pitch_classes
        if context.key_enforced:
            pcs = pcs & context.scale_pitch_classes
            if not pcs:
                pcs = {chord.root} & context.scale_pitch_classes or {context.key_root}
        return pcs

    def _snap_to_scale(
        self, pitch: int, context: LoopContext, options: GenerationOptions
    ) -> int:
        if not context.key_enforced:
            return pitch
        if pitch % 12 in context.scale_pitch_classes:
            return self._clamp_register(pitch, options)
        candidates = self._pitches_for_pcs(context.scale_pitch_classes, options)
        if not candidates:
            return pitch
        return min(candidates, key=lambda p: abs(p - pitch))

    def _pick_scale_neighbor(
        self,
        pitch: int,
        context: LoopContext,
        options: GenerationOptions,
        rng: random.Random,
        max_steps: int = 2,
    ) -> int:
        pitches = sorted(self._pitches_for_pcs(context.scale_pitch_classes, options))
        if not pitches:
            return pitch
        idx = min(range(len(pitches)), key=lambda i: abs(pitches[i] - pitch))
        step = rng.randint(-max_steps, max_steps)
        new_idx = max(0, min(len(pitches) - 1, idx + step))
        return pitches[new_idx]

    def _pick_chord_tone(
        self, chord: ChordEvent, context: LoopContext, options: GenerationOptions,
        rng: random.Random, register_center: bool = False, contour_bias: int | None = None,
        taste: TasteContext | None = None,
    ) -> int:
        valid = self._pitches_for_pcs(self._effective_chord_pcs(chord, context), options)
        if not valid:
            valid = [self._clamp_register(60 + chord.root, options)]
        if taste is not None and taste.last_pitch is not None:
            nearby = [p for p in valid if abs(p - taste.last_pitch) <= MAX_MELODIC_LEAP]
            if nearby:
                valid = nearby
        if contour_bias is not None:
            nearest = min(valid, key=lambda t: abs(t - contour_bias))
            if taste is not None:
                return self._refine_with_taste(nearest, taste, valid, rng)
            return nearest
        if register_center:
            center = (options.register_low + options.register_high) // 2
            nearest = min(valid, key=lambda t: abs(t - center))
            if taste is not None:
                return self._refine_with_taste(nearest, taste, valid, rng)
            return nearest
        if taste is not None:
            return best_pitch_among(valid, taste, rng)
        return rng.choice(valid)

    def _pick_scale_tone(
        self, chord: ChordEvent, context: LoopContext, options: GenerationOptions,
        rng: random.Random, contour_bias: int | None = None,
        taste: TasteContext | None = None,
    ) -> int:
        non_chord = context.scale_pitch_classes - chord.pitch_classes
        if not non_chord:
            return self._pick_chord_tone(chord, context, options, rng, contour_bias=contour_bias, taste=taste)
        pitches = self._pitches_for_pcs(non_chord, options)
        if taste is not None:
            pitches = [p for p in pitches if passing_tone_allowed(p % 12, taste)] or pitches
        if not pitches:
            return self._pick_chord_tone(chord, context, options, rng, taste=taste)
        if taste is not None and taste.last_pitch is not None:
            nearby = [p for p in pitches if abs(p - taste.last_pitch) <= MAX_MELODIC_LEAP]
            if nearby:
                pitches = nearby
        if contour_bias is not None:
            nearest = min(pitches, key=lambda t: abs(t - contour_bias))
            if taste is not None:
                return self._refine_with_taste(nearest, taste, pitches, rng)
            return nearest
        if taste is not None:
            return best_pitch_among(pitches, taste, rng)
        return rng.choice(pitches)

    def _pick_chromatic_approach(
        self, chord: ChordEvent, context: LoopContext, options: GenerationOptions,
        state: _GenState, rng: random.Random,
        taste: TasteContext | None = None,
    ) -> int:
        target = self._pick_chord_tone(chord, context, options, rng, taste=taste)
        approach = target + rng.choice([-1, 1])
        return self._clamp_register(approach, options)

    def _apply_motion_rules(
        self, pitch: int, state: _GenState, options: GenerationOptions,
        rng: random.Random, is_peak: bool, context: LoopContext,
    ) -> int:
        if state.last_pitch is None:
            return pitch

        interval = pitch - state.last_pitch
        abs_interval = abs(interval)

        if abs_interval == 0:
            state.consecutive_repeats += 1
            if state.consecutive_repeats > 2:
                if context.key_enforced:
                    pitch = self._pick_scale_neighbor(state.last_pitch, context, options, rng, max_steps=1)
                else:
                    pitch += rng.choice([-1, 1])
        else:
            state.consecutive_repeats = 0

        if abs_interval > MAX_MELODIC_LEAP:
            direction = 1 if interval > 0 else -1
            if context.key_enforced:
                pitch = state.last_pitch
                for _ in range(PREFERRED_LEAP):
                    pitch = self._scale_step(pitch, direction, context, options)
            else:
                pitch = state.last_pitch + direction * PREFERRED_LEAP
            state.pending_leap_recovery = True
            abs_interval = abs(pitch - state.last_pitch)

        if abs_interval > PREFERRED_LEAP:
            state.pending_leap_recovery = True
            state.consecutive_large_leaps += 1
            if state.consecutive_large_leaps > 1:
                if context.key_enforced:
                    pitch = self._pick_scale_neighbor(state.last_pitch, context, options, rng, max_steps=1)
                else:
                    direction = -1 if interval > 0 else 1
                    pitch = state.last_pitch + direction
                state.pending_leap_recovery = False
        else:
            state.consecutive_large_leaps = 0

        return pitch

    def _choose_duration(
        self, preset: dict, rng: random.Random, is_ending: bool, state: _GenState
    ) -> Fraction:
        if is_ending:
            opts = [Fraction(1, 1), Fraction(3, 2), Fraction(2, 1)]
        else:
            opts = preset["durations"]
            if rng.random() < preset["fast_note_prob"]:
                opts = opts + [Fraction(1, 4), Fraction(1, 8), Fraction(1, 6)]

        dur = rng.choice(opts)
        if dur == Fraction(1, 8):
            state.consecutive_32nds += 1
            if state.consecutive_32nds > 4:
                dur = Fraction(1, 4)
        else:
            state.consecutive_32nds = 0
        return quantize_duration(dur)

    def _choose_velocity(
        self, rng: random.Random, is_strong: bool, is_peak: bool,
        is_ending: bool, section: PhraseSection,
    ) -> int:
        if is_peak:
            return rng.randint(95, 115)
        if is_ending:
            return rng.randint(65, 90)
        if is_strong:
            return rng.randint(80, 105)
        return rng.randint(55, 85)

    def _clamp_register(self, pitch: int, options: GenerationOptions) -> int:
        while pitch < options.register_low:
            pitch += 12
        while pitch > options.register_high:
            pitch -= 12
        return max(options.register_low, min(options.register_high, pitch))

    def _apply_entry_delay(
        self, notes: list[MelodyNote], entry: Fraction, bar: int
    ) -> list[MelodyNote]:
        if entry == Fraction(0):
            return notes
        return [n for n in notes if not (n.bar == bar and n.position < entry)]

    def _validate_and_repair(
        self,
        notes: list[MelodyNote],
        context: LoopContext,
        options: GenerationOptions,
        beats_per_bar: int,
    ) -> list[MelodyNote]:
        loop_end = context.bars * beats_per_bar
        repaired: list[MelodyNote] = []

        for note in sorted(notes, key=lambda n: (n.bar, float(n.position))):
            start = (note.bar - 1) * beats_per_bar + float(note.position)
            if start >= loop_end:
                continue
            if note.duration <= 0:
                continue

            pitch = self._clamp_register(note.pitch, options)
            pitch = self._snap_to_scale(pitch, context, options)
            end = start + float(note.duration)
            if end > loop_end:
                note = MelodyNote(
                    pitch=pitch, velocity=note.velocity, bar=note.bar,
                    position=note.position,
                    duration=Fraction(loop_end - start).limit_denominator(24),
                )
            else:
                note = MelodyNote(
                    pitch=pitch, velocity=max(1, min(127, note.velocity)),
                    bar=note.bar, position=note.position, duration=note.duration,
                )

            if not options.legato and repaired:
                prev = repaired[-1]
                prev_end = (prev.bar - 1) * beats_per_bar + float(prev.position) + float(prev.duration)
                curr_start = (note.bar - 1) * beats_per_bar + float(note.position)
                if curr_start < prev_end:
                    gap = Fraction(curr_start - ((prev.bar - 1) * beats_per_bar + float(prev.position))).limit_denominator(24)
                    if gap > 0:
                        repaired[-1] = MelodyNote(
                            pitch=prev.pitch, velocity=prev.velocity,
                            bar=prev.bar, position=prev.position, duration=gap,
                        )
                    else:
                        continue

            if repaired:
                prev_pitch = repaired[-1].pitch
                leap = abs(pitch - prev_pitch)
                if leap > MAX_MELODIC_LEAP:
                    direction = 1 if pitch > prev_pitch else -1
                    nudged = prev_pitch
                    for _ in range(PREFERRED_LEAP):
                        if context.key_enforced:
                            nudged = self._scale_step(nudged, direction, context, options)
                        else:
                            nudged += direction
                    pitch = self._clamp_register(nudged, options)
                    note = MelodyNote(
                        pitch=pitch, velocity=note.velocity,
                        bar=note.bar, position=note.position, duration=note.duration,
                    )

            repaired.append(note)

        return repaired
