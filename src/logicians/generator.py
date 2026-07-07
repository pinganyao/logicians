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

ContourType = Literal["arch", "inverted_arch", "ascending", "descending", "static_hook", "call_response"]
PhraseSection = Literal["A", "A_prime", "B", "B_prime", "C", "C_prime", "D", "resolution"]

DENSITY_PRESETS = {
    "sparse": {
        "notes_per_bar": (2, 5),
        "durations": [Fraction(1, 1), Fraction(3, 2), Fraction(2, 1)],
        "rest_prob": 0.45,
        "fast_note_prob": 0.05,
    },
    "medium": {
        "notes_per_bar": (5, 9),
        "durations": [Fraction(1, 2), Fraction(1, 1), Fraction(3, 4), Fraction(1, 4)],
        "rest_prob": 0.25,
        "fast_note_prob": 0.15,
    },
    "busy": {
        "notes_per_bar": (8, 16),
        "durations": [Fraction(1, 2), Fraction(1, 4), Fraction(1, 6), Fraction(1, 8)],
        "rest_prob": 0.10,
        "fast_note_prob": 0.35,
    },
}

INTERVAL_WEIGHTS = [
    ("repeat", 0.08),
    ("step", 0.45),
    ("small_skip", 0.30),
    ("large_leap", 0.14),
    ("very_large", 0.03),
]

STRONG_BEAT_WEIGHTS = (0.75, 0.20, 0.05)
WEAK_BEAT_WEIGHTS = (0.45, 0.40, 0.15)
ENDING_WEIGHTS = (0.90, 0.10, 0.0)


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


@dataclass
class _GenState:
    rng: random.Random
    last_pitch: int | None = None
    consecutive_large_leaps: int = 0
    consecutive_repeats: int = 0
    consecutive_chromatic: int = 0
    consecutive_32nds: int = 0


class RuleBasedMelodyGenerator:
    """Detailed rule-based melody generator as a placeholder for future ML model."""

    def generate(self, context: LoopContext, options: GenerationOptions) -> MelodyClip:
        rng = random.Random(options.seed)
        beats_per_bar = context.time_signature[0]

        density_key = self._resolve_density(options, context)
        preset = DENSITY_PRESETS[density_key]

        phrase_plan = self._plan_phrase(context, options, rng)
        accent_map = self._build_accent_map(context, beats_per_bar)
        motif = self._create_motif(context, options, rng, preset)

        notes: list[MelodyNote] = []
        state = _GenState(rng=rng)

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
        notes = self._validate_and_repair(notes, context, options, beats_per_bar)

        return MelodyClip(bars=context.bars, notes=notes)

    def _resolve_density(self, options: GenerationOptions, context: LoopContext) -> str:
        density = options.density
        if context.rhythm_density > 0.7 and density == "medium":
            return "sparse"
        if context.rhythm_density < 0.3 and density == "medium":
            return "busy"
        return density

    def _plan_phrase(
        self, context: LoopContext, options: GenerationOptions, rng: random.Random
    ) -> PhrasePlan:
        bars = context.bars
        if bars == 4:
            sections: list[tuple[int, PhraseSection]] = [
                (1, "A"),
                (2, "A_prime"),
                (3, "B"),
                (4, "resolution"),
            ]
        else:
            sections = [
                (1, "A"), (2, "A_prime"),
                (3, "B"), (4, "B_prime"),
                (5, "C"), (6, "C_prime"),
                (7, "D"), (8, "resolution"),
            ]

        contour = rng.choice(["arch", "call_response", "arch", "ascending"])
        peak_bar = rng.randint(max(1, bars * 3 // 5), max(1, bars * 4 // 5))
        peak_position = Fraction(rng.choice([1, 2, 3]), 2)

        if options.density == "sparse":
            entry = Fraction(2)
        elif options.density == "busy":
            entry = Fraction(7, 8) if rng.random() < 0.5 else Fraction(0)
        else:
            entry = Fraction(rng.choice([0, 1, 1, 2]), 2)

        return PhrasePlan(
            sections=sections,
            contour=contour,  # type: ignore[arg-type]
            peak_bar=peak_bar,
            peak_position=peak_position,
            entry_position=entry,
        )

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
        chord = self._chord_at_bar(context, 1)
        anchor = self._pick_chord_tone(chord, context, options, rng, register_center=True)
        num_notes = rng.randint(2, 4)
        intervals = [0]
        current = anchor
        for _ in range(num_notes - 1):
            if context.key_enforced:
                next_pitch = self._pick_scale_neighbor(current, context, options, rng)
                intervals.append(next_pitch - current)
                current = next_pitch
            else:
                intervals.append(rng.choice([-2, -1, 1, 2, 3, -3, 4, -4]))
        durations = [rng.choice(preset["durations"]) for _ in range(num_notes)]
        return Motif(intervals=intervals, durations=durations, anchor_pitch=anchor)

    def _chord_at_bar(self, context: LoopContext, bar: int) -> ChordEvent:
        for c in context.chords:
            if c.bar == bar:
                return c
        return context.chords[-1] if context.chords else ChordEvent(
            bar=bar, beat=Fraction(0), root=context.key_root,
            quality="major", pitch_classes=context.scale_pitch_classes, name="C",
        )

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
        rng = state.rng
        target_count = rng.randint(*preset["notes_per_bar"])

        if section in ("A_prime", "B_prime", "C_prime"):
            target_count = max(2, target_count - rng.randint(0, 2))
        if section == "resolution":
            target_count = max(2, target_count - 1)

        positions = self._choose_positions(
            bar, target_count, preset, accent_map, beats_per_bar, rng, section
        )

        if section == "A" and is_first_bar:
            return self._motif_to_notes(motif, bar, chord, context, options, state, phrase_plan, section)

        if section == "A_prime":
            return self._vary_motif(motif, bar, chord, context, options, state, phrase_plan, section)

        notes: list[MelodyNote] = []
        for i, pos in enumerate(positions):
            is_strong = pos in (Fraction(0), Fraction(2)) or i == len(positions) - 1
            is_ending = section == "resolution" and i == len(positions) - 1
            is_peak = bar == phrase_plan.peak_bar and pos == phrase_plan.peak_position

            pitch = self._choose_pitch(
                chord, next_chord, context, options, state,
                is_strong=is_strong, is_ending=is_ending, is_peak=is_peak,
                bar=bar, section=section, phrase_plan=phrase_plan,
            )
            duration = self._choose_duration(preset, rng, is_ending, state)
            velocity = self._choose_velocity(rng, is_strong, is_peak, is_ending, section)

            notes.append(MelodyNote(
                pitch=pitch, velocity=velocity, bar=bar,
                position=pos, duration=duration,
            ))
            state.last_pitch = pitch

        return notes

    def _choose_positions(
        self,
        bar: int,
        count: int,
        preset: dict,
        accent_map: dict[Fraction, float],
        beats_per_bar: int,
        rng: random.Random,
        section: PhraseSection,
    ) -> list[Fraction]:
        candidates = [p for p in POSITION_GRID if p < beats_per_bar]
        rng.shuffle(candidates)

        scored = []
        for p in candidates:
            accent = accent_map.get(p, 0.0)
            offbeat_bonus = 0.3 if p % 1 != 0 else 0.0
            if preset["rest_prob"] > 0.3:
                offbeat_bonus *= 0.5
            score = accent * 0.5 + offbeat_bonus + rng.random()
            scored.append((score, p))

        scored.sort(reverse=True)
        positions = sorted([p for _, p in scored[:count]])

        filtered: list[Fraction] = []
        for p in positions:
            if filtered and p - filtered[-1] < Fraction(1, 8):
                continue
            filtered.append(p)
        return filtered[:count] if filtered else [Fraction(0), Fraction(2)]

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
        rhythm_shift = Fraction(rng.choice([0, 1, 2]), rng.choice([4, 8]))
        anchor = self._pick_chord_tone(chord, context, options, rng, register_center=True)
        pitch = anchor
        position = rhythm_shift
        notes: list[MelodyNote] = []
        for i, (interval, dur) in enumerate(zip(motif.intervals, motif.durations)):
            if i == 0:
                pitch = anchor
            else:
                inv = -interval if rng.random() < 0.3 else interval
                pitch = self._clamp_register(pitch + inv + (transposition if i == 1 else 0), options)
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
        is_strong: bool,
        is_ending: bool,
        is_peak: bool,
        bar: int,
        section: PhraseSection,
        phrase_plan: PhrasePlan,
    ) -> int:
        rng = state.rng

        if is_ending:
            weights = ENDING_WEIGHTS
        elif is_strong:
            weights = (
                options.chord_tone_weight,
                options.passing_tone_weight,
                options.chromatic_weight,
            )
        else:
            weights = WEAK_BEAT_WEIGHTS

        contour_bias = self._contour_bias(phrase_plan, bar, state.last_pitch, options)

        roll = rng.random()
        if context.key_enforced:
            chord_w, scale_w = weights[0], weights[1]
            threshold = chord_w + scale_w
            if roll * threshold < chord_w:
                pitch = self._pick_chord_tone(chord, context, options, rng, contour_bias=contour_bias)
            else:
                pitch = self._pick_scale_tone(chord, context, options, rng, contour_bias=contour_bias)
        elif roll < weights[0]:
            pitch = self._pick_chord_tone(chord, context, options, rng, contour_bias=contour_bias)
        elif roll < weights[0] + weights[1]:
            pitch = self._pick_scale_tone(chord, context, options, rng, contour_bias=contour_bias)
        else:
            pitch = self._pick_chromatic_approach(chord, context, options, state, rng)
            state.consecutive_chromatic += 1

        if state.last_pitch is not None:
            pitch = self._apply_motion_rules(pitch, state, options, rng, is_peak, context)

        pitch = self._clamp_register(pitch, options)
        return self._snap_to_scale(pitch, context, options)

    def _contour_bias(
        self, plan: PhrasePlan, bar: int, last_pitch: int | None, options: GenerationOptions
    ) -> int:
        center = (options.register_low + options.register_high) // 2
        if last_pitch is None:
            return center
        progress = bar / max(plan.sections[-1][0], 1)
        if plan.contour == "arch":
            target = center + int((options.register_high - center) * (1 - abs(progress - 0.7) * 2))
        elif plan.contour == "ascending":
            target = options.register_low + int((options.register_high - options.register_low) * progress)
        elif plan.contour == "descending":
            target = options.register_high - int((options.register_high - options.register_low) * progress)
        else:
            target = center
        return target

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
    ) -> int:
        valid = self._pitches_for_pcs(self._effective_chord_pcs(chord, context), options)
        if not valid:
            valid = [self._clamp_register(60 + chord.root, options)]
        if contour_bias is not None:
            return min(valid, key=lambda t: abs(t - contour_bias))
        if register_center:
            center = (options.register_low + options.register_high) // 2
            return min(valid, key=lambda t: abs(t - center))
        return rng.choice(valid)

    def _pick_scale_tone(
        self, chord: ChordEvent, context: LoopContext, options: GenerationOptions,
        rng: random.Random, contour_bias: int | None = None,
    ) -> int:
        non_chord = context.scale_pitch_classes - chord.pitch_classes
        if not non_chord:
            return self._pick_chord_tone(chord, context, options, rng, contour_bias=contour_bias)
        pitches = self._pitches_for_pcs(non_chord, options)
        if not pitches:
            return self._pick_chord_tone(chord, context, options, rng)
        if contour_bias is not None:
            return min(pitches, key=lambda t: abs(t - contour_bias))
        return rng.choice(pitches)

    def _pick_chromatic_approach(
        self, chord: ChordEvent, context: LoopContext, options: GenerationOptions,
        state: _GenState, rng: random.Random,
    ) -> int:
        target = self._pick_chord_tone(chord, context, options, rng)
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
            if state.consecutive_repeats > 3:
                if context.key_enforced:
                    pitch = self._pick_scale_neighbor(state.last_pitch, context, options, rng, max_steps=1)
                else:
                    pitch += rng.choice([-2, -1, 1, 2])
        else:
            state.consecutive_repeats = 0

        if abs_interval >= 6:
            state.consecutive_large_leaps += 1
            if state.consecutive_large_leaps > 2:
                if context.key_enforced:
                    pitch = self._pick_scale_neighbor(state.last_pitch, context, options, rng, max_steps=1)
                else:
                    direction = -1 if interval > 0 else 1
                    pitch = state.last_pitch + direction * rng.randint(1, 2)
        else:
            state.consecutive_large_leaps = 0

        if not is_peak and abs_interval > 12:
            if context.key_enforced:
                pitch = self._pick_scale_neighbor(state.last_pitch, context, options, rng, max_steps=3)
            else:
                pitch = state.last_pitch + rng.choice([-7, -5, 5, 7])

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

            repaired.append(note)

        return repaired
