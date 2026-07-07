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
        "notes_per_bar": (6, 10),
        "durations": [Fraction(1, 2), Fraction(1, 4), Fraction(1, 1)],
        "rest_prob": 0.15,
        "fast_note_prob": 0.35,
        "cell_size": Fraction(1, 4),
    },
    "medium": {
        "notes_per_bar": (12, 20),
        "durations": [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 1)],
        "rest_prob": 0.04,
        "fast_note_prob": 0.65,
        "cell_size": Fraction(1, 8),
    },
    "busy": {
        "notes_per_bar": (16, 28),
        "durations": [Fraction(1, 4), Fraction(1, 8), Fraction(1, 2)],
        "rest_prob": 0.02,
        "fast_note_prob": 0.80,
        "cell_size": Fraction(1, 8),
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
        return options.density

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
            entry = Fraction(1, 2)
        elif options.density == "busy":
            entry = Fraction(0)
        else:
            entry = Fraction(0)

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
        num_notes = rng.randint(4, 8)
        intervals = [0]
        current = anchor
        for _ in range(num_notes - 1):
            if context.key_enforced:
                step = rng.choice([-2, -1, 1, 2])
                next_pitch = self._pick_scale_neighbor(
                    current, context, options, rng, max_steps=max(1, abs(step))
                )
                if step < 0 and next_pitch >= current:
                    next_pitch = self._scale_step(current, -1, context, options)
                elif step > 0 and next_pitch <= current:
                    next_pitch = self._scale_step(current, 1, context, options)
                intervals.append(next_pitch - current)
                current = next_pitch
            else:
                intervals.append(rng.choice([-2, -1, 1, 2]))
        fast = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)]
        durations = [fast[i % len(fast)] for i in range(num_notes)]
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
        if section == "A" and is_first_bar:
            motif_notes = self._motif_to_notes(
                motif, bar, chord, context, options, state, phrase_plan, section
            )
            cursor = motif_notes[-1].position + motif_notes[-1].duration if motif_notes else Fraction(0)
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
                last = motif_notes[-1]
                cursor = last.position + last.duration
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
        is_resolution = section == "resolution"
        bar_end = Fraction(beats_per_bar)

        while cursor < bar_end - Fraction(1, 16):
            remaining = bar_end - cursor
            if is_resolution and remaining <= Fraction(1, 2):
                notes.append(self._make_sustained_note(
                    bar, cursor, remaining, chord, next_chord, context, options,
                    state, phrase_plan, section, is_ending=True,
                ))
                break

            roll = rng.random()
            if roll < 0.38:
                chunk, cursor = self._make_scalic_run(
                    bar, cursor, bar_end, chord, next_chord, context, options,
                    state, phrase_plan, section, accent_map,
                )
                notes.extend(chunk)
            elif roll < 0.82:
                chunk, cursor = self._make_quaver_stream(
                    bar, cursor, bar_end, chord, next_chord, context, options,
                    state, phrase_plan, section, preset, accent_map,
                )
                notes.extend(chunk)
            else:
                dur = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(1, 1)])
                dur = min(dur, remaining)
                notes.append(self._make_sustained_note(
                    bar, cursor, dur, chord, next_chord, context, options,
                    state, phrase_plan, section,
                    is_ending=False,
                ))
                cursor += dur

        return notes

    def _make_scalic_run(
        self,
        bar: int,
        cursor: Fraction,
        bar_end: Fraction,
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        phrase_plan: PhrasePlan,
        section: PhraseSection,
        accent_map: dict[Fraction, float],
    ) -> tuple[list[MelodyNote], Fraction]:
        rng = state.rng
        cell = Fraction(1, 8) if rng.random() < 0.55 else Fraction(1, 4)
        if rng.random() < 0.35:
            cell = Fraction(1, 4) if cell == Fraction(1, 8) else Fraction(1, 8)

        length = rng.randint(4, 8)
        direction = rng.choice([-1, 1])
        notes: list[MelodyNote] = []
        pos = cursor
        pitch = state.last_pitch or self._pick_chord_tone(
            chord, context, options, rng, register_center=True
        )

        for i in range(length):
            if pos + cell > bar_end:
                break
            if i > 0:
                pitch = self._scale_step(pitch, direction, context, options)
                if context.key_enforced:
                    pitch = self._snap_to_scale(pitch, context, options)
                else:
                    pitch = self._clamp_register(
                        pitch + rng.choice([-1, 0, 1]), options
                    )
            is_strong = pos in (Fraction(0), Fraction(2))
            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(rng, is_strong, False, False, section),
                bar=bar,
                position=pos,
                duration=cell,
            ))
            state.last_pitch = pitch
            pos += cell

        if not notes:
            return [], cursor + cell
        return notes, pos

    def _make_quaver_stream(
        self,
        bar: int,
        cursor: Fraction,
        bar_end: Fraction,
        chord: ChordEvent,
        next_chord: ChordEvent,
        context: LoopContext,
        options: GenerationOptions,
        state: _GenState,
        phrase_plan: PhrasePlan,
        section: PhraseSection,
        preset: dict,
        accent_map: dict[Fraction, float],
    ) -> tuple[list[MelodyNote], Fraction]:
        rng = state.rng
        if rng.random() < 0.45:
            cell = Fraction(1, 8)
        else:
            cell = Fraction(1, 4)

        length = rng.randint(3, 7)
        notes: list[MelodyNote] = []
        pos = cursor

        for i in range(length):
            if pos + cell > bar_end:
                break
            is_strong = pos in (Fraction(0), Fraction(2)) or i == length - 1
            is_peak = bar == phrase_plan.peak_bar and pos == phrase_plan.peak_position
            pitch = self._choose_pitch(
                chord, next_chord, context, options, state,
                is_strong=is_strong, is_ending=False, is_peak=is_peak,
                bar=bar, section=section, phrase_plan=phrase_plan,
            )
            notes.append(MelodyNote(
                pitch=pitch,
                velocity=self._choose_velocity(rng, is_strong, is_peak, False, section),
                bar=bar,
                position=pos,
                duration=cell,
            ))
            state.last_pitch = pitch
            pos += cell

        if not notes:
            return [], min(cursor + cell, bar_end)
        return notes, pos

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
            is_strong=True, is_ending=is_ending, is_peak=False,
            bar=bar, section=section, phrase_plan=phrase_plan,
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
        fast_durs = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 8)]
        notes: list[MelodyNote] = []
        for i, interval in enumerate(motif.intervals):
            dur = fast_durs[i % len(fast_durs)]
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
