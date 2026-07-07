"""Reactive drum generation.

Generates a *variation* of an existing drum loop that follows a generated
melody. It is intentionally self-contained (its own data models, options, and
generator) so it can be developed and tested without touching the melody
generation or scheduling code.

Design (conservative variation):
- Keep the original kick/backbeat-snare skeleton so it still reads as the same
  groove. If the loop has no usable drums, synthesize a basic backbeat.
- Rebuild the hi-hat layer, accenting hats that coincide with strong melody
  onsets.
- Add ghost snares in the melody's rests ("answer in the gaps").
- Drop a short tom fill on the last beat of the loop (phrase boundary).
- Hit a crash + kick on the melody's peak.

The generator is a pure function of (original drum track, melody clip, options)
and is deterministic for a fixed seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction

from .models import LoopContext, MelodyClip, MelodyNote

# --- General MIDI percussion (channel 10) -----------------------------------

KICK = 36
SNARE = 38
CLOSED_HAT = 42
OPEN_HAT = 46
CRASH = 49
# Toms high -> low, used for the phrase-boundary fill.
FILL_TOMS = [50, 47, 45, 43]

# Pitch classes of the *input* loop we recognize when extracting the skeleton.
INPUT_KICKS = {35, 36}
INPUT_SNARES = {37, 38, 40}

# Drum hits are one-shots; duration is nominal (matters only for note-off).
HIT_DURATION = Fraction(1, 8)


@dataclass
class DrumHit:
    """A single percussion strike. Unlike a melody note, hits are polyphonic:
    several hits may share the same (bar, position)."""

    pitch: int
    velocity: int
    bar: int
    position: Fraction
    duration: Fraction = HIT_DURATION


@dataclass
class DrumClip:
    bars: int
    hits: list[DrumHit] = field(default_factory=list)


@dataclass
class DrumOptions:
    seed: int | None = None
    # 0.0 = closest to the original groove, 1.0 = busiest reaction. Conservative
    # by default: keeps the skeleton, adds a modest reactive layer.
    variation: float = 0.3
    channel: int = 9  # GM channel 10 (0-indexed)


class RuleBasedDrumGenerator:
    """Produce a reactive drum variation that follows a melody clip."""

    def generate(
        self,
        context: LoopContext,
        melody: MelodyClip,
        options: DrumOptions | None = None,
    ) -> DrumClip:
        options = options or DrumOptions()
        rng = random.Random(options.seed)
        beats_per_bar = context.time_signature[0]
        bars = context.bars

        skeleton = self._extract_skeleton(
            context.tracks.get("drums", []), bars, beats_per_bar
        )
        onsets_by_bar, intervals_by_bar = self._melody_maps(melody)
        peak = self._melody_peak(melody)

        hits: list[DrumHit] = list(skeleton)
        hits += self._hat_layer(bars, beats_per_bar, onsets_by_bar, options, rng)
        hits += self._ghost_layer(bars, beats_per_bar, intervals_by_bar, options, rng)
        hits = self._apply_fill(hits, bars, beats_per_bar, rng)
        if peak is not None:
            hits += self._peak_accent(peak)

        return DrumClip(bars=bars, hits=self._validate(hits, bars, beats_per_bar))

    # -- skeleton -------------------------------------------------------------

    def _extract_skeleton(
        self, original: list, bars: int, beats_per_bar: int
    ) -> list[DrumHit]:
        """Keep the original kick + snare hits (the groove backbone). Fall back
        to a synthesized backbeat when the loop has no usable drums."""
        kept: list[DrumHit] = []
        for note in original:
            if note.pitch in INPUT_KICKS:
                kept.append(DrumHit(KICK, note.velocity, note.bar, note.position))
            elif note.pitch in INPUT_SNARES:
                kept.append(DrumHit(SNARE, note.velocity, note.bar, note.position))
        if kept:
            return kept
        return self._synth_skeleton(bars, beats_per_bar)

    def _synth_skeleton(self, bars: int, beats_per_bar: int) -> list[DrumHit]:
        """Basic backbeat: kick on downbeats/mid-bar, snare on the backbeats."""
        hits: list[DrumHit] = []
        for bar in range(1, bars + 1):
            for beat in range(beats_per_bar):
                pos = Fraction(beat)
                if beat % 2 == 0:
                    hits.append(DrumHit(KICK, 100, bar, pos))
                else:
                    hits.append(DrumHit(SNARE, 95, bar, pos))
        return hits

    # -- melody analysis ------------------------------------------------------

    def _melody_maps(
        self, melody: MelodyClip
    ) -> tuple[dict[int, list[tuple[Fraction, int]]], dict[int, list[tuple[Fraction, Fraction]]]]:
        """Return per-bar (onset position, velocity) pairs and per-bar sounding
        intervals [start, end) used to detect rests."""
        onsets: dict[int, list[tuple[Fraction, int]]] = {}
        intervals: dict[int, list[tuple[Fraction, Fraction]]] = {}
        for note in melody.notes:
            onsets.setdefault(note.bar, []).append((note.position, note.velocity))
            intervals.setdefault(note.bar, []).append(
                (note.position, note.position + note.duration)
            )
        return onsets, intervals

    def _melody_peak(self, melody: MelodyClip) -> MelodyNote | None:
        """The loudest melody note marks the phrase peak (the rule-based melody
        generator assigns peak notes the highest velocities). Ties -> earliest."""
        if not melody.notes:
            return None
        return max(
            melody.notes,
            key=lambda n: (n.velocity, -n.bar, -float(n.position)),
        )

    def _has_accent(
        self,
        bar: int,
        pos: Fraction,
        onsets_by_bar: dict[int, list[tuple[Fraction, int]]],
        threshold: int,
    ) -> bool:
        for onset_pos, vel in onsets_by_bar.get(bar, []):
            if abs(onset_pos - pos) <= Fraction(1, 4) and vel >= threshold:
                return True
        return False

    def _is_resting(
        self,
        bar: int,
        pos: Fraction,
        intervals_by_bar: dict[int, list[tuple[Fraction, Fraction]]],
    ) -> bool:
        for start, end in intervals_by_bar.get(bar, []):
            if start <= pos < end:
                return False
        return True

    # -- reactive layers ------------------------------------------------------

    def _hat_layer(
        self,
        bars: int,
        beats_per_bar: int,
        onsets_by_bar: dict[int, list[tuple[Fraction, int]]],
        options: DrumOptions,
        rng: random.Random,
    ) -> list[DrumHit]:
        """Steady hats, accented (louder + occasionally opened) where the melody
        lands a strong onset. Sixteenth grid only at high variation."""
        subdivision = 4 if options.variation > 0.6 else 2
        grid = [Fraction(i, subdivision) for i in range(subdivision * beats_per_bar)]
        hits: list[DrumHit] = []
        for bar in range(1, bars + 1):
            for pos in grid:
                if self._has_accent(bar, pos, onsets_by_bar, threshold=90):
                    pitch = OPEN_HAT if rng.random() < 0.3 else CLOSED_HAT
                    velocity = rng.randint(80, 100)
                else:
                    pitch = CLOSED_HAT
                    velocity = rng.randint(45, 65)
                hits.append(DrumHit(pitch, velocity, bar, pos))
        return hits

    def _ghost_layer(
        self,
        bars: int,
        beats_per_bar: int,
        intervals_by_bar: dict[int, list[tuple[Fraction, Fraction]]],
        options: DrumOptions,
        rng: random.Random,
    ) -> list[DrumHit]:
        """Ghost snares on offbeats where the melody is silent -- the drummer
        answering in the gaps. Density scales with variation."""
        prob = 0.15 + 0.4 * options.variation
        offbeats = [Fraction(2 * i + 1, 2) for i in range(beats_per_bar)]
        hits: list[DrumHit] = []
        for bar in range(1, bars + 1):
            for pos in offbeats:
                if self._is_resting(bar, pos, intervals_by_bar) and rng.random() < prob:
                    hits.append(DrumHit(SNARE, rng.randint(25, 45), bar, pos))
        return hits

    def _apply_fill(
        self,
        hits: list[DrumHit],
        bars: int,
        beats_per_bar: int,
        rng: random.Random,
    ) -> list[DrumHit]:
        """Replace the last beat's hats with a descending tom fill leading back
        into the loop."""
        region_start = Fraction(beats_per_bar - 1)
        kept = [
            h
            for h in hits
            if not (
                h.bar == bars
                and h.position >= region_start
                and h.pitch in (CLOSED_HAT, OPEN_HAT)
            )
        ]
        for k in range(4):
            pos = region_start + Fraction(k, 4)
            pitch = FILL_TOMS[k % len(FILL_TOMS)]
            kept.append(DrumHit(pitch, rng.randint(85, 110), bars, pos))
        return kept

    def _peak_accent(self, peak: MelodyNote) -> list[DrumHit]:
        """Crash + kick on the melody peak."""
        return [
            DrumHit(CRASH, 112, peak.bar, peak.position),
            DrumHit(KICK, 108, peak.bar, peak.position),
        ]

    # -- validation -----------------------------------------------------------

    def _validate(
        self, hits: list[DrumHit], bars: int, beats_per_bar: int
    ) -> list[DrumHit]:
        """Drop out-of-bounds hits, clamp velocity, and dedupe hits sharing a
        (bar, position, pitch), keeping the loudest."""
        best: dict[tuple[int, Fraction, int], DrumHit] = {}
        for h in hits:
            if not (1 <= h.bar <= bars):
                continue
            if not (Fraction(0) <= h.position < beats_per_bar):
                continue
            velocity = max(1, min(127, h.velocity))
            key = (h.bar, h.position, h.pitch)
            existing = best.get(key)
            if existing is None or velocity > existing.velocity:
                best[key] = DrumHit(h.pitch, velocity, h.bar, h.position, h.duration)
        return sorted(best.values(), key=lambda h: (h.bar, float(h.position), h.pitch))
