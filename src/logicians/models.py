"""Core data models for loop capture, analysis, and melody generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol


@dataclass
class NoteEvent:
    track: str
    pitch: int
    velocity: int
    start: float
    duration: float
    bar: int = 0
    position: Fraction = field(default_factory=lambda: Fraction(0))
    duration_beats: Fraction = field(default_factory=lambda: Fraction(1))


@dataclass
class ChordEvent:
    bar: int
    beat: Fraction
    root: int
    quality: str
    pitch_classes: set[int]
    name: str


@dataclass
class LoopContext:
    tempo_bpm: float
    time_signature: tuple[int, int]
    bars: int
    ppq: int
    key_root: int
    key_mode: str
    scale_pitch_classes: set[int]
    chords: list[ChordEvent]
    tracks: dict[str, list[NoteEvent]]
    bass_roots_by_bar: dict[int, int]
    rhythm_density: float
    key_enforced: bool = False


@dataclass
class MelodyNote:
    pitch: int
    velocity: int
    bar: int
    position: Fraction
    duration: Fraction


@dataclass
class MelodyClip:
    bars: int
    notes: list[MelodyNote]


@dataclass
class GenerationOptions:
    seed: int | None = None
    register_low: int = 60
    register_high: int = 84
    density: str = "medium"
    variation: float = 0.5
    chord_tone_weight: float = 0.65
    passing_tone_weight: float = 0.25
    chromatic_weight: float = 0.10
    rhythmic_complexity: float = 0.5
    phrase_length_bars: int | None = None
    start_on_loop_boundary: bool = True
    legato: bool = False


class MelodyGenerator(Protocol):
    def generate(self, context: LoopContext, options: GenerationOptions) -> MelodyClip:
        ...
