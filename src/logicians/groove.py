"""Runtime for the Groove-MIDI-trained tables (see scripts/train_groove.py).

Loads the committed JSON tables and provides:
- MarkovDrumModel: generate a drum groove in a learned style (the drum "style"
  generation mode), via a position-conditioned Markov chain over 16th-note
  instrument tokens, with velocities sampled from the dataset.
- style_bass_template(): the characteristic kick rhythm of a style, turned into
  bass groove slots (data-driven bass rhythm).

Data derived from the Groove MIDI Dataset (Magenta), CC-BY-4.0.
"""

from __future__ import annotations

import json
import os
import random
from fractions import Fraction
from functools import lru_cache

from .drums import CRASH, KICK, OPEN_HAT, SNARE, CLOSED_HAT, DrumClip, DrumHit

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Instrument class -> representative GM output pitch.
PITCH_BY_CLASS = {
    "kick": KICK,        # 36
    "snare": SNARE,      # 38
    "chat": CLOSED_HAT,  # 42
    "ohat": OPEN_HAT,    # 46
    "tom": 45,
    "crash": CRASH,      # 49
    "ride": 51,
}


@lru_cache(maxsize=None)
def _load(name: str) -> dict | None:
    """Load and cache a trained JSON table (read once per process)."""
    path = os.path.join(_DATA_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _style_table(styles: dict, style: str | None) -> dict | None:
    """A style's table, falling back to the 'all' aggregate bucket."""
    return styles.get(style) or styles.get("all")


class MarkovDrumModel:
    """Generate a drum groove in a trained style."""

    def __init__(self, data: dict):
        self.data = data
        self.steps_per_beat = data["meta"]["steps_per_beat"]

    @classmethod
    def load(cls) -> "MarkovDrumModel | None":
        data = _load("groove_drums.json")
        return cls(data) if data else None

    def styles(self) -> list[str]:
        return sorted(s for s in self.data["styles"] if s != "all")

    def _choice(self, distribution: dict[str, int], rng: random.Random) -> str:
        total = sum(distribution.values())
        if total <= 0:
            return ""
        threshold = rng.random() * total
        acc = 0
        for token, count in distribution.items():
            acc += count
            if threshold <= acc:
                return token
        return ""

    def generate(self, style: str, bars: int, beats_per_bar: int, rng: random.Random) -> DrumClip:
        table = _style_table(self.data["styles"], style)
        transitions = table["transitions"]
        velocity = table["velocity"]
        steps_per_bar = beats_per_bar * self.steps_per_beat

        seen: set[tuple[int, Fraction, int]] = set()
        hits: list[DrumHit] = []
        prev = ""
        for bar in range(1, bars + 1):
            for i in range(steps_per_bar):
                dist = transitions.get(f"{i}|{prev}") or transitions.get(f"{i}|")
                token = self._choice(dist, rng) if dist else ""
                if token:
                    position = Fraction(i, self.steps_per_beat)
                    for cls in token.split("+"):
                        pitch = PITCH_BY_CLASS.get(cls)
                        if pitch is None:
                            continue
                        mean, std = velocity.get(cls, [90.0, 8.0])
                        vel = max(1, min(127, int(rng.gauss(mean, std))))
                        key = (bar, position, pitch)
                        if key in seen:
                            continue
                        seen.add(key)
                        hits.append(DrumHit(pitch, vel, bar, position))
                prev = token
        return DrumClip(bars=bars, hits=sorted(hits, key=lambda h: (h.bar, float(h.position), h.pitch)))


KICK_TEMPLATE_THRESHOLD = 0.25  # kick appears this often across the corpus -> a bass slot


def style_bass_template(style: str, beats_per_bar: int):
    """The style's characteristic kick rhythm as bass groove slots, or None if
    unavailable. Returns a list of GrooveSlot (imported lazily to avoid a cycle)."""
    data = _load("groove_bass_templates.json")
    if not data:
        return None
    table = _style_table(data["styles"], style)
    if not table:
        return None
    from .bass import GrooveSlot

    steps_per_beat = data["meta"]["steps_per_beat"]
    freq = table["kick_freq"]
    peak = max(freq) or 1.0
    slots = []
    for step, f in enumerate(freq):
        on_beat = step % steps_per_beat == 0
        if f >= KICK_TEMPLATE_THRESHOLD or on_beat:
            slots.append(GrooveSlot(Fraction(step, steps_per_beat), min(1.0, f / peak + (0.3 if on_beat else 0.0)), on_beat))
    # guarantee a downbeat anchor
    if not any(s.position == 0 for s in slots):
        slots.insert(0, GrooveSlot(Fraction(0), 1.0, True))
    return slots
