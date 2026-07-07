"""Loop analysis: key, chord, rhythm density inference."""

from __future__ import annotations

from fractions import Fraction

from .midi_io import NOTE_NAMES, estimate_loop_bars, group_notes_by_track, split_mixed_harmony_notes
from .models import ChordEvent, LoopContext, NoteEvent

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
NATURAL_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
HARMONIC_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 11}

CHORD_QUALITIES: dict[str, set[int]] = {
    "major": {0, 4, 7},
    "minor": {0, 3, 7},
    "diminished": {0, 3, 6},
    "sus2": {0, 2, 7},
    "sus4": {0, 5, 7},
    "dom7": {0, 4, 7, 10},
    "maj7": {0, 4, 7, 11},
    "min7": {0, 3, 7, 10},
    "minMaj7": {0, 3, 7, 11},
}

QUALITY_SUFFIX = {
    "major": "",
    "minor": "m",
    "diminished": "dim",
    "sus2": "sus2",
    "sus4": "sus4",
    "dom7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "minMaj7": "mMaj7",
}

ROOT_BY_NAME: dict[str, int] = {
    "c": 0, "b#": 0,
    "c#": 1, "db": 1,
    "d": 2,
    "d#": 3, "eb": 3,
    "e": 4, "fb": 4,
    "f": 5, "e#": 5,
    "f#": 6, "gb": 6,
    "g": 7,
    "g#": 8, "ab": 8,
    "a": 9,
    "a#": 10, "bb": 10,
    "b": 11, "cb": 11,
}

MODE_ALIASES: dict[str, str] = {
    "": "major",
    "maj": "major",
    "major": "major",
    "m": "minor",
    "min": "minor",
    "minor": "minor",
    "harmonic": "harmonic_minor",
    "harmonicminor": "harmonic_minor",
    "harmonic_minor": "harmonic_minor",
    "hmin": "harmonic_minor",
    "naturalminor": "minor",
    "natural_minor": "minor",
}

MODE_TEMPLATES: dict[str, set[int]] = {
    "major": MAJOR_SCALE,
    "minor": NATURAL_MINOR_SCALE,
    "harmonic_minor": HARMONIC_MINOR_SCALE,
}


def parse_key(key: str) -> tuple[int, str, set[int]]:
    """Parse a key string like 'C', 'Am', 'F# minor', 'Bb harmonic minor'."""
    normalized = key.strip().lower().replace("_", " ")
    if not normalized:
        raise ValueError("Key cannot be empty")

    root_pc: int | None = None
    remainder = ""
    for length in (2, 1):
        candidate = normalized[:length]
        if candidate in ROOT_BY_NAME:
            root_pc = ROOT_BY_NAME[candidate]
            remainder = normalized[length:].strip()
            break

    if root_pc is None:
        raise ValueError(
            f"Unrecognized key: {key!r}. Use forms like C, Am, F# minor, Bb harmonic minor."
        )

    mode_key = remainder.replace(" ", "")
    if mode_key in MODE_ALIASES:
        mode = MODE_ALIASES[mode_key]
    elif remainder in ("harmonic minor",):
        mode = "harmonic_minor"
    elif remainder.endswith("m") and len(remainder) <= 2:
        mode = "minor"
    else:
        raise ValueError(
            f"Unrecognized mode in key {key!r}. Use major, minor, or harmonic minor."
        )

    template = MODE_TEMPLATES[mode]
    scale = _rotate_scale(template, root_pc)
    return root_pc, mode, scale


def format_key(key_root: int, key_mode: str) -> str:
    name = NOTE_NAMES[key_root]
    if key_mode == "major":
        return name
    if key_mode == "minor":
        return f"{name}m"
    return f"{name} {key_mode.replace('_', ' ')}"

def _rotate_scale(scale: set[int], root: int) -> set[int]:
    return {(pc + root) % 12 for pc in scale}


def _note_weight(note: NoteEvent, role: str) -> float:
    duration = float(note.duration_beats) if note.duration_beats else note.duration
    base = max(duration, 0.25)
    if role == "bass":
        return base * 3.0
    if role == "chords":
        return base * 2.0
    if role == "melody_reference":
        return base * 0.5
    return base


def infer_key(
    tracks: dict[str, list[NoteEvent]],
) -> tuple[int, str, set[int]]:
    """Return (key_root, mode, scale_pitch_classes)."""
    histogram = [0.0] * 12

    for role, notes in tracks.items():
        if role == "drums":
            continue
        for note in notes:
            weight = _note_weight(note, role)
            histogram[note.pitch % 12] += weight

    best_score = -1.0
    best_root = 0
    best_mode = "major"
    best_scale = MAJOR_SCALE

    for root in range(12):
        for mode, template in [
            ("major", MAJOR_SCALE),
            ("minor", NATURAL_MINOR_SCALE),
            ("harmonic_minor", HARMONIC_MINOR_SCALE),
        ]:
            scale = _rotate_scale(template, root)
            score = sum(histogram[pc] for pc in scale)
            non_scale = sum(histogram[pc] for pc in range(12) if pc not in scale)
            score -= non_scale * 0.3
            if score > best_score:
                best_score = score
                best_root = root
                best_mode = mode
                best_scale = scale

    return best_root, best_mode, best_scale


def _chord_name(root: int, quality: str) -> str:
    return f"{NOTE_NAMES[root]}{QUALITY_SUFFIX.get(quality, '')}"


def _chord_template(root: int, quality: str) -> set[int]:
    return {(root + interval) % 12 for interval in CHORD_QUALITIES[quality]}


def _diatonic_chords_in_major(key_root: int) -> dict[int, set[str]]:
    """Map pitch class -> likely qualities for chords diatonic to a major key."""
    # Scale degrees: I, ii, iii, IV, V, vi, vii°
    degree_specs = [
        (0, {"major", "maj7"}),
        (2, {"minor", "min7"}),
        (4, {"minor", "min7"}),
        (5, {"major", "maj7"}),
        (7, {"major", "dom7", "maj7"}),
        (9, {"minor", "min7"}),
        (11, {"diminished"}),
    ]
    result: dict[int, set[str]] = {}
    for semitone, qualities in degree_specs:
        pc = (key_root + semitone) % 12
        result[pc] = qualities
    return result


def _collect_bar_weights(notes: list[NoteEvent], bar: int) -> dict[int, float]:
    weights: dict[int, float] = {}
    for note in notes:
        if note.bar != bar:
            continue
        pc = note.pitch % 12
        duration = float(note.duration_beats) if note.duration_beats else note.duration
        weight = max(duration, 0.25) * (note.velocity / 127.0)
        weights[pc] = weights.get(pc, 0.0) + weight
    return weights


def _has_extension(weights: dict[int, float], root: int, semitone: int) -> bool:
    return weights.get((root + semitone) % 12, 0.0) > 0.1


def _score_chord_candidate(
    weights: dict[int, float],
    root: int,
    quality: str,
    bass_pc: int | None,
    scale: set[int],
    key_root: int,
    key_mode: str,
) -> float:
    template = _chord_template(root, quality)
    if not weights:
        return -1.0

    score = 0.0
    for pc, w in weights.items():
        if pc in template:
            score += w * 3.0
        elif pc in scale:
            score -= w * 0.4
        else:
            score -= w * 1.0

    for pc in template:
        if weights.get(pc, 0.0) < 0.1:
            score -= 0.5

    if bass_pc is not None:
        if bass_pc == root:
            score += 3.0
        elif bass_pc in template:
            score += 1.0
        else:
            score -= 0.5

    # Prefer chord qualities that match observed extensions.
    if quality == "maj7" and _has_extension(weights, root, 11):
        score += 1.5
    if quality in ("min7", "dom7") and _has_extension(weights, root, 10):
        score += 1.5
    if quality == "major" and not _has_extension(weights, root, 10) and not _has_extension(weights, root, 11):
        score += 0.5
    if quality in ("min7",) and not _has_extension(weights, root, 10):
        score -= 0.8
    if quality in ("maj7",) and not _has_extension(weights, root, 11):
        score -= 0.8

    # Penalise overly complex qualities without evidence.
    if quality in ("diminished", "sus2", "sus4", "minMaj7"):
        score -= 0.5

    template_in_scale = template <= scale
    if template_in_scale:
        score += 0.4

    if key_mode == "major":
        diatonic = _diatonic_chords_in_major(key_root)
        if root in diatonic and quality in diatonic[root]:
            score += 1.0
        elif root in diatonic:
            score += 0.2

    return score


def _resolve_harmony_tracks(
    tracks: dict[str, list[NoteEvent]],
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Return (chord_notes, bass_notes), splitting mixed live capture when needed."""
    chord_notes = list(tracks.get("chords", []))
    bass_notes = list(tracks.get("bass", []))
    if chord_notes or bass_notes:
        return chord_notes, bass_notes

    pooled: list[NoteEvent] = []
    for role, notes in tracks.items():
        if role in ("drums", "melody_reference"):
            continue
        pooled.extend(notes)

    if not pooled:
        return [], []

    return split_mixed_harmony_notes(pooled)


def _fallback_chord_quality(
    root: int,
    weights: dict[int, float],
    key_root: int,
    key_mode: str,
) -> str:
    if _has_extension(weights, root, 11):
        return "maj7"
    if _has_extension(weights, root, 10):
        if key_mode == "major":
            diatonic = _diatonic_chords_in_major(key_root)
            if root in diatonic and "dom7" in diatonic[root]:
                return "dom7"
            if root in diatonic and "min7" in diatonic[root]:
                return "min7"
        return "min7"
    if key_mode == "major":
        diatonic = _diatonic_chords_in_major(key_root)
        if root in diatonic:
            quals = diatonic[root]
            if "minor" in quals and "major" not in quals:
                return "minor"
            return "major"
    if root in _rotate_scale(NATURAL_MINOR_SCALE, key_root):
        return "minor"
    return "major"


def infer_chord_for_bar(
    bar: int,
    chord_notes: list[NoteEvent],
    bass_notes: list[NoteEvent],
    key_root: int,
    scale: set[int],
    key_mode: str = "major",
) -> ChordEvent:
    bar_bass = [n for n in bass_notes if n.bar == bar]
    bar_chord = [n for n in chord_notes if n.bar == bar]
    bar_weights = _collect_bar_weights(bar_chord + bar_bass, bar)

    bass_pc: int | None = None
    if bar_bass:
        bass_pc = min(bar_bass, key=lambda n: n.pitch).pitch % 12

    root_candidates = list(range(12))
    if bass_pc is not None:
        # Pop loops are usually root-position — search bass root first.
        root_candidates = [bass_pc] + [r for r in range(12) if r != bass_pc]

    best_score = -1.0
    best_root = bass_pc if bass_pc is not None else key_root
    best_quality = "major"

    for root in root_candidates:
        for quality in ("maj7", "min7", "dom7", "major", "minor"):
            score = _score_chord_candidate(
                bar_weights, root, quality, bass_pc, scale, key_root, key_mode
            )
            if score > best_score:
                best_score = score
                best_root = root
                best_quality = quality

    if best_score < 0.5:
        if bass_pc is not None:
            best_root = bass_pc
        elif bar_weights:
            best_root = max(bar_weights, key=bar_weights.get)
        else:
            best_root = key_root
        best_quality = _fallback_chord_quality(best_root, bar_weights, key_root, key_mode)

    pitch_classes = _chord_template(best_root, best_quality)
    return ChordEvent(
        bar=bar,
        beat=Fraction(0),
        root=best_root,
        quality=best_quality,
        pitch_classes=pitch_classes,
        name=_chord_name(best_root, best_quality),
    )


def infer_chords(
    tracks: dict[str, list[NoteEvent]],
    bars: int,
    key_root: int,
    scale: set[int],
    key_mode: str = "major",
) -> list[ChordEvent]:
    chord_notes, bass_notes = _resolve_harmony_tracks(tracks)
    return [
        infer_chord_for_bar(bar, chord_notes, bass_notes, key_root, scale, key_mode)
        for bar in range(1, bars + 1)
    ]


def bass_roots_by_bar(tracks: dict[str, list[NoteEvent]], bars: int) -> dict[int, int]:
    _, bass_notes = _resolve_harmony_tracks(tracks)
    result: dict[int, int] = {}
    for bar in range(1, bars + 1):
        bar_notes = [n for n in bass_notes if n.bar == bar]
        if bar_notes:
            lowest = min(bar_notes, key=lambda n: n.pitch)
            result[bar] = lowest.pitch % 12
    return result


def compute_rhythm_density(tracks: dict[str, list[NoteEvent]], bars: int, beats_per_bar: int) -> float:
    """Return normalized density score 0.0–1.0."""
    relevant = []
    for role in ("drums", "bass", "chords"):
        relevant.extend(tracks.get(role, []))

    if not relevant or bars == 0:
        return 0.5

    notes_per_bar = len(relevant) / bars
    onsets = sorted(n.start for n in relevant)
    if len(onsets) > 1:
        intervals = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
        avg_interval = sum(intervals) / len(intervals)
    else:
        avg_interval = beats_per_bar

    # Offbeat syncopation: notes not on integer beats
    offbeat_count = sum(
        1 for n in relevant
        if Fraction(n.start).limit_denominator(8) % 1 not in (Fraction(0),)
    )
    syncopation = offbeat_count / max(len(relevant), 1)

    density = min(1.0, notes_per_bar / 16.0) * 0.5
    density += min(1.0, 1.0 / max(avg_interval, 0.125)) * 0.3
    density += syncopation * 0.2
    return min(1.0, max(0.0, density))


def build_loop_context(
    notes: list[NoteEvent],
    tempo_bpm: float,
    time_signature: tuple[int, int],
    ppq: int,
    bars: int | None = None,
    key: str | None = None,
) -> LoopContext:
    tracks = group_notes_by_track(notes)
    loop_bars = bars or estimate_loop_bars(notes, time_signature)
    beats_per_bar = time_signature[0]

    if key is not None:
        key_root, key_mode, scale = parse_key(key)
    else:
        key_root, key_mode, scale = infer_key(tracks)
    chords = infer_chords(tracks, loop_bars, key_root, scale, key_mode)
    bass_roots = bass_roots_by_bar(tracks, loop_bars)
    density = compute_rhythm_density(tracks, loop_bars, beats_per_bar)

    return LoopContext(
        tempo_bpm=tempo_bpm,
        time_signature=time_signature,
        bars=loop_bars,
        ppq=ppq,
        key_root=key_root,
        key_mode=key_mode,
        scale_pitch_classes=scale,
        chords=chords,
        tracks=tracks,
        bass_roots_by_bar=bass_roots,
        rhythm_density=density,
        key_enforced=key is not None,
    )
