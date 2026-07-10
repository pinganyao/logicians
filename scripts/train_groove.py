"""Train Markov drum tables + bass groove templates from the Groove MIDI Dataset.

The Groove MIDI Dataset (GMD) is drum-only, CC-BY-4.0, by Google Magenta
(Gillick et al., "Learning to Groove with Inverse Sequence Transformations").
Download the tiny MIDI-only archive (3 MB):

    https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip

Then run:

    python scripts/train_groove.py --dataset /path/to/groove

This writes two small JSON tables into src/logicians/data/ (committed, so the
dataset itself is not needed at runtime):

- groove_drums.json    position-conditioned Markov transitions + per-instrument
                       velocity stats, per musical style. Drives the drum
                       "style" generation mode.
- groove_bass_templates.json  characteristic kick rhythm per style, reused as
                       data-driven bass groove templates.

Only 4/4 "beat" (looping groove) files are used; fills and odd meters skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import mido

STEPS_PER_BEAT = 4  # 16th-note grid

# Roland TD-11 pitch -> instrument class.
CLASS_BY_PITCH = {
    35: "kick", 36: "kick",
    37: "snare", 38: "snare", 40: "snare",
    42: "chat", 44: "chat", 22: "chat",
    46: "ohat", 26: "ohat",
    43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom", 58: "tom",
    49: "crash", 52: "crash", 55: "crash", 57: "crash",
    51: "ride", 53: "ride", 59: "ride",
}


def _token(classes) -> str:
    return "+".join(sorted(classes))


def _steps_from_midi(path: str, beats_per_bar: int) -> list[dict[str, int]]:
    """Return a list of steps; each step is {class: velocity} on the 16th grid."""
    mid = mido.MidiFile(path)
    ppq = mid.ticks_per_beat or 480
    steps: dict[int, dict[str, int]] = defaultdict(dict)
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                cls = CLASS_BY_PITCH.get(msg.note)
                if not cls:
                    continue
                step = int(round(tick / ppq * STEPS_PER_BEAT))
                # loudest wins if two same-class hits land on one step
                steps[step][cls] = max(steps[step].get(cls, 0), msg.velocity)
    if not steps:
        return []
    length = max(steps) + 1
    return [steps.get(i, {}) for i in range(length)]


def train(dataset_dir: str, out_dir: str) -> None:
    root = dataset_dir
    if os.path.isdir(os.path.join(root, "groove")):
        root = os.path.join(root, "groove")  # the zip nests a 'groove/' folder
    info_path = os.path.join(root, "info.csv")

    # style -> transition counts keyed by "stepindex|prevtoken" -> Counter(nexttoken)
    transitions: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    velocities: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    kick_hits: dict[str, list] = defaultdict(lambda: [0] * (STEPS_PER_BEAT * 8))
    kick_bars: dict[str, int] = defaultdict(int)
    used = 0

    with open(info_path) as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        if row["beat_type"] != "beat" or row["time_signature"] != "4-4":
            continue
        beats_per_bar = 4
        steps_per_bar = beats_per_bar * STEPS_PER_BEAT
        path = os.path.join(root, row["midi_filename"])
        if not os.path.exists(path):
            continue
        try:
            steps = _steps_from_midi(path, beats_per_bar)
        except Exception:
            continue
        if len(steps) < steps_per_bar:
            continue

        primary = row["style"].split("/")[0]
        for label in (primary, "all"):
            prev = ""
            for i, step in enumerate(steps):
                token = _token(step.keys())
                key = f"{i % steps_per_bar}|{prev}"
                transitions[label][key][token] += 1
                for cls, vel in step.items():
                    velocities[label][cls].append(vel)
                if "kick" in step:
                    kick_hits[label][i % steps_per_bar] += 1
                prev = token
            kick_bars[label] += len(steps) / steps_per_bar
        used += 1

    # serialize drum model
    drum_model = {
        "meta": {
            "source": "Groove MIDI Dataset (Magenta), CC-BY-4.0",
            "files_used": used,
            "steps_per_beat": STEPS_PER_BEAT,
        },
        "styles": {},
    }
    for label, table in transitions.items():
        vel_stats = {}
        for cls, vals in velocities[label].items():
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
            vel_stats[cls] = [round(mean, 1), round(std, 1)]
        drum_model["styles"][label] = {
            "transitions": {k: dict(c) for k, c in table.items()},
            "velocity": vel_stats,
        }

    # serialize bass groove templates from characteristic kick patterns
    bass_templates = {"meta": drum_model["meta"], "styles": {}}
    for label, hits in kick_hits.items():
        bars = max(kick_bars[label], 1)
        freq = [round(h / bars, 3) for h in hits[: STEPS_PER_BEAT * 4]]  # one 4/4 bar
        bass_templates["styles"][label] = {"kick_freq": freq}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "groove_drums.json"), "w") as fh:
        json.dump(drum_model, fh, separators=(",", ":"))
    with open(os.path.join(out_dir, "groove_bass_templates.json"), "w") as fh:
        json.dump(bass_templates, fh, separators=(",", ":"))

    print(f"Trained on {used} files. Styles: {sorted(k for k in drum_model['styles'] if k != 'all')}")
    print(f"Wrote {out_dir}/groove_drums.json and groove_bass_templates.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to the extracted GMD folder")
    default_out = os.path.join(os.path.dirname(__file__), "..", "src", "logicians", "data")
    ap.add_argument("--out", default=os.path.normpath(default_out))
    args = ap.parse_args()
    train(args.dataset, args.out)
