# Third-party data attribution

## Groove MIDI Dataset (GMD)

The trained tables in `src/logicians/data/groove_drums.json` and
`src/logicians/data/groove_bass_templates.json` are derived from the
**Groove MIDI Dataset** by Google Magenta.

- Source: https://magenta.withgoogle.com/datasets/groove
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Citation: Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, and David Bamman.
  "Learning to Groove with Inverse Sequence Transformations." ICML 2019.

The dataset is drums-only. Our tables capture (a) a position-conditioned Markov
model of drum patterns per style, and (b) characteristic kick rhythms reused as
bass groove templates. The dataset itself is not redistributed here; regenerate
the tables with `scripts/train_groove.py` (see that file for the download URL).
