#!/usr/bin/env python3
"""Generate a system architecture diagram for The Logicians presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
OUT_PNG = OUT_DIR / "system-architecture.png"
OUT_PDF = OUT_DIR / "system-architecture.pdf"

INK = "#100f14"
SURFACE = "#1a1822"
SURFACE_RAISED = "#211e2a"
ACCENT = "#9d8df1"
WARM = "#d4a574"
TEXT = "#f4f4f5"
TEXT_MUTED = "#9a96a8"
BORDER = "#2e2a3a"
LIVE = "#6ec8e8"
POSITIVE = "#7dd4a8"


def box(ax, x, y, w, h, label, sublabel=None, *, face=SURFACE_RAISED, edge=BORDER, accent=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5 if accent else 1.0,
        edgecolor=accent or edge,
        facecolor=face,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    y_text = y + h / 2 + (0.011 if sublabel else 0)
    ax.text(x + w / 2, y_text, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=TEXT, transform=ax.transAxes, zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.026, sublabel, ha="center", va="center",
                fontsize=8.2, color=TEXT_MUTED, transform=ax.transAxes, zorder=3)


def arrow(ax, x1, y1, x2, y2, *, color=TEXT_MUTED, lw=1.4, rad=0.0):
    style = f"Simple, tail_width=0.5, head_width=4, head_length=6"
    if rad:
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=lw,
            color=color, transform=ax.transAxes, zorder=1, connectionstyle=f"arc3,rad={rad}",
            shrinkA=3, shrinkB=3,
        ))
    else:
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=lw,
            color=color, transform=ax.transAxes, zorder=1, shrinkA=3, shrinkB=3,
        ))


def label(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=9, fontweight="bold",
            color=TEXT_MUTED, transform=ax.transAxes, zorder=3)


def build_diagram() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=INK)
    ax.set_facecolor(INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.935, "The Logicians", fontsize=30, fontweight="bold", color=TEXT, ha="left")
    ax.text(0.05, 0.89, "System Architecture", fontsize=15, color=ACCENT, ha="left")
    ax.text(0.05, 0.855, "AI-assisted live improvisation for Logic Pro", fontsize=11, color=TEXT_MUTED, ha="left")

    # Logic Pro (external)
    box(ax, 0.04, 0.40, 0.11, 0.16, "Logic Pro", "loop playback", face=SURFACE, accent=WARM)
    box(ax, 0.85, 0.40, 0.11, 0.16, "Logic Pro", "improvised parts", face=SURFACE, accent=WARM)

    # Interfaces
    label(ax, 0.20, 0.76, "INTERFACES")
    box(ax, 0.20, 0.64, 0.10, 0.085, "CLI", "cli.py")
    box(ax, 0.32, 0.64, 0.13, 0.085, "Live App", "desktop · server · UI")
    box(ax, 0.47, 0.64, 0.15, 0.085, "Live Session", "live_session.py", accent=ACCENT)

    # Input
    label(ax, 0.20, 0.56, "INPUT / CAPTURE")
    box(ax, 0.20, 0.42, 0.13, 0.095, "MIDI Input", "midi_io.py", accent=LIVE)
    box(ax, 0.35, 0.42, 0.11, 0.095, "MIDI File", "parse · ingest")
    box(ax, 0.48, 0.42, 0.11, 0.095, "Quantizer", "quantize.py")

    # Analysis
    label(ax, 0.20, 0.34, "ANALYSIS")
    box(ax, 0.20, 0.20, 0.17, 0.095, "Loop Analyzer", "key · chords · density", accent=ACCENT)
    box(ax, 0.39, 0.20, 0.14, 0.095, "LoopContext", "models.py", face=SURFACE)

    # Generation
    label(ax, 0.60, 0.56, "GENERATION")
    box(ax, 0.60, 0.42, 0.10, 0.095, "Melody", "generator.py", accent=POSITIVE)
    box(ax, 0.71, 0.42, 0.10, 0.095, "Drums", "drums · groove", accent=POSITIVE)
    box(ax, 0.82, 0.42, 0.10, 0.095, "Bass", "bass.py", accent=POSITIVE)

    # Output
    label(ax, 0.60, 0.34, "OUTPUT")
    box(ax, 0.60, 0.20, 0.15, 0.095, "Schedulers", "scheduler.py", accent=WARM)
    box(ax, 0.77, 0.20, 0.12, 0.095, "MIDI Export", "export.py")

    # Flow arrows
    arrow(ax, 0.15, 0.48, 0.20, 0.47, color=WARM)          # Logic → MIDI in
    arrow(ax, 0.33, 0.47, 0.35, 0.47)                    # MIDI in → file
    arrow(ax, 0.46, 0.47, 0.48, 0.47)                    # file → quant
    arrow(ax, 0.54, 0.42, 0.54, 0.30, color=ACCENT)      # quant → analyzer
    arrow(ax, 0.37, 0.25, 0.39, 0.25)                    # analyzer → context
    arrow(ax, 0.53, 0.25, 0.60, 0.47, color=ACCENT, rad=-0.15)  # context → melody
    arrow(ax, 0.53, 0.25, 0.71, 0.47, color=ACCENT)      # context → drums
    arrow(ax, 0.53, 0.25, 0.82, 0.47, color=ACCENT, rad=0.15)   # context → bass
    arrow(ax, 0.65, 0.42, 0.65, 0.30, color=POSITIVE)  # melody → sched
    arrow(ax, 0.76, 0.42, 0.70, 0.30, color=POSITIVE)  # drums → sched
    arrow(ax, 0.87, 0.42, 0.75, 0.30, color=POSITIVE)  # bass → sched
    arrow(ax, 0.75, 0.20, 0.77, 0.20)                     # sched → export
    arrow(ax, 0.89, 0.20, 0.87, 0.42, color=WARM, rad=0.2)  # export → logic (arc)
    arrow(ax, 0.675, 0.25, 0.85, 0.40, color=WARM, lw=1.8)  # sched → logic

    # Interface → session
    arrow(ax, 0.25, 0.64, 0.30, 0.60, color=TEXT_MUTED)
    arrow(ax, 0.385, 0.64, 0.42, 0.60, color=TEXT_MUTED)
    arrow(ax, 0.545, 0.64, 0.545, 0.53, color=ACCENT)

    # Footer
    ax.text(0.05, 0.075,
            "Live path: IAC Driver MIDI buses connect Logic Pro ↔ The Logicians",
            fontsize=9.5, color=TEXT_MUTED, ha="left")
    ax.text(0.05, 0.04,
            "MelodyGenerator protocol allows future neural model swap without changing the MIDI pipeline",
            fontsize=9.5, color=TEXT_MUTED, ha="left", style="italic")

    # Legend (bottom right, clean row)
    items = [(WARM, "Logic Pro / I/O"), (ACCENT, "Orchestration"), (POSITIVE, "Generators"), (LIVE, "Capture")]
    lx, ly = 0.58, 0.075
    for color, name in items:
        ax.add_patch(plt.Rectangle((lx, ly), 0.014, 0.014, facecolor=color, transform=ax.transAxes, zorder=3))
        ax.text(lx + 0.02, ly + 0.007, name, fontsize=8.5, color=TEXT_MUTED, va="center", transform=ax.transAxes)
        lx += 0.105

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_diagram()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=INK, edgecolor="none")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=INK, edgecolor="none")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
