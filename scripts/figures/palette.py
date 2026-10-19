"""Colour palette and matplotlib style shared by the desktop GUI and two Supplementary Figures
(patient vignettes, CT habitats).

Importing this module has no side effects; call apply_style() to set the style. Note that
render_supplementary_figures.py draws several figures in one process, so the style set by the
vignette figure also applies to the figures drawn after it, as when the paper's figures were made.
"""
from __future__ import annotations

import matplotlib

# Publication-style settings: Arial/Helvetica, compact type, thin lines, no top/right spines
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",  # keep exported SVG/PDF text editable, not outlined
    "pdf.fonttype": 42,
}


def apply_style():
    """Set the shared matplotlib style (the same settings the desktop GUI uses)."""
    matplotlib.rcParams.update(STYLE)


INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# Colour-blind-safe categorical order: blue, orange, aqua, amber, magenta
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def style_axes(ax):
    """Light axes: no top/right spines, grey left/bottom spines, small grey tick labels."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_axisbelow(True)
