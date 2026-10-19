"""Shared visual style for every figure in the paper.

One journal style for main and Supplementary Figures: 7.2-inch double-column
width, 7-pt Arial, bold lower-case panel letters, no in-image titles (legends
live in the manuscript), and one colour per endpoint and feature family.

Import what you need, e.g.:
    from scripts.figures.figure_style import BLUE, INK, letter, p_fmt, save
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "outputs" / "figures" / "main"
SUPP_FIG_DIR = ROOT / "outputs" / "figures" / "supplementary"
# outputs/ is not version-controlled; create the output folders on a fresh clone
for _d in (FIG_DIR, SUPP_FIG_DIR / "_parts"):
    _d.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
})

INK, SUB, MUTED, GRID = "#15181e", "#4a5361", "#8a929e", "#e3e7ed"
BLUE, ORANGE, GREEN, AMBER, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
PURPLE, TEAL, NAVY, RED = "#7a5cc0", "#0e9aa0", "#10306b", "#c8392b"
ENDPOINT = {"recurrence": BLUE, "egfr": PURPLE, "kras": PINK, "os": GREEN, "rfs": TEAL, "cluster": AMBER}
FAMILY = {"shape": NAVY, "firstorder": ORANGE, "texture": GREEN}
MM = [-2, -1, 0, 1, 2]
MM_SFX = {-2: "perturb_m2mm", -1: "perturb_m1mm", 1: "perturb_p1mm", 2: "perturb_p2mm"}
DAYS_PER_MONTH = 30.4375

NICE = {
    "shape_volume_mm3": "Volume", "shape_surface_area_mm2": "Surface area", "shape_sphericity": "Sphericity",
    "shape_elongation": "Elongation", "shape_flatness": "Flatness", "shape_roundness": "Roundness",
    "shape_equivalent_spherical_radius_mm": "Equiv. spherical radius", "shape_max_diameter_mm": "Max. 3D diameter",
    "firstorder_voxel_count": "Voxel count", "firstorder_mean_hu": "Mean HU", "firstorder_std_hu": "HU s.d.",
    "firstorder_min_hu": "Min. HU", "firstorder_max_hu": "Max. HU", "firstorder_median_hu": "Median HU",
    "firstorder_range_hu": "HU range", "firstorder_iqr_hu": "HU IQR", "firstorder_mad_hu": "HU MAD",
    "firstorder_skewness": "Skewness", "firstorder_kurtosis": "Kurtosis", "firstorder_energy": "Energy",
    "firstorder_entropy": "Entropy", "texture_contrast": "GLCM contrast", "texture_correlation": "GLCM correlation",
    "texture_energy": "GLCM energy", "texture_homogeneity": "GLCM homogeneity",
    "texture_dissimilarity": "GLCM dissimilarity",
}


def nice(f):
    return NICE.get(f, f)


def family(f):
    return f.split("_")[0]


def letter(fig, ax, s, dx=-0.045, dy=0.012):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, s, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, stem):
    for ext in ("png", "pdf"):
        try:
            fig.savefig(FIG_DIR / f"{stem}.{ext}", facecolor="white", bbox_inches="tight", pad_inches=0.04)
        except PermissionError as exc:
            print(f"  skipped {stem}.{ext} (file open elsewhere): {exc}")
    plt.close(fig)
    print(f"Saved outputs/figures/main/{stem}.png")


def p_fmt(p):
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}"
