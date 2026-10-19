"""Supplementary Fig. S6: the effect does not depend on the feature extractor or classifier.

Three panels:
  a) Sensitivity curves, custom extractor vs. PyRadiomics, one subplot per
     target (recurrence, EGFR, survival) -- same 5-level x-axis as Fig. 3c-e.
  b) Sensitivity curves, logistic regression vs. Random Forest, one
     subplot per classification target (recurrence, EGFR) -- same
     custom-extractor features, only the classifier changes.
  c) Summary bar chart: perturbation range per target under each of the
     three conditions (original custom-extractor + logistic regression;
     PyRadiomics + logistic regression; custom-extractor + Random
     Forest), so all three robustness axes (extractor, classifier) are
     visible on one shared scale.

Run: python scripts/figures/supp_extractor_classifier.py
Output: outputs/figures/supplementary/_parts/supp_fig_s7_implementation_robustness.png, .pdf
"""

from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures" / "supplementary" / "_parts"  # full render; render_supplementary_figures.py trims it
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

BLUE, ORANGE, GREEN, AMBER, GRAY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#c9960c", "#9aa3ad", "#7a5cc0"
INK, SUBINK = "#14161c", "#5b6470"
CUSTOM_C, PYRAD_C, RF_C = GRAY, PURPLE, AMBER

MM_LEVELS = [-2, -1, 0, 1, 2]

TARGETS3 = [("recurrence", "classification", "Recurrence\n(AUC)"),
            ("egfr", "classification", "EGFR mutation\n(AUC)"),
            ("recurrence_survival", "survival", "OS\n(concordance)")]
TARGETS2 = [("recurrence", "Recurrence\n(AUC)"), ("egfr", "EGFR mutation\n(AUC)")]


def panel_a(fig, x0, w):
    combined = pd.read_csv(DATA_DIR / "pyradiomics_downstream_comparison.csv")
    n = len(TARGETS3)
    sub_w = w / n
    for i, (target, paradigm, label) in enumerate(TARGETS3):
        ax = fig.add_axes([x0 + i * sub_w + 0.014, 0.16, sub_w - 0.03, 0.60])
        for extractor, color, lbl in [("custom", CUSTOM_C, "Study extractor"), ("pyradiomics", PYRAD_C, "PyRadiomics")]:
            sub = combined[(combined["paradigm"] == paradigm) & (combined["target"] == target) & (combined["extractor"] == extractor)]
            sub = sub.set_index("perturbation_mm").reindex(MM_LEVELS)
            ax.plot(MM_LEVELS, sub["mean_score"], color=color, linewidth=2.0, marker="o", markersize=5,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=3, label=lbl if i == 0 else None)
        ax.axhline(0.5, color="#c7cdd4", linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
        ax.axvline(0, color="#e3e5e9", linewidth=0.8, zorder=0)
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(0.3, 0.9)
        ax.set_xticks(MM_LEVELS, ["−2", "−1", "0", "+1", "+2"])
        ax.set_xlabel("Boundary shift (mm)")
        if i == 0:
            ax.set_ylabel("Mean nested-CV score")
            ax.legend(fontsize=6.4, frameon=False, loc="lower left", handletextpad=0.4, labelspacing=0.25)
        ax.set_title(label, fontsize=7.6)


def panel_b(fig, x0, w):
    lr = pd.read_csv(DATA_DIR / "perturbation_auc_comparison.csv")
    rf = pd.read_csv(DATA_DIR / "rf_robustness_results.csv")
    n = len(TARGETS2)
    sub_w = w / n
    for i, (target, label) in enumerate(TARGETS2):
        ax = fig.add_axes([x0 + i * sub_w + 0.014, 0.16, sub_w - 0.03, 0.60])

        lr_sub = lr[lr["target"] == target].set_index("perturbation_mm").reindex([m for m in MM_LEVELS if m != 0])
        orig_path = DATA_DIR / f"modeling_results_{target}.csv"
        orig_auc = pd.read_csv(orig_path)["auc"].mean() if orig_path.exists() else np.nan
        lr_means = [lr_sub["mean_auc"].get(-2), lr_sub["mean_auc"].get(-1), orig_auc,
                    lr_sub["mean_auc"].get(1), lr_sub["mean_auc"].get(2)]

        rf_sub = rf[rf["target"] == target].set_index("perturbation_mm").reindex(MM_LEVELS)

        ax.plot(MM_LEVELS, lr_means, color=CUSTOM_C, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3, label="Logistic regression" if i == 0 else None)
        ax.plot(MM_LEVELS, rf_sub["mean_auc"], color=RF_C, linewidth=2.0, marker="^", markersize=5.5,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3, label="Random forest" if i == 0 else None)
        ax.axhline(0.5, color="#c7cdd4", linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
        ax.axvline(0, color="#e3e5e9", linewidth=0.8, zorder=0)
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(0.3, 0.9)
        ax.set_xticks(MM_LEVELS, ["−2", "−1", "0", "+1", "+2"])
        ax.set_xlabel("Boundary shift (mm)")
        if i == 0:
            ax.set_ylabel("Mean nested-CV AUC")
            ax.legend(fontsize=6.4, frameon=False, loc="lower left", handletextpad=0.4, labelspacing=0.25)
        ax.set_title(label, fontsize=7.6)


def panel_c(ax):
    pyrad_swing = pd.read_csv(DATA_DIR / "pyradiomics_downstream_swing_summary.csv")
    rf_swing = pd.read_csv(DATA_DIR / "rf_robustness_swing_summary.csv")

    labels = ["Recurrence", "EGFR", "OS"]
    keys = [("recurrence", "classification"), ("egfr", "classification"), ("recurrence_survival", "survival")]

    custom_vals, pyrad_vals, rf_vals = [], [], []
    for target, paradigm in keys:
        c = pyrad_swing[(pyrad_swing["target"] == target) & (pyrad_swing["paradigm"] == paradigm) & (pyrad_swing["extractor"] == "custom")]["swing"]
        p = pyrad_swing[(pyrad_swing["target"] == target) & (pyrad_swing["paradigm"] == paradigm) & (pyrad_swing["extractor"] == "pyradiomics")]["swing"]
        custom_vals.append(float(c.iloc[0]) if len(c) else np.nan)
        pyrad_vals.append(float(p.iloc[0]) if len(p) else np.nan)
        r = rf_swing[rf_swing["target"] == target]["swing"]
        rf_vals.append(float(r.iloc[0]) if len(r) else np.nan)

    x = np.arange(len(labels))
    width = 0.26
    ax.bar(x - width, custom_vals, width, color=CUSTOM_C, edgecolor="black", linewidth=0.7, label="Study extractor + logistic regression")
    ax.bar(x, pyrad_vals, width, color=PYRAD_C, edgecolor="black", linewidth=0.7, label="PyRadiomics + logistic regression")
    ax.bar(x + width, rf_vals, width, color=RF_C, edgecolor="black", linewidth=0.7, label="Study extractor + random forest")

    for xi, vals in zip(x, zip(custom_vals, pyrad_vals, rf_vals)):
        for off, v in zip((-width, 0, width), vals):
            if not np.isnan(v):
                ax.text(xi + off, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=6.4, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Range across segmentations\n(highest − lowest score, −2 to +2 mm)")
    ax.set_title("Similar ranges across\nextractors and classifiers")
    finite = [v for v in custom_vals + pyrad_vals + rf_vals if not np.isnan(v)]
    ax.set_ylim(0, max(finite) * 1.3)
    ax.legend(fontsize=6.0, frameon=False, loc="upper right", handletextpad=0.4, labelspacing=0.3)


def main():
    fig = plt.figure(figsize=(15.5, 4.6))
    fig.patch.set_facecolor("white")
    fig.text(0.006, 0.99,
              "Supplementary Fig. S7 | Similar swings with a second extractor and a second classifier",
              fontsize=12, fontweight="bold", color=INK, ha="left", va="top")

    panel_a(fig, 0.03, 0.335)
    fig.text(0.03 - 0.018, 0.90, "a", fontsize=14, fontweight="bold", color=INK)

    panel_b(fig, 0.375, 0.235)
    fig.text(0.375 - 0.018, 0.90, "b", fontsize=14, fontweight="bold", color=INK)

    ax_c = fig.add_axes([0.665, 0.16, 0.30, 0.60])
    panel_c(ax_c)
    fig.text(0.665 - 0.03, 0.90, "c", fontsize=14, fontweight="bold", color=INK)

    out_png = FIG_DIR / "supp_fig_s7_implementation_robustness.png"
    out_pdf = FIG_DIR / "supp_fig_s7_implementation_robustness.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        print(f"Saved {out_pdf}")
    except PermissionError as exc:
        print(f"Skipped PDF (locked by another process): {exc}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
