"""Supplementary Fig. S11: stability filtering.

Three panels:
  a) Perturbation range (max - min mean score across the five boundary
     positions) for the full 26-feature model versus the 13-feature
     stable and unstable halves, one group of bars per target.
  b) Small-multiple sensitivity curves (mean score vs. boundary shift)
     for the stable half (blue) and unstable half (orange), one subplot
     per target, so the shape of each condition's curve is visible, not
     only its range.
  c) The synthetic stable/unstable feature split plotted against real
     inter-observer instability rank (as in Fig. 4c), colored by which
     half a feature fell into, showing that the split also separates
     features that are unstable and stable between readers.

Run: python scripts/figures/supp_stability_filtering.py
Output: outputs/figures/supplementary/_parts/supp_fig_s5_stability_filtering.png, .pdf

Requires data/processed/stability_filtering_results.csv (from
scripts/mitigation/run_stability_filtering_experiment.py), plus the primary
perturbation and interobserver outputs already used for Figs 3 and 4.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import modeling, perturbation  # noqa: E402

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

BLUE, ORANGE, GREEN, AMBER, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c9960c", "#9aa3ad"
INK, SUBINK = "#14161c", "#5b6470"
STABLE_C, UNSTABLE_C, FULL_C = GREEN, ORANGE, GRAY  # as in Fig. 6d

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures" / "supplementary" / "_parts"  # full render; render_supplementary_figures.py trims it
FIG_DIR.mkdir(parents=True, exist_ok=True)

MM_LEVELS = [-2, -1, 0, 1, 2]

TARGETS = [
    ("recurrence", "classification", "Recurrence\n(AUC)"),
    ("egfr", "classification", "EGFR mutation\n(AUC)"),
    ("recurrence_survival", "survival", "OS\n(concordance)"),
]


def _full_model_series(target, paradigm):
    """Mean score at each mm level for the ORIGINAL 26/27-feature model,
    reusing the same saved per-fold result files as Fig. 3c-e."""
    if paradigm == "classification":
        base = f"modeling_results_{target}"
        col = "auc"
    else:
        base = "survival_results"
        col = "concordance"
    suffix_map = {-2: "perturb_m2mm", -1: "perturb_m1mm", 1: "perturb_p1mm", 2: "perturb_p2mm"}
    means = []
    for mm in MM_LEVELS:
        if mm == 0:
            fname = base
        elif paradigm == "classification":
            fname = f"{base}__{suffix_map[mm]}"
        else:
            fname = f"survival__{suffix_map[mm]}_results"
        vals = pd.read_csv(DATA_DIR / f"{fname}.csv")[col].to_numpy()
        means.append(vals.mean())
    means = np.array(means)
    return means, float(means.max() - means.min())


def _condition_series(sf, target, paradigm, condition):
    sub = sf[(sf["target"] == target) & (sf["paradigm"] == paradigm) & (sf["condition"] == condition)]
    sub = sub.set_index("perturbation_mm").reindex(MM_LEVELS)
    return sub["mean_score"].to_numpy(), sub["std_score"].to_numpy()


def panel_a(ax, sf, sig):
    x = np.arange(len(TARGETS))
    width = 0.26
    conditions = [("full", FULL_C, "Full model (26 features)"), ("stable", STABLE_C, "Stable half (13 features)"),
                  ("unstable", UNSTABLE_C, "Unstable half (13 features)")]

    swing_vals = {c: [] for c, _, _ in conditions}
    swing_lo = {c: [] for c, _, _ in conditions}
    swing_hi = {c: [] for c, _, _ in conditions}
    for target, paradigm, _label in TARGETS:
        for cond, _color, _lbl in conditions:
            row = sig[(sig["target"] == target) & (sig["paradigm"] == paradigm) & (sig["condition"] == cond)].iloc[0]
            swing_vals[cond].append(row["observed_swing"])
            swing_lo[cond].append(row["observed_swing"] - row["swing_ci_lo"])
            swing_hi[cond].append(row["swing_ci_hi"] - row["observed_swing"])

    offsets = (-width, 0, width)
    for (cond, color, label), off in zip(conditions, offsets):
        vals = swing_vals[cond]
        err = [swing_lo[cond], swing_hi[cond]]
        ax.bar(x + off, vals, width, color=color, edgecolor="black", linewidth=0.7, label=label,
               yerr=err, capsize=2.5, error_kw={"linewidth": 1.0, "ecolor": INK})

    ymax = max(swing_hi[c][i] + swing_vals[c][i] for c in swing_vals for i in range(len(TARGETS)))
    for xi, target_row in enumerate(TARGETS):
        target, paradigm, _label = target_row
        p = sig[(sig["target"] == target) & (sig["paradigm"] == paradigm) &
                (sig["condition"] == "stable_minus_unstable_swing_diff")]["p_value"].iloc[0]
        ax.text(xi, ymax * 1.14, f"Stable vs unstable:\nP = {p:.2f}", ha="center", va="bottom", fontsize=6.0, color=SUBINK)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _t, _p, lbl in TARGETS])
    ax.set_ylabel("Range across segmentations\n(highest − lowest score, −2 to +2 mm)")
    ax.set_title("Stable and unstable halves do not\ndiffer significantly in range")
    ax.set_ylim(0, ymax * 1.55)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left", handletextpad=0.4, labelspacing=0.3)


def panel_b(fig, x0, w, sf):
    n = len(TARGETS)
    sub_w = w / n
    for i, (target, paradigm, label) in enumerate(TARGETS):
        ax = fig.add_axes([x0 + i * sub_w + 0.012, 0.16, sub_w - 0.028, 0.60])
        stable_means, stable_sd = _condition_series(sf, target, paradigm, "stable")
        unstable_means, unstable_sd = _condition_series(sf, target, paradigm, "unstable")
        full_means, _full_swing = _full_model_series(target, paradigm)

        ax.plot(MM_LEVELS, full_means, color=FULL_C, linewidth=1.4, marker="s", markersize=4,
                markeredgecolor="black", markeredgewidth=0.5, linestyle=(0, (2, 1.5)), zorder=2,
                label="Full" if i == 0 else None)
        ax.plot(MM_LEVELS, stable_means, color=STABLE_C, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3, label="Stable half" if i == 0 else None)
        ax.fill_between(MM_LEVELS, stable_means - stable_sd, stable_means + stable_sd, color=STABLE_C, alpha=0.12, zorder=1)
        ax.plot(MM_LEVELS, unstable_means, color=UNSTABLE_C, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3, label="Unstable half" if i == 0 else None)
        ax.fill_between(MM_LEVELS, unstable_means - unstable_sd, unstable_means + unstable_sd, color=UNSTABLE_C, alpha=0.12, zorder=1)

        ax.axhline(0.5, color="#c7cdd4", linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
        ax.axvline(0, color="#e3e5e9", linewidth=0.8, zorder=0)
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(0.3, 0.9)
        ax.set_xticks(MM_LEVELS, ["−2", "−1", "0", "+1", "+2"])
        ax.set_xlabel("Boundary shift (mm)")
        if i == 0:
            ax.set_ylabel("Mean nested-CV score")
            ax.legend(fontsize=6.2, frameon=False, loc="lower left", handletextpad=0.4, labelspacing=0.25)
        ax.set_title(label, fontsize=7.6)


def panel_c(ax):
    pert = pd.read_csv(DATA_DIR / "perturbation_features.csv")
    stability = perturbation.feature_stability_summary(pert, -1, 1)
    stability = stability[~stability["feature"].isin(modeling.NON_FEATURE_COLS)].reset_index(drop=True)
    n = len(stability)
    half = n // 2
    unstable_feats = set(stability.iloc[:half]["feature"])
    stable_feats = set(stability.iloc[half:]["feature"])

    iob_path = DATA_DIR / "interobserver_feature_stability.csv"
    iob = pd.read_csv(iob_path).set_index("feature")["real_interobserver_median_rel_change"]
    common = [f for f in stability["feature"] if f in iob.index]
    n_feat = len(common)

    synth_val = stability.set_index("feature").loc[common, "median_abs_rel_change"]
    real_val = iob.loc[common]
    synth_rank = synth_val.rank(ascending=False)
    real_rank = real_val.rank(ascending=False)

    for feats, color, label in [(stable_feats, STABLE_C, "Stable half"), (unstable_feats, UNSTABLE_C, "Unstable half")]:
        sel = [f for f in common if f in feats]
        ax.scatter(synth_rank[sel], real_rank[sel], s=34, color=color, alpha=0.9,
                   edgecolors="white", linewidths=0.5, zorder=3, label=label)

    stable_median = real_rank[[f for f in common if f in stable_feats]].median()
    unstable_median = real_rank[[f for f in common if f in unstable_feats]].median()
    ax.axhline(stable_median, color=STABLE_C, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1, alpha=0.6)
    ax.axhline(unstable_median, color=UNSTABLE_C, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1, alpha=0.6)

    lims = [0, n_feat + 1]
    ax.plot(lims, lims, color="#c7cdd4", linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Instability rank, ±1-mm adjustment\n(1 = least stable)")
    ax.set_ylabel("Instability rank between radiation\noncologists (1 = least stable)")
    ax.set_title("The split agrees with\ndisagreement between clinicians")
    ax.text(0.03, 0.97, f"Median rank between clinicians:\nstable half {stable_median:.0f}, unstable half {unstable_median:.0f} (of {n_feat})",
            transform=ax.transAxes, fontsize=6.4, color=INK, va="top", ha="left")
    ax.legend(fontsize=6.2, frameon=False, loc="lower right", handletextpad=0.4, labelspacing=0.25)


def main():
    sf_path = DATA_DIR / "stability_filtering_results.csv"
    if not sf_path.exists():
        raise SystemExit(
            "Missing data/processed/stability_filtering_results.csv -- run first:\n"
            "  python scripts/mitigation/run_stability_filtering_experiment.py"
        )
    sf = pd.read_csv(sf_path)
    sig_path = DATA_DIR / "swing_significance.csv"
    if not sig_path.exists():
        raise SystemExit(
            "Missing data/processed/swing_significance.csv -- run first:\n"
            "  python scripts/discovery/run_range_inference.py"
        )
    sig = pd.read_csv(sig_path)

    fig = plt.figure(figsize=(15.5, 4.6))
    fig.patch.set_facecolor("white")
    fig.text(0.006, 0.99,
              "Supplementary Fig. S5 | Stability filtering does not significantly reduce the swing",
              fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")

    ax_a = fig.add_axes([0.055, 0.16, 0.225, 0.60])
    panel_a(ax_a, sf, sig)
    fig.text(0.055 - 0.03, 0.90, "a", fontsize=14, fontweight="bold", color=INK)

    panel_b(fig, 0.335, 0.335, sf)
    fig.text(0.335 - 0.028, 0.90, "b", fontsize=14, fontweight="bold", color=INK)

    ax_c = fig.add_axes([0.735, 0.16, 0.235, 0.60])
    panel_c(ax_c)
    fig.text(0.735 - 0.03, 0.90, "c", fontsize=14, fontweight="bold", color=INK)

    out_png = FIG_DIR / "supp_fig_s5_stability_filtering.png"
    out_pdf = FIG_DIR / "supp_fig_s5_stability_filtering.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        print(f"Saved {out_pdf}")
    except PermissionError as exc:
        print(f"Skipped PDF (locked by another process): {exc}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
