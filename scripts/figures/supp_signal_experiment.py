"""Supplementary figure: effect of the segmentation as a function of model signal
(controlled-signal experiment, scripts/discovery/run_signal_simulation.py).

Paper:    Suppl. Figs S1-S13
Pipeline: python pipeline.py run supplementary_figures   (stage 'figures')

a  Patients reclassified by model variability and by a ±1-mm segmentation, against AUC
b  Misclassification of segmentation-dependent and segmentation-stable classifications, against AUC
Output: outputs/figures/supplementary/signal.png
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.figures.figure_style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, letter  # noqa: E402

runs = pd.read_csv(ROOT / "data" / "processed" / "signal_simulation_runs.csv")
syn = runs[runs.outcome == "synthetic"]
lev = syn.groupby("sigma", sort=False).mean(numeric_only=True).sort_values("auc")
real = runs[runs.outcome != "synthetic"]
# sensitivity analysis: outcome generated from the mean of the -1-mm, expert and +1-mm features
avg = pd.read_csv(ROOT / "data" / "processed" / "signal_simulation_runs_avg.csv")
lev_avg = avg[avg.outcome == "synthetic"].groupby("sigma", sort=False).mean(numeric_only=True).sort_values("auc")
from matplotlib.lines import Line2D  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32, "left": 0.08, "right": 0.98, "bottom": 0.17, "top": 0.88})
ax = axes[0]
for col, c, lab in (("outline_reclassification", ORANGE, "±1-mm adjustment (same model)"), ("model_variability", MUTED, "Model-development variability (same segmentation)")):
    ax.scatter(syn.auc, 100 * syn[col], s=9, color=c, alpha=0.35, lw=0)
    ax.plot(lev.auc, 100 * lev[col], "-o", color=c, ms=3.5, lw=1.4, label=lab)
    ax.plot(lev_avg.auc, 100 * lev_avg[col], "--", color=c, lw=1.0)
for _, r in real.iterrows():
    c = BLUE if r.outcome.startswith("recurrence") else PURPLE
    for col, m in (("outline_reclassification", "D"), ("model_variability", "s")):
        ax.plot(r.auc, 100 * r[col], m, color=c, mfc="white", mew=1.1, ms=5)
    ax.annotate(r.outcome.split(" ")[0].capitalize() if r.outcome.startswith("rec") else "EGFR", (r.auc, 100 * r.outline_reclassification),
                xytext=(4, 6), textcoords="offset points", fontsize=6, color=c)
ax.set_xlabel("Discrimination of the model (AUC)")
ax.set_ylabel("Patients reclassified (%)")
ax.set_ylim(0, 40)
ax.set_title("Reclassification by segmentation and by model development", color=INK)
h, lab_ = ax.get_legend_handles_labels()
h.append(Line2D([], [], color=MUTED, ls="--", lw=1.0))
lab_.append("Outcome from averaged segmentations")
ax.add_artist(ax.legend(h, lab_, frameon=False, loc="lower left"))
ax.legend([Line2D([], [], marker="D", ls="", color=INK, mfc="white", mew=1.1, ms=4),
           Line2D([], [], marker="s", ls="", color=INK, mfc="white", mew=1.1, ms=4)],
          ["Real outcome, ±1-mm adjustment", "Real outcome, model development"],
          frameon=False, loc="upper right", borderaxespad=0.1, handletextpad=0.3, labelspacing=0.3)
letter(fig, ax, "a")

ax = axes[1]
for col, c, lab in (("error_if_outline_dependent", ORANGE, "Segmentation-dependent classification"), ("error_if_outline_stable", GREEN, "Segmentation-stable classification")):
    ax.scatter(syn.auc, 100 * syn[col], s=9, color=c, alpha=0.35, lw=0)
    ax.plot(lev.auc, 100 * lev[col], "-o", color=c, ms=3.5, lw=1.4, label=lab)
    ax.plot(lev_avg.auc, 100 * lev_avg[col], "--", color=c, lw=1.0)
ax.set_xlabel("Discrimination of the model (AUC)")
ax.set_ylabel("Discordant with simulated outcome (%)")
ax.set_ylim(0, 65)
ax.set_title("Discordance with the simulated outcome", color=INK)
ax.legend(frameon=False, loc="lower left")
letter(fig, ax, "b")

out = ROOT / "outputs" / "figures" / "supplementary" / "signal.png"
fig.savefig(out, dpi=300)
print("Saved", out)
