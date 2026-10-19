"""Supplementary figure: segmentation sensitivity versus model confidence (LIDC-IDRI).

a  Selective prediction: disagreement with the consensus label among the
   predictions retained when the least trustworthy are withheld by low
   confidence, high segmentation sensitivity or both (95% patient-bootstrap CIs).
b  Disagreement with the consensus label by confidence quintile, for
   segmentations flagged or not flagged by the ±1-mm check.

Paper:    Suppl. Figs S1-S13
Pipeline: python pipeline.py run supplementary_figures   (stage 'figures')
Needs:    data/external/lidc/lidc_risk_coverage.csv, lidc_confidence_check.json
          (scripts/external/lidc_confidence_check.py)
Output:   outputs/figures/supplementary/confidence.png
Run:      python scripts/figures/supp_confidence.py
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.figures.figure_style import GREEN, INK, MUTED, ORANGE, PURPLE, letter  # noqa: E402

D = ROOT / "data" / "external" / "lidc"
rc = pd.read_csv(D / "lidc_risk_coverage.csv")
strata = pd.DataFrame(json.loads((D / "lidc_confidence_check.json").read_text())["confidence_strata"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.3, "left": 0.08, "right": 0.98, "bottom": 0.17, "top": 0.88})
ax = axes[0]
for name, c, lab in (("confidence", PURPLE, "Low model confidence withheld first"),
                     ("sensitivity", ORANGE, "High segmentation sensitivity withheld first"),
                     ("combined", GREEN, "Both (sum of ranks)")):
    g = rc[rc.strategy == name].sort_values("coverage")
    x = 100 * (1 - g.coverage)
    ax.fill_between(x, 100 * g.ci_low, 100 * g.ci_high, color=c, alpha=0.12, lw=0)
    ax.plot(x, 100 * g.error_retained, "-o", color=c, ms=3.5, lw=1.4, label=lab)
ax.set_xlabel("Predictions withheld (%)")
ax.set_ylabel("Disagreement with consensus label\namong retained predictions (%)")
ax.set_title("Selective prediction", color=INK)
ax.set_ylim(0, None)
ax.legend(frameon=False, loc="upper right")
letter(fig, ax, "a")

ax = axes[1]
x = np.arange(len(strata))
w = 0.38
ax.bar(x - w / 2, 100 * strata.error_if_flagged.fillna(0), w, color=ORANGE, label="Flagged by ±1-mm check")
ax.bar(x + w / 2, 100 * strata.error_if_not_flagged.fillna(0), w, color=MUTED, label="Not flagged")
for xi, v in zip(x, strata.error_if_flagged):
    if v == 0:  # a true zero, not a missing bar
        ax.text(xi - w / 2, 0.4, "0", ha="center", va="bottom", fontsize=6, color=ORANGE)
ax.set_xticks(x, [f"Q{q}" for q in strata.confidence_quintile])
ax.set_xlabel("Model confidence quintile (Q1 = closest to the threshold)")
ax.set_ylabel("Disagreement with consensus label (%)")
ax.set_title("Within strata of model confidence", color=INK)
ax.legend(frameon=False, loc="upper right")
letter(fig, ax, "b")

out = ROOT / "outputs" / "figures" / "supplementary" / "confidence.png"
fig.savefig(out, dpi=300, facecolor="white")
print("Saved", out)
