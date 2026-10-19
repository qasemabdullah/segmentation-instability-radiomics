"""Calibration of the classification models with significance tests and resampling.

For recurrence and EGFR at every boundary position, from the pooled
out-of-fold predictions of the primary split (no refitting):
  calibration-in-the-large  observed rate minus mean predicted probability
  calibration slope         logistic regression of the outcome on the logit
  Brier score
  Spiegelhalter z test      H0: predictions are calibrated (two-sided P)
  95% intervals             2,000 fold-stratified bootstrap resamples

Paper:    Suppl. Table S24
Pipeline: python pipeline.py run calibration_inference   (stage 'discovery')
Needs:    data/processed/modeling_predictions_{recurrence,egfr}[__perturb_*].csv
Output:   data/processed/calibration_inference.csv
Run:      python scripts/discovery/run_calibration_inference.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
POSITIONS = [(-2, "__perturb_m2mm"), (-1, "__perturb_m1mm"), (0, ""), (1, "__perturb_p1mm"), (2, "__perturb_p2mm")]
N_BOOT = 2000


def stats(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6).fit(lp, y).coef_[0][0] if 0 < y.sum() < len(y) else np.nan
    return {"citl": y.mean() - p.mean(), "slope": slope, "brier": np.mean((p - y) ** 2)}


def spiegelhalter_p(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.sum((y - p) * (1 - 2 * p)) / np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    return 2 * norm.sf(abs(z))


def main():
    rng = np.random.default_rng(20260930)
    rows = []
    for tgt in ("recurrence", "egfr"):
        for mm, sfx in POSITIONS:
            d = pd.read_csv(P / f"modeling_predictions_{tgt}{sfx}.csv")
            y, p = d.y_true.to_numpy(), d.y_proba.to_numpy()
            point = stats(y, p)
            idx = [np.flatnonzero(d.fold.to_numpy() == f) for f in sorted(d.fold.unique())]
            boot = []
            for _ in range(N_BOOT):
                b = np.concatenate([rng.choice(i, len(i), replace=True) for i in idx])
                boot.append(stats(y[b], p[b]))
            boot = pd.DataFrame(boot)
            row = {"endpoint": tgt, "mm": mm, "n": len(y), "observed_rate": y.mean(), "mean_predicted": p.mean(),
                   "spiegelhalter_p": spiegelhalter_p(y, p)}
            for k, v in point.items():
                row[k] = v
                row[f"{k}_lo"], row[f"{k}_hi"] = np.nanpercentile(boot[k], [2.5, 97.5])
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(P / "calibration_inference.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
