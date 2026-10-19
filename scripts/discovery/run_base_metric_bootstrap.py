"""Bootstrap 95% CIs of the baseline nested cross-validated scores.

Paper:    Suppl. Table S2
Pipeline: python pipeline.py run bootstrap_ci   (stage 'discovery')

Run: python scripts/discovery/run_base_metric_bootstrap.py
Output: data/processed/base_metric_bootstrap.csv
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

N_BOOT = 2000
SEED = 0

CLASSIFICATION_COMBOS = [
    ("recurrence", "radiomics", "modeling_predictions_recurrence"),
    ("recurrence", "+clinical", "modeling_predictions_recurrence__radiomics_clinical"),
    ("recurrence", "+clinical+rnaseq", "modeling_predictions_recurrence__radiomics_clinical_rnaseq"),
    ("recurrence", "embeddings", "modeling_predictions_recurrence__embeddings"),
    ("recurrence", "+embeddings", "modeling_predictions_recurrence__radiomics_embeddings"),
    ("egfr", "radiomics", "modeling_predictions_egfr"),
    ("egfr", "+clinical", "modeling_predictions_egfr__radiomics_clinical"),
    ("egfr", "+clinical+rnaseq", "modeling_predictions_egfr__radiomics_clinical_rnaseq"),
    ("egfr", "embeddings", "modeling_predictions_egfr__embeddings"),
    ("egfr", "+embeddings", "modeling_predictions_egfr__radiomics_embeddings"),
    ("kras", "radiomics", "modeling_predictions_kras"),
]


def _auc_metric(sub):
    if sub["y_true"].nunique() < 2:
        return np.nan
    return roc_auc_score(sub["y_true"], sub["y_proba"])


def _concordance_metric(sub):
    try:
        return concordance_index_censored(sub["event"].astype(bool), sub["duration_days"], sub["risk_score"])[0]
    except (ValueError, ZeroDivisionError):
        return np.nan


def _bootstrap_metric(df, metric_fn, rng, n_boot=N_BOOT):
    fold_indices = {f: g.index.to_numpy() for f, g in df.groupby("fold")}
    boot_scores = np.empty(n_boot)
    for b in range(n_boot):
        fold_scores = []
        for _f, idx in fold_indices.items():
            resample = rng.choice(idx, size=len(idx), replace=True)
            sub = df.loc[resample]
            s = metric_fn(sub)
            if not np.isnan(s):
                fold_scores.append(s)
        boot_scores[b] = np.mean(fold_scores) if fold_scores else np.nan
    return boot_scores


def main():
    rows = []
    rng = np.random.RandomState(SEED)

    for target, feature_set, fname in CLASSIFICATION_COMBOS:
        path = DATA_DIR / f"{fname}.csv"
        if not path.exists():
            print(f"Missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        observed = np.mean([_auc_metric(g) for _f, g in df.groupby("fold")])
        boot = _bootstrap_metric(df, _auc_metric, rng)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append({
            "paradigm": "classification", "target": target, "feature_set": feature_set,
            "metric": "AUC", "observed": observed, "ci_lo": lo, "ci_hi": hi,
        })
        print(f"{target} | {feature_set}: AUC={observed:.3f} (95% CI {lo:.3f}-{hi:.3f})")

    surv_path = DATA_DIR / "survival_predictions.csv"
    if surv_path.exists():
        df = pd.read_csv(surv_path)
        observed = np.mean([_concordance_metric(g) for _f, g in df.groupby("fold")])
        boot = _bootstrap_metric(df, _concordance_metric, rng)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append({
            "paradigm": "survival", "target": "recurrence_survival", "feature_set": "radiomics",
            "metric": "concordance", "observed": observed, "ci_lo": lo, "ci_hi": hi,
        })
        print(f"survival | radiomics: concordance={observed:.3f} (95% CI {lo:.3f}-{hi:.3f})")

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / "base_metric_bootstrap.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
