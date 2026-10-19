"""Discovery cohort: models on segmentations adjusted by −2 to +2 mm, and the
agreement between the controlled adjustment and real reader disagreement.

1. Classification (recurrence, EGFR) and overall-survival models re-run under
   nested cross-validation on the features of every adjusted segmentation,
   with the same folds as the expert segmentation (paired comparison).
2. Feature instability under the ±1-mm adjustment compared with instability
   between the five radiation oncologists (Spearman rank correlation).

Paper:    Fig. 3c-e, Table 3, Fig. 4c, Suppl. Fig. S11c, Suppl. Table S21
Pipeline: python pipeline.py run adjusted_models   (stage 'discovery'); python pipeline.py run adjustment_vs_readers   (stage 'readers')
Needs:   data/processed/perturbation_features.csv (scripts/data/adjusted_segmentation_features.py)
         data/processed/interobserver_feature_stability.csv (scripts/readers/analyze_interobserver.py)
Output:  data/processed/modeling_results_*__perturb_*.csv, survival__perturb_*_results.csv,
         interobserver_vs_synthetic_comparison.csv
Run:     python scripts/discovery/run_adjusted_segmentation_models.py [models] [comparison]
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, perturbation  # noqa: E402

SHIFTS_MM = (-2, -1, 1, 2)


def models():
    perturbation.run_auc_comparison(targets=("recurrence", "egfr"), mm_levels=SHIFTS_MM)
    perturbation.run_survival_perturbation_comparison(mm_levels=SHIFTS_MM)


def comparison():
    d = config.PROCESSED_DIR
    real = pd.read_csv(d / "interobserver_feature_stability.csv", index_col=0).iloc[:, 0]
    real.name = "real_interobserver_median_rel_change"
    adjusted = (perturbation.feature_stability_summary(pd.read_csv(d / "perturbation_features.csv"), -1, 1)
                .set_index("feature")["median_abs_rel_change"].rename("synthetic_perturbation_median_rel_change"))
    comp = pd.concat([real, adjusted], axis=1).dropna().drop(index="texture_n_slices", errors="ignore")
    comp.to_csv(d / "interobserver_vs_synthetic_comparison.csv")
    rho, p = spearmanr(comp.iloc[:, 0], comp.iloc[:, 1])
    print(f"Instability under ±1-mm adjustment vs between readers: rho = {rho:.3f}, P = {p:.2g}, n = {len(comp)}")


STEPS = {"models": models, "comparison": comparison}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(STEPS):
        STEPS[name]()
