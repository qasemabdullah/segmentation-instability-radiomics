"""Benjamini-Hochberg correction across the eleven pre-specified tests.

Paper:    Methods, Suppl. Table S27
Pipeline: python pipeline.py run multiple_testing   (stage 'reporting')

The eleven tests are computed from the stored results exactly as in the figures and tables
that report them, and corrected together as one family:
  - Spearman correlation of feature instability between readers and under the ±1-mm adjustment (Fig. 4c)
  - log-rank test of overall survival above and below the median out-of-fold risk (Fig. 2b)
  - Mann-Whitney tests of mean attenuation and of tumor volume by recurrence (Suppl. Fig. S4)
  - chi-square test of radiomic phenotype against recurrence (Suppl. Fig. S7)
  - Spearman correlations of SHAP importance against feature instability, for recurrence, EGFR and KRAS (Fig. 6c)
  - bootstrap tests of the stable-minus-unstable range of performance, for recurrence, EGFR and OS (Fig. 6d)

Output: data/processed/multiple_testing.csv
Run:    python scripts/reporting/multiple_testing.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
P = ROOT / "data" / "processed"


def benjamini_hochberg(p):
    """q values of the Benjamini-Hochberg procedure for an array of P values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    q_sorted = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1)
    return q


def main():
    from lifelines.statistics import logrank_test

    from nsclc_pipeline import interpretation
    from scripts.figures.supp_phenotypes import run_clustering

    tests = {}

    # Fig. 4c: instability between readers against instability under the ±1-mm adjustment
    comp = pd.read_csv(P / "interobserver_vs_synthetic_comparison.csv").set_index("feature")
    tests["Spearman, instability rank between readers vs under ±1-mm adjustment (Fig. 4c)"] = spearmanr(
        comp["real_interobserver_median_rel_change"], comp["synthetic_perturbation_median_rel_change"]).pvalue

    # Fig. 2b: overall survival above and below the median out-of-fold risk
    d = pd.read_csv(P / "survival_predictions.csv")
    hi = d.risk_score >= d.risk_score.median()
    tests["Log-rank, OS by median risk (Fig. 2b)"] = logrank_test(
        d.duration_days[hi], d.duration_days[~hi], d.event[hi], d.event[~hi]).p_value

    # Suppl. Fig. S4: attenuation and volume by recurrence
    rad = pd.read_csv(P / "radiomics_features.csv").merge(
        pd.read_csv(P / "clinical.csv")[["case_id", "recurrence"]], on="case_id", how="left")
    rad = rad[rad.recurrence.isin(["yes", "no"])]
    for col, name in (("firstorder_mean_hu", "mean attenuation"), ("shape_volume_mm3", "tumor volume")):
        tests[f"Mann-Whitney, {name} by recurrence"] = mannwhitneyu(
            rad.loc[rad.recurrence == "no", col], rad.loc[rad.recurrence == "yes", col]).pvalue

    # Suppl. Fig. S7: phenotype against recurrence
    tests["χ², phenotype vs recurrence"] = run_clustering()[6]

    # Fig. 6c: SHAP importance against instability (ranks)
    inst = comp["synthetic_perturbation_median_rel_change"].rank(ascending=False)
    for target, name in (("egfr", "EGFR"), ("kras", "KRAS"), ("recurrence", "recurrence")):
        sv, _ = interpretation.compute_shap(target, "radiomics")
        imp = pd.Series(np.abs(np.asarray(sv.values)).mean(0), index=sv.feature_names).rank(ascending=False)
        common = imp.index.intersection(inst.index)
        tests[f"Spearman, SHAP importance vs stability, {name}"] = spearmanr(imp[common], inst[common]).pvalue

    # Fig. 6d: stable-minus-unstable range of performance
    sw = pd.read_csv(P / "swing_significance.csv")
    sw = sw[sw.condition == "stable_minus_unstable_swing_diff"].set_index("target").p_value
    for target, name in (("egfr", "EGFR"), ("recurrence", "recurrence"), ("recurrence_survival", "OS")):
        tests[f"Bootstrap, stable-minus-unstable range, {name}"] = sw[target]

    out = pd.DataFrame({"test": list(tests), "p": list(tests.values())})
    out["q"] = benjamini_hochberg(out.p)
    out["significant_q_below_0.05"] = out.q < 0.05
    out = out.sort_values("p").reset_index(drop=True)
    out.to_csv(P / "multiple_testing.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
