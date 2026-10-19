"""Range of performance across segmentations: bootstrap CIs and the stable-vs-unstable test.

Paper:    Table 3, Fig. 6d, Suppl. Fig. S11, Suppl. Table S21
Pipeline: python pipeline.py run range_inference   (stage 'ai')

Every number reported for those two analyses so far has been a point
estimate (mean nested-CV score at each of 5 perturbation levels) with only
a 5-fold s.d. band -- it says nothing about whether the range between the
highest- and lowest-scoring level, or the difference between the stable-
and unstable-feature halves' ranges, is distinguishable from fold-to-fold
sampling noise. This script answers that directly with a fold-stratified
bootstrap, exploiting a fact already true of this pipeline (Methods): the
outer 5-fold assignment is identical, patient-for-patient, across every
perturbation level and every stability-filtering condition for a given
target (verified empirically here, not assumed). That means the same
per-fold bootstrap resample can be reused across levels and conditions,
giving a *paired* bootstrap rather than an unpaired one -- the correct
choice whenever the same patients are compared under different conditions.

Procedure, per target:
  1. For each of the 5 outer folds, resample that fold's patients with
     replacement (2,000 replicates), reusing the exact same resampled
     patient indices for every perturbation level and every condition
     (full / stable / unstable).
  2. Within each replicate, recompute the per-fold AUC (or concordance)
     on the resampled patients for each fold, average the 5 folds -- the
     same statistic (mean across outer folds) already reported everywhere
     else in this paper -- to get one bootstrap score per level.
  3. swing_b = max - min across the 5 levels, for that replicate. The
     resulting distribution over 2,000 replicates gives a 95% CI on the
     range itself.
  4. For the stability-filtering conditions, additionally compute
     diff_b = swing_stable_b - swing_unstable_b per replicate, and report
     its 95% CI and a two-sided bootstrap p-value (the fraction of
     replicates on the less-common side of zero, doubled).

Run: python scripts/discovery/run_range_inference.py
Output: data/processed/swing_significance.csv
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "processed"
MM_LEVELS = [-2, -1, 0, 1, 2]
MM_TAG = {-2: "m2mm", -1: "m1mm", 0: "p0mm", 1: "p1mm", 2: "p2mm"}
N_BOOT = 2000
SEED = 0


def _auc_metric(sub):
    if sub["y_true"].nunique() < 2:
        return np.nan
    return roc_auc_score(sub["y_true"], sub["y_proba"])


def _concordance_metric(sub):
    try:
        c = concordance_index_censored(sub["event"].astype(bool), sub["duration_days"], sub["risk_score"])[0]
        return c
    except (ValueError, ZeroDivisionError):
        return np.nan


def _load_classification_condition(target, condition):
    paths = {}
    for mm in MM_LEVELS:
        if condition == "full":
            fname = f"modeling_predictions_{target}" if mm == 0 else f"modeling_predictions_{target}__perturb_{MM_TAG[mm]}"
        else:
            fname = f"modeling_predictions_{target}__stabfilt_{condition}_{MM_TAG[mm]}"
        paths[mm] = DATA_DIR / f"{fname}.csv"
    return {mm: pd.read_csv(p).sort_values("case_id").reset_index(drop=True) for mm, p in paths.items()}


def _load_survival_condition(condition):
    paths = {}
    for mm in MM_LEVELS:
        if condition == "full":
            fname = "survival_predictions" if mm == 0 else f"survival__perturb_{MM_TAG[mm]}_predictions"
        else:
            fname = f"survival__stabfilt_{condition}_{MM_TAG[mm]}_predictions"
        paths[mm] = DATA_DIR / f"{fname}.csv"
    return {mm: pd.read_csv(p).sort_values("case_id").reset_index(drop=True) for mm, p in paths.items()}


def _check_fold_consistency(level_dfs):
    ref = level_dfs[MM_LEVELS[0]][["case_id", "fold"]]
    for mm in MM_LEVELS[1:]:
        other = level_dfs[mm][["case_id", "fold"]]
        assert (ref.reset_index(drop=True) == other.reset_index(drop=True)).all().all(), \
            f"fold/case_id mismatch at mm={mm}"


def _bootstrap_scores(level_dfs, metric_fn, rng, n_boot=N_BOOT):
    """Returns a DataFrame (n_boot rows x len(MM_LEVELS) cols) of bootstrap
    mean-across-fold scores, using the SAME per-fold resample indices
    across all levels (paired bootstrap)."""
    ref = level_dfs[MM_LEVELS[0]]
    fold_indices = {f: g.index.to_numpy() for f, g in ref.groupby("fold")}

    out = {mm: np.empty(n_boot) for mm in MM_LEVELS}
    for b in range(n_boot):
        resample_per_fold = {f: rng.choice(idx, size=len(idx), replace=True) for f, idx in fold_indices.items()}
        for mm in MM_LEVELS:
            df = level_dfs[mm]
            fold_scores = []
            for f, idx in resample_per_fold.items():
                sub = df.loc[idx]
                s = metric_fn(sub)
                if not np.isnan(s):
                    fold_scores.append(s)
            out[mm][b] = np.mean(fold_scores) if fold_scores else np.nan
    return pd.DataFrame(out)


def _ci(x, lo=2.5, hi=97.5):
    return float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))


def _two_sided_p(diffs):
    diffs = diffs[~np.isnan(diffs)]
    p_pos = (diffs > 0).mean()
    p_neg = (diffs < 0).mean()
    return float(2 * min(p_pos, p_neg))


def run_target(target, paradigm, loader, metric_fn, seed):
    rng = np.random.RandomState(seed)
    conditions = ["full", "stable", "unstable"]
    boot = {}
    orig_swing = {}
    for cond in conditions:
        level_dfs = loader(cond)
        _check_fold_consistency(level_dfs)
        boot_df = _bootstrap_scores(level_dfs, metric_fn, rng)
        boot[cond] = boot_df
        # observed (non-bootstrap) per-level mean, same statistic reported elsewhere
        obs_means = {mm: np.mean([metric_fn(g) for _, g in level_dfs[mm].groupby("fold")]) for mm in MM_LEVELS}
        orig_swing[cond] = max(obs_means.values()) - min(obs_means.values())

    rows = []
    for cond in conditions:
        swings = boot[cond].max(axis=1) - boot[cond].min(axis=1)
        lo, hi = _ci(swings)
        rows.append({
            "target": target, "paradigm": paradigm, "condition": cond,
            "observed_swing": orig_swing[cond],
            "boot_swing_median": float(np.nanmedian(swings)),
            "swing_ci_lo": lo, "swing_ci_hi": hi,
        })

    diff = boot["stable"].max(axis=1) - boot["stable"].min(axis=1) - (boot["unstable"].max(axis=1) - boot["unstable"].min(axis=1))
    lo, hi = _ci(diff)
    p = _two_sided_p(diff.to_numpy())
    rows.append({
        "target": target, "paradigm": paradigm, "condition": "stable_minus_unstable_swing_diff",
        "observed_swing": orig_swing["stable"] - orig_swing["unstable"],
        "boot_swing_median": float(np.nanmedian(diff)),
        "swing_ci_lo": lo, "swing_ci_hi": hi, "p_value": p,
    })
    return rows


def main():
    all_rows = []

    all_rows += run_target(
        "egfr", "classification",
        lambda cond: _load_classification_condition("egfr", cond),
        _auc_metric, seed=SEED,
    )
    all_rows += run_target(
        "recurrence", "classification",
        lambda cond: _load_classification_condition("recurrence", cond),
        _auc_metric, seed=SEED + 1,
    )
    all_rows += run_target(
        "recurrence_survival", "survival",
        lambda cond: _load_survival_condition(cond),
        _concordance_metric, seed=SEED + 2,
    )

    out = pd.DataFrame(all_rows)
    out_path = DATA_DIR / "swing_significance.csv"
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
