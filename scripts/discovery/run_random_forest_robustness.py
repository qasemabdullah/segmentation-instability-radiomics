"""Second classifier (random forest) under segmentation adjustment.

Paper:    Suppl. Fig. S6
Pipeline: python pipeline.py run random_forest   (stage 'discovery')

Reuses modeling.load_dataset() (classifier-agnostic) for feature loading;
builds its own pipeline since modeling._build_pipeline() hardcodes
logistic regression.

Run: python scripts/discovery/run_random_forest_robustness.py
Output: data/processed/rf_robustness_results.csv, rf_robustness_swing_summary.csv
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import modeling  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
MM_LEVELS = (-2, -1, 0, 1, 2)
TARGETS = ("recurrence", "egfr")
RANDOM_STATE = 0

PARAM_GRID = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [3, 5, None],
}


def _build_rf_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def nested_cv_rf_auc(X, y, case_ids, outer_folds=5, inner_folds=5, random_state=RANDOM_STATE):
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    fold_aucs = []
    for fold_i, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = _build_rf_pipeline()
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        search = GridSearchCV(pipe, PARAM_GRID, scoring="roc_auc", cv=inner, n_jobs=1)
        search.fit(X_train, y_train)

        proba = search.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)
        print(f"    outer fold {fold_i}: AUC={auc:.3f} (n_test={len(test_idx)}, best={search.best_params_})")
    return np.array(fold_aucs)


def main():
    pert = pd.read_csv(DATA_DIR / "perturbation_features.csv")
    rows = []

    for target in TARGETS:
        for mm in MM_LEVELS:
            print(f"\n=== {target} | {mm:+g} mm (Random Forest) ===")
            if mm == 0:
                table = None  # use the real radiomics_features.csv (original, unperturbed)
            else:
                table = pert[pert["perturbation_mm"] == mm].drop(columns=["perturbation_mm"])
            X, y, case_ids, numeric_cols, categorical_cols, gene_cols, embed_cols = modeling.load_dataset(
                target, "radiomics", radiomics_path=table
            )
            aucs = nested_cv_rf_auc(X, y, case_ids)
            print(f"  mean AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}")
            rows.append({
                "target": target, "perturbation_mm": mm,
                "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std()), "n": len(y),
            })

    results = pd.DataFrame(rows)
    out_path = DATA_DIR / "rf_robustness_results.csv"
    results.to_csv(out_path, index=False)

    swing = results.groupby("target")["mean_auc"].agg(lambda s: s.max() - s.min()).reset_index()
    swing.columns = ["target", "swing"]
    swing_path = DATA_DIR / "rf_robustness_swing_summary.csv"
    swing.to_csv(swing_path, index=False)

    print("\n\n================ RANDOM FOREST SWING SUMMARY ================")
    print(swing.to_string(index=False))
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {swing_path}")

    # side-by-side with the original logistic-regression swings, for direct comparison
    lr = pd.read_csv(DATA_DIR / "perturbation_auc_comparison.csv")
    lr_orig = {}
    for target in TARGETS:
        orig_path = DATA_DIR / f"modeling_results_{target}.csv"
        if orig_path.exists():
            lr_orig[target] = pd.read_csv(orig_path)["auc"].mean()
    print("\nLogistic-regression comparison (already-reported numbers, Fig. 6b):")
    for target in TARGETS:
        sub = lr[lr["target"] == target]
        levels = dict(zip(sub["perturbation_mm"], sub["mean_auc"]))
        if target in lr_orig:
            levels[0] = lr_orig[target]
        lr_swing = max(levels.values()) - min(levels.values()) if levels else float("nan")
        rf_swing = swing.loc[swing["target"] == target, "swing"].iloc[0]
        print(f"  {target}: logistic-regression swing={lr_swing:.3f}, random-forest swing={rf_swing:.3f}")


if __name__ == "__main__":
    main()
