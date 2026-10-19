"""SHAP interpretation: which features drive each model's predictions, and
in which direction.

Nested cross-validation (modeling.py) evaluates five models, one per outer
fold, each on patients it was not trained on. Interpretation instead uses
one model fitted on all patients, so it describes what the model learned
and says nothing about generalization.

`texture_n_slices` (the number of slices with enough tumor for texture) is
excluded from the models through modeling.NON_FEATURE_COLS: it is an
extraction count that mainly reflects tumor size, and an early SHAP run
ranked it among the top features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import modeling


def fit_full_pipeline(target: str, feature_set: str = "radiomics", random_state: int = 0):
    """Fit modeling._build_pipeline on ALL available data (not a CV fold)
    -- for descriptive interpretation only, never reported as a
    performance number (that's what nested_cv_auc is for)."""
    X, y, case_ids, numeric_cols, categorical_cols, gene_cols, embed_cols = modeling.load_dataset(
        target, feature_set
    )
    pipe = modeling._build_pipeline(numeric_cols, categorical_cols, gene_cols, embed_cols)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipe, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, scoring="roc_auc", cv=inner
    )
    search.fit(X, y)
    return search.best_estimator_, X, y, case_ids


def compute_shap(target: str, feature_set: str = "radiomics", random_state: int = 0):
    """Returns (shap.Explanation, case_ids) with clean feature names
    (the ColumnTransformer's "num__"/"cat__" prefixes stripped) and, for
    one-hot-encoded categoricals, the specific level name kept (e.g.
    "gender_Male") so a SHAP plot is readable without cross-referencing
    modeling.py's column lists."""
    pipeline, X, y, case_ids = fit_full_pipeline(target, feature_set, random_state)
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    X_t = pre.transform(X)
    if hasattr(X_t, "toarray"):  # a one-hot block can make this sparse
        X_t = X_t.toarray()
    feature_names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]

    # Explicit full-sample masker -- shap.maskers.Independent otherwise
    # silently subsamples to 100 background rows by default, which would
    # be a real (if small) change of method at n=144, not just a speed knob.
    masker = shap.maskers.Independent(X_t, max_samples=X_t.shape[0])
    explainer = shap.LinearExplainer(clf, masker)
    shap_values = explainer(X_t)
    shap_values.feature_names = feature_names
    return shap_values, case_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(modeling.TARGETS), default="recurrence")
    parser.add_argument("--feature-set", choices=list(modeling.FEATURE_SETS), default="radiomics")
    args = parser.parse_args()

    sv, case_ids = compute_shap(args.target, args.feature_set)
    mean_abs = np.abs(sv.values).mean(axis=0)
    ranking = pd.DataFrame({"feature": sv.feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    print(f"Top features for {args.target} ({args.feature_set}):")
    print(ranking.head(15).to_string(index=False))
