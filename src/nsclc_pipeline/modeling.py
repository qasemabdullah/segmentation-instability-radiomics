"""Nested cross-validation modeling: radiomics (+ optionally clinical
baseline covariates, + optionally RNA-seq) -> clinical/genomic outcome.

At n~144 (fewer still for genomic targets with missing labels, fewer yet
once RNA-seq is required), a plain train/test split -- or even k-fold CV
where hyperparameters are tuned on the same folds the score is reported
on -- overstates performance. That is the single most common failure mode
in published small-cohort radiogenomics papers. This uses *nested* CV: an
outer loop gives an unbiased performance estimate, an inner loop (fit only
on each outer training fold) picks hyperparameters, so the reported AUC is
never computed on data that influenced model selection. RNA-seq's ~22,125
genes are reduced via PCA fit *only* inside each training fold too (via
the sklearn Pipeline/ColumnTransformer below) -- fitting PCA on the full
dataset first would leak test-fold information into the "features" the
model sees, just as surely as tuning on the test fold would.

Clinical covariates are restricted to baseline/pre-treatment
characteristics (age, stage, grade, smoking history, ...) deliberately --
not treatment fields (chemo/radiation) or anything downstream of the
outcome being predicted, which would make the "prediction" partly
circular.

Framing: treat results here as a methodology demonstration and exploratory
signal, not a validated biomarker claim -- the confidence interval on an
AUC from ~5 folds of ~30 patients each is wide, and this hasn't been
externally validated on an independent cohort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

TARGETS = {
    "recurrence": {
        "column": "recurrence",
        "positive": "yes",
        "description": "Tumor recurrence after treatment",
    },
    "egfr": {
        "column": "egfr_mutation_status",
        "positive": "Mutant",
        "description": "EGFR mutation status",
    },
    "kras": {
        "column": "kras_mutation_status",
        "positive": "Mutant",
        "description": "KRAS mutation status",
    },
}

NON_FEATURE_COLS = {
    "case_id", "ct_series_uid", "seg_series_uid",
    # extraction-pipeline bookkeeping, not a radiological feature -- it's
    # how many axial slices had enough tumor pixels to compute texture
    # on, which mostly just re-encodes tumor size (already captured
    # properly by shape_volume_mm3). Caught via SHAP: it was ranking as a
    # top "predictive" feature, which is a modeling smell, not a finding
    # -- kept in radiomics_features.csv as useful QC info, excluded here.
    "texture_n_slices",
}

# Baseline/pre-treatment only -- see module docstring for why treatment
# and outcome-adjacent fields are excluded.
CLINICAL_NUMERIC = ["age_at_histological_diagnosis", "pack_years"]
CLINICAL_CATEGORICAL = [
    "gender",
    "smoking_status",
    "histology",
    "pathological_t_stage",
    "pathological_n_stage",
    "pathological_m_stage",
    "histopathological_grade",
    "lymphovascular_invasion",
    "pleural_invasion_elastic_visceral_or_parietal",
]

FEATURE_SETS = {
    "radiomics": {"radiomics": True, "clinical": False, "rnaseq": False, "embeddings": False},
    "radiomics+clinical": {"radiomics": True, "clinical": True, "rnaseq": False, "embeddings": False},
    "radiomics+clinical+rnaseq": {"radiomics": True, "clinical": True, "rnaseq": True, "embeddings": False},
    "embeddings": {"radiomics": False, "clinical": False, "rnaseq": False, "embeddings": True},
    "radiomics+embeddings": {"radiomics": True, "clinical": False, "rnaseq": False, "embeddings": True},
}
RNA_SEQ_PCA_COMPONENTS = 10  # same order of magnitude as Bakr et al.'s 10 co-expression metagenes
EMBEDDING_PCA_COMPONENTS = 10  # same reduction applied to the 512-dim CNN embeddings, see embeddings.py


def load_dataset(target: str, feature_set: str = "radiomics", radiomics_path=None):
    """radiomics_path lets a caller swap in an alternate radiomics table
    with the same schema as radiomics_features.csv -- e.g. features
    re-extracted from a perturbed (eroded/dilated) mask, to measure how
    much the reported AUC moves under segmentation variation alone, with
    everything else about the pipeline held fixed. Defaults to the real
    (unperturbed) radiomics_features.csv."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}, choose from {list(TARGETS)}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}, choose from {list(FEATURE_SETS)}")
    spec = TARGETS[target]
    use = FEATURE_SETS[feature_set]

    if isinstance(radiomics_path, pd.DataFrame):
        rad = radiomics_path
    else:
        rad = pd.read_csv(radiomics_path or config.RADIOMICS_OUT)
    radiomics_cols = [c for c in rad.columns if c not in NON_FEATURE_COLS]

    # Every feature set still starts from the radiomics table's case_id
    # list (it's the "which patients have a usable mask" anchor), even
    # when radiomics features themselves aren't used (e.g. "embeddings").
    df = rad[["case_id"]].copy()
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    if use["radiomics"]:
        df = rad.copy()
        numeric_cols += radiomics_cols

    clin = pd.read_csv(config.CLINICAL_OUT)
    clin_cols = ["case_id", spec["column"]]
    if use["clinical"]:
        clin_cols += CLINICAL_NUMERIC + CLINICAL_CATEGORICAL
        numeric_cols += CLINICAL_NUMERIC
        categorical_cols += CLINICAL_CATEGORICAL
    df = df.merge(clin[clin_cols], on="case_id", how="left")

    gene_cols: list[str] = []
    if use["rnaseq"]:
        rna = pd.read_csv(config.RNA_SEQ_OUT, index_col=0).T
        rna.index.name = "case_id"
        rna = rna.reset_index()
        gene_cols = [c for c in rna.columns if c != "case_id"]
        # inner join -- only patients with radiomics AND RNA-seq survive
        df = df.merge(rna, on="case_id", how="inner")

    embed_cols: list[str] = []
    if use["embeddings"]:
        embeddings_df = pd.read_csv(config.PROCESSED_DIR / "embeddings.csv")
        embed_cols = [c for c in embeddings_df.columns if c.startswith("embed_")]
        # inner join -- only patients with an extracted embedding survive
        df = df.merge(embeddings_df[["case_id"] + embed_cols], on="case_id", how="inner")

    df = df.dropna(subset=[spec["column"]])
    # Sort by case_id: radiomics_features.csv's row order depends on which
    # order a (possibly resumed) extraction batch happened to process
    # patients in, not a fixed order -- without this, StratifiedKFold's
    # fold assignment (and therefore the reported AUCs) would depend on
    # that incidental history rather than just --seed as documented.
    df = df.sort_values("case_id").reset_index(drop=True)

    y = (df[spec["column"]] == spec["positive"]).astype(int).to_numpy()
    case_ids = df["case_id"].to_numpy()
    X = df[numeric_cols + categorical_cols + gene_cols + embed_cols]
    return X, y, case_ids, numeric_cols, categorical_cols, gene_cols, embed_cols


def _build_pipeline(numeric_cols, categorical_cols, gene_cols, embed_cols=()):
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    if gene_cols:
        # Cufflinks reports NA for an undetected gene -- impute 0 (not
        # detected), not the median, which would fabricate expression.
        transformers.append(
            (
                "gene",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("pca", PCA(n_components=RNA_SEQ_PCA_COMPONENTS, random_state=0)),
                        ("scale", StandardScaler()),
                    ]
                ),
                gene_cols,
            )
        )
    if embed_cols:
        transformers.append(
            (
                "embed",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("pca", PCA(n_components=EMBEDDING_PCA_COMPONENTS, random_state=0)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(embed_cols),
            )
        )
    pre = ColumnTransformer(transformers)
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def nested_cv_auc(
    X, y, case_ids, numeric_cols, categorical_cols, gene_cols, embed_cols=(),
    outer_folds=5, inner_folds=5, random_state=0,
):
    """One AUC per outer fold, each computed on a held-out fold the model
    (including its hyperparameter, tuned only on that fold's training
    split) never saw during selection. Also returns out-of-fold predicted
    probabilities for every patient (each predicted only by a model that
    never trained on them), used for the ROC curve."""
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    param_grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}

    fold_aucs = []
    chosen_c = []
    oof_rows = []
    for fold_i, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = _build_pipeline(numeric_cols, categorical_cols, gene_cols, embed_cols)
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=inner, n_jobs=1)
        search.fit(X_train, y_train)

        proba = search.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)
        chosen_c.append(search.best_params_["clf__C"])
        for cid, yt, p in zip(case_ids[test_idx], y_test, proba):
            oof_rows.append({"case_id": cid, "fold": fold_i, "y_true": int(yt), "y_proba": float(p)})
        print(
            f"  outer fold {fold_i}: AUC={auc:.3f} "
            f"(n_test={len(test_idx)}, best C={search.best_params_['clf__C']})"
        )

    return np.array(fold_aucs), chosen_c, pd.DataFrame(oof_rows)


def run(
    target: str = "recurrence",
    feature_set: str = "radiomics",
    outer_folds: int = 5,
    inner_folds: int = 5,
    random_state: int = 0,
    radiomics_path=None,
    result_suffix: str | None = None,
    save: bool = True,
):
    X, y, case_ids, numeric_cols, categorical_cols, gene_cols, embed_cols = load_dataset(
        target, feature_set, radiomics_path=radiomics_path
    )
    spec = TARGETS[target]
    n_pos = int(y.sum())
    n_features = (
        len(numeric_cols) + len(categorical_cols)
        + (RNA_SEQ_PCA_COMPONENTS if gene_cols else 0)
        + (EMBEDDING_PCA_COMPONENTS if embed_cols else 0)
    )
    extra = ""
    if gene_cols:
        extra += f", {RNA_SEQ_PCA_COMPONENTS} RNA-seq PCA components from {len(gene_cols)} genes"
    if embed_cols:
        extra += f", {EMBEDDING_PCA_COMPONENTS} embedding PCA components from {len(embed_cols)} CNN dims"
    print(
        f"Target: {spec['description']} ({target}) -- feature set: {feature_set}\n"
        f"n={len(y)}, positive={n_pos} ({n_pos / len(y):.1%}), "
        f"features={n_features} ({len(numeric_cols)} numeric, "
        f"{len(categorical_cols)} categorical{extra})"
    )

    min_class = min(n_pos, len(y) - n_pos)
    if min_class < outer_folds:
        raise ValueError(
            f"Only {min_class} patients in the minority class -- too few for "
            f"{outer_folds}-fold stratified CV. Reduce --outer-folds or pick a "
            f"target with more labels."
        )

    aucs, chosen_c, oof = nested_cv_auc(
        X, y, case_ids, numeric_cols, categorical_cols, gene_cols, embed_cols,
        outer_folds=outer_folds, inner_folds=inner_folds, random_state=random_state,
    )
    print(f"\nNested CV AUC: {aucs.mean():.3f} +/- {aucs.std():.3f}  (across {outer_folds} outer folds)")
    print("Per-fold:", [round(a, 3) for a in aucs])
    print(
        "\nCaveat: n is small (especially per fold), so this AUC is a noisy "
        "estimate -- treat it as exploratory signal, not a validated "
        "biomarker. Try a different --seed to see how much the estimate moves."
    )

    if not save:
        return aucs, oof

    suffix = result_suffix or (target if feature_set == "radiomics" else f"{target}__{feature_set.replace('+', '_')}")
    out_path = config.PROCESSED_DIR / f"modeling_results_{suffix}.csv"
    pd.DataFrame({"fold": range(1, len(aucs) + 1), "auc": aucs, "best_C": chosen_c}).to_csv(
        out_path, index=False
    )
    oof_path = config.PROCESSED_DIR / f"modeling_predictions_{suffix}.csv"
    oof.to_csv(oof_path, index=False)
    print(f"\nSaved fold-level results -> {out_path}")
    print(f"Saved out-of-fold predictions -> {oof_path}")
    return aucs, oof


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(TARGETS), default="recurrence")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="radiomics")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        args.target,
        feature_set=args.feature_set,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        random_state=args.seed,
    )
