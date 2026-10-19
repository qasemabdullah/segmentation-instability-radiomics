"""Radiomic phenotypes: Ward clustering of the discovery cohort and its
stability under segmentation adjustment.

The reference phenotypes are fitted once on the expert segmentations
(k = 4, Ward linkage on median-imputed, standardized features). Adjusted
segmentations are then assigned to the nearest fixed phenotype centre, so
only the assignment can change, never the phenotype definitions.

Used by: Fig. 3f,h, Fig. 1, Supplementary Fig. 7, Supplementary Data 15 and
the AI-segmentation analysis (scripts/run_ai_segmentation_models.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from . import config

N_PHENOTYPES = 4
SHIFTS_MM = [-2, -1, 0, 1, 2]
# identifiers and extraction metadata, never used as features
NON_FEATURE_COLS = {"case_id", "ct_series_uid", "seg_series_uid", "texture_n_slices"}


def fit_reference_phenotypes():
    """Fit the reference phenotypes on the expert segmentations.

    Returns (features table, feature columns, fitted scaler, fitted imputer,
    phenotype labels, phenotype centres in standardized space, standardized
    feature matrix).
    """
    df = pd.read_csv(config.RADIOMICS_OUT)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = SimpleImputer(strategy="median").fit_transform(df[feat_cols].to_numpy())
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    labels = AgglomerativeClustering(n_clusters=N_PHENOTYPES, linkage="ward").fit_predict(Xz)
    centroids = np.array([Xz[labels == c].mean(axis=0) for c in range(N_PHENOTYPES)])
    return df, feat_cols, scaler, SimpleImputer(strategy="median").fit(X), labels, centroids, Xz


def reassignment_by_shift():
    """Adjusted Rand index and share of patients reassigned to another
    phenotype when each patient's segmentation is shifted by −2 to +2 mm."""
    df, feat_cols, scaler, imputer, orig_labels, centroids, _ = fit_reference_phenotypes()
    case_order = df["case_id"].tolist()
    pert = pd.read_csv(config.PROCESSED_DIR / "perturbation_features.csv")
    rows = []
    for mm in SHIFTS_MM:
        sub = pert[pert["perturbation_mm"] == mm].set_index("case_id").reindex(case_order)
        Xpz = scaler.transform(imputer.transform(sub[feat_cols].to_numpy()))
        new_labels = np.linalg.norm(Xpz[:, None, :] - centroids[None, :, :], axis=2).argmin(axis=1)
        rows.append({"mm": mm, "ari": adjusted_rand_score(orig_labels, new_labels),
                     "pct_reassigned": float((new_labels != orig_labels).mean() * 100)})
    return pd.DataFrame(rows)


def bootstrap_ari(Xz, orig_labels, n_boot=100, seed=0):
    """Baseline reproducibility: adjusted Rand index between the reference
    phenotypes and phenotypes re-fitted on bootstrap resamples of patients."""
    rng = np.random.RandomState(seed)
    n = Xz.shape[0]
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_labels = AgglomerativeClustering(n_clusters=N_PHENOTYPES, linkage="ward").fit_predict(Xz[idx])
        aris.append(adjusted_rand_score(orig_labels[idx], boot_labels))
    return np.array(aris)
