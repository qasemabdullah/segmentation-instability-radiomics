"""PyRadiomics: phenotype stability and reader agreement with the reference extractor.

Paper:    Suppl. Tables S13-S14, Suppl. Fig. S6
Pipeline: python pipeline.py run pyradiomics_analyses   (stage 'pyradiomics')

Run in the MAIN environment (only reads already-extracted PyRadiomics
CSVs; no pyradiomics import needed):
  python scripts/pyradiomics/run_pyradiomics_clustering_interobserver.py

Output: data/processed/pyradiomics_clustering_summary.csv,
        data/processed/pyradiomics_interobserver_comparison.csv
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
K = 4


def clustering_check():
    df = pd.read_csv(DATA_DIR / "pyradiomics_features.csv")
    feat_cols = [c for c in df.columns if c != "case_id"]
    X = SimpleImputer(strategy="median").fit_transform(df[feat_cols].to_numpy())
    Xz = StandardScaler().fit_transform(X)

    sils = {}
    for k in range(3, 8):
        lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xz)
        sils[k] = silhouette_score(Xz, lbl)
    labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Xz)
    sizes = pd.Series(labels).value_counts().sort_index().tolist()

    print(f"PyRadiomics clustering ({len(feat_cols)} features): silhouette at k=4 = {sils[K]:.3f}")
    print(f"  silhouette by k: {sils}")
    print(f"  cluster sizes: {sizes}")

    out = pd.DataFrame([{"k": k, "silhouette": s} for k, s in sils.items()])
    out.to_csv(DATA_DIR / "pyradiomics_clustering_summary.csv", index=False)
    print(f"Saved -> {DATA_DIR / 'pyradiomics_clustering_summary.csv'}")
    return sils[K], sizes


def interobserver_check():
    real = pd.read_csv(DATA_DIR / "pyradiomics_interobserver_features.csv")
    feat_cols = [c for c in real.columns if c not in ("patient_id", "observer")]

    real_rows = []
    for patient_id, group in real.groupby("patient_id"):
        group = group.set_index("observer")
        obs = sorted(group.index)
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                a, b = group.loc[obs[i]], group.loc[obs[j]]
                for col in feat_cols:
                    va, vb = a[col], b[col]
                    denom = max(abs(va), 1e-6)
                    real_rows.append({"feature": col, "rel_change": abs(va - vb) / denom})
    real_df = pd.DataFrame(real_rows)
    real_summary = real_df.groupby("feature")["rel_change"].median()

    pert = pd.read_csv(DATA_DIR / "pyradiomics_perturbation_features.csv")
    a = pert[pert["perturbation_mm"] == -1].set_index("case_id")
    b = pert[pert["perturbation_mm"] == 1].set_index("case_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    synth_rows = []
    for col in feat_cols:
        if col not in a.columns:
            continue
        va, vb = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        denom = np.maximum(np.abs(va), 1e-6)
        synth_rows.append({"feature": col, "median_rel_change": np.median(np.abs(va - vb) / denom)})
    synth_summary = pd.DataFrame(synth_rows).set_index("feature")["median_rel_change"]

    compare = pd.concat([real_summary.rename("real"), synth_summary.rename("synthetic")], axis=1).dropna()
    rho, p = spearmanr(compare["real"], compare["synthetic"])
    print(f"\nPyRadiomics interobserver validation: Spearman rho={rho:.3f}, p={p:.4f}, n_features={len(compare)}")
    compare.to_csv(DATA_DIR / "pyradiomics_interobserver_comparison.csv")
    print(f"Saved -> {DATA_DIR / 'pyradiomics_interobserver_comparison.csv'}")
    return rho, p, len(compare)


def main():
    sil, sizes = clustering_check()
    rho, p, n_feat = interobserver_check()
    summary = pd.DataFrame([{
        "silhouette_k4": sil, "cluster_sizes": str(sizes),
        "interobserver_rho": rho, "interobserver_p": p, "interobserver_n_features": n_feat,
    }])
    summary.to_csv(DATA_DIR / "pyradiomics_clustering_interobserver_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
