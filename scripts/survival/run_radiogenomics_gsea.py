"""Gene-set enrichment of the overall-survival risk score (high vs low tertile).

Paper:    Fig. 7c, Suppl. Table S25
Pipeline: python pipeline.py run radiogenomics   (stage 'discovery')

Design mirrors a standard radiogenomics GSEA comparison (high- vs
low-risk groups, preranked Hallmark GSEA), restricted to real data this
study already has: the 117 patients with both RNA-seq (GSE103584) and a
radiomics OS risk score (Methods, "Clinicopathological adjustment and
risk-group stratification"; km_tertile_os.csv), comparing the High and
Low OS-risk tertiles (Intermediate excluded, standard practice for a
two-group differential comparison). This is descriptive/exploratory,
like the SHAP interpretation fit elsewhere in this paper: a full-cohort
comparison, not a cross-validated predictive claim.

Ranking metric: Welch's t-statistic per gene (log2(FPKM+1), NaN/"not
detected" filled to 0 per the Cufflinks convention already used
elsewhere in this pipeline) between High and Low risk groups, fed to
gseapy's preranked GSEA against the MSigDB Hallmark gene sets (fetched
live from Enrichr; identical collection version used by the reference
habitat-imaging study this analysis is modeled on).

Run: python scripts/survival/run_radiogenomics_gsea.py
Output: data/processed/radiogenomics_gsea_results.csv (full Hallmark table)
        data/processed/radiogenomics_gsea_top.csv (FDR<0.2 subset)
        data/processed/radiogenomics_de_genes.csv (per-gene ranking metric)
"""
from __future__ import annotations

import pathlib
import sys

import gseapy
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "processed"
FDR_THRESHOLD = 0.2  # same threshold the reference habitat-imaging study used


def build_groups():
    tert = pd.read_csv(DATA_DIR / "km_tertile_os.csv")
    rna = pd.read_csv(DATA_DIR / "rna_seq_expression.csv.gz")
    rna_patients = [c for c in rna.columns if c != "gene_symbol"]

    tert = tert[tert["case_id"].isin(rna_patients)]
    high_ids = tert.loc[tert["risk_tertile"] == "High", "case_id"].tolist()
    low_ids = tert.loc[tert["risk_tertile"] == "Low", "case_id"].tolist()
    print(f"High-risk with RNA-seq: n={len(high_ids)}, Low-risk with RNA-seq: n={len(low_ids)}")

    expr = rna.set_index("gene_symbol")
    expr = expr.fillna(0.0)  # not-detected genes, matching this pipeline's existing FPKM convention
    log_expr = np.log2(expr + 1.0)
    return log_expr, high_ids, low_ids


def differential_expression(log_expr, high_ids, low_ids):
    high = log_expr[high_ids].to_numpy()
    low = log_expr[low_ids].to_numpy()

    nonzero = ~((high == 0).all(axis=1) & (low == 0).all(axis=1))
    genes = log_expr.index[nonzero]
    high, low = high[nonzero], low[nonzero]

    t_stat, p_val = ttest_ind(high, low, axis=1, equal_var=False, nan_policy="omit")
    mean_diff = high.mean(axis=1) - low.mean(axis=1)

    de = pd.DataFrame({"gene_symbol": genes, "t_stat": t_stat, "p_value": p_val, "log2fc_high_minus_low": mean_diff})
    de = de.dropna(subset=["t_stat"]).sort_values("t_stat", ascending=False).reset_index(drop=True)
    return de


def main():
    log_expr, high_ids, low_ids = build_groups()
    de = differential_expression(log_expr, high_ids, low_ids)
    de.to_csv(DATA_DIR / "radiogenomics_de_genes.csv", index=False)
    print(f"Ranked {len(de)} genes by Welch's t-statistic (High vs Low OS-risk tertile)")

    rnk = de.set_index("gene_symbol")["t_stat"]
    gene_sets = gseapy.get_library(name="MSigDB_Hallmark_2020", organism="human")
    print(f"Fetched {len(gene_sets)} MSigDB Hallmark gene sets from Enrichr")

    pre_res = gseapy.prerank(
        rnk=rnk, gene_sets=gene_sets, min_size=15, max_size=500,
        permutation_num=1000, seed=0, outdir=None, threads=4,
    )
    res = pre_res.res2d.copy()
    for col in ["ES", "NES", "NOM p-val", "FDR q-val"]:
        res[col] = pd.to_numeric(res[col], errors="coerce")
    res = res.sort_values("NES", ascending=False).reset_index(drop=True)
    res.to_csv(DATA_DIR / "radiogenomics_gsea_results.csv", index=False)

    top = res[res["FDR q-val"] < FDR_THRESHOLD].copy()
    top.to_csv(DATA_DIR / "radiogenomics_gsea_top.csv", index=False)

    print(f"\n{len(top)} of {len(res)} Hallmark pathways below FDR q<{FDR_THRESHOLD}:")
    if len(top):
        print(top[["Term", "NES", "FDR q-val"]].to_string(index=False))
    else:
        print("(none)")
    print(f"\nSaved -> {DATA_DIR / 'radiogenomics_gsea_results.csv'}")
    print(f"Saved -> {DATA_DIR / 'radiogenomics_gsea_top.csv'}")


if __name__ == "__main__":
    main()
