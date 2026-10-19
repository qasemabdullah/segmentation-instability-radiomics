"""Fetch and tidy the paired RNA-seq data for this cohort.

This is not part of the TCIA image collection -- it's an external resource
the data submitters point to: GEO GSE103584, "Identification of
relationships between Molecular and Imaging Phenotypes in NSCLC using
radiogenomics" (Bakr et al.), gene-level Cufflinks FPKM for 130 of the 211
patients (the Stanford/VA "R01-*" subset only -- the AMC cohort has no
paired RNA-seq). Case IDs in the matrix header match `case_id` in
clinical.csv exactly, so joining is a plain merge.

Data processing per GEO: reads aligned with TopHat2 to hg19, gene/isoform
expression estimated with Cufflinks. Values are FPKM; missing/undetected
genes are NA (Cufflinks emits "NA" natively -- no extra cleanup needed
since pandas already treats "NA" as a missing value).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from . import config


def fetch(url: str | None = None, out_path: Path | None = None, timeout: int = 120) -> Path:
    """Download the GEO supplementary matrix if not already present."""
    url = url or config.RNA_SEQ_URL
    out_path = out_path or config.RNA_SEQ_RAW
    if out_path.exists():
        return out_path

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return out_path


def load_rna_seq(path: Path | None = None) -> pd.DataFrame:
    """Load the raw matrix as genes (rows) x case_id (columns), FPKM."""
    path = path or config.RNA_SEQ_RAW
    df = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    df.index.name = "gene_symbol"
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt to (case_id, gene_symbol, fpkm) -- convenient for joins/filtering,
    but large (~2.9M rows for the full matrix); prefer the wide form for
    storage and melt on demand for a gene subset."""
    long_df = df.reset_index().melt(id_vars="gene_symbol", var_name="case_id", value_name="fpkm")
    return long_df.dropna(subset=["fpkm"])


def build(fetch_if_missing: bool = True, out_path: Path | None = None) -> pd.DataFrame:
    out_path = out_path or config.RNA_SEQ_OUT
    if fetch_if_missing:
        fetch()
    df = load_rna_seq()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, compression="gzip")
    return df


if __name__ == "__main__":
    expr = build()
    print(
        f"RNA-seq: {expr.shape[0]} genes x {expr.shape[1]} patients "
        f"-> {config.RNA_SEQ_OUT}"
    )
