"""Metadata of the discovery collection: clinical, genomic, series and annotation tables (no image download).

Paper:    Table 2, Suppl. Table S1, Suppl. Figs S1-S2
Pipeline: python pipeline.py run metadata   (stage 'data')

Builds, in order:
  data/processed/clinical.csv        tidy clinical + genomic table
  data/processed/series_index.csv    patient -> CT/PT/SEG series join
  data/processed/aim_annotations.csv semantic nodule characteristics (long)
  data/processed/aim_locators.csv    single-slice locator circles + UIDs

Image download (potentially ~98GB) is intentionally separate -- see
src/nsclc_pipeline/download.py and the README.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse

from nsclc_pipeline import aim_annotations, clinical, rna_seq, series_index


def main(fetch_rna_seq: bool = False):
    clin = clinical.build()
    print(f"clinical.csv: {len(clin)} patients")

    idx = series_index.build()
    print(f"series_index.csv: {len(idx)} series, {idx['case_id'].nunique()} patients")

    annotations, locators = aim_annotations.build()
    print(
        f"aim_annotations.csv: {len(annotations)} rows, "
        f"aim_locators.csv: {len(locators)} rows, "
        f"{annotations['case_id'].nunique()} patients with AIM files"
    )

    if fetch_rna_seq:
        expr = rna_seq.build()
        print(f"rna_seq_expression.csv.gz: {expr.shape[0]} genes x {expr.shape[1]} patients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rna-seq",
        action="store_true",
        help="Also fetch+tidy the GEO GSE103584 RNA-seq matrix (~7MB download, external to TCIA)",
    )
    args = parser.parse_args()
    main(fetch_rna_seq=args.rna_seq)
