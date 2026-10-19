"""Discovery cohort: convert every expert DICOM-SEG segmentation to a NIfTI mask.

Downloads each patient's SEG series from TCIA (small files) and writes the
binary tumour mask on the CT grid it was drawn on. Patients whose mask
already exists are skipped, so the step can be interrupted and resumed.
This is the command-line equivalent of the GUI's "Convert ALL" button.

Paper:    all discovery-cohort analyses
Pipeline: python pipeline.py run discovery_masks   (stage 'data')
         "Tumor segmentation"); every downstream discovery-cohort analysis.
Needs:   data/processed/series_index.csv (scripts/data/build_metadata.py)
Output:  data/processed/masks/<case_id>_seg.nii.gz
Run:     python scripts/data/build_discovery_masks.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, download, seg_to_mask  # noqa: E402


def main():
    seg = pd.read_csv(config.SERIES_INDEX_OUT)
    seg = seg[seg["modality"] == "SEG"].sort_values("case_id")
    written = skipped = failed = 0
    for i, row in enumerate(seg.itertuples(), start=1):
        out_path = config.MASKS_DIR / f"{row.case_id}_seg.nii.gz"
        if out_path.exists():
            skipped += 1
            continue
        try:
            seg_dir = download.download_series(row.series_uid)
            seg_to_mask.build_one(row.case_id, seg_dir)
            written += 1
            print(f"{i}/{len(seg)} {row.case_id}: written")
        except Exception as e:  # noqa: BLE001 -- keep going, report at the end
            print(f"{i}/{len(seg)} {row.case_id}: FAILED ({type(e).__name__}: {e})")
            failed += 1
    print(f"Masks: {written} written, {skipped} already present, {failed} failed.")


if __name__ == "__main__":
    main()
