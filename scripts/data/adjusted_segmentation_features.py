"""Discovery cohort: features of every segmentation adjusted by −2 to +2 mm (parallel).

Paper:    Fig. 3a-b
Pipeline: python pipeline.py run segmentation_adjustment   (stage 'data')

Output: data/processed/perturbation_features.csv
Run:    python scripts/data/adjusted_segmentation_features.py [n_workers]
"""
from __future__ import annotations

import pathlib
import sys
from multiprocessing import Pool

import pandas as pd
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, perturbation, radiomics, viz  # noqa: E402


def one(case_id):
    from scripts.data.lung1_features import crop_to_tumour
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    si = pd.read_csv(config.SERIES_INDEX_OUT)
    try:
        ct_dir, mask_path, _, _ = radiomics.resolve_ct_and_mask(case_id, si)
        ct = viz.load_ct_volume(ct_dir)
        mask = sitk.Cast(viz.load_mask_on_ct(mask_path, ct) > 0, sitk.sitkUInt8)
        ct, mask = crop_to_tumour(ct, mask)
        arr = sitk.GetArrayFromImage(ct)
        return [f for mm in perturbation.DEFAULT_MM_LEVELS
                if (f := perturbation.extract_perturbed_features(case_id, arr, ct, mask, mm)) is not None]
    except Exception as e:  # noqa: BLE001
        print(f"FAILED {case_id}: {e!r}", flush=True)
        return []


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ids = sorted(pd.read_csv(config.RADIOMICS_OUT).case_id)
    rows = []
    with Pool(n) as pool:
        for i, r in enumerate(pool.imap_unordered(one, ids), 1):
            rows += r
            print(f"[{i}/{len(ids)}]", flush=True)
    df = pd.DataFrame(rows).sort_values(["case_id", "perturbation_mm"])
    df.to_csv(config.PROCESSED_DIR / "perturbation_features.csv", index=False)
    v = df.pivot_table(index="case_id", columns="perturbation_mm", values="shape_volume_mm3")
    print("median volume change vs original:", {mm: round(float((v[mm] / v[0] - 1).median()), 3) for mm in v.columns})


if __name__ == "__main__":
    main()
