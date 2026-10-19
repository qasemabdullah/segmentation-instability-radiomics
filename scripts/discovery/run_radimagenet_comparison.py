"""RadImageNet versus ImageNet image embeddings.

Paper:    Suppl. Table S3
Pipeline: python pipeline.py run radimagenet   (stage 'discovery')

modeling.load_dataset() hardcodes embeddings.csv's path and column
prefix, so rather than touching that (and risking the already-verified
ImageNet-embedding results elsewhere in this paper), this script builds
the same feature matrix by hand and calls modeling._build_pipeline() /
nested_cv_auc() directly -- the same underlying functions, just with
RadImageNet's embed_cols instead of ImageNet's.

Run: python scripts/discovery/run_radimagenet_comparison.py
Output: data/processed/radimagenet_comparison.csv
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import config, modeling  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
TARGETS = ("recurrence", "egfr")


def build_xy(target, radimg_df, radiomics_df, with_radiomics):
    spec = modeling.TARGETS[target]
    clin = pd.read_csv(config.CLINICAL_OUT)

    embed_cols = [c for c in radimg_df.columns if c.startswith("radimg_")]
    if with_radiomics:
        radiomics_cols = [c for c in radiomics_df.columns if c not in modeling.NON_FEATURE_COLS]
        df = radiomics_df.copy()
        numeric_cols = list(radiomics_cols)
    else:
        df = radimg_df[["case_id"]].copy()
        numeric_cols = []
    df = df.merge(radimg_df[["case_id"] + embed_cols], on="case_id", how="inner")
    df = df.merge(clin[["case_id", spec["column"]]], on="case_id", how="left")
    df = df.dropna(subset=[spec["column"]])
    df = df.sort_values("case_id").reset_index(drop=True)

    y = (df[spec["column"]] == spec["positive"]).astype(int).to_numpy()
    case_ids = df["case_id"].to_numpy()
    X = df[numeric_cols + embed_cols]
    return X, y, case_ids, numeric_cols, embed_cols


def main():
    radimg_df = pd.read_csv(DATA_DIR / "radimagenet_embeddings.csv")
    radiomics_df = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    print(f"RadImageNet embeddings: {len(radimg_df)} patients, "
          f"{sum(1 for c in radimg_df.columns if c.startswith('radimg_'))} dims")

    rows = []
    for target in TARGETS:
        for with_radiomics, label in [(False, "embeddings"), (True, "radiomics+embeddings")]:
            print(f"\n=== {target} | {label} (RadImageNet) ===")
            X, y, case_ids, numeric_cols, embed_cols = build_xy(target, radimg_df, radiomics_df, with_radiomics)
            print(f"n={len(y)}, positive={int(y.sum())} ({y.mean():.1%}), "
                  f"radiomics_features={len(numeric_cols)}, embed_dims={len(embed_cols)}")
            aucs, chosen_c, oof = modeling.nested_cv_auc(X, y, case_ids, numeric_cols, [], [], embed_cols)
            print(f"Nested CV AUC: {aucs.mean():.3f} +/- {aucs.std():.3f}")
            rows.append({
                "target": target, "feature_set": label, "extractor": "radimagenet",
                "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std()), "n": len(y),
            })
            oof.to_csv(DATA_DIR / f"modeling_predictions_{target}__{label.replace('+', '_')}_radimagenet.csv", index=False)

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / "radimagenet_comparison.csv"
    out.to_csv(out_path, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
