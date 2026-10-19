"""Tumor volume alone as a predictor (size confounding).

Paper:    Fig. 2a, Results
Pipeline: python pipeline.py run volume_only   (stage 'discovery')

Two checks, reusing the existing nested-CV pipeline unchanged:
  1. A volume-only, single-feature classifier (identical nested-CV
     protocol, just one column) for each target, compared against the
     already-reported 26-feature radiomics-alone AUC. If a single
     feature matches the full panel, the panel's signal is not adding
     much beyond volume.
  2. Spearman correlation between each target's out-of-fold predicted
     probability (already saved from the main radiomics-alone run) and
     tumor volume, per target -- does the model's actual output track
     size directly?

Run: python scripts/discovery/run_volume_confounding_check.py
Output: data/processed/volume_confounding_check.csv
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import modeling  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
TARGETS = ("recurrence", "egfr", "kras")


def main():
    rad = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    volume_only = rad[["case_id", "shape_volume_mm3"]].copy()

    rows = []
    for target in TARGETS:
        print(f"\n=== {target}: volume-only single-feature classifier ===")
        aucs, oof = modeling.run(
            target=target, feature_set="radiomics", radiomics_path=volume_only,
            result_suffix=f"{target}__volume_only", save=True,
        )
        volume_auc = float(aucs.mean())

        full_path = DATA_DIR / f"modeling_results_{target}.csv"
        full_auc = float(pd.read_csv(full_path)["auc"].mean()) if full_path.exists() else float("nan")

        pred_path = DATA_DIR / f"modeling_predictions_{target}.csv"
        rho, p = float("nan"), float("nan")
        if pred_path.exists():
            preds = pd.read_csv(pred_path).merge(rad[["case_id", "shape_volume_mm3"]], on="case_id", how="left")
            rho, p = spearmanr(preds["y_proba"], preds["shape_volume_mm3"])

        rows.append({
            "target": target, "volume_only_auc": volume_auc, "full_26feature_auc": full_auc,
            "auc_gap_full_minus_volume": full_auc - volume_auc,
            "spearman_prediction_vs_volume_rho": rho, "spearman_prediction_vs_volume_p": p,
        })
        print(f"  volume-only AUC={volume_auc:.3f}, full-panel AUC={full_auc:.3f}, "
              f"prediction-vs-volume rho={rho:.3f} (p={p:.4f})")

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / "volume_confounding_check.csv"
    out.to_csv(out_path, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
