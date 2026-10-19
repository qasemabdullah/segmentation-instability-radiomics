"""PyRadiomics: primary classification and survival analyses with the reference features.

Paper:    Suppl. Tables S13-S14, Suppl. Fig. S6
Pipeline: python pipeline.py run pyradiomics_analyses   (stage 'pyradiomics')

Run in the MAIN environment (not .venv_pyradiomics -- this only reads the
already-extracted pyradiomics_features.csv, no pyradiomics import needed):
  python scripts/pyradiomics/run_pyradiomics_primary_analyses.py

Output: result_suffix="{target}__pyradiomics_primary" files alongside the
existing custom-extractor results (modeling_results_*__pyradiomics_primary.csv,
survival__pyradiomics_primary_results.csv, etc.) -- purely additive.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import modeling, survival  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
TARGETS = ("recurrence", "egfr", "kras")


def main():
    pyrad = pd.read_csv(DATA_DIR / "pyradiomics_features.csv")
    print(f"PyRadiomics baseline table: {len(pyrad)} patients, {len(pyrad.columns) - 1} features")

    rows = []
    for target in TARGETS:
        print(f"\n=== {target}: PyRadiomics-primary classification ===")
        aucs, _oof = modeling.run(
            target=target, feature_set="radiomics", radiomics_path=pyrad,
            result_suffix=f"{target}__pyradiomics_primary", save=True,
        )
        rows.append({"paradigm": "classification", "target": target, "mean_score": float(aucs.mean()), "std_score": float(aucs.std())})

    print("\n=== survival: PyRadiomics-primary Cox model ===")
    scores = survival.run(radiomics_path=pyrad, result_suffix="survival__pyradiomics_primary", save=True)
    rows.append({"paradigm": "survival", "target": "recurrence_survival", "mean_score": float(scores.mean()), "std_score": float(scores.std())})

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / "pyradiomics_primary_summary.csv"
    out.to_csv(out_path, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
