"""Multi-reader NSCLC cohort: overall-survival reader analysis with a model that
cannot share patients with the reader cohort.

The main reader analysis scores OS with a Cox model trained on Lung1, which
comes from the same institution as the multi-reader cohort; patient overlap
cannot be excluded from public identifiers. Here the five radiation
oncologists' segmentations are instead scored by a Cox model trained on all
144 discovery-cohort (Stanford / VA Palo Alto) expert segmentations, which
cannot contain these patients. Classification, pairwise disagreement and the
share of patients with any disagreement are computed exactly as in
run_reader_decisions.py (median of the scores over the readers' original
segmentations).

Paper:    Results, Suppl. Table S18
Pipeline: python pipeline.py run reader_os_discovery   (stage 'readers')
Needs:    data/processed/reader_features_pm1.csv (scripts/readers/run_reader_decisions.py)
Output:   data/processed/reader_os_discovery_model.json
Run:      python scripts/readers/run_reader_os_discovery.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, survival  # noqa: E402
from nsclc_pipeline.modeling import NON_FEATURE_COLS  # noqa: E402

P = config.PROCESSED_DIR


def main():
    from scripts.external.lung1_models import fit_cox
    feat = [c for c in pd.read_csv(config.RADIOMICS_OUT, nrows=1).columns if c not in NON_FEATURE_COLS]
    X, y, ids, _ = survival.load_survival_dataset(pd.read_csv(config.RADIOMICS_OUT), endpoint="os")
    model = fit_cox(X[feat], y)
    d = pd.read_csv(P / "reader_features_pm1.csv")
    d = d[d.perturbation_mm == 0].copy()
    d["score"] = model.predict(d[feat])
    d["high"] = d.score >= d.score.median()
    base = d.pivot(index="case_id", columns="reader", values="high")
    sc = d.pivot(index="case_id", columns="reader", values="score")
    pairs = list(itertools.combinations(base.columns, 2))
    res = {"model": "ridge Cox, all 144 discovery-cohort expert segmentations (OS)",
           "patients": int(base.shape[0]),
           "pairwise_disagreement": float(np.mean([(base[a] != base[b]).mean() for a, b in pairs])),
           "patients_with_any_disagreement": float((base.nunique(axis=1) > 1).mean()),
           "mean_reader_rank_rho": float(np.mean([spearmanr(sc[a], sc[b]).correlation for a, b in pairs]))}
    (P / "reader_os_discovery_model.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
