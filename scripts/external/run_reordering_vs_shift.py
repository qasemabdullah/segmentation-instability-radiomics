"""Why concordance can stay stable while individual classifications change (reordering vs shift).

Paper:    Fig. 7d-g, Suppl. Table S26
Pipeline: python pipeline.py run lung1_models   (stage 'external')

A segmentation change can (a) shift everyone's risk score in the same direction,
which moves patients across a fixed threshold without changing their order,
or (b) reorder patients. This separates the two, for the Lung1 cohort (expert
segmentation vs its -1/+1 mm versions) and the discovery cohort (expert vs AI and
-1/+1 mm), using models trained on original expert segmentations in each outer fold:

  fixed-threshold flips   decision relative to the training-set median
  rank flips              decision relative to the median of the scores being
                          compared (removes any uniform shift)
  Spearman rho            agreement of the two risk orderings
  median shift            median change in score, in training-set s.d. units

Output: data/processed/flip_decomposition.csv
Run:    python scripts/external/run_reordering_vs_shift.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import scripts.external.lung1_models as L  # noqa: E402
from nsclc_pipeline import config  # noqa: E402


def decompose(p0, p1, thr, sd):
    fixed = np.mean((p0 >= thr) != (p1 >= thr))
    rank = np.mean((p0 >= np.median(p0)) != (p1 >= np.median(p1)))
    return {"fixed_threshold_flips": fixed, "rank_flips": rank, "spearman": spearmanr(p0, p1).correlation,
            "median_shift_sd": float(np.median(p1 - p0) / sd)}


def lung1():
    V, y, ids, feat, _ = L.load()
    out = {mm: [] for mm in (-1, 1)}
    for tr, te in KFold(5, shuffle=True, random_state=0).split(V[0]):
        m = L.fit_cox(V[0].iloc[tr], y[tr])
        ptr = m.predict(V[0].iloc[tr])
        p0 = m.predict(V[0].iloc[te])
        for mm in (-1, 1):
            out[mm].append(decompose(p0, m.predict(V[mm].iloc[te]), np.median(ptr), ptr.std()))
    return [{"cohort": "Lung1", "endpoint": "os", "comparison": f"expert vs {mm:+d} mm",
             **pd.DataFrame(v).mean().to_dict()} for mm, v in out.items()]


def stanford():
    d = pd.read_csv(config.PROCESSED_DIR / "bamf_paired_predictions.csv")
    rows = []
    for ep, g in d.groupby("endpoint"):
        for k in ("ai", "-1mm", "+1mm"):
            per = []
            for _, f in g.groupby("fold"):
                # training s.d. is not stored; the pooled s.d. of expert scores in the fold is used instead
                per.append(decompose(f.p_expert.to_numpy(), f[f"p_{k}"].to_numpy(), f.threshold.iloc[0],
                                     f.p_expert.std()))
            rows.append({"cohort": "Stanford", "endpoint": ep, "comparison": f"expert vs {k}",
                         **pd.DataFrame(per).mean().to_dict()})
    return rows


def main():
    res = pd.DataFrame(lung1() + stanford())
    res.to_csv(config.PROCESSED_DIR / "flip_decomposition.csv", index=False)
    pd.set_option("display.width", 160)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
