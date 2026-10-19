"""Threshold-free instability: how far does a patient's score move?

Classification changes depend on where the threshold lies. This analysis
removes the threshold and compares continuous scores directly: each score is
expressed as a percentile rank (0-100) within the reference distribution, and
instability is the absolute shift in percentile rank of the same patient
(nodule) between two segmentations scored by the same model, or between two
model-development runs scored on the same segmentation. Reported: median and
interquartile range of the shift, and the share of shifts larger than 10 and
25 percentile points. Computed from stored held-out predictions, no refitting.

  discovery  expert vs ±1-mm and vs AI segmentation (bamf_paired_predictions.csv);
             model-development: same expert segmentation, two seeds (repeated_cv_oof.csv.gz)
  readers    pairs of radiation oncologists, fixed recurrence and EGFR models
             (reader_decisions.csv; OS omitted because its model was trained on Lung1)
  nodules    pairs of radiologists vs two seeds on the same segmentation (lidc_oof.csv.gz)

Paper:    Results, Suppl. Table S12
Pipeline: python pipeline.py run rank_shift   (stage 'external')
Output:   data/processed/rank_shift.csv
Run:      python scripts/discovery/run_rank_shift.py
"""
from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
E = ROOT / "data" / "external"


def pct(ref, x):
    ref = np.sort(np.asarray(ref, float))
    return 100 * np.searchsorted(ref, np.asarray(x, float), side="right") / len(ref)


def summarize(setting, endpoint, comparison, shifts):
    s = np.abs(np.asarray(shifts, float))
    return {"setting": setting, "endpoint": endpoint, "comparison": comparison, "n": int(len(s)),
            "median_shift": float(np.median(s)), "q1": float(np.percentile(s, 25)), "q3": float(np.percentile(s, 75)),
            "share_gt10": float((s > 10).mean()), "share_gt25": float((s > 25).mean())}


def discovery(rows):
    d = pd.read_csv(P / "bamf_paired_predictions.csv")
    rc = pd.read_csv(P / "repeated_cv_oof.csv.gz")
    rc = rc[rc.mm == 0]
    for ep, g in d.groupby("endpoint"):
        g = g.dropna(subset=["p_expert", "p_-1mm", "p_+1mm", "p_ai"])
        base = pct(g.p_expert, g.p_expert)
        adj = np.r_[pct(g.p_expert, g["p_-1mm"]) - base, pct(g.p_expert, g["p_+1mm"]) - base]
        rows.append(summarize("Discovery", ep, "±1-mm adjustment (same model)", adj))
        rows.append(summarize("Discovery", ep, "AI segmentation (same model)", pct(g.p_expert, g.p_ai) - base))
        w = rc[rc.endpoint == ep].pivot_table(index="case_id", columns="seed", values="pred")
        r = {s: pct(w[s], w[s]) for s in w.columns}
        mdv = np.concatenate([r[a] - r[b] for a, b in itertools.combinations(r, 2)])
        rows.append(summarize("Discovery", ep, "Model-development variability (same segmentation)", mdv))


def readers(rows):
    d = pd.read_csv(P / "reader_decisions.csv")
    d = d[(d.perturbation_mm == 0) & (d.endpoint != "os")]
    for ep, g in d.groupby("endpoint"):
        g = g.assign(r=pct(g.score, g.score))
        w = g.pivot_table(index="case_id", columns="reader", values="r")
        sh = np.concatenate([(w[a] - w[b]).to_numpy() for a, b in itertools.combinations(w.columns, 2)])
        rows.append(summarize("Multi-reader NSCLC", ep, "Another radiation oncologist (same model)", sh))


def nodules(rows):
    o = pd.read_csv(E / "lidc" / "lidc_oof.csv.gz")
    o = o[o.perturbation_mm == 0].copy()
    o["r"] = o.groupby("seed").p.transform(lambda v: pct(v, v))
    reader_sh = []
    for _, g in o.groupby(["seed", "PatientID", "nodule"]):
        rr = g.r.to_numpy()
        reader_sh += [rr[i] - rr[j] for i, j in itertools.combinations(range(len(rr)), 2)]
    rows.append(summarize("Multi-reader nodules", "malignancy", "Another radiologist (same model)", reader_sh))
    w = o.pivot_table(index=["PatientID", "nodule", "reader"], columns="seed", values="r")
    mdv = np.concatenate([(w[a] - w[b]).dropna().to_numpy() for a, b in itertools.combinations(w.columns, 2)])
    rows.append(summarize("Multi-reader nodules", "malignancy", "Model-development variability (same segmentation)", mdv))


def main():
    rows = []
    for f in (discovery, readers, nodules):
        f(rows)
    out = pd.DataFrame(rows)
    out.to_csv(P / "rank_shift.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
