"""Is segmentation-induced reclassification an artifact of the median threshold?

Repeats the main reclassification comparisons with the classification
threshold at the 25th, 33rd, 50th, 67th and 75th percentile of the scores
from the reference segmentation, using the stored held-out predictions (no
model is refitted):

  discovery  ±1-mm and AI segmentations scored by the model developed on
             expert segmentations (131 patients; bamf_paired_predictions.csv);
             model-development variability = classifications of the same
             expert segmentation in two cross-validation seeds (repeated_cv_oof.csv.gz)
  readers    pairs of radiation oncologists, fixed models (reader_decisions.csv)
  nodules    pairs of radiologists and model-development variability between
             seeds, LIDC-IDRI (lidc_oof.csv.gz)
  liver      expert versus AI segmentation and model-development variability
             between seeds (crlm_oof.csv.gz)

Percentiles are taken within each endpoint (and within each seed where the
predictions come from separate cross-validation runs). A threshold at the
q-th percentile classifies the top (100 − q)% as high risk / positive.

Paper:    Results, Suppl. Table S11
Pipeline: python pipeline.py run threshold_sensitivity   (stage 'external')
Output:   data/processed/threshold_sensitivity.csv
Run:      python scripts/discovery/run_threshold_sensitivity.py
"""
from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
E = ROOT / "data" / "external"
QS = [25, 33, 50, 67, 75]


def cls(scores, q):
    return (scores >= np.percentile(scores, q)).astype(int)


def discovery(rows):
    d = pd.read_csv(P / "bamf_paired_predictions.csv")
    rc = pd.read_csv(P / "repeated_cv_oof.csv.gz")
    rc = rc[rc.mm == 0]
    for ep, g in d.groupby("endpoint"):
        g = g.dropna(subset=["p_expert", "p_-1mm", "p_+1mm", "p_ai"])
        r = rc[rc.endpoint == ep].pivot_table(index="case_id", columns="seed", values="pred")
        for q in QS:
            thr = np.percentile(g.p_expert, q)
            c0 = (g.p_expert >= thr).astype(int)
            adj = np.mean([((g[k] >= thr).astype(int) != c0).mean() for k in ("p_-1mm", "p_+1mm")])
            ai = ((g.p_ai >= thr).astype(int) != c0).mean()
            seeds = {s: cls(r[s].to_numpy(), q) for s in r.columns}
            mdv = np.mean([(seeds[a] != seeds[b]).mean() for a, b in itertools.combinations(seeds, 2)])
            rows += [{"setting": "Discovery (131 patients)", "endpoint": ep, "comparison": "±1-mm adjustment", "percentile": q, "reclassified": adj},
                     {"setting": "Discovery (131 patients)", "endpoint": ep, "comparison": "AI segmentation", "percentile": q, "reclassified": ai},
                     {"setting": "Discovery (144 patients)", "endpoint": ep, "comparison": "Model-development variability", "percentile": q, "reclassified": mdv}]


def readers(rows):
    d = pd.read_csv(P / "reader_decisions.csv")
    d = d[d.perturbation_mm == 0]
    for ep, g in d.groupby("endpoint"):
        for q in QS:
            thr = np.percentile(g.score, q)
            w = g.assign(c=(g.score >= thr).astype(int)).pivot_table(index="case_id", columns="reader", values="c")
            pairs = [(w[a] != w[b]).mean() for a, b in itertools.combinations(w.columns, 2)]
            rows.append({"setting": "Multi-reader NSCLC (21 patients)", "endpoint": ep, "comparison": "Another radiation oncologist",
                         "percentile": q, "reclassified": float(np.mean(pairs))})


def nodules(rows):
    o = pd.read_csv(E / "lidc" / "lidc_oof.csv.gz")
    o = o[o.perturbation_mm == 0]
    for q in QS:
        per_seed, cls_by_seed = [], {}
        for s, g in o.groupby("seed"):
            g = g.assign(c=(g.p >= np.percentile(g.p, q)).astype(int))
            cls_by_seed[s] = g.set_index(["PatientID", "nodule", "reader"]).c
            dis = []
            for _, n in g.groupby(["PatientID", "nodule"]):
                cc = n.c.to_numpy()
                dis += [cc[i] != cc[j] for i, j in itertools.combinations(range(len(cc)), 2)]
            per_seed.append(np.mean(dis))
        seeds = list(cls_by_seed)
        mdv = np.mean([(cls_by_seed[a] != cls_by_seed[b].reindex(cls_by_seed[a].index)).mean()
                       for a, b in itertools.combinations(seeds, 2)])
        rows += [{"setting": "Multi-reader nodules (589 nodules)", "endpoint": "malignancy", "comparison": "Another radiologist", "percentile": q,
                  "reclassified": float(np.mean(per_seed))},
                 {"setting": "Multi-reader nodules (589 nodules)", "endpoint": "malignancy", "comparison": "Model-development variability", "percentile": q,
                  "reclassified": float(mdv)}]


def liver(rows):
    c = pd.read_csv(E / "crlm" / "crlm_oof.csv.gz")
    for ep, g in c.groupby("endpoint"):
        for q in QS:
            ai_rate, cls_by_seed = [], {}
            for s, gs in g.groupby("seed"):
                w = gs.pivot_table(index="PatientID", columns="variant", values="risk")
                thr = np.percentile(w["expert"].dropna(), q)
                both = w[["expert", "ai"]].dropna()
                ai_rate.append(((both.expert >= thr) != (both.ai >= thr)).mean())
                cls_by_seed[s] = (w["expert"] >= thr).astype(int)
            seeds = list(cls_by_seed)
            mdv = np.mean([(cls_by_seed[a] != cls_by_seed[b].reindex(cls_by_seed[a].index)).mean()
                           for a, b in itertools.combinations(seeds, 2)])
            rows += [{"setting": "Liver metastases (185 patients)", "endpoint": ep, "comparison": "AI segmentation", "percentile": q, "reclassified": float(np.mean(ai_rate))},
                     {"setting": "Liver metastases (197 patients)", "endpoint": ep, "comparison": "Model-development variability", "percentile": q, "reclassified": float(mdv)}]


def main():
    rows = []
    for f in (discovery, readers, nodules, liver):
        f(rows)
    out = pd.DataFrame(rows)
    out.to_csv(P / "threshold_sensitivity.csv", index=False)
    wide = out.pivot_table(index=["setting", "endpoint", "comparison"], columns="percentile", values="reclassified")
    pd.set_option("display.width", 200)
    print((100 * wide).round(1).to_string())


if __name__ == "__main__":
    main()
