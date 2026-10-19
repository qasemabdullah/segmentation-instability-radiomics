"""LIDC-IDRI: the ±1-mm flag against misclassification and reader disagreement, with patient-bootstrap CIs.

Paper:    Fig. 5a-f, Suppl. Table S17, Suppl. Fig. S9
Pipeline: python pipeline.py run lidc_models   (stage 'external')

For every reader's outline (threshold 0.5, as in the main analysis):
  flag   = classifications at −1, 0 and +1 mm do not all agree
  error  = classification at 0 mm differs from the consensus label
  disagree (pairwise) = classification differs from another reader's
Reports flag rate, error rate and disagreement by flag status, the share of
errors and of disagreements that were flagged, and odds ratios with 95%
intervals from a bootstrap that resamples patients (so correlated readers,
nodules and repetitions stay together).

Output: data/external/lidc/flag_clinical_check.json
"""
from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "data" / "external" / "lidc"


def tables(o: pd.DataFrame):
    w = o.pivot_table(index=["seed", "PatientID", "nodule", "reader", "y"], columns="perturbation_mm", values="p").dropna().reset_index()
    c = {k: (w[k] >= 0.5).astype(int) for k in (-1, 0, 1)}
    w["cls"] = c[0]
    w["flag"] = ((c[-1] != c[0]) | (c[1] != c[0])).astype(int)
    w["error"] = (w.cls != w.y).astype(int)
    pairs = []
    for (s, pid, nod), g in w.groupby(["seed", "PatientID", "nodule"]):
        for (_, a), (_, b) in itertools.permutations(g.iterrows(), 2):
            pairs.append((s, pid, a.flag, int(a.cls != b.cls)))
    pr = pd.DataFrame(pairs, columns=["seed", "PatientID", "flag", "disagree"])
    return w, pr


def odds(flag, out):
    a = ((flag == 1) & (out == 1)).sum() + 0.5
    b = ((flag == 1) & (out == 0)).sum() + 0.5
    c = ((flag == 0) & (out == 1)).sum() + 0.5
    d = ((flag == 0) & (out == 0)).sum() + 0.5
    return (a * d) / (b * c)


def summary(w, pr):
    return {
        "outlines": int(len(w)),
        "flag_rate": float(w.flag.mean()),
        "error_if_flagged": float(w.error[w.flag == 1].mean()),
        "error_if_not_flagged": float(w.error[w.flag == 0].mean()),
        "share_of_errors_flagged": float(w.flag[w.error == 1].mean()),
        "or_error": float(odds(w.flag, w.error)),
        "disagree_if_flagged": float(pr.disagree[pr.flag == 1].mean()),
        "disagree_if_not_flagged": float(pr.disagree[pr.flag == 0].mean()),
        "share_of_disagreements_flagged": float(pr.flag[pr.disagree == 1].mean()),
        "or_disagree": float(odds(pr.flag, pr.disagree)),
    }


def main(n_boot: int = 1000):
    o = pd.read_csv(D / "lidc_oof.csv.gz")
    w, pr = tables(o)
    out = summary(w, pr)
    pts = w.PatientID.unique()
    wg, pg = dict(tuple(w.groupby("PatientID"))), dict(tuple(pr.groupby("PatientID")))
    rng = np.random.default_rng(0)
    boot = {"or_error": [], "or_disagree": []}
    for _ in range(n_boot):
        s = rng.choice(pts, len(pts), replace=True)
        wb = pd.concat([wg[p] for p in s])
        pb = pd.concat([pg[p] for p in s if p in pg])
        boot["or_error"].append(odds(wb.flag, wb.error))
        boot["or_disagree"].append(odds(pb.flag, pb.disagree))
    for k, v in boot.items():
        out[k + "_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["n_patients"] = int(len(pts))
    out["bootstrap"] = f"{n_boot} patient-level resamples"
    (D / "flag_clinical_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
