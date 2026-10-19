"""Overall outcome of replacing the expert segmentation by the public AI tool.

Reclassification rates for the AI segmentation are computed in the patients
whose tumor the tool detected. This script places detection failures and
reclassification in one summary: of all patients with an AI segmentation on
the expert CT (and a known label for the endpoint), the share whose tumor was
missed, the share detected and classified as from the expert segmentation,
and the share detected but reclassified; the last two use the 20-repetition
same-patient analysis (per patient, the share of the 20 repetitions in which
the AI segmentation changed the classification), with the primary allocation
alongside. 95% intervals from 2,000 patient-level bootstrap resamples.

Paper:    Results, Suppl. Table S16
Pipeline: python pipeline.py run ai_outcome   (stage 'ai')
Needs:    bamf_agreement.csv, decomposition_predictions.csv.gz, bamf_paired_predictions.csv, clinical outcomes
Outputs:  data/processed/ai_outcome_summary.csv
Run:      python scripts/ai_segmentation/run_ai_outcome_summary.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, modeling, survival  # noqa: E402

P = config.PROCESSED_DIR


def labelled(ep):
    exp = pd.read_csv(config.RADIOMICS_OUT)
    if ep == "os":
        return set(survival.load_survival_dataset(exp, endpoint="os")[2])
    return set(modeling.load_dataset(ep, "radiomics", exp)[2])


def main():
    ag = pd.read_csv(P / "bamf_agreement.csv")
    ag = ag[ag.source == "ai"] if "source" in ag else ag
    dec = pd.read_csv(P / "decomposition_predictions.csv.gz")
    dec = dec[dec.cohort == "discovery"]
    prim = pd.read_csv(P / "bamf_paired_predictions.csv")
    rng = np.random.default_rng(20261005)
    rows = []
    for ep in ("recurrence", "egfr", "os"):
        a = ag[ag.case_id.isin(labelled(ep))].set_index("case_id")
        d = dec[dec.endpoint == ep].pivot_table(index="case_id", columns=["segmentation", "seed"], values="high",
                                                aggfunc="first")
        flip20 = (d["expert"].astype(float) != d["ai"].astype(float)).where(d["ai"].notna()).mean(axis=1)
        pp = prim[prim.endpoint == ep].set_index("case_id")
        flip1 = ((pp.p_expert >= pp.threshold) != (pp.p_ai >= pp.threshold)).astype(float)
        per = pd.DataFrame({"missed": (~a.detected.astype(bool)).astype(float)})
        for name, f in (("20 repetitions", flip20), ("primary allocation", flip1)):
            per["reclassified"] = f.reindex(per.index).fillna(0.0).where(per.missed == 0, 0.0)
            per["changed_or_missed"] = per.missed + per.reclassified
            per["same"] = 1 - per.changed_or_missed
            idx = rng.integers(0, len(per), size=(2000, len(per)))
            row = {"endpoint": ep, "analysis": name, "patients_with_ai_segmentation": len(per),
                   "missed_n": int(per.missed.sum()), "detected_n": int((per.missed == 0).sum()),
                   "reclassified_among_detected": float(per.reclassified[per.missed == 0].mean())}
            for c in ("missed", "same", "reclassified", "changed_or_missed"):
                v = per[c].to_numpy()
                row[c] = v.mean()
                row[f"{c}_lo"], row[f"{c}_hi"] = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(P / "ai_outcome_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
