"""Discovery cohort: the ±1-mm flag against AI-induced reclassification.

Paper:    Fig. 3g, Fig. 4g, Suppl. Table S15
Pipeline: python pipeline.py run ai_models   (stage 'ai')

For each held-out patient (models trained on expert segmentations; predictions from
run_ai_segmentation_models.py), the self-check flags the patient as segmentation-sensitive if
the decisions from the expert segmentation and its -1 mm and +1 mm versions do not
all agree. We then compare the expert-vs-AI decision flip rate in flagged and
unflagged patients (odds ratio with a 95% interval and P value from 2,000 patient
bootstrap resamples, the same inference as for the reader analyses), and report sensitivity, specificity
and the share of all AI-induced flips that the flag captures.

Output: data/processed/instability_flag.csv
Run:    python scripts/ai_segmentation/run_instability_flag.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import clustered, config  # noqa: E402


def main():
    d = pd.read_csv(config.PROCESSED_DIR / "bamf_paired_predictions.csv")
    rows = []
    for ep, g in d.groupby("endpoint"):
        hi = {k: g[f"p_{k}"] >= g.threshold for k in ("expert", "-1mm", "+1mm", "ai")}
        flag = ~((hi["expert"] == hi["-1mm"]) & (hi["expert"] == hi["+1mm"]))
        flip = hi["expert"] != hi["ai"]
        a, b = int((flag & flip).sum()), int((flag & ~flip).sum())
        c, e = int((~flag & flip).sum()), int((~flag & ~flip).sum())
        cb = clustered.cluster_bootstrap_or(flag.to_numpy(), flip.to_numpy(), g.case_id.to_numpy(), seed=12)
        rows.append({"endpoint": ep, "n": len(g), "flagged": int(flag.sum()),
                     "flip_rate_flagged": a / max(a + b, 1), "flip_rate_unflagged": c / max(c + e, 1),
                     **cb, "sensitivity": a / max(a + c, 1), "specificity": e / max(b + e, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(config.PROCESSED_DIR / "instability_flag.csv", index=False)
    pd.set_option("display.width", 160)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
