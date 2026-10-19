"""Computational cost: wall-clock time of every analysis step, for the runtime table.

Paper:    Methods, Suppl. Table S30
Pipeline: python pipeline.py run runtime_table   (stage 'reporting')

pipeline.py appends the wall-clock time of every command it runs to
data/interim/runtime_log.csv. This script sums the latest run of each step,
groups the steps by stage, and adds the per-patient cost of the image steps
(feature extraction at five segmentation positions, foundation-model
inference), measured separately because those steps read the CT images and
are run once (data/interim/feature_timing.json).

Outputs: data/processed/runtime_summary.csv, data/processed/runtime_summary.json
Run:     python scripts/reporting/runtime_summary.py
"""
from __future__ import annotations

import json
import os
import pathlib
import platform

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG = ROOT / "data" / "interim" / "runtime_log.csv"
FEAT = ROOT / "data" / "interim" / "feature_timing.json"
OUT = ROOT / "data" / "processed"
RENAMED_STEPS = {"reviewer_checks": "sensitivity_checks"}  # logs written before a step was renamed


def main():
    import pipeline
    stage_of = {st.name: stage for stage, steps in pipeline.STAGES.items() for st in steps}
    steps_by_name = {st.name: st for steps in pipeline.STAGES.values() for st in steps}
    log = pd.read_csv(LOG)
    log["step"] = log.step.replace(RENAMED_STEPS)
    # the latest complete run of each step: its last n commands, where n is the step's number of commands
    rows = []
    for step, g in log.groupby("step", sort=False):
        if step not in steps_by_name:  # e.g. manuscript steps, which are not part of this pipeline
            continue
        ncmd = len(steps_by_name[step].commands)
        rows.append({"stage": stage_of[step], "step": step, "wall_s": g.wall_s.tail(ncmd).sum()})
    df = pd.DataFrame(rows)
    df["wall_min"] = df.wall_s / 60
    df.to_csv(OUT / "runtime_summary.csv", index=False)
    f = json.loads(FEAT.read_text()) if FEAT.exists() else {}
    res = {"machine": f"{platform.processor()}; {os.cpu_count()} logical CPUs",
           "total_analysis_hours": float(df.wall_s.sum() / 3600),
           "by_stage_minutes": df.groupby("stage").wall_min.sum().round(1).to_dict(),
           "longest_steps_minutes": df.nlargest(5, "wall_s").set_index("step").wall_min.round(1).to_dict(), **f}
    (OUT / "runtime_summary.json").write_text(json.dumps(res, indent=2))
    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
