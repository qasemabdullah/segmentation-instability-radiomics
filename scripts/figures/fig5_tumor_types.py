"""Fig. 5: segmentation-dependent classifications across tumor types.

Paper:    Figs 1-7, Suppl. Fig. S5
Pipeline: python pipeline.py run main_figures   (stage 'figures')

  a  one LIDC-IDRI nodule with its four radiologists' segmentations
  b  size of segmentation disagreement in each setting (average symmetric surface distance)
  c  decisions changed by segmentation choice vs by refitting the model, in every setting,
     with each model's discrimination
  d  LIDC: error rate of nodules whose decision is stable vs reader-dependent
  e  the +-1 mm self-check against disagreement with independent readers
  f  LIDC: discrimination vs reader stability for radiomics and foundation-model features
  g  colorectal liver metastases: concordance from each segmentation (no prognostic signal)

Reads data/external/lidc, data/external/crlm and data/processed outputs.
Run:    python scripts/figures/fig5_tumor_types.py
Output: outputs/figures/main/figure5_generalization.png / .pdf
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.figures.main_figures import (AMBER, BLUE, DATA, GREEN, GRID, INK, MUTED, NAVY, ORANGE, PURPLE, RED, SUB,  # noqa: E402
                               TEAL, letter, save)

LIDC, CRLM = ROOT / "data" / "external" / "lidc", ROOT / "data" / "external" / "crlm"
READER_COLORS = [BLUE, ORANGE, GREEN, PURPLE]
RNG = np.random.default_rng(0)


def example_nodule():
    pairs = pd.read_csv(LIDC / "lidc_pairs.csv")
    ann = pd.read_csv(LIDC / "annotations.csv")
    four = ann.groupby(["PatientID", "nodule"]).size()
    four = four[four == 4].index
    pi = pairs.set_index(["PatientID", "nodule"])
    med = pi.loc[pi.index.intersection(four)].groupby(level=[0, 1]).assd_mm.median()
    vol = ann.groupby(["PatientID", "nodule"]).pylidc_volume.median().loc[med.index]
    cand = med[(vol > 1500) & (vol < 8000)]
    key = (cand - cand.median()).abs().idxmin()
    return key, float(cand.loc[key])


def panel_a(ax):
    (pid, k), assd = example_nodule()
    d = np.load(LIDC / "crops" / f"{pid}_n{k}.npz")
    z = int(np.argmax(d["masks"].sum(axis=(0, 2, 3))))
    img = np.clip((d["ct"][z].astype(float) + 1000) / 1400, 0, 1)
    ys, xs = np.nonzero(d["masks"][:, z].any(axis=0))
    pad = 12
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
    ax.imshow(img[y0:y1, x0:x1], cmap="gray", interpolation="lanczos")
    for i, m in enumerate(d["masks"][:, z]):
        ax.contour(m[y0:y1, x0:x1], levels=[0.5], colors=[READER_COLORS[i % 4]], linewidths=1.0)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    sp = float(d["spacing"][0])
    ax.plot([2, 2 + 5 / sp], [y1 - y0 - 3] * 2, color="white", lw=1.5)
    ax.text(2 + 2.5 / sp, y1 - y0 - 5, "5 mm", color="white", fontsize=5.5, va="bottom", ha="center",
            path_effects=[patheffects.withStroke(linewidth=1.5, foreground="black")])
    ax.set_title(f"LIDC nodule, four readers\n(median ASSD {assd:.2f} mm)", fontsize=6.6)


def panel_b(ax):
    ia = pd.read_csv(DATA / "interobserver_surface_distance.csv").assd_mm
    la = pd.read_csv(LIDC / "lidc_pairs.csv").assd_mm
    sa = pd.read_csv(DATA / "bamf_agreement.csv")
    sa = sa[(sa.source == "ai") & sa.detected].assd_mm
    ca = pd.read_csv(CRLM / "crlm_agreement.csv").ai_assd_mm.dropna()
    groups = [("NSCLC\nreader", ia, NAVY), ("Nodule\nreader", la, BLUE),
              ("NSCLC\nAI", sa, RED), ("Liver\nAI", ca, AMBER)]
    for i, (lab, v, c) in enumerate(groups):
        v = np.clip(v.to_numpy(), 0.05, 25)
        ax.scatter(i + RNG.uniform(-0.22, 0.22, len(v)), v, s=2.5, color=c, alpha=0.25, lw=0, rasterized=True)
        q = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.3, i + 0.3], [q[1]] * 2, color=INK, lw=1.4)
        ax.plot([i, i], [q[0], q[2]], color=INK, lw=0.8)
        ax.text(i, 42, f"{q[1]:.2f}", ha="center", fontsize=5.8, color=SUB)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 70)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=5.2)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("Average symmetric surface distance (mm)")
    ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))


def lidc_pairwise(tag):
    """Per seed: mean AUC over reader contours, pairwise reader disagreement."""
    o = pd.read_csv(LIDC / f"lidc_oof{tag}.csv.gz")
    o = o[o.perturbation_mm == 0].assign(d=lambda x: x.p >= 0.5)
    rows = []
    for s, g in o.groupby("seed"):
        w = g.pivot_table(index=["PatientID", "nodule"], columns="reader", values="d", aggfunc="first")
        pw = g.pivot_table(index=["PatientID", "nodule"], columns="reader", values="p", aggfunc="first")
        y = g.drop_duplicates(["PatientID", "nodule"]).set_index(["PatientID", "nodule"]).y.loc[w.index]
        dis = np.nanmean([np.nanmean((w[i] != w[j])[w[i].notna() & w[j].notna()])
                          for i, j in itertools.combinations(w.columns, 2)])
        auc = np.mean([roc_auc_score(y[pw[r].notna()], pw[r].dropna()) for r in pw.columns if pw[r].notna().sum() > 20])
        rows.append({"seed": s, "auc": auc, "disagree": dis})
    return pd.DataFrame(rows), o


def panel_c(ax):
    L = json.loads((LIDC / "lidc_model_summary.json").read_text())["threshold_0.5"]
    F = json.loads((LIDC / "lidc_model_summary_fm.json").read_text())["threshold_0.5"]
    C = json.loads((CRLM / "crlm_results.json").read_text())
    sig = {"recurrence": "AUC 0.56", "egfr": "AUC 0.60", "os": "C 0.62"}
    nm = {"recurrence": "recurrence", "egfr": "EGFR", "os": "OS"}
    # NSCLC rows: same patients, same models and threshold rule for both quantities, with 95% bootstrap
    # intervals (scripts/discovery/run_same_patient_comparison.py)
    dc = pd.read_csv(DATA / "decomposition_summary.csv").set_index(["endpoint", "comparison"])
    rows = []
    for ep in ("egfr", "recurrence", "os"):
        r = dc.loc[(ep, "another radiation oncologist (cohort median)")]
        rows.append((f"NSCLC readers, {nm[ep]}", r.segmentation_rate, r.model_rate, sig[ep], NAVY,
                     (r.segmentation_lo, r.segmentation_hi)))
    for ep in ("egfr", "recurrence", "os"):
        r = dc.loc[(ep, "AI")]
        rows.append((f"NSCLC AI, {nm[ep]}", r.segmentation_rate, r.model_rate, sig[ep], RED,
                     (r.segmentation_lo, r.segmentation_hi)))
    rows.append(("Nodule readers, radiomics", L["pairwise_disagreement"], L["refit_noise"], f"AUC {L['auc_by_reader_mean']:.2f}", BLUE, None))
    rows.append(("Nodule readers, foundation", F["pairwise_disagreement"], F["refit_noise"], f"AUC {F['auc_by_reader_mean']:.2f}", TEAL, None))
    for ep, lab in (("os", "OS"), ("dfs", "DFS")):
        rows.append((f"Liver met. AI, {lab}", C[ep]["flips_expert_vs_ai"], C[ep]["refit_noise"], f"C {C[ep]['c_expert']:.2f}", AMBER, None))
    for i, (lab, cont, refit, s, c, ci) in enumerate(rows[::-1]):
        ax.plot([100 * refit, 100 * cont], [i, i], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        if ci is not None:
            ax.plot([100 * ci[0], 100 * ci[1]], [i, i], color=c, lw=0.8, zorder=2)
        ax.scatter(100 * refit, i, s=18, facecolor="white", edgecolor=MUTED, lw=0.9, zorder=2)
        ax.scatter(100 * cont, i, s=20, color=c, zorder=3)
        ax.text(52, i, s, va="center", fontsize=5.6, color=SUB)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=5.8)
    ax.set_xlim(-2, 60)
    ax.set_xlabel("Patients reclassified (%)")
    ax.scatter([], [], s=20, color=INK, label="Segmentation choice")
    ax.scatter([], [], s=18, facecolor="white", edgecolor=MUTED, label="Model-development variability")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8)


def panel_d(ax):
    """Error rate among nodules with the same decision for every reader vs a reader-dependent
    decision; per-seed rates averaged over seeds (as in the Supplementary Table), 95% interval
    from 1,000 bootstrap resamples of patients (a patient's nodules are correlated and stay together,
    as in every other LIDC interval; Methods)."""
    for j, (tag, lab, c) in enumerate((("", "Radiomics", BLUE), ("_fm", "Foundation", TEAL))):
        o = pd.read_csv(LIDC / f"lidc_oof{tag}.csv.gz")
        o = o[o.perturbation_mm == 0].assign(d=lambda x: x.p >= 0.5)
        per = o.groupby(["seed", "PatientID", "nodule"]).agg(call=("d", "mean"), y=("y", "first")).reset_index()
        per["unstable"] = (per.call > 0) & (per.call < 1)
        per["wrong"] = (per.call >= 0.5) != per.y
        keys = per[["PatientID"]].drop_duplicates().reset_index(drop=True)

        def rates(pp):
            r = pp.groupby(["seed", "unstable"]).wrong.mean().unstack()
            return r.mean().reindex([False, True]).to_numpy()

        est = rates(per)
        boots = []
        for b in range(1000):
            k = keys.sample(len(keys), replace=True, random_state=b)
            boots.append(rates(k.merge(per, on="PatientID")))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        x = np.array([0, 1]) + (j - 0.5) * 0.36
        ax.bar(x, 100 * est, width=0.34, color=c, alpha=0.9, label=lab, edgecolor="white", lw=0.5)
        for xi, e, l_, h_ in zip(x, est, lo, hi):
            ax.plot([xi, xi], [100 * l_, 100 * h_], color=INK, lw=0.8)
            ax.text(xi, 100 * h_ + 1.5, f"{100 * e:.0f}%", ha="center", fontsize=5.6)
    ax.set_xticks([0, 1], ["Same for\nall readers", "Depends on\nthe reader"], fontsize=5.8)
    ax.set_xlabel("Classification", fontsize=6.3)
    ax.set_ylabel("Disagreement with consensus label (%)")
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, loc="upper left")
    # the headline comparison is unadjusted; print the confidence-adjusted result beside it
    dep = json.loads((LIDC / "lidc_dependence_check.json").read_text())
    o_, (l_, h_) = dep["point"]["or_dependent_adj_conf"], dep["ci95"]["or_dependent_adj_conf"]
    ax.text(0.03, 0.62, f"Bars unadjusted.\nAdjusted for\nconfidence:\nOR {o_:.2f}\n({l_:.2f}–{h_:.2f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=INK)


def panel_e(ax):
    L = json.loads((LIDC / "lidc_model_summary.json").read_text())["threshold_0.5"]
    rd = pd.read_csv(DATA / "reader_decision_summary.csv").set_index("endpoint")
    rows = [("Nod.", L["disagree_if_flagged"], L["disagree_if_not_flagged"], L["flag_odds_ratio"], BLUE)]
    for ep, lab in (("egfr", "EGFR"), ("recurrence", "Recur."), ("os", "OS")):
        r = rd.loc[ep]
        rows.append((lab, r.disagree_if_flagged, r.disagree_if_not_flagged, r.odds_ratio, NAVY))
    for i, (lab, a, b, orr, c) in enumerate(rows):
        ax.bar(i - 0.18, 100 * a, 0.34, color=c)
        ax.bar(i + 0.18, 100 * b, 0.34, color="white", edgecolor=c, lw=0.8)
        ax.text(i, 100 * max(a, b) + 2, f"OR\n{orr:.1f}", ha="center", fontsize=5.4, linespacing=0.9)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=5.8)
    ax.set_ylabel("Disagreement with other readers (%)")
    ax.text(0, -0.2, "LIDC", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.6, color=SUB)
    ax.text(2, -0.2, "NSCLC readers", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.6, color=SUB)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=NAVY, label="Flagged by own ±1-mm check"),
                       Patch(facecolor="white", edgecolor=NAVY, label="Not flagged")],
              frameon=False, loc="upper left", fontsize=5.5, ncol=1, handlelength=1.2)


def panel_f(ax):
    for tag, lab, c in (("", "Radiomics", BLUE), ("_fm", "Foundation model", TEAL)):
        s, _ = lidc_pairwise(tag)
        ax.scatter(s.auc, 100 * s.disagree, s=12, color=c, alpha=0.8, lw=0, label=lab)
        J = json.loads((LIDC / f"lidc_model_summary{tag}.json").read_text())["threshold_0.5"]
        ax.axhline(100 * J["refit_noise"], color=c, lw=0.7, ls=(0, (3, 2)))
    for tag, c in (("", BLUE), ("_fm", TEAL)):
        rn = json.loads((LIDC / f"lidc_model_summary{tag}.json").read_text())["threshold_0.5"]["refit_noise"]
        ax.text(0.998, 100 * rn + 0.15, "model-development variability", fontsize=5.4, color=c, ha="right")
    ax.set_xlabel("AUC (consensus malignancy rating)")
    ax.set_ylabel("Reader pairs with different classifications (%)")
    ax.set_xlim(0.84, 1.0)
    ax.set_ylim(0, 8)
    ax.legend(frameon=False, loc="upper right")


def panel_g(ax):
    from sksurv.metrics import concordance_index_censored
    o = pd.read_csv(CRLM / "crlm_oof.csv.gz")
    ai_pts = set(o[o.variant == "ai"].PatientID)
    ep_cols = {"os": ("overall_survival_months", "vital_status"), "dfs": ("months_to_dfs_progression", "vital_status_dfs")}
    variants = [("expert", "Exp."), ("expert-1", "−1"), ("expert+1", "+1"), ("ai", "AI")]
    for j, (ep, c) in enumerate((("os", GREEN), ("dfs", TEAL))):
        t, e = ep_cols[ep]
        g0 = o[(o.endpoint == ep) & o.PatientID.isin(ai_pts)]
        for i, (v, _) in enumerate(variants):
            cs = [concordance_index_censored(g[e].astype(bool), g[t], g.risk)[0]
                  for _, g in g0[g0.variant == v].groupby("seed")]
            x = i + (j - 0.5) * 0.3
            ax.scatter(x + RNG.uniform(-0.06, 0.06, len(cs)), cs, s=5, color=c, alpha=0.6, lw=0)
            ax.plot([x - 0.12, x + 0.12], [np.mean(cs)] * 2, color=INK, lw=1.1)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(range(len(variants)), [v[1] for v in variants])
    ax.set_ylabel("Concordance index")
    ax.set_xlabel("Segmentation", fontsize=6.3)
    ax.set_ylim(0.4, 0.7)
    ax.scatter([], [], s=8, color=GREEN, label="OS")
    ax.scatter([], [], s=8, color=TEAL, label="DFS")
    ax.legend(frameon=False, loc="upper right", ncol=2)


def main():
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.42)
    top = gs[0].subgridspec(1, 3, width_ratios=[0.75, 1.05, 1.45], wspace=1.05)
    bot = gs[1].subgridspec(1, 4, width_ratios=[1, 1.1, 1, 1], wspace=0.55)
    axes = {k: fig.add_subplot(s) for k, s in zip("abc", top)}
    axes.update({k: fig.add_subplot(s) for k, s in zip("defg", bot)})
    for k, f in zip("abcdefg", (panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g)):
        f(axes[k])
    for k in "abcdefg":
        letter(fig, axes[k], k, dx=-0.05 if k not in "c" else -0.16)
    save(fig, "figure5_generalization")


if __name__ == "__main__":
    main()
