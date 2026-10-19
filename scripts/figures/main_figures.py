"""Figs 2, 3, 4, 6 and 7, and Supplementary Fig. S5 (ROC curves), in the shared journal style.

Paper:    Figs 1-7, Suppl. Fig. S5
Pipeline: python pipeline.py run main_figures   (stage 'figures')

  key 2    Fig. 2  Radiomic models in the discovery cohort (AUCs, Kaplan-Meier,
                  incremental concordance).
  key 3    Fig. 3  Millimetre adjustment of the segmentation: features, accuracy,
                  phenotypes and reclassification.
  key 4    Fig. 4  Radiation oncologists' and AI segmentations; PyRadiomics.
  key 5    Fig. 6  Feature importance versus stability, and mitigation strategies.
  key 6    Fig. 7  Prognostic value, transcriptomic correlates and the external cohort.
  key roc  Supplementary Fig. 5  ROC curves for every endpoint and feature set.
(Fig. 1 is drawn by fig1_study_design.py, Fig. 5 by fig5_tumor_types.py.)

Every value is read from this repository's saved pipeline output
(data/processed) or recomputed from it with the pipeline's own functions.

Run: python scripts/figures/main_figures.py [2 3 4 5 6 roc]
Output: outputs/figures/main/figure{2,3,4,6,7}_*.png / .pdf, outputs/figures/supplementary/cvdetail.png
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures" / "main"

from scripts.figures.figure_style import (AMBER, BLUE, DAYS_PER_MONTH, ENDPOINT, FAMILY, GREEN, GRID, INK, MM, MM_SFX,  # noqa: E402,F401
                          MUTED, NAVY, NICE, ORANGE, PINK, PURPLE, RED, SUB, TEAL, family, letter, nice, p_fmt, save)


def fold_scores(name, col):
    return pd.read_csv(DATA / f"{name}.csv")[col].to_numpy()


# shared survival helpers

def km_panel(ax, groups, colors, labels, ticks=(0, 24, 48, 72, 96), at_risk_ax=None, p=None):
    from lifelines import KaplanMeierFitter
    for (t, e), c, lab in zip(groups, colors, labels):
        kmf = KaplanMeierFitter().fit(t, e)
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=c, lw=1.4, label=f"{lab} (n = {len(t)})")
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post", color=c, alpha=0.10, lw=0)
        cens = t[~e.astype(bool)]
        ax.plot(cens, kmf.survival_function_at_times(cens), "|", color=c, ms=3, mew=0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, ticks[-1] + 6)
    ax.set_xticks(ticks)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower left", frameon=False, handlelength=1.2, borderaxespad=0.2)
    if p is not None:
        ax.text(0.97, 0.97, f"Log-rank {p_fmt(p)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    if at_risk_ax is not None:
        at_risk_ax.set_xlim(ax.get_xlim())
        at_risk_ax.set_ylim(-0.5, len(groups) - 0.5)
        at_risk_ax.axis("off")
        for i, ((t, _), c) in enumerate(zip(groups, colors)):
            y = len(groups) - 1 - i
            for tk in ticks:
                at_risk_ax.text(tk, y, str(int((t >= tk).sum())), ha="center", va="center", fontsize=6.2, color=c)
        at_risk_ax.text(-0.02, 1.05, "No. at risk", transform=at_risk_ax.transAxes, ha="left", va="bottom",
                        fontsize=6.2, color=SUB, style="italic")


# Figure 2 -- baseline prediction

FS = [("", "Rad.", NAVY), ("__radiomics_clinical", "+Clin.", BLUE),
      ("__radiomics_clinical_rnaseq", "+Clin.\n+RNA", GREEN), ("__embeddings", "CNN", AMBER),
      ("__radiomics_embeddings", "Rad.\n+CNN", PINK)]


def fig2():
    from lifelines.statistics import logrank_test
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1.25], hspace=0.42, wspace=0.42,
                          left=0.07, right=0.985, top=0.95, bottom=0.08)

    # a -- AUC by feature set, one facet per classification target
    sub = gs[0, :].subgridspec(1, 3, wspace=0.08, width_ratios=[5.6, 5.6, 2.2])
    rng = np.random.RandomState(1)
    axes_a = []
    for k, (tgt, lab) in enumerate([("recurrence", "Recurrence"), ("egfr", "EGFR mutation"), ("kras", "KRAS mutation")]):
        ax = fig.add_subplot(sub[0, k], sharey=axes_a[0] if axes_a else None)
        axes_a.append(ax)
        sets = FS if tgt != "kras" else FS[:1]
        xs = np.arange(len(sets) + 1)
        for i, (sfx, name, c) in enumerate(sets):
            f = DATA / f"modeling_results_{tgt}{sfx}.csv"
            v = pd.read_csv(f)["auc"].to_numpy()
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=7, color=c, alpha=0.45, lw=0, zorder=2)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=0), fmt="o", color=c, ms=4.2, mec="white", mew=0.6,
                        elinewidth=1.1, capsize=2, zorder=3)
        v = fold_scores(f"modeling_results_{tgt}__volume_only", "auc")
        i = len(sets)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=7, color=MUTED, alpha=0.45, lw=0, zorder=2)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=0), fmt="o", color=MUTED, mfc="white", ms=4.2, mew=1.0,
                    elinewidth=1.1, capsize=2, zorder=3)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(xs, [s[1] for s in sets] + ["Vol.\nonly"], fontsize=6.2)
        ax.set_xlim(-0.6, len(sets) + 0.6)
        ax.set_title(lab, fontsize=7.2, color=ENDPOINT[tgt])
        if k == 0:
            ax.set_ylabel("Nested-CV AUC")
        else:
            ax.tick_params(labelleft=False)
    axes_a[0].set_ylim(0.2, 1.0)
    axes_a[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    letter(fig, axes_a[0], "a")

    # b, c -- Kaplan-Meier by median out-of-fold risk
    for col, (fname, lab, let) in enumerate([("survival_predictions", "Overall survival", "b"),
                                             ("survival_rfs_predictions", "Recurrence-free survival", "c")]):
        d = pd.read_csv(DATA / f"{fname}.csv")
        d["m"] = d["duration_days"] / DAYS_PER_MONTH
        hi = d["risk_score"] >= d["risk_score"].median()
        p = logrank_test(d.m[hi], d.m[~hi], d.event[hi], d.event[~hi]).p_value
        cell = gs[1, col].subgridspec(2, 1, height_ratios=[4.2, 1.0], hspace=0.55)
        ax = fig.add_subplot(cell[0])
        ar = fig.add_subplot(cell[1])
        km_panel(ax, [(d.m[~hi].to_numpy(), d.event[~hi].to_numpy()), (d.m[hi].to_numpy(), d.event[hi].to_numpy())],
                 [BLUE, ORANGE], ["Low risk", "High risk"], at_risk_ax=ar, p=p)
        ax.set_ylabel(f"{lab} probability" if col == 0 else "RFS probability")
        ax.set_xlabel("Months from CT", labelpad=1)
        ax.set_title(lab, fontsize=7.2)
        letter(fig, ax, let)

    # d -- incremental concordance
    ax = fig.add_subplot(gs[1, 2])
    steps = ["Clinical", "+ Volume", "+ Radiomics"]
    for ep, c, off in [("os", GREEN, -0.06), ("rfs", TEAL, 0.06)]:
        s = pd.read_csv(DATA / f"survival_staircase_{ep}.csv")
        x = np.arange(3) + off
        ax.errorbar(x, s["mean_concordance"], yerr=s["std_concordance"], color=c, marker="o", ms=4, mec="white",
                    mew=0.6, lw=1.3, capsize=2, label="OS" if ep == "os" else "RFS")
    for ep, fname, c in [("os", "survival_results", GREEN), ("rfs", "survival_rfs_results", TEAL)]:
        v = fold_scores(fname, "concordance")
        ax.scatter(np.full(len(v), 3.1 if ep == "os" else 3.3), v, s=8, color=c, alpha=0.5, lw=0)
        ax.errorbar(3.1 if ep == "os" else 3.3, v.mean(), yerr=v.std(ddof=0), fmt="D", color=c, ms=3.8,
                    mec="white", mew=0.6, capsize=2)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(2.6, color=GRID, lw=0.8)
    ax.set_xticks([0, 1, 2, 3.2], steps + ["Radiomics\nonly"], fontsize=6.1)
    ax.set_ylim(0.3, 0.85)
    ax.set_ylabel("Nested-CV concordance index")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    ax.set_title("Incremental discrimination", fontsize=7.2)
    letter(fig, ax, "d")
    save(fig, "figure2_baseline_prediction")


# Figure 3 -- segmentation perturbation

def _cluster_by_mm():
    from nsclc_pipeline.phenotypes import bootstrap_ari, fit_reference_phenotypes, reassignment_by_shift
    ci = reassignment_by_shift()
    _df, _fc, _sc, _imp, labels, _cent, Xz = fit_reference_phenotypes()
    return ci, bootstrap_ari(Xz, labels)


def fig3():
    from scripts.figures.fig1_study_design import MM_COLORS, PIPE_CASE, compute_perturbation_contours
    gray, masks = compute_perturbation_contours(PIPE_CASE)
    comp = pd.read_csv(DATA / "interobserver_vs_synthetic_comparison.csv")
    ci, boot = _cluster_by_mm()

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.6, 1.0, 1.05], hspace=0.6, wspace=0.55,
                          left=0.07, right=0.985, top=0.965, bottom=0.06)

    # a -- real contours
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(gray, cmap="gray", interpolation="bilinear")
    for mm in (-2, -1, 1, 2, 0):
        ax.contour(masks[mm], levels=[0.5], colors=[INK if mm == 0 else MM_COLORS[mm]],
                   linewidths=1.5 if mm == 0 else 1.0, linestyles=["solid"] if mm == 0 else [(0, (3, 1.3))])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(True)
    hs = [Line2D([], [], color=INK if mm == 0 else MM_COLORS[mm], lw=1.5 if mm == 0 else 1.0,
                 ls="-" if mm == 0 else (0, (3, 1.3))) for mm in (2, 1, 0, -1, -2)]
    ax.legend(hs, ["+2 mm", "+1 mm", "Expert", "−1 mm", "−2 mm"], loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="none", handlelength=1.8, borderaxespad=0.3, fontsize=5.9)
    ax.set_title(f"Segmentation adjusted by ±1 and ±2 mm ({PIPE_CASE})", fontsize=7.2)
    letter(fig, ax, "a", dx=-0.02)

    # b -- per-feature instability
    ax = fig.add_subplot(gs[0, 2:4])
    c2 = comp.sort_values("synthetic_perturbation_median_rel_change", ascending=True)
    y = np.arange(len(c2))
    ax.barh(y, c2["synthetic_perturbation_median_rel_change"], color=[FAMILY[family(f)] for f in c2["feature"]],
            height=0.7)
    ax.set_xscale("log")
    ax.set_yticks(y, [nice(f) for f in c2["feature"]], fontsize=5.6)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("Median relative change, −1 vs +1 mm (log)")
    ax.legend([Line2D([], [], color=c, lw=4) for c in FAMILY.values()], ["Shape", "First-order", "GLCM texture"],
              frameon=False, loc="lower right", handlelength=1.0)
    ax.set_title("Feature instability (n = 144)", fontsize=7.2)
    letter(fig, ax, "b", dx=-0.095)

    # c-f -- sensitivity curves, one per paradigm
    specs = [("egfr", "EGFR AUC", lambda m: "modeling_results_egfr" if m == 0 else f"modeling_results_egfr__{MM_SFX[m]}", "auc"),
             ("recurrence", "Recurrence AUC", lambda m: "modeling_results_recurrence" if m == 0 else f"modeling_results_recurrence__{MM_SFX[m]}", "auc"),
             ("os", "OS concordance", lambda m: "survival_results" if m == 0 else f"survival__{MM_SFX[m]}_results", "concordance")]
    # seed-averaged curves: each point is one of 20 fold seeds (thin lines join a seed across shifts)
    rep = pd.read_csv(DATA / "repeated_cv_scores.csv")
    for k, (key, lab, fn, col) in enumerate(specs):
        ax = fig.add_subplot(gs[1, k])
        w = rep[rep.endpoint == key].pivot(index="seed", columns="mm", values="score")[MM]
        mu, sd = w.mean().to_numpy(), w.std(ddof=0).to_numpy()
        c = ENDPOINT[key]
        for _, row in w.iterrows():
            ax.plot(MM, row.to_numpy(), color=c, lw=0.4, alpha=0.25)
        ax.fill_between(MM, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
        ax.plot(MM, mu, color=c, lw=1.6, marker="o", ms=3.5, mec="white", mew=0.5)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(0.2, 1.0)
        ax.set_xticks(MM, ["−2", "−1", "0", "+1", "+2"])
        ax.set_xlabel("Boundary shift (mm)")
        ax.set_ylabel(lab)
        ax.set_title((lab.split()[0] if key != "os" else "Survival (OS)") + "\nmodel re-developed", fontsize=7.2, color=c)
        letter(fig, ax, "cde"[k])
    ax = fig.add_subplot(gs[1, 3])
    ax.axhspan(boot.mean() - boot.std(), boot.mean() + boot.std(), color=GRID, alpha=0.9, lw=0)
    ax.axhline(boot.mean(), color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.plot(ci["mm"], ci["ari"], color=AMBER, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.5)
    for _, r in ci.iterrows():
        if r["mm"] != 0:
            ax.text(r["mm"] + (0.22 if r["mm"] < 0 else -0.05), r["ari"] - 0.035, f"{r['pct_reassigned']:.0f}%",
                    ha="left" if r["mm"] < 0 else "right", va="top", fontsize=5.8, color="#9a6b00")
    ax.text(2.3, boot.mean() - boot.std() - 0.015, "resampling baseline", ha="right", va="top", fontsize=5.6,
            color=SUB)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(MM, ["−2", "−1", "0", "+1", "+2"])
    ax.set_xlabel("Boundary shift (mm)")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_title("Phenotype clusters", fontsize=7.2, color=AMBER)
    letter(fig, ax, "f")

    # g -- reclassification: model development vs segmentation, same patients, models and threshold rule,
    # with 95% patient-bootstrap intervals (scripts/discovery/run_same_patient_comparison.py)
    ax = fig.add_subplot(gs[2, 0:2])
    dc = pd.read_csv(DATA / "decomposition_summary.csv").set_index(["endpoint", "comparison"])
    eps = [("recurrence", "Recurrence"), ("egfr", "EGFR"), ("os", "OS")]
    rd_ = "another radiation oncologist (cohort median)"
    srcs = [("Model development (discovery)", lambda e: dc.loc[(e, "±1 mm"), ["model_rate", "model_lo", "model_hi"]], GRID),
            ("±1-mm adjustment", lambda e: dc.loc[(e, "±1 mm"), ["segmentation_rate", "segmentation_lo", "segmentation_hi"]], "#f3b38f"),
            ("AI segmentation", lambda e: dc.loc[(e, "AI"), ["segmentation_rate", "segmentation_lo", "segmentation_hi"]], RED),
            ("Model development (reader cohort)", lambda e: dc.loc[(e, rd_), ["model_rate", "model_lo", "model_hi"]], "#b9c0cc"),
            ("Another radiation oncologist", lambda e: dc.loc[(e, rd_), ["segmentation_rate", "segmentation_lo", "segmentation_hi"]], TEAL)]
    w = 0.16
    for k, (lab, fn, c) in enumerate(srcs):
        v = np.array([100 * fn(e).to_numpy(dtype=float) for e, _ in eps])
        x = np.arange(len(eps)) + (k - 2) * w
        ax.bar(x, v[:, 0], width=w * 0.95, color=c, label=lab, edgecolor="white", lw=0.3,
               hatch="////" if "reader cohort" in lab else None)
        ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2)
        for xi, vi, hi in zip(x, v[:, 0], v[:, 2]):
            ax.text(xi, hi + 1, f"{vi:.0f}", ha="center", fontsize=5.0, color=SUB)
    ax.set_xticks(range(len(eps)), [e[1] for e in eps])
    ax.set_ylim(0, 74)
    ax.set_ylabel("Patients reclassified (%)")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left", handlelength=1.1, columnspacing=0.8)
    ax.set_title("Reclassification, model fixed:\nmodel development vs segmentation", fontsize=7.2)
    letter(fig, ax, "g", dx=-0.085)

    # h -- bootstrap resampling ARI distribution vs perturbed ARI
    ax = fig.add_subplot(gs[2, 2:4])
    ax.hist(boot, bins=np.linspace(0, 1, 21), color=GRID, edgecolor="white", lw=0.5, label="Bootstrap reclustering")
    top = ax.get_ylim()[1]
    for _, r in ci.iterrows():
        if r["mm"] != 0:
            mm = int(r["mm"])
            ax.axvline(r["ari"], color=MM_COLORS[mm], lw=1.3)
            ax.text(r["ari"] + (-0.012 if mm < 0 else 0.012), top * (0.97 if abs(mm) == 2 else 0.88),
                    f"{mm:+d} mm".replace("-", "−"), ha="right" if mm < 0 else "left", va="top", fontsize=5.8,
                    color=INK, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("Adjusted Rand index vs original clusters")
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title("Cluster stability: resampling vs boundary shift (mm)", fontsize=7.2)
    letter(fig, ax, "h")
    save(fig, "figure3_segmentation_perturbation")


# Figure 4 -- real interobserver validation

def fig4():
    from scripts.figures.fig1_study_design import compute_interobserver
    sd = pd.read_csv(DATA / "interobserver_surface_distance.csv")
    comp = pd.read_csv(DATA / "interobserver_vs_synthetic_comparison.csv")
    xc = pd.read_csv(DATA / "pyradiomics_crosscheck.csv")
    per = sd.groupby("patient")["assd_mm"].median().sort_values()
    lo_id, hi_id = per.index[len(per) // 2], per.index[-2]

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.05, 1.0], hspace=0.62, wspace=0.6,
                          left=0.07, right=0.985, top=0.97, bottom=0.055)
    rc = [BLUE, ORANGE, GREEN, PINK, AMBER]
    for k, pid in enumerate([lo_id, hi_id]):
        ax = fig.add_subplot(gs[0, k])
        gray, obs, _, _ = compute_interobserver(pid)
        ax.imshow(gray, cmap="gray", interpolation="bilinear")
        for i, o in enumerate(obs):
            ax.contour(o, levels=[0.5], colors=[rc[i]], linewidths=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(True)
        ax.set_title(f"{pid.replace('interobs', 'Patient ')}: median ASSD {per[pid]:.1f} mm", fontsize=6.6)
        if k == 0:
            letter(fig, ax, "a", dx=-0.02)
    ax.legend([Line2D([], [], color=c, lw=1.4) for c in rc], [f"Reader {i + 1}" for i in range(5)],
              loc="upper center", bbox_to_anchor=(-0.2, -0.03), ncol=5, frameon=False, handlelength=1.2,
              columnspacing=0.8)

    # b -- ASSD by patient
    ax = fig.add_subplot(gs[0, 2:4])
    order = per.index.tolist()
    for i, pid in enumerate(order):
        v = sd.loc[sd["patient"] == pid, "assd_mm"].to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.RandomState(i).uniform(-0.18, 0.18, len(v)), v, s=6,
                   color=TEAL, alpha=0.55, lw=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=INK, lw=1.0)
    for y, ls, lab in [(1, (0, (3, 2)), "1 mm"), (2, (0, (1, 1.5)), "2 mm")]:
        ax.axhline(y, color=ORANGE, lw=0.9, ls=ls)
        ax.text(len(order) - 0.4, y + 0.05, lab, ha="right", va="bottom", fontsize=6, color=ORANGE)
    m = sd["assd_mm"].mean()
    ax.set_xticks(range(len(order)), [p.replace("interobs", "") for p in order], fontsize=5.4, rotation=90)
    ax.set_xlabel("Patient (sorted by median disagreement)")
    ax.set_ylabel("Pairwise ASSD (mm)")
    ax.set_title(f"Disagreement between radiation oncologists (210 pairs, mean {m:.2f} mm)", fontsize=7.2)
    letter(fig, ax, "b")

    # c -- rank agreement real vs synthetic
    ax = fig.add_subplot(gs[1, 0:2])
    rr = comp["real_interobserver_median_rel_change"].rank(ascending=False)
    rs = comp["synthetic_perturbation_median_rel_change"].rank(ascending=False)
    rho, p = spearmanr(comp["real_interobserver_median_rel_change"], comp["synthetic_perturbation_median_rel_change"])
    ax.plot([0, 27], [0, 27], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.scatter(rs, rr, s=16, c=[FAMILY[family(f)] for f in comp["feature"]], edgecolor="white", lw=0.4, zorder=3)
    for f, x, yv in zip(comp["feature"], rs, rr):
        if x <= 4 or x >= 24 or abs(x - yv) >= 8:
            ax.annotate(nice(f), (x, yv), xytext=(4, -1), textcoords="offset points", fontsize=5.6, color=SUB)
    ax.set_xlim(0, 27.5)
    ax.set_ylim(0, 27.5)
    ax.set_xlabel("Instability rank, ±1-mm adjustment (1 = least stable)")
    ax.set_ylabel("Instability rank, radiation oncologists")
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}\n{p_fmt(p)}, n = {len(comp)} features", transform=ax.transAxes,
            va="top", fontsize=6.5)
    ax.legend([Line2D([], [], marker="o", ls="", color=c) for c in FAMILY.values()], ["Shape", "First-order", "GLCM"],
              frameon=False, loc="lower right", handletextpad=0.2)
    ax.set_title("Adjustment reproduces disagreement between clinicians", fontsize=7.2)
    letter(fig, ax, "c")

    # d -- custom vs PyRadiomics
    ax = fig.add_subplot(gs[1, 2:4])
    rho, p = spearmanr(xc["custom_extractor_instability"], xc["pyradiomics_instability"])
    # a feature with exactly zero change cannot sit on a log axis; floor it for display only
    xv = xc["custom_extractor_instability"].clip(lower=1e-3)
    yv = xc["pyradiomics_instability"].clip(lower=1e-3)
    lo, hi = min(xv.min(), yv.min()) * 0.6, max(xv.max(), yv.max()) * 1.6
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.scatter(xv, yv, s=16, c=[FAMILY[family(f)] for f in xc["feature"]], edgecolor="white", lw=0.4, zorder=3)
    for f, a, b in zip(xc["feature"], xv, yv):
        if f in ("firstorder_energy", "firstorder_mean_hu", "shape_volume_mm3", "texture_energy", "firstorder_min_hu"):
            lab = nice(f) + (" (no change)" if f == "firstorder_min_hu" else "")
            ax.annotate(lab, (a, b), xytext=(5, -3), textcoords="offset points", fontsize=5.8, color=SUB)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}\n{p_fmt(p)}, n = {len(xc)} features, 30 patients",
            transform=ax.transAxes, va="top", fontsize=6.5)
    ax.set_xlabel("Instability, study extractor (±1 mm)")
    ax.set_ylabel("Instability, PyRadiomics (±1 mm)")
    ax.set_title("Agreement between feature extractors", fontsize=7.2)
    letter(fig, ax, "d")
    fig4_ai_row(fig, gs)
    save(fig, "figure4_interobserver_validation")


def _ai_example(case):
    import SimpleITK as sitk
    from nsclc_pipeline import config, viz
    rad = pd.read_csv(config.RADIOMICS_OUT).set_index("case_id")
    ct = viz.load_ct_volume(ROOT / "data" / "interim" / "dicom" / rad.loc[case, "ct_series_uid"])
    e = sitk.GetArrayFromImage(viz.load_mask_on_ct(config.PROCESSED_DIR / "masks" / f"{case}_seg.nii.gz", ct)) > 0
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(config.PROCESSED_DIR / "masks_bamf" / f"{case}_ai.nii.gz"))) > 0
    z = int(np.argmax(e.reshape(e.shape[0], -1).sum(1)))
    ys, xs = np.where(e[z] | a[z])
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    h = int(max(np.ptp(ys), np.ptp(xs)) * 0.65) + 8
    sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
    ct_arr = sitk.GetArrayFromImage(ct)[z]
    return viz.window_slice(ct_arr, *viz.WINDOW_PRESETS["Lung"])[sl], e[z][sl], a[z][sl]


def fig4_ai_row(fig, gs):
    """Row 3: a public nnU-Net segmentation (BAMF/AIMI) as a second contour source."""
    ag = pd.read_csv(DATA / "bamf_agreement.csv")
    ag = ag[ag.source == "ai"].copy()
    det = ag[ag.detected]
    ex = [det.iloc[(det.dice - det.dice.quantile(0.8)).abs().argmin()].case_id,
          det.iloc[(det.dice - det.dice.quantile(0.2)).abs().argmin()].case_id]
    for k, case in enumerate(ex):
        ax = fig.add_subplot(gs[2, k])
        g, e, a = _ai_example(case)
        ax.imshow(g, cmap="gray", interpolation="bilinear")
        ax.contour(e, levels=[0.5], colors=[INK], linewidths=1.1)
        ax.contour(a, levels=[0.5], colors=[RED], linewidths=1.1, linestyles=[(0, (3, 1.2))])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(True)
        d = float(det.loc[det.case_id == case, "dice"].iloc[0])
        ax.set_title(f"{case}: Dice {d:.2f}", fontsize=6.6)
        if k == 0:
            letter(fig, ax, "e", dx=-0.02)
    ax.legend([Line2D([], [], color=INK, lw=1.2), Line2D([], [], color=RED, lw=1.2, ls=(0, (3, 1.2)))],
              ["Expert", "AI (nnU-Net)"], loc="upper center", bbox_to_anchor=(-0.2, -0.03), ncol=2, frameon=False)

    # f -- agreement vs tumor size
    ax = fig.add_subplot(gs[2, 2])
    from nsclc_pipeline import config
    vox_ml = pd.read_csv(config.RADIOMICS_OUT).set_index("case_id")["shape_volume_mm3"] / 1000
    ag["vol"] = ag.case_id.map(vox_ml)
    ax.scatter(det.case_id.map(vox_ml), det.dice, s=9, color=RED, alpha=0.7, lw=0, label=f"Detected (n = {len(det)})")
    miss = ag[~ag.detected]
    ax.scatter(miss.vol, np.zeros(len(miss)) - 0.04, s=14, marker="x", color=INK, lw=0.8, label=f"Missed (n = {len(miss)})")
    ax.axvline(4 / 3 * np.pi * 15 ** 3 / 1000, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(4 / 3 * np.pi * 15 ** 3 / 1000 * 0.9, 0.2, "30-mm\nsphere", fontsize=5.6, color=SUB, va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
    ax.set_xscale("log")
    ax.set_ylim(-0.08, 1.0)
    ax.set_xlabel("Expert tumor volume (cm³, log)")
    ax.set_ylabel("Dice, AI vs expert")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="upper right", fontsize=5.8, handletextpad=0.1, borderaxespad=0.1)
    ax.set_title("Agreement falls with size", fontsize=7.2)
    letter(fig, ax, "f")

    # g -- decision flips, same model, different contour
    ax = fig.add_subplot(gs[2, 3])
    s = pd.read_csv(DATA / "bamf_model_summary.csv")
    s = s[s.analysis.str.startswith("deployment")]
    eps = [("recurrence", "Recur."), ("egfr", "EGFR"), ("os", "OS")]
    for k, (cnt, lab, c) in enumerate([("ai", "AI segmentation", RED), ("-1mm", "−1 mm", ORANGE), ("+1mm", "+1 mm", BLUE)]):
        v = [100 * s[(s.endpoint == e) & (s.contour == cnt)].decision_flip_rate.iloc[0] for e, _ in eps]
        ax.bar(np.arange(3) + (k - 1) * 0.26, v, width=0.25, color=c, label=lab)
    ax.set_xticks(range(3), [e[1] for e in eps])
    ax.set_ylabel("Patients reclassified (%)")
    ax.set_ylim(0, 55)
    ax.legend(frameon=False, loc="upper left", fontsize=5.8, handlelength=1)
    ax.set_title("Model fixed, other segmentation", fontsize=7.2)
    letter(fig, ax, "g")


# Figure 5 -- interpretation vs stability; stability filtering

def _shap(target):
    from nsclc_pipeline import interpretation
    sv, _ = interpretation.compute_shap(target, "radiomics")
    return sv


def fig5():
    comp = pd.read_csv(DATA / "interobserver_vs_synthetic_comparison.csv").set_index("feature")
    inst = comp["synthetic_perturbation_median_rel_change"]
    shaps = {t: _shap(t) for t in ("recurrence", "egfr", "kras")}

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.62,
                          left=0.12, right=0.985, top=0.955, bottom=0.075)

    # a -- beeswarm, recurrence
    sv = shaps["recurrence"]
    vals, data, names = np.asarray(sv.values), np.asarray(sv.data, dtype=float), list(sv.feature_names)
    order = np.argsort(np.abs(vals).mean(0))[::-1][:10]
    ax = fig.add_subplot(gs[0, 0:2])
    cmap = LinearSegmentedColormap.from_list("fv", [BLUE, "#d9d4ee", RED])
    for row, j in enumerate(order[::-1]):
        x = vals[:, j]
        z = data[:, j]
        zr = (z - np.nanpercentile(z, 5)) / (np.nanpercentile(z, 95) - np.nanpercentile(z, 5) + 1e-12)
        bins = np.round(x / (np.ptp(x) / 40 + 1e-12))
        jit = np.zeros_like(x)
        for b in np.unique(bins):
            idx = np.where(bins == b)[0]
            jit[idx] = (np.arange(len(idx)) - (len(idx) - 1) / 2) * 0.045
        ax.scatter(x, row + np.clip(jit, -0.38, 0.38), s=4, c=np.clip(zr, 0, 1), cmap=cmap, vmin=0, vmax=1, lw=0)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(order)), [nice(names[j]) for j in order[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("SHAP value (effect on log-odds of recurrence)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, ticks=[0, 1], aspect=30)
    cb.ax.set_yticklabels(["Low\nvalue", "High\nvalue"], fontsize=5.8)
    cb.outline.set_visible(False)
    ax.set_title("What the recurrence model uses", fontsize=7.2)
    letter(fig, ax, "a", dx=-0.1)

    # b -- mean |SHAP| across targets (top features)
    ax = fig.add_subplot(gs[0, 2])
    imp = pd.DataFrame({t: pd.Series(np.abs(np.asarray(s.values)).mean(0), index=s.feature_names)
                        for t, s in shaps.items()})
    top = sorted(set().union(*[imp[t].nlargest(5).index for t in imp]), key=lambda f: -imp.loc[f].max())
    m = imp.loc[top] / imp.max()
    ax.imshow(m.to_numpy(), cmap=LinearSegmentedColormap.from_list("w", ["white", NAVY]), vmin=0, vmax=1,
              aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            r = int(imp[m.columns[j]].rank(ascending=False)[top[i]])
            ax.text(j, i, str(r), ha="center", va="center", fontsize=5.8, color="white" if m.iat[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), ["Recur.", "EGFR", "KRAS"])
    ax.set_yticks(range(len(top)), [nice(f) for f in top], fontsize=6.1)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("Importance rank by target", fontsize=7.2)
    letter(fig, ax, "b", dx=-0.1)

    # c -- importance rank vs instability rank
    ax = fig.add_subplot(gs[1, 0])
    ax.axvspan(0, 8.5, ymin=0, ymax=8.5 / 36, color="#fde3d6", lw=0, zorder=0)
    ax.text(0.6, 8.9, "important and unstable", fontsize=5.6, color=ORANGE, va="bottom")
    ir = inst.rank(ascending=False)
    for t in ("recurrence", "egfr", "kras"):
        s = imp[t].rank(ascending=False)
        common = s.index.intersection(ir.index)
        rho = spearmanr(s[common], ir[common]).correlation
        ax.scatter(s[common], ir[common], s=9, color=ENDPOINT[t], alpha=0.8, lw=0, label=f"{t.upper() if t != 'recurrence' else 'Recurrence'} (ρ = {rho:.2f})".replace("-", "−"))
    ax.set_xlim(0, 27.5)
    ax.set_ylim(0, 36)
    ax.set_yticks([1, 10, 20, 26])
    ax.set_xlabel("SHAP importance rank (1 = most)")
    ax.set_ylabel("Instability rank (1 = least stable)")
    ax.legend(frameon=False, loc="upper center", fontsize=5.8, handletextpad=0.1, borderaxespad=0.1, ncol=1)
    ax.set_title("Importance is unrelated to stability", fontsize=7.2)
    letter(fig, ax, "c", dx=-0.075)

    # d -- stability filtering: swing with 95% CI
    ax = fig.add_subplot(gs[1, 1])
    sw = pd.read_csv(DATA / "swing_significance.csv")
    tgts = [("egfr", "EGFR"), ("recurrence", "Recur."), ("recurrence_survival", "OS")]
    conds = [("full", "All 26", MUTED), ("stable", "Stable 13", GREEN), ("unstable", "Unstable 13", ORANGE)]
    for i, (t, lab) in enumerate(tgts):
        for k, (cnd, clab, c) in enumerate(conds):
            r = sw[(sw.target == t) & (sw.condition == cnd)].iloc[0]
            x = i + (k - 1) * 0.24
            ax.plot([x, x], [r.swing_ci_lo, r.swing_ci_hi], color=c, lw=1.8, alpha=0.4, solid_capstyle="round")
            ax.plot(x, r.observed_swing, "o", color=c, ms=4, mec="white", mew=0.5, label=clab if i == 0 else None)
        p = sw[(sw.target == t) & (sw.condition == "stable_minus_unstable_swing_diff")].p_value.iloc[0]
        ax.text(i, 0.455, f"P = {p:.2f}", ha="center", fontsize=5.8, color=SUB)
    ax.set_xticks(range(3), [t[1] for t in tgts])
    ax.set_ylim(0, 0.48)
    ax.set_ylabel("Range across segmentations (95% CI)")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=5.8, handletextpad=0.1)
    ax.set_title("Does stability filtering help?", fontsize=7.2)
    letter(fig, ax, "d", dx=-0.075)

    # e -- contour ensemble: expert-vs-AI decision flips by strategy
    ax = fig.add_subplot(gs[1, 2])
    cr = pd.read_csv(DATA / "contour_robust_summary.csv")
    eps = [("recurrence", "Recur."), ("egfr", "EGFR"), ("os", "OS")]
    strats = [("baseline", "Single segmentation", MUTED), ("tte", "Averaged over segmentations", GREEN),
              ("aug", "Augmented training", AMBER), ("aug+tte", "Both", TEAL)]
    for k, (s, lab, c) in enumerate(strats):
        for i, (e, _) in enumerate(eps):
            r = cr[(cr.endpoint == e) & (cr.strategy == s)].iloc[0]
            x = i + (k - 1.5) * 0.2
            ax.bar(x, 100 * r.flip_rate, width=0.19, color=c, label=lab if i == 0 else None)
            if s != "baseline" and r.ci_hi < 0:
                ax.text(x, 100 * r.flip_rate + 1, "*", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(3), [e[1] for e in eps])
    ax.set_ylim(0, 58)
    ax.set_ylabel("Reclassified, expert vs AI segmentation (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.6, handlelength=1, ncol=1)
    ax.set_title("Averaging over segmentations", fontsize=7.2)
    letter(fig, ax, "e", dx=-0.075)
    save(fig, "figure6_interpretation_stability")


# Figure 6 -- prognostic value and biology

def fig6():
    from lifelines.statistics import multivariate_logrank_test
    fig = plt.figure(figsize=(7.2, 7.9))
    H = [1.0, 1.15, 1.0]
    gs = fig.add_gridspec(3, 2, height_ratios=H, hspace=0.55, wspace=0.12,
                          left=0.215, right=0.985, top=0.95, bottom=0.06)
    gsb = fig.add_gridspec(3, 2, height_ratios=H, hspace=0.55, wspace=0.75,
                           left=0.09, right=0.985, top=0.95, bottom=0.06)
    gsc = fig.add_gridspec(3, 4, height_ratios=H, hspace=0.55, wspace=0.62,
                           left=0.07, right=0.985, top=0.95, bottom=0.06)

    # a -- multivariable Cox forest, OS and RFS
    labels = None
    axes = []
    for k, (ep, lab, c) in enumerate([("os", "Overall survival", GREEN), ("rfs", "Recurrence-free survival", TEAL)]):
        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)
        uni = pd.read_csv(DATA / f"cox_univariate_{ep}.csv")
        mv = pd.read_csv(DATA / f"cox_multivariate_{ep}.csv")
        labels = mv["variable"].tolist()
        y = np.arange(len(mv))[::-1]
        for yy, (_, r), (_, u) in zip(y, mv.iterrows(), uni.iterrows()):
            ax.plot([u.HR_lower, u.HR_upper], [yy + 0.18] * 2, color=MUTED, lw=0.9)
            ax.plot(u.HR, yy + 0.18, "o", color=MUTED, mfc="white", ms=3, mew=0.8)
            ax.plot([r.HR_lower, r.HR_upper], [yy - 0.12] * 2, color=c, lw=1.4)
            ax.plot(r.HR, yy - 0.12, "s", color=c, ms=3.8, mec="white", mew=0.4)
            ax.text(7.6, yy, f"{r.HR:.2f}", ha="right", va="center", fontsize=5.9,
                    color=INK, fontweight="bold" if r.p < 0.05 else "normal")
        ax.axvline(1, color=INK, lw=0.6)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlim(0.2, 8.0)
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        ax.set_yticks(y, [l.replace(" vs. ", " vs ").replace("Radiomics risk", "Radiomic risk").replace("Pathological ", "Path. ").replace("Tumor", "Tumor") for l in labels] if k == 0 else [""] * len(y),
                      fontsize=6.1)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Hazard ratio (95% CI)")
        ax.set_title(lab, fontsize=7.2, color=c)
    axes[0].legend([Line2D([], [], marker="o", color=MUTED, mfc="white", lw=0.9), Line2D([], [], marker="s", color=GREEN, lw=1.4)],
                   ["Univariable", "Multivariable (all covariates)"], frameon=False, loc="lower center",
                   bbox_to_anchor=(1.05, 1.09), ncol=2, fontsize=6.2)
    letter(fig, axes[0], "a", dx=-0.2)

    # b -- OS tertiles
    ax = fig.add_subplot(gsb[1, 0])
    d = pd.read_csv(DATA / "km_tertile_os.csv")
    d["m"] = d["duration_days"] / DAYS_PER_MONTH
    p = multivariate_logrank_test(d.m, d.risk_tertile, d.event).p_value
    grp = [(d.m[d.risk_tertile == g].to_numpy(), d.event[d.risk_tertile == g].to_numpy()) for g in ("Low", "Intermediate", "High")]
    km_panel(ax, grp, [BLUE, AMBER, ORANGE], ["Low", "Intermediate", "High"], p=p)
    ax.set_xlabel("Months from CT")
    ax.set_ylabel("Overall survival probability")
    ax.set_title("OS by radiomic risk tertile", fontsize=7.2)
    letter(fig, ax, "b", dx=-0.075)

    # c -- GSEA
    ax = fig.add_subplot(gsb[1, 1])
    g = pd.read_csv(DATA / "radiogenomics_gsea_results.csv")
    g = g[g["FDR q-val"] < 0.2].sort_values("NES")
    g["Term"] = g["Term"].str.replace("Pperoxisome", "Peroxisome")
    y = np.arange(len(g))
    ax.barh(y, g["NES"], color=[ORANGE if v > 0 else BLUE for v in g["NES"]], height=0.7)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(y, g["Term"].str.replace("Epithelial Mesenchymal Transition", "EMT"), fontsize=5.6)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("NES, high vs low OS-risk tertile")
    ax.set_title(f"Hallmark pathways (FDR < 0.2; {len(g)} pathways)", fontsize=7.2)
    letter(fig, ax, "c", dx=-0.21)
    fig6_lung1_row(fig, gsc)
    save(fig, "figure7_prognosis_biology")


def fig6_lung1_row(fig, gs):
    """Row 3: external replication in NSCLC-Radiomics (Lung1)."""
    import json
    from lifelines.statistics import logrank_test
    L1 = ROOT / "data" / "external" / "lung1"
    res = json.loads((L1 / "lung1_results.json").read_text())
    oof = pd.read_csv(L1 / "lung1_oof_predictions.csv")
    fc = pd.read_csv(L1 / "lung1_fold_concordance.csv")

    # d -- KM by median out-of-fold risk
    ax = fig.add_subplot(gs[2, 0])
    oof["m"] = oof.time / DAYS_PER_MONTH
    hi = oof["risk_+0"] >= oof["risk_+0"].median()
    p = logrank_test(oof.m[hi], oof.m[~hi], oof.event[hi], oof.event[~hi]).p_value
    km_panel(ax, [(oof.m[~hi].to_numpy(), oof.event[~hi].to_numpy()), (oof.m[hi].to_numpy(), oof.event[hi].to_numpy())],
             [BLUE, ORANGE], ["Low risk", "High risk"], ticks=(0, 24, 48, 72, 96), p=p)
    ax.legend(loc="upper right", frameon=False, fontsize=5.6, handlelength=1)
    for t in ax.texts:
        t.set_position((0.97, 0.62))
        t.set_fontsize(6)
    ax.set_xlabel("Months")
    ax.set_ylabel("Overall survival")
    ax.set_title(f"Lung1: OS (n = {res['n']})", fontsize=7.2)
    letter(fig, ax, "d")

    # e -- concordance across shifts
    ax = fig.add_subplot(gs[2, 1])
    mu = [res[f"c_mean_{m:+d}"] for m in MM]
    for m in MM:
        v = fc[fc.mm == m].c
        ax.scatter(np.full(len(v), m), v, s=5, color=GREEN, alpha=0.4, lw=0)
    ax.plot(MM, mu, color=GREEN, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.5)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(MM, ["−2", "−1", "0", "+1", "+2"])
    ax.set_ylim(0.3, 0.75)
    ax.set_xlabel("Boundary shift (mm)")
    ax.set_ylabel("OS concordance")
    lo, hi_ = res["swing_ci"]
    ax.text(0.03, 0.97, f"Range {res['swing']:.2f}\n95% CI {lo:.2f}–{hi_:.2f}", transform=ax.transAxes,
            va="top", fontsize=6)
    fd = pd.read_csv(DATA / "flip_decomposition.csv")
    fd = fd[fd.cohort == "Lung1"].set_index("comparison")
    ax.text(0.97, 0.03, "Risk order vs original\n(model fixed)\n"
            f"−1 mm: ρ = {fd.loc['expert vs -1 mm', 'spearman']:.2f}\n+1 mm: ρ = {fd.loc['expert vs +1 mm', 'spearman']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=ORANGE)
    ax.set_title("Lung1: concordance\n(model re-developed)", fontsize=7.2)
    letter(fig, ax, "e")

    # f -- cluster reassignment
    ax = fig.add_subplot(gs[2, 2])
    cl = res["clustering"]
    vals = [100 * cl[f"reassigned_{m:+d}"] for m in MM]
    ax.bar(MM, vals, color=[AMBER if m else GRID for m in MM], width=0.7)
    for m, v in zip(MM, vals):
        ax.text(m, v + 1.5, f"{v:.0f}%", ha="center", fontsize=5.8)
    ax.set_xticks(MM, ["−2", "−1", "0", "+1", "+2"])
    ax.set_ylim(0, max(vals) * 1.25 + 5)
    ax.set_xlabel("Boundary shift (mm)")
    ax.set_ylabel("Patients reassigned (%)")
    ax.text(0.97, 0.97, f"resampling ARI\n{cl['bootstrap_ari_mean']:.2f} ± {cl['bootstrap_ari_sd']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=SUB)
    ax.set_title("Lung1: phenotypes", fontsize=7.2)
    letter(fig, ax, "f")

    # g -- feature instability, Lung1 vs Stanford
    ax = fig.add_subplot(gs[2, 3])
    l1 = pd.read_csv(L1 / "lung1_feature_instability.csv", index_col=0).iloc[:, 0]
    st = pd.read_csv(DATA / "interobserver_vs_synthetic_comparison.csv").set_index("feature")
    common = l1.index.intersection(st.index)
    xr = st.loc[common, "synthetic_perturbation_median_rel_change"].rank(ascending=False)
    yr = l1[common].rank(ascending=False)
    ax.plot([0, 27], [0, 27], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.scatter(xr, yr, s=12, c=[FAMILY[family(f)] for f in common], edgecolor="white", lw=0.3)
    ins = res["instability"]
    ax.text(0.03, 0.97, f"ρ = {ins['rho_vs_stanford_synthetic']:.2f}", transform=ax.transAxes, va="top", fontsize=6.2)
    ax.set_xlim(0, 27.5)
    ax.set_ylim(0, 27.5)
    ax.set_xlabel("Instability rank, discovery cohort")
    ax.set_ylabel("Instability rank, Lung1")
    ax.set_title("Same fragile features", fontsize=7.2)
    letter(fig, ax, "g")


def supp_roc():
    """Supplementary figure: pooled out-of-fold ROC curves for every endpoint and
    feature set (the fold-level AUCs are shown in Fig. 2a). Replaces an older
    one-row-per-model stack that had grown to more than 60 inches."""
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.34)
    sets = FS + [("__volume_only", "Volume only", MUTED)]
    for ax, (tgt, lab) in zip(axes, [("recurrence", "Recurrence"), ("egfr", "EGFR mutation"), ("kras", "KRAS mutation")]):
        for sfx, name, c in sets:
            f = DATA / f"modeling_predictions_{tgt}{sfx}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            fpr, tpr, _ = roc_curve(d.y_true, d.y_proba)
            ax.plot(fpr, tpr, color=c, lw=1.2, ls=(0, (3, 1.5)) if sfx == "__volume_only" else "-",
                    label=f"{name.replace(chr(10), ' ')} ({roc_auc_score(d.y_true, d.y_proba):.2f}, n = {len(d)})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("1 − specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(lab, fontsize=7.2, color=ENDPOINT[tgt])
        ax.legend(fontsize=5.4, frameon=False, loc="upper left", bbox_to_anchor=(-0.05, -0.3), handlelength=1.4,
                  labelspacing=0.2)
    for ax, s in zip(axes, "abc"):
        letter(fig, ax, s, dx=-0.05)
    out = ROOT / "outputs" / "figures" / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cvdetail.png", facecolor="white", bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print("Saved outputs/figures/supplementary/cvdetail.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["2", "3", "4", "5", "6"]
    for w in which:
        {"2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "roc": supp_roc}[w]()
