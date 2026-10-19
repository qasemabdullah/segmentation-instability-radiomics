"""Supplementary figures: patient flow of the discovery cohort (flow.png) and
the validation design used in each cohort (cv.png).

Paper:    Suppl. Figs S1-S13
Pipeline: python pipeline.py run supplementary_figures   (stage 'figures')

Numbers are read from the data where they are computed; the validation
settings mirror modeling.py, survival.py, lung1_models.py, lidc_models.py and
crlm_models.py.
Output: outputs/figures/supplementary/flow.png, outputs/figures/supplementary/cv.png
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.figures.figure_style import BLUE, GREEN, INK, MUTED, ORANGE, SUB, letter  # noqa: E402

OUT = ROOT / "outputs" / "figures" / "supplementary"
EXC_FC, EXC_EC = "#fbeee8", "#d9774f"
INC_FC, INC_EC = "#eef3fb", BLUE


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=6.6, weight="normal", color=INK, lw=0.9, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 1.5
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, fontweight=weight, color=color, linespacing=1.35)


def line(ax, pts, color=MUTED, arrow=True, lw=0.9):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt")
    if arrow:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7))


def note(ax, x, y, w, h, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#f6f7f9", ec=MUTED, lw=0.9))
    ax.text(x + 2, y + h - 1.6, title, ha="left", va="top", fontsize=6.4, fontweight="bold", color=INK)
    ax.text(x + 2, y + h - 4.6, body, ha="left", va="top", fontsize=6.1, color=INK, linespacing=1.4)


# patient flow
def flow():
    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    box(ax, 30, 47, 40, 7, "NSCLC Radiogenomics collection\n211 patients (Stanford, n = 118; VA Palo Alto, n = 93)", fc=INC_FC, ec=INC_EC, weight="bold")
    box(ax, 75, 47.5, 23, 6, "ALK rearrangement not modeled:\n2 positive of 157 tested", fc=EXC_FC, ec=EXC_EC, color="#8a3b1c", fs=6.2)
    line(ax, [(70, 50.5), (75, 50.5)], color=EXC_EC)
    box(ax, 75, 37.5, 23, 6, "Excluded (n = 67): no volumetric\nexpert tumor segmentation", fc=EXC_FC, ec=EXC_EC, color="#8a3b1c", fs=6.2)
    line(ax, [(50, 47), (50, 40.5), (75, 40.5)], color=EXC_EC)
    box(ax, 30, 30, 40, 7, "Discovery cohort: 144 patients with an expert segmentation\nmedian follow-up 62.9 months from CT (reverse Kaplan–Meier)", fc=INC_FC, ec=INC_EC, weight="bold")
    line(ax, [(50, 40.5), (50, 37)])

    cols = [(1, "Recurrence\nn = 144\n40 events (27.8%)", None),
            (21, "EGFR mutation\nn = 117\n23 mutant (19.7%)", "Excluded (n = 27):\nstatus not recorded"),
            (41, "KRAS mutation\nn = 115\n27 mutant (23.5%)", "Excluded (n = 29):\nstatus not recorded"),
            (61, "Overall and recurrence-free\nsurvival, n = 144\n51 deaths (35.4%)", None),
            (81, "Gene expression\nn = 117 with RNA sequencing\n74 in extreme OS tertiles", "Excluded (n = 27):\nno RNA sequencing")]
    w = 18
    line(ax, [(50, 30), (50, 26)], arrow=False)
    line(ax, [(1 + w / 2, 26), (81 + w / 2, 26)], arrow=False)
    for x, txt, exc in cols:
        line(ax, [(x + w / 2, 26), (x + w / 2, 22.5)])
        box(ax, x, 13.5, w, 9, txt, fs=6.3)
        if exc:
            line(ax, [(x + w / 2, 13.5), (x + w / 2, 10)], color=EXC_EC)
            box(ax, x, 4.5, w, 5.5, exc, fc=EXC_FC, ec=EXC_EC, color="#8a3b1c", fs=6.0)
    ax.text(1, 1.2, "Blue, study population; black, analysis set for each endpoint; orange, excluded or not modeled, with reason and number.",
            fontsize=5.8, color=SUB, va="bottom")
    fig.savefig(OUT / "flow.png", dpi=300, facecolor="white")
    plt.close(fig)


# validation design
def cv():
    fig = plt.figure(figsize=(7.2, 5.6))
    # a: nested cross-validation
    ax = fig.add_axes([0.02, 0.50, 0.96, 0.46])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    letter(fig, ax, "a", dx=-0.005, dy=0.0)
    ax.text(0, 49, "Nested cross-validation, as used for every refitted model", fontsize=7.2, fontweight="bold", color=INK, va="top")

    ax.text(0, 42.5, "Outer split\n(five folds)", fontsize=6.3, color=SUB, va="center")
    for i in range(5):
        c = ORANGE if i == 0 else BLUE
        ax.add_patch(Rectangle((13 + i * 8.2, 40), 7.6, 5, fc=c, ec="white"))
        ax.text(13 + i * 8.2 + 3.8, 42.5, str(i + 1), ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
    ax.text(13, 38.2, "held-out test fold", color=ORANGE, fontsize=6, va="top")
    ax.text(29.4, 38.2, "training folds (2–5)", color=BLUE, fontsize=6, va="top")

    ax.text(0, 29.5, "Inner split of the\ntraining folds", fontsize=6.3, color=SUB, va="center")
    for i in range(5):
        c = GREEN if i == 2 else "#aab1bc"
        ax.add_patch(Rectangle((13 + i * 8.2, 27), 7.6, 5, fc=c, ec="white"))
    ax.text(13, 25.2, "inner validation fold (green) rotates; each regularization value is scored on all five", color=SUB, fontsize=6, va="top")
    line(ax, [(33, 36), (33, 32.3)])

    steps = ["Select the regularization strength\nwith the best inner-validation score",
             "Refit on all training folds;\nclassification threshold = median training score",
             "Score the held-out patients from every segmentation\n(AUC or concordance; above or below threshold)"]
    y = [15, 7.5, 0]
    line(ax, [(33, 22.4), (33, 20.8)])
    for yy, s in zip(y, steps):
        box(ax, 13, yy, 40, 5.6, s, fs=6.2, ec=MUTED)
    for a, b in ((15, 13.2), (7.5, 5.7)):
        line(ax, [(33, a), (33, b)])

    note(ax, 60, 30, 39, 13, "Repeat for each outer fold",
         "Every patient is scored once, by a model that never\nsaw them during fitting or tuning; imputation and\nscaling are fitted on the training folds only")
    note(ax, 60, 11, 39, 14, "Repeat with new fold seeds",
         "Model-development variability: share of patients\nclassified differently by two seeds from the same segmentation\nSegmentation effect: share classified differently by\nthe same model from another segmentation")

    # b: validation in each cohort
    bx = fig.add_axes([0.02, 0.02, 0.96, 0.44])
    bx.set_xlim(0, 100)
    bx.set_ylim(0, 44)
    bx.axis("off")
    letter(fig, bx, "b", dx=-0.005, dy=0.0)
    bx.text(0, 43.5, "Validation in each cohort", fontsize=7.2, fontweight="bold", color=INK, va="top")
    L = json.loads((ROOT / "data/external/lidc/lidc_model_summary.json").read_text())
    head = ["Cohort", "Model", "Validation", "Repetitions", "Segmentations scored"]
    xs = [0, 22, 49, 70, 81]
    rows = [
        ("Discovery\n(n = 144)", "Logistic regression (recurrence,\nEGFR, KRAS); ridge Cox (OS, RFS)", "Nested, 5 × 5 folds\n(stratified for classification)", "20 fold seeds", "Expert, ±1 and ±2 mm, AI"),
        ("Multi-reader NSCLC\n(n = 21)", "The 100 discovery-cohort models\n(recurrence, EGFR, OS)", "None (no refitting;\nnever trained on these patients)", "100 models\n(20 × 5 folds)", "5 radiation oncologists,\n±1 mm"),
        ("External cohort,\nLung1 (n = 419)", "Ridge Cox (OS); discovery-cohort\nmodel also applied unchanged", "Nested, 5 × 5 folds", "1", "Expert, ±1 and ±2 mm"),
        (f"Multi-reader nodules,\nLIDC-IDRI ({L['n_nodules']} nodules)", "Logistic regression, trained on one\nrandomly chosen reader per nodule", "Nested, 5 × 5 folds,\ngrouped by patient", f"{L['n_seeds']} fold seeds", "Every reader, ±1 mm;\nfoundation model"),
        ("Liver metastases\n(n = 197)", "Ridge Cox (OS, disease-free\nsurvival)", "Nested, 5 × 5 folds,\nstratified", "20 fold seeds", "Expert, AI, ±1 mm"),
    ]
    y0 = 37
    for x, h in zip(xs, head):
        bx.text(x, y0, h, fontsize=6.4, fontweight="bold", color=INK, va="center")
    bx.plot([0, 100], [y0 - 2.2, y0 - 2.2], color=INK, lw=0.6)
    for i, r in enumerate(rows):
        yy = y0 - 6.5 - i * 7
        if i % 2 == 0:
            bx.add_patch(Rectangle((0, yy - 3.4), 100, 6.8, fc="#f4f6f9", ec="none", zorder=0))
        for x, v in zip(xs, r):
            bx.text(x + 0.5, yy, v, fontsize=6.0, color=INK, va="center", linespacing=1.3)
    bx.plot([0, 100], [y0 - 6.5 - 4 * 7 - 3.4] * 2, color=INK, lw=0.6)
    fig.savefig(OUT / "cv.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flow()
    cv()
    print("Saved flow.png and cv.png")
