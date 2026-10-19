"""Reproduce the analyses of the paper: every step, in order, with what it produces.

    Tumor segmentation induces individual-level instability in radiomic risk classification.

This file is the single map of the project. Each step is one or more scripts (or library
modules) and lists the paper items it feeds. Steps are grouped into stages that must run
in the order listed; a step can also be run on its own once its inputs exist.

Usage
    python pipeline.py list                 all stages and steps
    python pipeline.py list external        steps of one stage
    python pipeline.py run figures          run one stage
    python pipeline.py run lidc_models      run one step
    python pipeline.py run all --dry-run    print every command without running it

Tags
    [network]          downloads public images from TCIA / NCI Imaging Data Commons
                       (tens of GB in total; takes hours)
    [images]           reads the CT images or segmentation masks downloaded by the
                       [network] steps (they are not part of the repository)
    [pyradiomics-env]  runs in .venv_pyradiomics (Python 3.11), because PyRadiomics
                       does not install on Python 3.12

Every other step runs from the processed results in data/, which are part of the
repository. Figures are written to outputs/figures/ (not version-controlled).
Paper items use the numbering of the submitted manuscript ("S" = Supplementary).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable
PYRAD_PY = ROOT / ".venv_pyradiomics" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
RUNTIME_LOG = ROOT / "data" / "interim" / "runtime_log.csv"
OUTPUT_DIRS = ("outputs/figures/main", "outputs/figures/supplementary/_parts")


@dataclass
class Step:
    name: str
    commands: list[list[str]]
    purpose: str
    paper: str = ""
    tags: list[str] = field(default_factory=list)


def s(script, *args):
    """Command for a script, found by file name in its topic folder under scripts/."""
    found = list((ROOT / "scripts").glob(f"*/{script}"))
    if len(found) != 1:
        raise SystemExit(f"script {script}: expected one match under scripts/, found {len(found)}")
    return [found[0].relative_to(ROOT).as_posix(), *map(str, args)]


def m(module, *args):
    """Command for a library module in src/nsclc_pipeline/."""
    return ["-m", f"nsclc_pipeline.{module}", *map(str, args)]


BASELINE_MODELS = [m("modeling", "--target", t, "--feature-set", fs)
                   for t in ("recurrence", "egfr")
                   for fs in ("radiomics", "radiomics+clinical", "radiomics+clinical+rnaseq", "embeddings", "radiomics+embeddings")]
BASELINE_MODELS += [m("modeling", "--target", "kras", "--feature-set", "radiomics"),
                    m("survival", "--endpoint", "os"), m("survival", "--endpoint", "rfs")]

STAGES: dict[str, list[Step]] = {
    "data": [
        Step("metadata", [s("build_metadata.py", "--rna-seq")],
             "Clinical, genomic and series tables of the NSCLC Radiogenomics collection; RNA sequencing (GEO GSE103584)",
             "Table 2, Suppl. Table S1, Suppl. Figs S1-S2", ["network"]),
        Step("discovery_masks", [s("build_discovery_masks.py")],
             "Expert DICOM-SEG segmentations -> NIfTI masks (144 patients)", "all discovery-cohort analyses", ["network"]),
        Step("discovery_features", [m("radiomics")],
             "26 radiomic features from each expert segmentation (downloads each CT)", "Fig. 2", ["network"]),
        Step("segmentation_adjustment", [s("validate_segmentation_adjustment.py"), s("adjusted_segmentation_features.py")],
             "Validate the ±1/±2-mm adjustment on spheres; features of every adjusted segmentation", "Fig. 3a-b", ["images"]),
        Step("cnn_embeddings", [m("embeddings"), m("radimagenet_embeddings")],
             "ImageNet and RadImageNet CNN embeddings of the tumor", "Fig. 2a, Suppl. Table S3", ["images"]),
        Step("cohort_overlap", [s("cohort_overlap.py")],
             "Scans shared by the multi-reader NSCLC cohort and Lung1, matched by image content", "Table 2, Methods", ["images"]),
    ],
    "discovery": [
        Step("baseline_models", BASELINE_MODELS,
             "Nested cross-validation for every endpoint and feature set", "Fig. 2a-d, Suppl. Table S2, Suppl. Fig. S5"),
        Step("volume_only", [s("run_volume_confounding_check.py")], "Tumor volume alone as a predictor", "Fig. 2a, Results"),
        Step("radimagenet", [s("run_radimagenet_comparison.py")], "RadImageNet versus ImageNet embeddings", "Suppl. Table S3"),
        Step("bootstrap_ci", [s("run_base_metric_bootstrap.py")], "Bootstrap 95% CIs of baseline scores", "Suppl. Table S2"),
        Step("adjusted_models", [s("run_adjusted_segmentation_models.py", "models")],
             "Models on segmentations adjusted by -2 to +2 mm", "Fig. 3c-e, Table 3"),
        Step("repeated_cv", [s("run_repeated_cv.py", 20, 6)],
             "20 fold seeds: model-development variability across allocations", "Fig. 3c-e,g, Table 3, Suppl. Table S6"),
        Step("paired_segmentation_effect", [s("run_paired_segmentation_effect.py")],
             "Segmentation effect on performance paired within repetitions, with patient-bootstrap intervals",
             "Table 3, Suppl. Table S7"),
        Step("random_forest", [s("run_random_forest_robustness.py")], "Second classifier (random forest)", "Suppl. Fig. S6"),
        Step("calibration_inference", [s("run_calibration_inference.py")],
             "Calibration statistics with fold-stratified bootstrap intervals", "Suppl. Table S24"),
        Step("survival_tables", [s("run_survival_cox_tables.py"), s("run_survival_incremental_value.py")],
             "Cox models with clinicopathological covariates, tertiles, incremental value",
             "Fig. 2d, Fig. 7a-b, Suppl. Tables S5, S23, Suppl. Fig. S12"),
        Step("competing_risks", [s("run_competing_risks.py")],
             "Censoring of deaths without recurrence, disease-free survival, cause-specific and Fine-Gray models",
             "Methods, Results, Suppl. Table S4"),
        Step("radiogenomics", [s("run_radiogenomics_gsea.py")], "Gene-set enrichment of the OS risk score", "Fig. 7c, Suppl. Table S25"),
        Step("habitats", [s("run_ct_habitat_supervoxels_3d.py")], "Exploratory 3D intratumoral habitats", "Suppl. Fig. S8", ["images"]),
        Step("signal_experiment", [s("run_signal_simulation.py", 20, 5, 6), s("run_signal_simulation.py", 20, 5, 6, "--latent-average")]
             + [s("run_signal_simulation.py", 20, 5, 6, f"--signal={k}") for k in ("sparse", "stable", "nonlinear", "external")],
             "Controlled-signal experiment (linear, averaged-segmentation, sparse, stable-feature, nonlinear and "
             "out-of-panel signal)", "Suppl. Fig. S10, Suppl. Table S20"),
    ],
    "readers": [
        Step("interobserver", [s("export_interobserver_masks.py"), s("analyze_interobserver.py")],
             "Five radiation oncologists' segmentations (NSCLC-Radiomics-Interobserver1)", "Fig. 4a-b", ["network"]),
        Step("adjustment_vs_readers", [s("run_adjusted_segmentation_models.py", "comparison")],
             "Feature instability under ±1 mm versus between readers", "Fig. 4c, Suppl. Fig. S11c, Suppl. Table S21"),
        Step("reader_classifications", [s("run_reader_decisions.py")],
             "Reclassification between radiation oncologists; ±1-mm flag", "Fig. 5e, Suppl. Table S18", ["images"]),
        Step("reader_os_discovery", [s("run_reader_os_discovery.py")],
             "OS reader analysis with the discovery-cohort model (no patients shared with the reader cohort)",
             "Results, Suppl. Table S18"),
    ],
    "ai": [
        Step("ai_masks", [s("ai_segmentation_masks.py")], "Public nnU-Net segmentations (AIMI) matched to each tumor",
             "Fig. 4e-f", ["network"]),
        Step("ai_models", [s("run_ai_segmentation_features.py"), s("run_ai_segmentation_models.py"), s("run_instability_flag.py")],
             "Expert versus AI segmentation; ±1-mm flag", "Fig. 3g, Fig. 4g, Suppl. Table S15", ["images"]),
        Step("averaging", [s("run_segmentation_averaging.py")], "Averaging over segmentations and augmented training",
             "Fig. 6e, Suppl. Table S22"),
        Step("stability_filtering", [s("run_stability_filtering_experiment.py"), s("run_stability_filtering_infold.py")],
             "Models restricted to stable or unstable features", "Fig. 6d, Suppl. Fig. S11, Suppl. Table S21"),
        # after stability_filtering, whose predictions it reads for the stable and unstable halves
        Step("range_inference", [s("run_range_inference.py")],
             "Range of performance across segmentations with bootstrap CIs (all, stable and unstable features)",
             "Table 3, Fig. 6d, Suppl. Fig. S11, Suppl. Table S21"),
        Step("confidence_nsclc", [s("nsclc_confidence_check.py")],
             "Segmentation sensitivity beyond model confidence (radiation oncologists and AI)",
             "Results, Suppl. Tables S15, S18"),
        Step("same_patient_comparison", [s("run_same_patient_comparison.py", 20, 6),
                                         s("run_same_patient_comparison.py", "--drop-roundness", 20, 6)],
             "Model-development vs segmentation instability in the same patients (26 and 25 features)",
             "Fig. 1f, Fig. 3g, Fig. 5c, Suppl. Table S9"),
        Step("ai_outcome", [s("run_ai_outcome_summary.py")],
             "Detection failures and reclassification by the AI tool in one summary", "Results, Suppl. Table S16"),
        Step("refit_bootstrap", [s("run_refit_bootstrap.py", 200, 20, 18)],
             "Same-patient comparison with model development repeated in every bootstrap replicate",
             "Methods, Results, Suppl. Table S10"),
    ],
    "external": [
        Step("lung1_data", [s("lung1_download.py"), s("lung1_features.py")], "NSCLC-Radiomics (Lung1): CT, segmentations, features",
             "Table 2", ["network"]),
        Step("lung1_models", [s("lung1_models.py"), s("run_reordering_vs_shift.py")],
             "External cohort, no-refit external validation, reordering versus shift", "Fig. 7d-g, Suppl. Table S26"),
        Step("fixed_vs_refit", [s("run_fixed_vs_refit.py")],
             "Performance at each segmentation with the model fixed or re-developed (discovery and Lung1)",
             "Table 3, Results, Suppl. Table S8"),
        Step("lidc_data", [s("lidc_download.py"), s("lidc_features.py"), s("lidc_fm_features.py")],
             "LIDC-IDRI: every reader's segmentation, features and foundation-model features", "Table 2", ["network"]),
        Step("lidc_models", [s("lidc_models.py"), s("lidc_models.py", "--fm"), s("lidc_loo_check.py"), s("flag_clinical_check.py"),
                             s("lidc_confidence_check.py"), s("lidc_dependence_check.py")],
             "Reader-dependent classifications, reader-consensus-label disagreement, the ±1-mm flag and model confidence",
             "Fig. 5a-f, Suppl. Table S17, Suppl. Fig. S9"),
        Step("crlm", [s("crlm_features.py"), s("crlm_models.py")], "Colorectal liver metastases: expert versus AI",
             "Fig. 5b,c,g, Suppl. Table S19", ["network"]),
        Step("threshold_sensitivity", [s("run_threshold_sensitivity.py")],
             "Reclassification at thresholds from the 25th to the 75th percentile (discovery, readers, AI, nodules, liver)",
             "Results, Suppl. Table S11"),
        Step("rank_shift", [s("run_rank_shift.py")],
             "Threshold-free instability: shift in percentile rank of continuous scores", "Results, Suppl. Table S12"),
        # after the AI and external-cohort steps, whose outputs these read
        Step("sensitivity_checks", [s("run_sensitivity_checks.py")],
             "Borderline patients, AI design range, Lung1 reordering, calibration",
             "Results, Suppl. Tables S15, S24, S26"),
        Step("acquisition", [s("acquisition_summary.py")],
             "CT acquisition parameters of every analyzed series", "Methods, Suppl. Table S29", ["network"]),
    ],
    "pyradiomics": [
        Step("pyradiomics_extraction", [s("run_pyradiomics_crosscheck.py"), s("run_pyradiomics_full_extraction.py"),
                                        s("run_pyradiomics_primary_baseline.py"), s("run_pyradiomics_interobserver.py")],
             "IBSI-compliant reference features (PyRadiomics)", "Fig. 4d, Suppl. Tables S13-S14", ["pyradiomics-env"]),
        Step("pyradiomics_analyses", [s("run_pyradiomics_downstream_comparison.py"), s("run_pyradiomics_primary_analyses.py"),
                                      s("run_pyradiomics_clustering_interobserver.py")],
             "Models and phenotypes on PyRadiomics features", "Suppl. Tables S13-S14, Suppl. Fig. S6"),
    ],
    "figures": [
        Step("main_figures", [s("fig1_study_design.py"), s("main_figures.py", 2, 3, 4, 5, 6, "roc"),
                              s("fig5_tumor_types.py")], "Figs 1-7 and Suppl. Fig. S5", "Figs 1-7, Suppl. Fig. S5", ["images"]),
        Step("supplementary_figures", [s("supp_patient_flow_validation.py"), s("supp_cohort_distributions_tertiles.py"),
                                       s("supp_signal_experiment.py"), s("supp_confidence.py"),
                                       s("render_supplementary_figures.py")],
             "Supplementary Figures", "Suppl. Figs S1-S13", ["images"]),
    ],
    "reporting": [
        Step("multiple_testing", [s("multiple_testing.py")],
             "Benjamini-Hochberg correction across the eleven pre-specified tests", "Methods, Suppl. Table S27"),
        Step("runtime_table", [s("runtime_summary.py")], "Wall-clock time of every analysis step", "Methods, Suppl. Table S30"),
    ],
}


def interpreter(step):
    return str(PYRAD_PY) if "pyradiomics-env" in step.tags else PY


def log_runtime(step, cmd, wall):
    """Append one line per command (wall-clock seconds) for the runtime table (Suppl. Table S30)."""
    import platform
    new = not RUNTIME_LOG.exists()
    RUNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNTIME_LOG, "a", encoding="utf-8") as f:
        if new:
            f.write("timestamp,step,command,wall_s,logical_cpus,machine\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{step.name},\"{' '.join(cmd)}\",{wall:.1f},"
                f"{os.cpu_count()},\"{platform.processor() or platform.machine()}\"\n")


def run_step(step, dry_run=False):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONIOENCODING="utf-8")
    for cmd in step.commands:
        full = [interpreter(step), *cmd]
        print(f"$ {' '.join(full[1:]) if full[0] == PY else ' '.join(full)}", flush=True)
        if dry_run:
            continue
        t = time.time()
        r = subprocess.run(full, cwd=ROOT, env=env)
        if r.returncode:
            raise SystemExit(f"step {step.name} failed: {' '.join(cmd)}")
        wall = time.time() - t
        log_runtime(step, cmd, wall)
        print(f"  done in {wall:.0f} s", flush=True)


def resolve(target):
    if target == "all":
        return [st for steps in STAGES.values() for st in steps]
    if target in STAGES:
        return STAGES[target]
    for steps in STAGES.values():
        for st in steps:
            if st.name == target:
                return [st]
    raise SystemExit(f"unknown stage or step: {target} (see: python pipeline.py list)")


def list_steps(stage=None):
    for name, steps in STAGES.items():
        if stage and name != stage:
            continue
        print(f"\n[{name}]")
        for st in steps:
            tags = f"  [{', '.join(st.tags)}]" if st.tags else ""
            print(f"  {st.name:26s} {st.purpose}{tags}")
            if st.paper:
                print(f"  {'':26s} -> {st.paper}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="show stages and steps")
    ls.add_argument("stage", nargs="?")
    rn = sub.add_parser("run", help="run a stage, a step or 'all'")
    rn.add_argument("target")
    rn.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = ap.parse_args()
    if a.cmd == "list":
        list_steps(a.stage)
    else:
        for d in OUTPUT_DIRS:
            (ROOT / d).mkdir(parents=True, exist_ok=True)
        for st in resolve(a.target):
            print(f"\n== {st.name}: {st.purpose}", flush=True)
            run_step(st, a.dry_run)


if __name__ == "__main__":
    main()
