# Scripts

One script per analysis step, grouped by topic. Run them through `pipeline.py` (which sets the
Python path and records runtimes), for example `python pipeline.py run repeated_cv`, or directly
from the repository root, for example `python scripts/discovery/run_repeated_cv.py`.
The header of each script lists the paper items it produces, its inputs and outputs, and how to run it.

## data/

Downloads, masks and features of the discovery cohort; cohort overlap and CT acquisition parameters.

| Script | Purpose | Pipeline step |
|---|---|---|
| `acquisition_summary.py` | CT acquisition parameters of every analyzed series, read from the DICOM headers. | `acquisition` |
| `adjusted_segmentation_features.py` | Discovery cohort: features of every segmentation adjusted by −2 to +2 mm (parallel). | `segmentation_adjustment` |
| `build_discovery_masks.py` | Discovery cohort: convert every expert DICOM-SEG segmentation to a NIfTI mask. | `discovery_masks` |
| `build_metadata.py` | Metadata of the discovery collection: clinical, genomic, series and annotation tables (no image download). | `metadata` |
| `cohort_overlap.py` | Patient overlap between the multi-reader NSCLC cohort and Lung1. | `cohort_overlap` |
| `crlm_features.py` | Liver-metastasis cohort: features of the expert, ±1-mm and AI segmentations of the largest lesion. | `crlm` |
| `export_interobserver_masks.py` | Multi-reader NSCLC cohort: export each reader's segmentation and the CT as NIfTI. | `interobserver` |
| `lidc_download.py` | LIDC-IDRI step 1: index every nodule segmented by three or four radiologists. | `lidc_data` |
| `lidc_features.py` | LIDC-IDRI step 2: features of every reader's segmentation at −1, 0 and +1 mm. | `lidc_data` |
| `lidc_fm_features.py` | LIDC-IDRI: foundation-model features (Pai et al. 2024) seeded at each reader's segmentation. | `lidc_data` |
| `lung1_download.py` | External validation cohort (Lung1): download CT and segmentations from Imaging Data Commons. | `lung1_data` |
| `lung1_features.py` | External validation cohort (Lung1): features at every segmentation adjustment. | `lung1_data` |
| `validate_segmentation_adjustment.py` | Geometric validation of the ±1/±2-mm segmentation adjustment on synthetic spheres. | `segmentation_adjustment` |

## discovery/

Discovery-cohort models, segmentation adjustment, repeated validation and the controlled-signal experiment.

| Script | Purpose | Pipeline step |
|---|---|---|
| `run_adjusted_segmentation_models.py` | Discovery cohort: models on segmentations adjusted by −2 to +2 mm, and the agreement between the controlled adjustment and real reader disagreement. | `adjusted_models`, `adjustment_vs_readers` |
| `run_base_metric_bootstrap.py` | Bootstrap 95% CIs of the baseline nested cross-validated scores. | `bootstrap_ci` |
| `run_calibration_inference.py` | Calibration of the classification models with significance tests and resampling. | `calibration_inference` |
| `run_ct_habitat_supervoxels_3d.py` | Exploratory 3D intratumoral habitats (supervoxels, k-means). | `habitats` |
| `run_paired_segmentation_effect.py` | Paired analysis of the segmentation effect on performance across the 20 repetitions. | `paired_segmentation_effect` |
| `run_radimagenet_comparison.py` | RadImageNet versus ImageNet image embeddings. | `radimagenet` |
| `run_random_forest_robustness.py` | Second classifier (random forest) under segmentation adjustment. | `random_forest` |
| `run_range_inference.py` | Range of performance across segmentations: bootstrap CIs and the stable-vs-unstable test. | `range_inference` |
| `run_rank_shift.py` | Threshold-free instability: how far does a patient's score move? | `rank_shift` |
| `run_refit_bootstrap.py` | Same-patient comparison with model development repeated in every bootstrap replicate. | `refit_bootstrap` |
| `run_repeated_cv.py` | Repeated nested cross-validation (20 fold seeds): model variability and the re-allocation null. | `repeated_cv` |
| `run_same_patient_comparison.py` | Model-development versus segmentation instability on the same patients. | `same_patient_comparison` |
| `run_sensitivity_checks.py` | Robustness checks: borderline patients, AI design range, Lung1 reordering and calibration. | `sensitivity_checks` |
| `run_signal_simulation.py` | Does the effect of the segmentation depend on how much signal a model carries? | `signal_experiment` |
| `run_threshold_sensitivity.py` | Is segmentation-induced reclassification an artifact of the median threshold? | `threshold_sensitivity` |
| `run_volume_confounding_check.py` | Tumor volume alone as a predictor (size confounding). | `volume_only` |

## survival/

Cox models, competing risks, incremental value and gene-set enrichment.

| Script | Purpose | Pipeline step |
|---|---|---|
| `run_competing_risks.py` | Deaths without recurrence: censoring dates, disease-free survival and competing risks. | `competing_risks` |
| `run_radiogenomics_gsea.py` | Gene-set enrichment of the overall-survival risk score (high vs low tertile). | `radiogenomics` |
| `run_survival_cox_tables.py` | Univariable and multivariable Cox models and survival by risk tertile. | `survival_tables` |
| `run_survival_incremental_value.py` | Incremental value of radiomics over clinicopathological variables, and exploratory covariates. | `survival_tables` |

## readers/

Multi-reader NSCLC cohort (five radiation oncologists).

| Script | Purpose | Pipeline step |
|---|---|---|
| `analyze_interobserver.py` | Multi-reader NSCLC cohort: disagreement between five radiation oncologists and feature instability between them. | `interobserver` |
| `nsclc_confidence_check.py` | NSCLC: does segmentation sensitivity predict reclassification by another segmentation beyond model confidence? | `confidence_nsclc` |
| `run_reader_decisions.py` | Multi-reader NSCLC cohort: reclassification between radiation oncologists and the ±1-mm flag. | `reader_classifications` |
| `run_reader_os_discovery.py` | Multi-reader NSCLC cohort: overall-survival reader analysis with a model that cannot share patients with the reader cohort. | `reader_os_discovery` |

## ai_segmentation/

Expert versus public AI (nnU-Net) segmentations.

| Script | Purpose | Pipeline step |
|---|---|---|
| `ai_segmentation_masks.py` | AI segmentation step 1: match the public nnU-Net (AIMI) segmentations to each expert tumor. | `ai_masks` |
| `run_ai_outcome_summary.py` | Overall outcome of replacing the expert segmentation by the public AI tool. | `ai_outcome` |
| `run_ai_segmentation_features.py` | Features of the AI segmentations adjusted by ±1 mm. | `ai_models` |
| `run_ai_segmentation_models.py` | AI segmentation step 2: expert versus AI segmentation (deployment and re-development). | `ai_models` |
| `run_instability_flag.py` | Discovery cohort: the ±1-mm flag against AI-induced reclassification. | `ai_models` |

## mitigation/

Stability filtering and averaging over segmentations.

| Script | Purpose | Pipeline step |
|---|---|---|
| `run_segmentation_averaging.py` | Mitigation: averaging over segmentations and training on deliberately varied segmentations. | `averaging` |
| `run_stability_filtering_experiment.py` | Mitigation: models restricted to the most or least stable features. | `stability_filtering` |
| `run_stability_filtering_infold.py` | Stability filtering with the stable and unstable halves chosen inside each outer fold. | `stability_filtering` |

## external/

Lung1, LIDC-IDRI and liver-metastasis cohorts; threshold and rank checks.

| Script | Purpose | Pipeline step |
|---|---|---|
| `crlm_models.py` | Liver-metastasis cohort: survival models, reclassification by AI segmentation and model variability. | `crlm` |
| `flag_clinical_check.py` | LIDC-IDRI: the ±1-mm flag against misclassification and reader disagreement, with patient-bootstrap CIs. | `lidc_models` |
| `lidc_confidence_check.py` | LIDC-IDRI: does segmentation sensitivity add information beyond model confidence? | `lidc_models` |
| `lidc_dependence_check.py` | LIDC-IDRI: are reader-dependent classifications more often discordant with the reader-consensus label beyond their proximity to the threshold? | `lidc_models` |
| `lidc_loo_check.py` | LIDC-IDRI: each segmentation judged only by the other readers' ratings (removes circularity). | `lidc_models` |
| `lidc_models.py` | LIDC-IDRI step 3: reader-dependent classifications, their errors and model variability. | `lidc_models` |
| `lung1_models.py` | External validation cohort (Lung1): survival model, reclassification and phenotypes under adjustment. | `lung1_models` |
| `run_fixed_vs_refit.py` | Performance at each segmentation with the model kept fixed or re-developed. | `fixed_vs_refit` |
| `run_reordering_vs_shift.py` | Why concordance can stay stable while individual classifications change (reordering vs shift). | `lung1_models` |

## pyradiomics/

IBSI-compliant reference features; extraction steps run in .venv_pyradiomics (Python 3.11).

| Script | Purpose | Pipeline step |
|---|---|---|
| `run_pyradiomics_clustering_interobserver.py` | PyRadiomics: phenotype stability and reader agreement with the reference extractor. | `pyradiomics_analyses` |
| `run_pyradiomics_crosscheck.py` | PyRadiomics: instability ranking compared with the study extractor (30 patients). | `pyradiomics_extraction` |
| `run_pyradiomics_downstream_comparison.py` | PyRadiomics: model performance at every segmentation adjustment. | `pyradiomics_analyses` |
| `run_pyradiomics_full_extraction.py` | PyRadiomics: features of all 144 patients at every segmentation adjustment. | `pyradiomics_extraction` |
| `run_pyradiomics_interobserver.py` | PyRadiomics: features of the five radiation oncologists' segmentations. | `pyradiomics_extraction` |
| `run_pyradiomics_primary_analyses.py` | PyRadiomics: primary classification and survival analyses with the reference features. | `pyradiomics_analyses` |
| `run_pyradiomics_primary_baseline.py` | PyRadiomics: reference features at the expert segmentation. | `pyradiomics_extraction` |

## figures/

Every main and supplementary figure; the shared style is in figure_style.py and palette.py.

| Script | Purpose | Pipeline step |
|---|---|---|
| `fig1_study_design.py` | Fig. 1: study design and patient cohorts. | `main_figures` |
| `fig5_tumor_types.py` | Fig. 5: segmentation-dependent classifications across tumor types. | `main_figures` |
| `figure_style.py` | Shared visual style for every figure in the paper. | helper |
| `main_figures.py` | Figs 2, 3, 4, 6 and 7, and Supplementary Fig. S5 (ROC curves), in the shared journal style. | `main_figures` |
| `palette.py` | Color palette and matplotlib style shared by the desktop GUI and two Supplementary Figures (patient vignettes, CT habitats). | helper |
| `render_supplementary_figures.py` | Draws Supplementary Figs. S6, S7, S8, S11 and S13 into outputs/figures/supplementary/. | `supplementary_figures` |
| `supp_cohort_distributions_tertiles.py` | Supplementary figures: cohort characteristics, radiomic feature distributions and survival by risk tertile. | `supplementary_figures` |
| `supp_confidence.py` | Supplementary figure: segmentation sensitivity versus model confidence (LIDC-IDRI). | `supplementary_figures` |
| `supp_ct_habitats.py` | Supplementary Fig. S8: exploratory 3D intratumoral habitats. | helper |
| `supp_extractor_classifier.py` | Supplementary Fig. S6: the effect does not depend on the feature extractor or classifier. | helper |
| `supp_patient_flow_validation.py` | Supplementary figures: patient flow of the discovery cohort and the validation design used in each cohort. | `supplementary_figures` |
| `supp_patient_vignettes.py` | Supplementary Fig. S13: one patient from each overall-survival risk tertile. | helper |
| `supp_phenotypes.py` | Supplementary Fig. S7: radiomic phenotypes of the discovery cohort. | helper |
| `supp_signal_experiment.py` | Supplementary Fig. S10: effect of the segmentation as a function of model signal (controlled-signal experiment). | `supplementary_figures` |
| `supp_stability_filtering.py` | Supplementary Fig. S11: stability filtering. | helper |

## reporting/

Multiple-testing correction and the runtime table.

| Script | Purpose | Pipeline step |
|---|---|---|
| `multiple_testing.py` | Benjamini-Hochberg correction across the eleven pre-specified tests. | `multiple_testing` |
| `runtime_summary.py` | Computational cost: wall-clock time of every analysis step, for the runtime table. | `runtime_table` |

## gui/

Optional desktop viewer (Tkinter) for browsing data, results and figures; see docs/data_and_gui.md.

| Script | Purpose | Pipeline step |
|---|---|---|
| `gui.py` | Desktop GUI (Tkinter) for browsing the data, running steps and viewing results. | helper |
