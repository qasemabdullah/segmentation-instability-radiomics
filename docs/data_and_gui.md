# Data sources, curation notes and the desktop GUI

This page describes the discovery cohort (the Stanford NSCLC Radiogenomics collection on The
Cancer Imaging Archive), how its files are curated, the desktop GUI and the command-line tools
that the GUI calls. Results of the study are in the paper; this page does not report them.

- Collection: https://doi.org/10.7937/K9/TCIA.2017.7hs46erv
- Reference: Bakr et al., *Sci Data* 5, 180202 (2018).

The collection is released under CC BY 3.0. Work that uses it must cite the collection (see the
DOI page for the citation and the data usage policy).

## What is in the repository and what is downloaded

`data/raw/` holds everything except the pixel data (about 870 KB):

| File | Contents |
|---|---|
| `NSCLCR01Radiogenomic_DATA_LABELS_*.csv` | clinical and genomic table, 211 patients |
| `NSCLC_Radiogenomics-*-nbia-digest.xlsx` | series-level metadata for all 1,351 image series |
| `NSCLC_Radiogenomics-*.tcia` | manifest of the series UIDs in the collection |
| `AIM_files_updated-*.zip` | 190 AIM XML files with semantic annotations |

The DICOM images (727 CT, 480 PET and 144 SEG series, about 98 GB) are not included. They are
downloaded from TCIA when needed (see below).

### External resources

- **RNA sequencing:** GEO [GSE103584](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103584),
  gene-level FPKM for 130 of the 211 patients (the `R01-*` subset; the `AMC-*` patients have no
  RNA sequencing). The GEO sample titles match `case_id`, so the two tables are joined directly.
  Handled by `rna_seq.py`.
- **Imaging Data Commons:** [the same collection on IDC](https://portal.imaging.datacommons.cancer.gov/explore/filters/?collection_id=nsclc_radiogenomics),
  which hosts the same images in cloud storage.
- **AIMI nnU-Net segmentations:** [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8345959),
  automatic lung and nodule segmentations for several TCIA lung collections. They are used as
  the AI segmentation in the study (`scripts/ai_segmentation/`).

### Data caveats

The AIM files are not tumor masks. Each marks the tumor with a circle on one CT slice (a center
and one edge point) and lists the radiologist's semantic characteristics (margin, attenuation,
calcification, shape, location). The volumetric tumor masks are the 144 SEG series, converted
by `seg_to_mask.py`. For `R01-143`, `R01-147`, `R01-151`, `R01-152` and `R01-154`, the AIM
locator refers to a CT series that is not in the current release, so these locators are not
used.

The `time_to_death_days` column of the clinical table differs from `date_of_death - ct_date` by
up to several hundred days, which suggests it was measured from surgery. All durations are
computed from the CT date instead.

`radiomics.py` is not PyRadiomics. It computes shape features with SimpleITK, first-order
intensity features with NumPy and texture from per-slice 2D GLCMs (scikit-image), averaged
across slices and weighted by tumor area. The feature categories match PyRadiomics, but the
values differ (texture is 2.5D rather than 3D over 13 directions). The PyRadiomics comparison in
the paper is run separately in `scripts/pyradiomics/`, in its own environment.

## Setup

```
pip install -r requirements.txt
```

## GUI

The Tkinter desktop app puts the main steps behind buttons, with a log panel:

```
python scripts/gui/gui.py
```

Every task runs on a background thread, so the window stays responsive during downloads. The
buttons call the same functions as the command-line tools below.

**Pipeline stages**

- **Build metadata tables** and **Fetch RNA-seq** run directly.
- **Download images...** chooses a modality and a set of patients from a searchable list of the
  case IDs that have series of that modality (leave the selection empty for all patients).
  **Preview** shows the number of series and the approximate download size before anything is
  downloaded; a warning appears above 300 series. The download shows a progress bar.
- **Convert SEG -> mask...** lists only the 144 patients with a SEG series, finds the series and
  downloads it if needed, so a mask cannot be saved under the wrong patient. "Convert ALL"
  converts every available mask and skips patients already converted.
- **Extract radiomics...** extracts the features (see `radiomics.py`) for one patient or, with
  "Extract ALL", for every patient with a SEG series, downloading each CT as it goes. Results
  are saved after every patient, so an interrupted run resumes where it stopped.
- **Run modeling...** chooses the target (recurrence, EGFR or KRAS), the feature set, the
  numbers of outer and inner folds and the random seed. Before running, it shows how many
  patients have a label for that combination and the class balance (requiring RNA sequencing,
  for example, reduces the cohort). It then runs nested cross-validation and prints the AUC of
  each fold.
- **Segmentation adjustment study...** step 1 re-extracts the features from segmentations
  adjusted inwards and outwards (−2 to +2 mm, see `mask_shift.py` and `perturbation.py`);
  step 2 compares feature stability between −1 and +1 mm and re-runs the same model on the
  adjusted feature tables.
- **Run survival model...** fits a Cox proportional-hazards model on follow-up time
  (`survival.py`) and prints the concordance index of each fold.
- **Extract embeddings...** extracts ImageNet-pretrained ResNet18 features from the central
  tumor slice (`embeddings.py`), with the same single-patient, "Extract ALL" and resume options
  as radiomics extraction.

**Other buttons**

- **View images...** opens a slice viewer for any patient with a mask. It finds the CT series
  the mask was drawn on from the SEG object itself (a patient can have several CT series),
  downloads it if needed and shows axial slices with a slider, lung, mediastinal and bone
  windows, and a tumor overlay that can be switched off.
- **Dashboard...** shows charts in the window:
  - **Paper: main figures** and **Paper: supplementary figures** browse the files in
    `outputs/figures/main` and `outputs/figures/supplementary` after
    `python pipeline.py run figures`, with fit-to-width, zoom and buttons to open the image or
    its folder. If the environment variable `NSCLC_PAPER_DIR` points at a built manuscript,
    each figure is shown with its number, title and legend from the paper.
  - The **Explore** tabs (feature sets, cohort, radiomics, modeling, interpretation,
    segmentation adjustment, survival) draw quick charts from the files in `data/processed/`.
    They are for exploring the data and are not the figures of the paper.
- **Open output folder** opens `data/processed/`; **View a table...** opens any
  `data/processed/*.csv` or `*.csv.gz` file in a sortable table.

## Command line

The full analysis is run with `python pipeline.py` (see the main README). The tools below are
the individual steps that the GUI uses.

**1. Metadata tables** (no image download; takes seconds):

```
python scripts/data/build_metadata.py
```

This writes to `data/processed/`:

- `clinical.csv`: the clinical and genomic table (211 rows), with REDCap checkbox columns
  combined, dates and numbers parsed, and `Not Collected`, `N/A` and similar values set to
  missing.
- `series_index.csv`: one row per DICOM series (1,351 rows), linking `case_id` to the series
  UID, modality and scanner, checked against the `.tcia` manifest.
- `aim_annotations.csv`: semantic tumor characteristics in long format (4,451 rows).
- `aim_locators.csv`: one row per locator circle, with the study, series and SOP instance UIDs
  of the CT slice it marks (191 rows).

**2. RNA sequencing** (optional, about 7 MB, 130 patients):

```
python scripts/data/build_metadata.py --rna-seq
# or
python -m nsclc_pipeline.rna_seq
```

This writes `data/processed/rna_seq_expression.csv.gz` (22,126 genes by 130 patients, FPKM, wide
format; `rna_seq.to_long()` converts a subset of genes to long format).

**3. Images** (the full collection is about 98 GB; this uses the public TCIA REST API and needs
no NBIA Data Retriever):

```
# one series
python -m nsclc_pipeline.download --series-uid <SeriesInstanceUID>

# all SEG (tumor mask) series, small
python -m nsclc_pipeline.download --modality SEG

# all series of selected patients
python -m nsclc_pipeline.download --case-id R01-038 --case-id AMC-003
```

The DICOM files are saved to `data/interim/dicom/<series_uid>/`.

**4. SEG series to NIfTI mask:**

```
python -m nsclc_pipeline.seg_to_mask <case_id> data/interim/dicom/<seg_series_uid>
```

This writes `data/processed/masks/<case_id>_seg.nii.gz`. The geometry of the mask (spacing,
origin, direction) is taken from the SEG object, so the CT is not needed for this step.

**5. Radiomic features** (downloads each patient's CT automatically):

```
python -m nsclc_pipeline.radiomics
```

This writes `data/processed/radiomics_features.csv`, one row per patient. Patients already
extracted are skipped on a re-run.

**6. Models** (nested cross-validation):

```
python -m nsclc_pipeline.modeling --target recurrence
python -m nsclc_pipeline.modeling --target egfr --feature-set radiomics+clinical
python -m nsclc_pipeline.modeling --target egfr --feature-set radiomics+clinical+rnaseq --seed 1
```

- Targets: `recurrence`, `egfr` and `kras`. ALK is not offered because the cohort has only two
  ALK-positive patients.
- Feature sets:
  - `radiomics` (default);
  - `radiomics+clinical`, which adds age, stage, grade, smoking and histology (baseline fields
    only; treatment fields are excluded, see the docstring of `modeling.py`);
  - `radiomics+clinical+rnaseq`, which adds the RNA sequencing data reduced to 10 principal
    components fitted inside each training fold;
  - `embeddings`, the CNN embeddings instead of radiomics;
  - `radiomics+embeddings`, both together.
- Options: `--outer-folds`, `--inner-folds` and `--seed`.
- The AUC of each outer fold is printed and saved to
  `data/processed/modeling_results_<target>[__<feature_set>].csv`.
- `load_dataset` sorts patients by `case_id` before splitting, so the result depends only on
  `--seed` and not on the order in which features were extracted.

**7. Segmentation adjustment** (`perturbation.py`, using `mask_shift.py`):

```
python -m nsclc_pipeline.perturbation          # re-extract features at -2, -1, 0, +1 and +2 mm
python -c "from nsclc_pipeline import perturbation; perturbation.run_auc_comparison()"
```

The boundary is moved with a signed distance map, so the adjustment follows each scan's voxel
spacing, including anisotropic spacing. `run_auc_comparison` re-runs the same nested
cross-validation with the adjusted feature table in place of the original one, through the
`radiomics_path` argument of `modeling.run` (which also accepts a DataFrame).
`feature_stability_summary(df, -1, 1)` ranks the features by their median relative change
between −1 and +1 mm.

**8. Survival** (`survival.py`):

```
python -m nsclc_pipeline.survival
python -m nsclc_pipeline.survival --endpoint rfs
```

A Cox proportional-hazards model on the time from CT, with patients without an event censored,
evaluated with the same nested cross-validation as `modeling.py` and Harrell's concordance
index. Options: `--endpoint` (`os`, `rfs` or `dfs`), `--outer-folds`, `--inner-folds` and
`--seed`.

**9. Image embeddings** (`embeddings.py`):

```
python -m nsclc_pipeline.embeddings
python -m nsclc_pipeline.modeling --target recurrence --feature-set embeddings
```

An ImageNet-pretrained ResNet18 applied to a single 2D slice: a general transfer-learning
baseline that runs on a CPU, not a medical-imaging foundation model. (The CT foundation model
used in the paper, Pai et al. 2024, is run separately in `scripts/data/lidc_fm_features.py`.)
The axial slice with the largest tumor area is cropped to a padded square around the tumor,
lung-windowed, resized to 224 × 224 and copied to three channels; the 512-dimensional output of
the layer before the classifier is kept. For modeling, the embeddings are reduced to 10
principal components inside each training fold and passed to the same logistic-regression
pipeline as the radiomic features.

**10. SHAP interpretation** (`interpretation.py`):

The radiomics model of each target is fitted once on all available patients (descriptive only;
performance comes from the nested cross-validation), and exact SHAP values are computed with
`shap.LinearExplainer`. `texture_n_slices`, the number of slices used for texture, is kept in
`radiomics_features.csv` for quality control but excluded from the models through
`modeling.NON_FEATURE_COLS`, because it mainly reflects tumor size.

## Package layout

```
src/nsclc_pipeline/
  config.py                  paths
  clinical.py                raw clinical CSV -> clinical table
  series_index.py            NBIA digest + .tcia manifest -> series index
  aim_annotations.py         AIM XML -> semantic annotations and locators
  rna_seq.py                 GEO GSE103584 -> expression matrix
  download.py                TCIA REST API downloader
  seg_to_mask.py             DICOM SEG -> NIfTI mask
  viz.py                     HU windowing and mask overlays (GUI)
  radiomics.py               shape, first-order and texture features
  mask_shift.py              inward and outward segmentation adjustment
  perturbation.py            features and models under segmentation adjustment
  modeling.py                nested cross-validation models
  survival.py                Cox models and concordance index
  interpretation.py          SHAP importance
  clustered.py               patient-level clustered statistics
  phenotypes.py              radiomic phenotypes
  embeddings.py              ImageNet ResNet18 embeddings
  radimagenet_embeddings.py  RadImageNet embeddings
scripts/data/build_metadata.py   metadata steps in order
scripts/gui/gui.py               desktop GUI
data/raw/                        original TCIA files (tracked)
data/interim/                    downloaded DICOM (not tracked)
data/processed/                  derived tables (tracked) and masks/ (not tracked)
```
