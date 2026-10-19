"""Central path configuration for the NSCLC Radiogenomics pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLINICAL_CSV = RAW_DIR / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
NBIA_DIGEST_XLSX = RAW_DIR / "NSCLC_Radiogenomics-6-1-21-Version-4-nbia-digest.xlsx"
TCIA_MANIFEST = RAW_DIR / "NSCLC_Radiogenomics-6-1-21-Version-4.tcia"
AIM_ZIP = RAW_DIR / "AIM_files_updated-11-10-2020.zip"

# External resource, not part of the original TCIA bundle: GEO GSE103584,
# RNA-seq (Cufflinks FPKM) for the 130-patient Stanford/VA "R01" subset.
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103584
RNA_SEQ_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE103nnn/GSE103584/suppl/"
    "GSE103584_R01_NSCLC_RNAseq.txt.gz"
)
RNA_SEQ_RAW = RAW_DIR / "GSE103584_R01_NSCLC_RNAseq.txt.gz"

CLINICAL_OUT = PROCESSED_DIR / "clinical.csv"
SERIES_INDEX_OUT = PROCESSED_DIR / "series_index.csv"
AIM_ANNOTATIONS_OUT = PROCESSED_DIR / "aim_annotations.csv"
AIM_LOCATORS_OUT = PROCESSED_DIR / "aim_locators.csv"
RNA_SEQ_OUT = PROCESSED_DIR / "rna_seq_expression.csv.gz"
RADIOMICS_OUT = PROCESSED_DIR / "radiomics_features.csv"

DICOM_DIR = INTERIM_DIR / "dicom"
MASKS_DIR = PROCESSED_DIR / "masks"

for d in (INTERIM_DIR, PROCESSED_DIR, DICOM_DIR, MASKS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def idc_index_file() -> Path:
    """The NCI Imaging Data Commons series index installed with the idc-index-data package."""
    import idc_index_data
    return Path(idc_index_data.IDC_INDEX_PARQUET_FILEPATH)


def idc_clinical_table(collection: str) -> Path:
    """Clinical table of an Imaging Data Commons collection (e.g. 'lidc_idri_clinical'),
    downloaded by idc-index into its own data folder on first use."""
    import glob

    from idc_index import IDCClient
    client = IDCClient()
    folder = Path(client.idc_data_dir) / "clinical_data" / collection
    if not glob.glob(str(folder / "*.parquet")):
        client.fetch_index("clinical_index")
    return Path(glob.glob(str(folder / "*.parquet"))[0])
