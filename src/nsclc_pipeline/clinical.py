"""Load and tidy the NSCLC Radiogenomics clinical/genomic table.

Source: NSCLCR01Radiogenomic_DATA_LABELS_*.csv (REDCap export, 211 patients).
Turns the wide, checkbox-style REDCap export into a tidy table with
snake_case columns, real dtypes (dates, numerics, booleans), consistent
missing-value handling, and a single categorical tumor-location column
collapsed from the six "Tumor Location (choice=...)" checkbox columns.
"""

from __future__ import annotations

import re

import pandas as pd

from . import config

# Values REDCap/TCIA use in place of a real missing value.
_MISSING_TOKENS = {
    "n/a",
    "na",
    "not collected",
    "not recorded in database",
    "not assessed",
    "not collected/other, unknown",
    "unknown",
}

_TUMOR_LOCATION_COLS = {
    "Tumor Location (choice=RUL)": "RUL",
    "Tumor Location (choice=RML)": "RML",
    "Tumor Location (choice=RLL)": "RLL",
    "Tumor Location (choice=LUL)": "LUL",
    "Tumor Location (choice=LLL)": "LLL",
    "Tumor Location (choice=L Lingula)": "Lingula",
    "Tumor Location (choice=Unknown)": "Unknown",
}

_DATE_COLS = [
    "Date of Recurrence",
    "Date of Last Known Alive",
    "Date of Death",
    "CT Date",
    "PET Date",
]

_NUMERIC_COLS = [
    "Age at Histological Diagnosis",
    "Pack Years",
    "Quit Smoking Year",
    "Days between CT and surgery",
    "Time to Death (days)",
]


def _snake_case(col: str) -> str:
    col = col.strip()
    col = re.sub(r"\(choice=.*\)", "", col)
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col.lower()


def _clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    def clean_cell(v):
        if isinstance(v, str) and v.strip().lower() in _MISSING_TOKENS:
            return pd.NA
        return v

    return df.map(clean_cell)


def _collapse_tumor_location(df: pd.DataFrame) -> pd.DataFrame:
    present_cols = [c for c in _TUMOR_LOCATION_COLS if c in df.columns]

    def row_location(row):
        hits = [_TUMOR_LOCATION_COLS[c] for c in present_cols if row[c] == "Checked"]
        return ",".join(hits) if hits else pd.NA

    df["tumor_location"] = df.apply(row_location, axis=1)
    return df.drop(columns=present_cols)


def load_clinical(csv_path=None) -> pd.DataFrame:
    """Load the raw clinical CSV and return a tidy DataFrame."""
    csv_path = csv_path or config.CLINICAL_CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)

    df = _clean_missing(df)
    df = _collapse_tumor_location(df)

    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = [
        "tumor_location" if c == "tumor_location" else _snake_case(c) for c in df.columns
    ]

    df = df.rename(columns={"case_id": "case_id", "patient_affiliation": "site"})

    # Recurrence-free / dead-or-alive flags are more model-friendly than strings.
    if "recurrence" in df.columns:
        df["recurrence_bool"] = df["recurrence"].str.lower().map({"yes": True, "no": False})
    if "survival_status" in df.columns:
        df["deceased"] = df["survival_status"].str.lower().map({"dead": True, "alive": False})

    return df.sort_values("case_id").reset_index(drop=True)


def build(csv_path=None, out_path=None) -> pd.DataFrame:
    """Load, tidy, and persist the clinical table to data/processed/."""
    out_path = out_path or config.CLINICAL_OUT
    df = load_clinical(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    tidy = build()
    print(f"Wrote {len(tidy)} patients -> {config.CLINICAL_OUT}")
    print(tidy.head())
