"""Download DICOM series from TCIA for series listed in the series index.

Uses the public TCIA REST API (no Java NBIA Data Retriever install
required): each Series Instance UID maps to a zip of that series' DICOM
files at the getImage endpoint. This collection ("NSCLC Radiogenomics",
DOI 10.7937/K9/TCIA.2017.7hs46erv) is public and requires no API key.

The full collection is ~98GB (727 CT + 480 PT + 144 SEG series across 211
patients) -- do not run `download_all` casually. Prefer `download_series`
for one series at a time, or filter series_index.csv by modality/case_id
first (e.g. just the 144 SEG series, or one patient's CT+SEG for a pilot).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config

_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"


def download_series(series_uid: str, out_dir: Path | None = None, timeout: int = 120) -> Path:
    """Download one series' DICOM files as a zip and extract them.

    Returns the directory the DICOM files were extracted into. Skips the
    request entirely if that directory already exists and is non-empty.
    """
    out_dir = out_dir or (config.DICOM_DIR / series_uid)
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir

    resp = requests.get(_BASE_URL, params={"SeriesInstanceUID": series_uid}, timeout=timeout)
    resp.raise_for_status()
    if not resp.content.startswith(b"PK"):
        raise RuntimeError(
            f"Response for series {series_uid} is not a zip file "
            f"(first bytes: {resp.content[:80]!r}) -- likely an API error page."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(out_dir)
    return out_dir


def filter_series(
    series_index_path=None,
    modality: str | None = None,
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Apply the same filters download_by_filter uses, without downloading.
    Useful to preview count/total size before committing to a download."""
    series_index_path = series_index_path or config.SERIES_INDEX_OUT
    df = pd.read_csv(series_index_path)

    if modality:
        df = df[df["modality"] == modality]
    if case_ids:
        df = df[df["case_id"].isin(case_ids)]
    if limit:
        df = df.head(limit)
    return df


def download_by_filter(
    series_index_path=None,
    modality: str | None = None,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    progress_cb=None,
) -> list[Path]:
    """Download a filtered subset of the series index. Use this, not
    download_all, unless you actually intend to pull the full ~98GB.

    progress_cb(i, total, row), if given, is called before each series is
    fetched (i is 1-based) -- lets a caller drive a progress bar."""
    df = filter_series(series_index_path, modality, case_ids, limit)
    total = len(df)

    paths = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"Downloading {row['case_id']} / {row['modality']} / {row['series_uid']} ...")
        if progress_cb:
            progress_cb(i, total, row)
        paths.append(download_series(row["series_uid"]))
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["CT", "PT", "SEG"], default=None)
    parser.add_argument("--case-id", action="append", dest="case_ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--series-uid", default=None, help="Download a single series by UID")
    args = parser.parse_args()

    if args.series_uid:
        out = download_series(args.series_uid)
        print(f"Extracted -> {out}")
    else:
        out_paths = download_by_filter(modality=args.modality, case_ids=args.case_ids, limit=args.limit)
        print(f"Downloaded {len(out_paths)} series -> {config.DICOM_DIR}")
