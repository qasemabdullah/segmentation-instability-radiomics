"""Build a tidy series-level index joining patients to their CT/PT/SEG series.

Source: the NBIA "digest" spreadsheet (one row per DICOM series, with the
Patient ID, Series/Study UIDs, modality, scanner make/model, etc.) plus the
.tcia manifest, which lists the Series Instance UIDs actually included in
this collection's download. We cross-check the two so a mismatch (a series
in the digest but not the manifest, or vice versa) is caught early rather
than silently dropping data during download.
"""

from __future__ import annotations

import pandas as pd

from . import config

_KEEP_COLS = {
    "Patient ID": "case_id",
    "Patient Sex": "sex",
    "Study Instance UID": "study_uid",
    "Study Date": "study_date",
    "Study Description": "study_description",
    "Series Instance UID": "series_uid",
    "Modality": "modality",
    "Series Description": "series_description",
    "Body Part Examined": "body_part",
    "Manufacturer": "manufacturer",
    "Manufacturer Model Name": "scanner_model",
    "Image Count": "image_count",
    "File Size": "file_size_bytes",
}


def _read_manifest_uids(manifest_path) -> set[str]:
    uids = set()
    in_list = False
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("ListOfSeriesToDownload"):
            in_list = True
            continue
        if in_list:
            uids.add(line)
    return uids


def load_series_index(digest_path=None, manifest_path=None) -> pd.DataFrame:
    digest_path = digest_path or config.NBIA_DIGEST_XLSX
    manifest_path = manifest_path or config.TCIA_MANIFEST

    raw = pd.read_excel(digest_path, sheet_name="Metadata")
    df = raw[list(_KEEP_COLS)].rename(columns=_KEEP_COLS)
    df["study_date"] = pd.to_datetime(df["study_date"], errors="coerce")

    manifest_uids = _read_manifest_uids(manifest_path)
    digest_uids = set(df["series_uid"])

    only_in_digest = digest_uids - manifest_uids
    only_in_manifest = manifest_uids - digest_uids
    if only_in_digest or only_in_manifest:
        print(
            f"[series_index] WARNING: {len(only_in_digest)} series in digest but not "
            f"in manifest, {len(only_in_manifest)} in manifest but not in digest."
        )

    df["in_manifest"] = df["series_uid"].isin(manifest_uids)
    return df.sort_values(["case_id", "modality", "series_uid"]).reset_index(drop=True)


def build(digest_path=None, manifest_path=None, out_path=None) -> pd.DataFrame:
    out_path = out_path or config.SERIES_INDEX_OUT
    df = load_series_index(digest_path, manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    idx = build()
    print(f"Wrote {len(idx)} series across {idx['case_id'].nunique()} patients -> {config.SERIES_INDEX_OUT}")
    print(idx["modality"].value_counts())
