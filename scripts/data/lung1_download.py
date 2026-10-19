"""External validation cohort (Lung1): download CT and segmentations from Imaging Data Commons.

Paper:    Table 2
Pipeline: python pipeline.py run lung1_data   (stage 'external')

Output: data/external/lung1/<PatientID>/<Modality>_<SeriesInstanceUID>/*.dcm
        data/external/lung1/series.csv
Run:    python scripts/data/lung1_download.py
"""
import pathlib

from idc_index import IDCClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "lung1"
OUT.mkdir(parents=True, exist_ok=True)

c = IDCClient()
s = c.sql_query("""
SELECT PatientID, Modality, SeriesInstanceUID, SeriesDescription, analysis_result_id, series_size_MB
FROM index WHERE collection_id = 'nsclc_radiomics'
  AND (Modality = 'CT' OR (Modality = 'SEG' AND analysis_result_id IS NULL AND SeriesDescription = 'Segmentation'))
""")
s.to_csv(OUT / "series.csv", index=False)
print(s.groupby("Modality").agg(n=("SeriesInstanceUID", "count"), GB=("series_size_MB", lambda x: round(x.sum() / 1024, 1))))
c.download_from_selection(seriesInstanceUID=s.SeriesInstanceUID.tolist(), downloadDir=str(OUT),
                          dirTemplate="%PatientID/%Modality_%SeriesInstanceUID", quiet=False)
print("done")
