"""Parse the AIM (Annotation and Image Markup) XML files.

These AIM files are not volumetric tumor segmentation masks.
Each ImageAnnotation is a single-slice locator (almost always a
"TwoDimensionCircle": a center point + one edge point placed on one DICOM
frame) plus a set of controlled-vocabulary semantic characteristics
(nodule margins, attenuation, calcification, anatomic location, etc.) that
a radiologist assigned to that lesion. The actual volumetric masks ship as
DICOM-SEG series in the image collection itself (144 SEG series in the
NBIA digest) and are handled by seg_to_mask.py once the images are
downloaded.

This module turns the AIM XML into two tidy tables:
  - annotations: one row per (lesion, semantic characteristic) -- long format
  - locators: one row per lesion's single-slice locator circle, with the
    SOP/Series/Study UIDs needed to find the corresponding DICOM frame
"""

from __future__ import annotations

import math
import zipfile
from xml.etree import ElementTree as ET

import pandas as pd

from . import config

_NS = {"aim": "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"}
_XSI = "{http://www.w3.org/2001/XMLSchema-instance}type"


def _tag(name: str) -> str:
    return f"aim:{name}"


def _findtext_value(elem, path):
    node = elem.find(_tag(path), _NS)
    if node is None:
        return None
    return node.get("value") or None


def _findtext_root(elem, path):
    """Like _findtext_value but for UID elements, which carry root= not value=."""
    node = elem.find(_tag(path), _NS)
    if node is None:
        return None
    return node.get("root") or None


def _parse_one(xml_bytes: bytes, case_id_hint: str):
    root = ET.fromstring(xml_bytes)
    equipment = root.find(_tag("equipment"), _NS)
    manufacturer = _findtext_value(equipment, "manufacturerName") if equipment is not None else None
    software_version = _findtext_value(equipment, "softwareVersion") if equipment is not None else None

    person = root.find(_tag("person"), _NS)
    person_id = _findtext_value(person, "id") if person is not None else None
    case_id = person_id or case_id_hint

    annotation_rows = []
    locator_rows = []

    for ann in root.iterfind(f"{_tag('imageAnnotations')}/{_tag('ImageAnnotation')}", _NS):
        lesion_name = _findtext_value(ann, "name")
        comment = _findtext_value(ann, "comment")
        ann_uid_node = ann.find(_tag("uniqueIdentifier"), _NS)
        lesion_uid = ann_uid_node.get("root") if ann_uid_node is not None else None

        base = dict(
            case_id=case_id,
            lesion_uid=lesion_uid,
            lesion_name=lesion_name,
            lesion_comment=comment,
            manufacturer=manufacturer,
            software_version=software_version,
        )

        # Anatomic location (imagingPhysicalEntityCollection) and semantic
        # characteristics (imagingObservationEntityCollection ->
        # imagingObservationCharacteristicCollection) both use the same
        # typeCode{code, codeSystem} + label shape, so we walk them uniformly.
        entity_paths = [
            (
                "imagingPhysicalEntityCollection/ImagingPhysicalEntity",
                "anatomic_location",
            ),
        ]
        for path, group in entity_paths:
            for entity in ann.iterfind(_tag_path(path), _NS):
                annotation_rows.append(
                    {**base, "group": group, **_type_code_row(entity)}
                )

        for obs in ann.iterfind(
            _tag_path("imagingObservationEntityCollection/ImagingObservationEntity"), _NS
        ):
            annotation_rows.append(
                {**base, "group": "observation", **_type_code_row(obs)}
            )
            for char in obs.iterfind(
                _tag_path(
                    "imagingObservationCharacteristicCollection/ImagingObservationCharacteristic"
                ),
                _NS,
            ):
                annotation_rows.append(
                    {**base, "group": "characteristic", **_type_code_row(char)}
                )

        # Markup (locator) + image reference
        for markup in ann.iterfind(
            _tag_path("markupEntityCollection/MarkupEntity"), _NS
        ):
            shape_type = markup.get(_XSI)
            frame_number = _findtext_value(markup, "referencedFrameNumber")
            markup_sop_uid = _findtext_root(markup, "imageReferenceUid")
            points = []
            for coord in markup.iterfind(
                _tag_path(
                    "twoDimensionSpatialCoordinateCollection/TwoDimensionSpatialCoordinate"
                ),
                _NS,
            ):
                x = float(_findtext_value(coord, "x"))
                y = float(_findtext_value(coord, "y"))
                points.append((x, y))

            row = {
                **base,
                "shape_type": shape_type,
                "frame_number": int(frame_number) if frame_number else None,
                "n_points": len(points),
                "sop_instance_uid": markup_sop_uid,
            }
            if shape_type == "TwoDimensionCircle" and len(points) == 2:
                (cx, cy), (ex, ey) = points
                row["center_x"] = cx
                row["center_y"] = cy
                row["radius_px"] = math.hypot(ex - cx, ey - cy)
            else:
                row["points"] = str(points)
            locator_rows.append(row)

        for ref in ann.iterfind(
            _tag_path("imageReferenceEntityCollection/ImageReferenceEntity"), _NS
        ):
            study_uid = _findtext_root(ref.find(_tag("imageStudy"), _NS), "instanceUid")
            series = ref.find(f"{_tag('imageStudy')}/{_tag('imageSeries')}", _NS)
            series_uid = _findtext_root(series, "instanceUid") if series is not None else None
            image = ref.find(
                f"{_tag('imageStudy')}/{_tag('imageSeries')}/{_tag('imageCollection')}/{_tag('Image')}",
                _NS,
            )
            sop_uid = _findtext_root(image, "sopInstanceUid") if image is not None else None
            for lr in locator_rows:
                if lr["case_id"] == case_id and lr["lesion_uid"] == lesion_uid:
                    lr.setdefault("study_uid", study_uid)
                    lr.setdefault("series_uid", series_uid)
                    lr.setdefault("sop_instance_uid", sop_uid)

    return annotation_rows, locator_rows


def _tag_path(path: str) -> str:
    return "/".join(_tag(p) for p in path.split("/"))


def _type_code_row(entity) -> dict:
    label = _findtext_value(entity, "label")
    comment = _findtext_value(entity, "comment")
    type_code = entity.find(_tag("typeCode"), _NS)
    code = type_code.get("code") if type_code is not None else None
    code_system = type_code.get("codeSystem") if type_code is not None else None
    code_system_name = type_code.get("codeSystemName") if type_code is not None else None
    return {
        "label": label,
        "comment": comment,
        "code": code,
        # NB: in this export codeSystem often holds the human-readable value
        # (e.g. codeSystem="peripheral" for label="Axial Location") rather
        # than a coding-system name -- kept as-is, not reinterpreted.
        "value_text": code_system,
        "code_system_name": code_system_name,
    }


def parse_aim_zip(zip_path=None):
    zip_path = zip_path or config.AIM_ZIP
    annotation_rows, locator_rows = [], []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            case_id_hint = name.rsplit("/", 1)[-1].removesuffix(".xml")
            xml_bytes = zf.read(name)
            a_rows, l_rows = _parse_one(xml_bytes, case_id_hint)
            annotation_rows.extend(a_rows)
            locator_rows.extend(l_rows)

    annotations = pd.DataFrame(annotation_rows)
    locators = pd.DataFrame(locator_rows)
    return annotations, locators


def build(zip_path=None, annotations_out=None, locators_out=None):
    annotations_out = annotations_out or config.AIM_ANNOTATIONS_OUT
    locators_out = locators_out or config.AIM_LOCATORS_OUT
    annotations, locators = parse_aim_zip(zip_path)
    annotations_out.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(annotations_out, index=False)
    locators.to_csv(locators_out, index=False)
    return annotations, locators


if __name__ == "__main__":
    annotations, locators = build()
    print(f"Wrote {len(annotations)} annotation rows -> {config.AIM_ANNOTATIONS_OUT}")
    print(f"Wrote {len(locators)} locator rows -> {config.AIM_LOCATORS_OUT}")
    print(f"Patients with AIM files: {annotations['case_id'].nunique()}")
