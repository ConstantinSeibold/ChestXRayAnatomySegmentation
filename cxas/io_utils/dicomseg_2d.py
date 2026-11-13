"""DICOM output for projection radiographs (CR / DX) with highdicom.

* ``write_seg``: ONE multi-segment DICOM Segmentation per image (binary), one segment per non-empty structure, in
  the frame of the source image. This is the form IDC and MHub expect (the previous writer produced one file per
  class, each coded as "Lung, Right").
* ``write_landmark_sr``: a TID 1500 Measurement Report (Comprehensive 3D SR) with one planar ROI per landmark,
  graphic type POINT in the source image's pixel coordinates.

Structures are coded as SNOMED CT "Anatomical Structure" with a property type from the private coding scheme
``99CXAS`` (code value = the structure's slug).
TODO(release): map property types to SNOMED CT where a code exists (e.g. via MHub SegDB), keep 99CXAS otherwise.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pydicom

PRIVATE_SCHEME = "99CXAS"
MANUFACTURER = "CXAS"


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").upper()[:16] or "UNKNOWN"


def _code(name: str):
    from highdicom.sr import CodedConcept

    return CodedConcept(value=_slug(name), scheme_designator=PRIVATE_SCHEME, meaning=name.strip()[:64])


def write_seg(
    dcm_file: str,
    masks: Dict[str, np.ndarray],
    out_path: str,
    algorithm_name: str = "CXAS",
    algorithm_version: str = "unknown",
    series_description: str = "CXAS anatomy segmentation",
) -> Optional[str]:
    """masks: {structure name: bool (rows, columns)} -> one DICOM SEG at out_path (None if all masks are empty)."""
    import highdicom as hd
    from pydicom.sr.codedict import codes
    from pydicom.uid import JPEG2000Lossless

    src = pydicom.dcmread(dcm_file)
    names = [n for n, m in masks.items() if m is not None and np.asarray(m).any()]
    if not names:
        return None
    shape = (int(src.Rows), int(src.Columns))
    algo = hd.AlgorithmIdentificationSequence(
        name=algorithm_name, version=algorithm_version, family=codes.cid7162.ArtificialIntelligence
    )
    descriptions, planes = [], []
    for k, name in enumerate(names, start=1):
        m = np.asarray(masks[name]).astype(bool)
        assert m.shape == shape, f"mask {name} has shape {m.shape}, image is {shape}"
        descriptions.append(hd.seg.SegmentDescription(
            segment_number=k,
            segment_label=name.strip()[:64],
            segmented_property_category=codes.SCT.AnatomicalStructure,
            segmented_property_type=_code(name),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algo,
        ))
        planes.append(m)
    seg = hd.seg.Segmentation(
        source_images=[src],
        # (frames=1, rows, cols, segments) as a view of a segment-major stack: every segment highdicom reads is then
        # contiguous (a channel-last copy plus strided per-segment reads took ~20 s on a 3000 px radiograph)
        pixel_array=np.moveaxis(np.stack(planes, axis=0).view(np.uint8), 0, -1)[None],
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=descriptions,
        series_instance_uid=hd.UID(),
        series_number=300,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=MANUFACTURER,
        manufacturer_model_name=algorithm_name,
        software_versions=algorithm_version,
        device_serial_number="0",
        series_description=series_description,
        transfer_syntax_uid=JPEG2000Lossless,
        workers=int(os.environ.get("CXAS_ENCODE_WORKERS", 0)),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    seg.save_as(out_path)
    return out_path


def write_landmark_sr(
    dcm_file: str,
    landmarks: Iterable[dict],
    out_path: str,
    algorithm_name: str = "CXAS landmarks",
    algorithm_version: str = "unknown",
    min_confidence: float = 0.0,
) -> Optional[str]:
    """landmarks: [{class, landmark, x, y, confidence}] in source-image pixels -> TID 1500 SR at out_path."""
    import highdicom as hd
    from highdicom.sr import (
        CodedConcept, Comprehensive3DSR, DeviceObserverIdentifyingAttributes, FindingSite, GraphicTypeValues,
        ImageRegion, MeasurementReport, ObservationContext, ObserverContext,
        PlanarROIMeasurementsAndQualitativeEvaluations, QualitativeEvaluation, SourceImageForRegion,
        TrackingIdentifier,
    )
    from pydicom.sr.codedict import codes

    src = pydicom.dcmread(dcm_file, stop_before_pixels=True)
    ref = SourceImageForRegion(
        referenced_sop_class_uid=src.SOPClassUID, referenced_sop_instance_uid=src.SOPInstanceUID
    )
    groups: List = []
    for lm in landmarks:
        if float(lm.get("confidence", 1.0)) < min_confidence:
            continue
        x, y = float(lm["x"]), float(lm["y"])
        if not (0 <= x < int(src.Columns) and 0 <= y < int(src.Rows)):
            continue
        label = f"{lm['class'].strip()}: {lm['landmark']}"
        groups.append(PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(uid=hd.UID(), identifier=label[:64]),
            referenced_region=ImageRegion(
                graphic_type=GraphicTypeValues.POINT,
                graphic_data=np.array([[x, y]], dtype=float),
                source_image=ref,
            ),
            finding_type=codes.SCT.AnatomicalStructure,
            finding_sites=[FindingSite(anatomic_location=_code(lm["class"]))],
            qualitative_evaluations=[QualitativeEvaluation(
                name=CodedConcept(value="LANDMARK", scheme_designator=PRIVATE_SCHEME, meaning="Landmark"),
                value=CodedConcept(value=_slug(lm["landmark"]), scheme_designator=PRIVATE_SCHEME,
                                   meaning=str(lm["landmark"])[:64]),
            )],
        ))
    if not groups:
        return None
    observer = ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
            uid=hd.UID(), name=algorithm_name, model_name=algorithm_version
        ),
    )
    report = MeasurementReport(
        observation_context=ObservationContext(observer_device_context=observer),
        procedure_reported=CodedConcept(value="36643-5", scheme_designator="LN", meaning="Chest X-ray 2 Views"),
        imaging_measurements=groups,
    )
    sr = Comprehensive3DSR(
        evidence=[src],
        content=report,
        series_instance_uid=hd.UID(),
        series_number=301,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=MANUFACTURER,
        series_description="CXAS anatomical landmarks",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sr.save_as(out_path)
    return out_path


def write_dicom_seg(metainfo: str, dcm_file: str, mask: np.ndarray, out_dir: str, id_label_dict: dict) -> Optional[str]:
    """Backwards-compatible entry point: (C, H, W) mask in a label space -> one SEG file in out_dir."""
    masks = {id_label_dict[str(i)]: mask[i] for i in range(mask.shape[0])}
    stem = os.path.splitext(os.path.basename(dcm_file))[0]
    return write_seg(dcm_file, masks, os.path.join(out_dir, stem + "_seg.dcm"))
