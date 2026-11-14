"""Physical units (mm) for measurements on radiographs with pixel spacing in their DICOM header.

Spacing, in order of preference (DICOM PS3.3, C.8.11.4 / C.7.6.3):
  1. PixelSpacing (0028,0030): calibrated to the patient.
  2. ImagerPixelSpacing (0018,1164) divided by the magnification: EstimatedRadiographicMagnificationFactor
     (0018,1114), or source-to-detector / source-to-patient distance (0018,1110 / 0018,1111).
  3. ImagerPixelSpacing alone: at the detector, so anatomy appears ~5-15% larger than it is.
Every result says which one it used. Images without spacing (PNG, JPG, DICOM without these tags) stay in pixels.

Features are computed on the 512 x 512 grid of the model input, which stretches non-square images differently
along x and y; conversions here therefore use one factor per axis, and perimeters and distances are measured after
scaling the geometry into mm rather than by scaling a pixel length.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def physical_spacing(ds) -> Optional[dict]:
    """{"row_mm", "col_mm", "source"} from a pydicom dataset, or None if the header has no usable spacing."""
    def pair(tag):
        v = ds.get(tag, None)
        try:
            r, c = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError):
            return None
        return (r, c) if r > 0 and c > 0 else None

    ps = pair("PixelSpacing")
    if ps:
        return {"row_mm": ps[0], "col_mm": ps[1], "source": "PixelSpacing"}
    ips = pair("ImagerPixelSpacing")
    if not ips:
        return None
    try:
        ermf = float(ds.get("EstimatedRadiographicMagnificationFactor", 0) or 0)
    except (TypeError, ValueError):
        ermf = 0.0
    if ermf > 0:
        return {"row_mm": ips[0] / ermf, "col_mm": ips[1] / ermf,
                "source": "ImagerPixelSpacing / EstimatedRadiographicMagnificationFactor"}
    try:
        sid = float(ds.get("DistanceSourceToDetector", 0) or 0)
        sod = float(ds.get("DistanceSourceToPatient", 0) or 0)
    except (TypeError, ValueError):
        sid = sod = 0.0
    if sid > 0 and sod > 0 and sid >= sod:
        mag = sid / sod
        return {"row_mm": ips[0] / mag, "col_mm": ips[1] / mag,
                "source": "ImagerPixelSpacing / (DistanceSourceToDetector / DistanceSourceToPatient)"}
    return {"row_mm": ips[0], "col_mm": ips[1], "source": "ImagerPixelSpacing (detector plane, not magnification-corrected)"}


def grid_spacing(spacing: dict, native_hw: Tuple[int, int], grid_hw: Tuple[int, int] = (512, 512)) -> Tuple[float, float]:
    """(mm per grid row, mm per grid column) for masks resized from native_hw to grid_hw."""
    H, W = int(native_hw[0]), int(native_hw[1])
    return spacing["row_mm"] * H / grid_hw[0], spacing["col_mm"] * W / grid_hw[1]


def _perimeter_mm(mask: np.ndarray, sy: float, sx: float) -> float:
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return -1.0
    # the largest outline (as the pixel version measures one), simplified with a 1-pixel tolerance first: a
    # pixel-by-pixel staircase overestimates curved outlines by 5-9%, simplified it is within 1% (discs, any aspect)
    c = cv2.approxPolyDP(max(cs, key=len), 1.0, True)[:, 0, :].astype(np.float64)
    c[:, 0] *= sx
    c[:, 1] *= sy
    return float(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum())


def _scd_mm(mask: np.ndarray, sy: float, sx: float) -> float:
    """Spine-centre distance in mm: mean perpendicular distance of the vertebral-body centres to the line fitted
    through them, with the geometry scaled to mm first."""
    from .label_mapper import label_mapper
    pts = []
    for i in label_mapper["all vertebrae"]:
        m = mask[i] if mask[i].ndim == 2 else mask[i][0]
        if m.any():
            ys, xs = np.nonzero(m)
            pts.append((xs.mean() * sx, ys.mean() * sy))
    if len(pts) < 3:
        return -1.0
    p = np.array(pts)
    a, b = np.polyfit(p[:, 1], p[:, 0], 1)                  # x = a*y + b (a near-vertical spine)
    return float(np.mean(np.abs(p[:, 0] - (a * p[:, 1] + b)) / np.sqrt(1 + a * a)))


def features_in_mm(method: str, feats: dict, mask_grid: np.ndarray, spacing: Optional[dict],
                   native_hw: Tuple[int, int]) -> Dict[str, float]:
    """Physical-unit counterparts of a pixel feature dict (computed on the 512 grid); {} without spacing.

    Keys get a unit suffix (_mm, _mm2) next to the unchanged pixel values; unitless features (CTR, compactness,
    convexity) are not repeated. The spacing used is reported under spacing_row_mm / spacing_col_mm / spacing_source.
    """
    if not spacing:
        return {}
    sy, sx = grid_spacing(spacing, native_hw, mask_grid.shape[-2:])
    out: Dict[str, float] = {"spacing_row_mm": spacing["row_mm"], "spacing_col_mm": spacing["col_mm"],
                             "spacing_source": spacing["source"]}
    m = method.lower()
    if m == "area":
        for k, v in feats.items():
            if k.endswith("_area"):
                out[k + "_mm2"] = float(v) * sx * sy if v is not None and v >= 0 else -1.0
    elif m == "perimeter":
        from .label_mapper import id2label_dict
        for i in range(mask_grid.shape[0]):
            out[id2label_dict[str(i)] + "_perimeter_mm"] = _perimeter_mm(mask_grid[i], sy, sx) if mask_grid[i].any() else -1.0
    elif m == "centroid":
        for k, v in feats.items():
            if k.endswith("_cx"):
                out[k + "_mm"] = float(v) * sx if v >= 0 else -1.0
            elif k.endswith("_cy"):
                out[k + "_mm"] = float(v) * sy if v >= 0 else -1.0
    elif m == "box":
        for k, v in feats.items():
            if k.endswith(("_x", "_width")):
                out[k + "_mm"] = float(v) * sx
            elif k.endswith(("_y", "_height")):
                out[k + "_mm"] = float(v) * sy
    elif m in ("ctr", "cardio-thoracic ratio"):
        for k in ("cardiac_width", "thoracic_width"):
            if k in feats and feats[k] is not None and feats[k] >= 0:
                out[k + "_mm"] = float(feats[k]) * sx            # both widths are horizontal (along image rows)
    elif m in ("scd", "spine-center distance"):
        out["score_mm"] = _scd_mm(mask_grid, sy, sx)
    return out


def landmarks_in_mm(landmarks: list, spacing: Optional[dict]) -> list:
    """Adds x_mm / y_mm (from the image's top-left corner) to landmarks given in native image pixels."""
    if not spacing:
        return landmarks
    return [dict(lm, x_mm=lm["x"] * spacing["col_mm"], y_mm=lm["y"] * spacing["row_mm"]) for lm in landmarks]
