"""Example chest radiographs for the demo notebooks, downloaded from the NCI Imaging Data Commons (IDC).

All three images come from CC BY collections:
  - frontal_pa.dcm, lateral.dcm: one patient's PA + lateral pair from LIDC-IDRI (CC BY 3.0),
    https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
  - frontal_ap_portable.dcm: a portable AP film with lines and leads from COVID-19-NY-SBU (CC BY 4.0),
    https://doi.org/10.7937/TCIA.BBAG-2923
IDC: Fedorov et al., RadioGraphics 43(12), 2023, https://doi.org/10.1148/rg.230180

About 30 MB in total; needs `pip install idc-index`. The frontal PA is also saved as PNG and JPG (DICOM window applied)
to show the non-DICOM inputs.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

LIDC_SERIES = "1.3.6.1.4.1.14519.5.2.1.6279.6001.239451150743822952667769822351"
SBU_SERIES = "1.3.6.1.4.1.14519.5.2.1.99.1071.14770952329205836403990748256437"
IMAGES = {  # file name -> (series, SOPInstanceUID)
    "frontal_pa.dcm": (LIDC_SERIES, "1.3.6.1.4.1.14519.5.2.1.6279.6001.259024827419628700387407156445"),
    "lateral.dcm": (LIDC_SERIES, "1.3.6.1.4.1.14519.5.2.1.6279.6001.203849802556601198380234975276"),
    "frontal_ap_portable.dcm": (SBU_SERIES, "1.3.6.1.4.1.14519.5.2.1.99.1071.33585871997107358488394652161423"),
}


def _to_8bit(ds):
    import numpy as np
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    a = apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds, index=0).astype(np.float32)
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        a = 1.0 - a
    return (a * 255).round().astype(np.uint8)


def get_example_images(dest: str | os.PathLike = "example_data") -> dict:
    """Downloads the example images once (skipped when present) and returns {name: path}.

    Names: frontal_pa, lateral, frontal_ap_portable (DICOM), frontal_pa_png, frontal_pa_jpg, and dir (the folder).
    """
    import pydicom
    from PIL import Image

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not all((dest / n).exists() for n in IMAGES):
        from idc_index import IDCClient
        raw = dest / "_download"
        IDCClient().download_from_selection(downloadDir=str(raw), seriesInstanceUID=[LIDC_SERIES, SBU_SERIES],
                                            dirTemplate="%SeriesInstanceUID", quiet=True)
        by_sop = {}
        for p in raw.rglob("*.dcm"):
            by_sop[pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID] = p
        for name, (_, sop) in IMAGES.items():
            shutil.copy(by_sop[sop], dest / name)
        shutil.rmtree(raw)
    out = {Path(n).stem: str(dest / n) for n in IMAGES}
    png, jpg = dest / "png" / "frontal_pa.png", dest / "jpg" / "frontal_pa.jpg"
    if not png.exists() or not jpg.exists():
        img = Image.fromarray(_to_8bit(pydicom.dcmread(out["frontal_pa"])))
        png.parent.mkdir(exist_ok=True)
        jpg.parent.mkdir(exist_ok=True)
        img.save(png)
        img.save(jpg, quality=95)
    out.update(frontal_pa_png=str(png), frontal_pa_jpg=str(jpg), dir=str(dest))
    return out


# ---------------------------------------------------------------------------------------------------------------
# plotting helpers shared by the notebooks

COLOURS = ["#4e79a7", "#76b7b2", "#e15759", "#f28e2b", "#edc948", "#59a14f", "#b07aa1", "#ff9da7", "#9c755f",
           "#bab0ac", "#d4a6c8", "#8cd17d"]
NAMED_COLOURS = {"right lung": "#4e79a7", "left lung": "#76b7b2", "lung": "#4e79a7", "heart": "#e15759",
                 "aorta": "#f28e2b", "trachea": "#edc948", "spine": "#d4a6c8", "sternum": "#9c755f",
                 "clavicles": "#9c755f", "diaphragm": "#59a14f", "scapulas": "#bab0ac"}


def display_image(path: str):
    """Greyscale float image in [0, 1] for display: DICOM window applied, MONOCHROME1 inverted; PNG/JPG as is."""
    import numpy as np
    if str(path).lower().endswith(".dcm"):
        import pydicom
        return _to_8bit(pydicom.dcmread(path)).astype(np.float32) / 255
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255


def overlay(ax, image, masks: dict, alpha: float = 0.45, legend: bool = True, title: str | None = None):
    """Draws {name: bool mask} over a greyscale image, one colour per name; masks are resized to the image."""
    import cv2
    import matplotlib.colors as mc
    import numpy as np
    from matplotlib.patches import Patch
    rgb = np.dstack([image] * 3).astype(np.float32)
    handles = []
    for i, (name, m) in enumerate(masks.items()):
        m = np.asarray(m).astype(np.uint8)
        if m.shape != image.shape:
            m = cv2.resize(m, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
        if not m.any():
            continue
        colour = NAMED_COLOURS.get(name, COLOURS[i % len(COLOURS)])
        c = np.array(mc.to_rgb(colour))
        sel = m.astype(bool)
        rgb[sel] = (1 - alpha) * rgb[sel] + alpha * c
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cs, -1, tuple(float(v) for v in c), max(1, image.shape[0] // 400))
        handles.append(Patch(color=colour, label=name))
    ax.imshow(np.clip(rgb, 0, 1))
    ax.axis("off")
    if legend and handles:
        ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.8)
    if title:
        ax.set_title(title, fontsize=10)
