"""DICOM in, DICOM out: anatomy segmentation (DICOM SEG) and named landmarks (DICOM SR, TID 1500) per radiograph.

The entry point for Imaging Data Commons / MHub style batch processing:

    cxas_idc -i /data/dicom -o /data/out                      # every .dcm below /data/dicom
    cxas_idc -i image.dcm -o out --no-landmarks --view frontal

Per input image (single-frame CR / DX):
    <out>/<SOPInstanceUID>/seg.dcm             one multi-segment DICOM SEG, native resolution, lossless JPEG 2000
    <out>/<SOPInstanceUID>/landmarks_sr.dcm    TID 1500 measurement report, one POINT per landmark
    <out>/<SOPInstanceUID>/result.json         view used, per-structure scores, landmarks (pixel coordinates)

The view is taken from ViewPosition (0018,5101); when it is missing it is decided from the frontal model's
output (left and right lung side by side). ``--view`` forces it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pydicom


def _version() -> str:
    try:
        return version("cxas")
    except PackageNotFoundError:
        return "dev"


def _inputs(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob.glob(os.path.join(p, "**", "*.dcm"), recursive=True))
        else:
            yield p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cxas_idc", description=__doc__.split("\n\n")[0])
    ap.add_argument("-i", "--input", nargs="+", required=True, help="DICOM files and/or directories")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--model", default="M2F_anatomy_v2", choices=["M2F_anatomy_v2", "UNet_ResNet50_default"])
    ap.add_argument("--view", default="auto", choices=["auto", "frontal", "lateral"])
    ap.add_argument("--no-landmarks", action="store_true")
    ap.add_argument("--cxas-classes-only", action="store_true",
                    help="write only the 159 CXAS classes, not the model's additional ones")
    ap.add_argument("--score-floor", type=float, default=None, help="default: the model's own (0.05)")
    ap.add_argument("--landmark-min-confidence", type=float, default=0.1,
                    help="landmarks below this heatmap confidence are left out of the DICOM SR (all are kept in "
                         "result.json). Default 0.1: drops ~4%% on typical radiographs and the unstable argmax "
                         "positions of absent structures")
    ap.add_argument("-g", "--gpus", default="auto",
                    help="'auto' (GPU 0 if CUDA is available, else CPU), 'cpu', or a GPU id")
    a = ap.parse_args(argv)

    import torch

    from .segmentor import CXAS
    from .io_utils.dicomseg_2d import write_landmark_sr, write_seg
    from .landmarks.measurements import UNITS, measure
    from .units import landmarks_in_mm, physical_spacing

    if a.gpus == "auto":
        a.gpus = "0" if torch.cuda.is_available() else "cpu"
    elif a.gpus != "cpu" and not torch.cuda.is_available():
        print(f"CUDA is not available; running on CPU instead of GPU {a.gpus}", file=sys.stderr)
        a.gpus = "cpu"
    print(f"device: {'cpu' if a.gpus == 'cpu' else 'cuda:' + a.gpus.split(',')[0]}", flush=True)

    os.makedirs(a.output, exist_ok=True)
    seg_model = CXAS(a.model, gpus=a.gpus)
    is_bundle = hasattr(seg_model.model, "bundles")
    if is_bundle:
        seg_model.model.view_override = None if a.view == "auto" else a.view
        seg_model.model.score_floor = a.score_floor
    names = [seg_model.model.cxas_names[i] for i in range(159)] if is_bundle else None
    if names is None:
        from .label_mapper import id2label_dict
        names = [id2label_dict[str(i)].strip() for i in range(len(id2label_dict))]

    det = None
    if not a.no_landmarks:
        from .landmarks import LandmarkDetector
        det = LandmarkDetector(device="cpu" if a.gpus == "cpu" else f"cuda:{a.gpus.split(',')[0]}")

    algo = f"CXAS {a.model}"
    n_ok = n_fail = 0
    for f in _inputs(a.input):
        t0 = time.time()
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if ds.get("Modality") not in ("CR", "DX"):
                print(f"skip {f}: modality {ds.get('Modality')}", file=sys.stderr)
                continue
            out_dir = os.path.join(a.output, str(ds.SOPInstanceUID))
            pred = seg_model.process_file(f)
            if is_bundle:
                view = pred["view"][0]
                masks = {n: pred["segmentation_preds"][0][i].numpy() for i, n in enumerate(names)}
                if not a.cxas_classes_only:
                    masks.update(pred["extra_segmentation"][0])
                scores = pred["instances"][0]
            else:
                fd = seg_model.fileloader.load_file(f)
                pred["orig_data"] = fd["orig_data"]
                view = a.view if a.view != "auto" else (fd.get("view") or "frontal")
                full = seg_model.resize_to_numpy(pred["segmentation_preds"][0], fd["file_size"])
                masks = {n: full[i] for i, n in enumerate(names)}
                scores = {}
            seg_path = write_seg(f, masks, os.path.join(out_dir, "seg.dcm"), algo, _version())
            lms, sr_path, measures = [], None, {}
            spacing = physical_spacing(pydicom.dcmread(f, stop_before_pixels=True))
            if det is not None:
                img = np.ascontiguousarray(np.asarray(pred["orig_data"]).transpose(1, 2, 0))
                lms = det.predict(img, view=view)
                H, W = img.shape[:2]
                for lm in lms:      # same rule as the SR writer: confident and inside the image
                    lm["in_sr"] = bool(lm["confidence"] >= a.landmark_min_confidence
                                       and 0 <= lm["x"] < W and 0 <= lm["y"] < H)
                measures = measure(lms, view, spacing=spacing, min_confidence=a.landmark_min_confidence)
                lms = landmarks_in_mm(lms, spacing)
                sr_path = write_landmark_sr(f, lms, os.path.join(out_dir, "landmarks_sr.dcm"),
                                            "CXAS landmarks_v1", _version(),
                                            min_confidence=a.landmark_min_confidence,
                                            measurements=measures, units=UNITS)
            json.dump({"input": os.path.abspath(f), "view": view, "dicom_view_position": str(ds.get("ViewPosition", "")),
                       "model": a.model, "cxas_version": _version(), "seg": seg_path, "landmarks_sr": sr_path,
                       "structures": scores, "spacing": spacing,
                       "measurements": {k: (None if v != v else v) for k, v in measures.items()},
                       "measurement_units": {k: UNITS.get(k) for k in measures},
                       "landmark_min_confidence": a.landmark_min_confidence, "landmarks": lms},
                      open(os.path.join(out_dir, "result.json"), "w"), indent=1)
            n_ok += 1
            print(f"{f}: view={view} structures={sum(bool(np.asarray(m).any()) for m in masks.values())} "
                  f"landmarks={sum(l.get('in_sr', True) for l in lms)}/{len(lms)} {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # keep going over a cohort; report at the end
            n_fail += 1
            print(f"FAILED {f}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"done: {n_ok} ok, {n_fail} failed")
    return 1 if n_fail and not n_ok else 0


if __name__ == "__main__":
    sys.exit(main())
