#!/usr/bin/env python
"""Reference landmark positions for landmark-based registration (cxas/data/landmark_reference_v1.json).

Built from real radiographs: CC BY chest CR/DX series from the Imaging Data Commons
(scripts/landmark_reference_idc_series.txt; download them with the `idc` CLI or s5cmd). For every radiograph:
view from the DICOM ViewPosition (else CXAS's view check), landmarks from the CXAS landmark model (confidence >= 0.3),
positions normalised to the 512 x 512 registration frame, laterals mirrored to face right. Per view, the
reference is the generalised Procrustes mean (similarity alignment, per-landmark median, 6 iterations) of the
landmarks present on at least half of the radiographs. The frontal mean is then moved so that its T4 and T10
vertebral-body centres sit on the T4/T10 positions of the default registration reference, so both registration
methods produce the same framing; the lateral mean keeps the centroid and size of the per-landmark median shape.

Evaluated on 600 held-out MIMIC-CXR radiographs (fit on half of the landmarks, scatter of the other half across
images): frontal 0.148 vs 0.183 for T4/T10, lateral 0.152 vs 0.379 unregistered (fractions of the reference size).

    python scripts/build_landmark_reference.py --dicom-dir /data/idc_chest --gpus 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

MIN_CONF = 0.3
FRONTAL_CORNERS = ("superior_left", "superior_right", "inferior_left", "inferior_right")


def normalised(landmarks, hw, view):
    H, W = hw
    P = {(l["class"].strip(), l["landmark"]): np.array([l["x"] * 512 / W, l["y"] * 512 / H])
         for l in landmarks if l["confidence"] >= MIN_CONF}
    if view == "lateral":
        st = [p[0] for (c, _), p in P.items() if c == "sternum"]
        vt = [p[0] for (c, _), p in P.items() if c.startswith("vertebrae")]
        if st and vt and np.mean(st) < np.mean(vt):
            P = {k: np.array([511.0 - p[0], p[1]]) for k, p in P.items()}
    return P


def procrustes_mean(shapes, keys, iterations=6):
    from cxas.registration.landmark_fit import weighted_similarity
    mean = {k: np.median([s[k] for s in shapes if k in s], 0) for k in keys}
    aligned = []
    for _ in range(iterations):
        aligned = []
        for s in shapes:
            ks = [k for k in keys if k in s]
            if len(ks) < 6:
                continue
            M = weighted_similarity(np.array([s[k] for k in ks]), np.array([mean[k] for k in ks]), np.ones(len(ks)))
            aligned.append({k: M[:, :2] @ s[k] + M[:, 2] for k in ks})
        mean = {k: np.median([a[k] for a in aligned if k in a], 0) for k in keys}
    return mean, len(aligned)


def two_point_similarity(src, dst):
    vs, vd = src[1] - src[0], dst[1] - dst[0]
    sc = np.linalg.norm(vd) / np.linalg.norm(vs)
    a = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0])
    R = sc * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.hstack([R, (dst[0] - R @ src[0])[:, None]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "cxas/data/landmark_reference_v1.json"))
    ap.add_argument("--gpus", default="0")
    a = ap.parse_args()

    from cxas import CXAS
    from cxas.file_io import FileLoader
    from cxas.landmarks import LandmarkDetector
    from cxas.registration.landmark_fit import weighted_similarity
    from cxas.registration.reference import get_default_reference_landmarks, load_reference

    device = "cpu" if a.gpus == "cpu" else f"cuda:{a.gpus}"
    det, seg, fl = LandmarkDetector(device=device), None, FileLoader(a.gpus)
    shapes = {"frontal": [], "lateral": []}
    for p in sorted(glob.glob(os.path.join(a.dicom_dir, "**", "*.dcm"), recursive=True)):
        fd = fl.load_file(p)
        view = fd.get("view")
        if view is None:
            seg = seg or CXAS("M2F_anatomy_v2", gpus=a.gpus)
            view = seg.process_file(p)["view"][0]
        img = np.ascontiguousarray(fd["orig_data"].transpose(1, 2, 0))
        shapes[view].append(normalised(det.predict(img, view=view), img.shape[:2], view))

    default = get_default_reference_landmarks(load_reference(None))
    ref = {"source": "Imaging Data Commons, CC BY chest radiograph series; landmarks predicted by the CXAS landmark "
                     "model, generalised Procrustes mean (median), laterals mirrored to face right",
           "frame": "512 x 512, positions normalised by image size", "views": {}}
    for view, sh in shapes.items():
        cnt = {}
        for s in sh:
            for k in s:
                cnt[k] = cnt.get(k, 0) + 1
        keys = sorted(k for k, c in cnt.items() if c >= 0.5 * len(sh))
        init = {k: np.median([s[k] for s in sh if k in s], 0) for k in keys}
        mean, n_used = procrustes_mean(sh, keys)
        if view == "frontal":
            cen = [np.mean([mean[(f"vertebrae {t}", c)] for c in FRONTAL_CORNERS], 0) for t in ("T4", "T10")]
            M = two_point_similarity(np.array(cen), np.array([[default.t4.x, default.t4.y], [default.t10.x, default.t10.y]]))
        else:
            M = weighted_similarity(np.array([mean[k] for k in keys]), np.array([init[k] for k in keys]), np.ones(len(keys)))
        mean = {k: M[:, :2] @ v + M[:, 2] for k, v in mean.items()}
        ref["views"][view] = {"n_images": n_used, "landmarks": [
            {"class": c, "landmark": n, "x": round(float(mean[(c, n)][0]), 3), "y": round(float(mean[(c, n)][1]), 3),
             "n": int(cnt[(c, n)])} for (c, n) in keys]}
        print(view, len(sh), "radiographs,", n_used, "used,", len(keys), "landmarks")
    json.dump(ref, open(a.out, "w"), indent=0)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
