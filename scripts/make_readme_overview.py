#!/usr/bin/env python
"""README overview figure: what CXAS predicts on a frontal and a lateral chest radiograph.

Panels (all from M2F_anatomy_v2 and the landmark model, at native resolution, shown downsampled):
heart & great vessels | bones | lung zones | named landmarks (frontal) | segmentation + landmarks (lateral).
Masks are translucent fills with thin white outlines (the style of the previous overview); landmarks are coloured
by anatomical region with the palette of the landmark paper.

    python scripts/make_readme_overview.py --frontal 41460.png --lateral lateral.dcm --out readme_images/overview.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

BG, INK, MUTED = "#0d1117", "#e6edf3", "#8b949e"
REGION_COLOR = {"Heart & mediastinum": "#e76f51", "Vessels": "#b5179e", "Airway": "#e9c46a",
                "Devices / lines": "#f4a261", "Lungs": "#2a9d8f", "Pleura & diaphragm": "#6b97a8",
                "Bones": "#7b8cad", "Soft tissue": "#c9a0a6", "Abdomen / other": "#9aa0a6", "Meta": "#d5d8dc"}
GROUP_REGION = {"G01": "Heart & mediastinum", "G02": "Pleura & diaphragm", "G03": "Heart & mediastinum",
                "G04": "Pleura & diaphragm", "G05": "Lungs", "G06": "Lungs", "G07": "Heart & mediastinum",
                "G08": "Heart & mediastinum", "G09": "Vessels", "G10": "Vessels", "G11": "Vessels", "G12": "Airway",
                "G13": "Airway", "G14": "Devices / lines", "G15": "Devices / lines", "G16": "Heart & mediastinum",
                "G17": "Bones", "G18": "Bones", "G19": "Bones", "G20": "Bones", "G21": "Lungs", "G22": "Lungs",
                "G23": "Soft tissue", "G24": "Bones", "G25": "Vessels", "G26": "Abdomen / other", "G27": "Meta"}
LEGEND = ["Heart & mediastinum", "Vessels", "Airway", "Lungs", "Pleura & diaphragm", "Bones"]

ORD = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, 13)]
VESSELS = ["heart", "ascending aorta", "aortic arch", "descending aorta", "pulmonary artery", "trachea", "esophagus"]
BONES = (["scapula right", "scapula left"]
         + [f"posterior {o} rib {s}" for o in ORD for s in ("right", "left")]
         + [f"vertebrae {p}{i}" for p, n in (("C", 7), ("T", 12), ("L", 5)) for i in range(1, n + 1)]
         + ["clavicle right", "clavicle left"])
ZONES = [f"{s} {z}" for s in ("right", "left") for z in ("apical zone lung", "upper zone lung", "mid zone lung", "lung base")]
LATERAL = (["apical zone lung", "upper zone lung", "mid zone lung", "lung base", "heart", "trachea", "sternum"]
           + [f"vertebrae {p}{i}" for p, n in (("C", 7), ("T", 12), ("L", 5)) for i in range(1, n + 1)])
CONTEXT = {"frontal": ["right lung", "left lung", "heart", "trachea", "right hemidiaphragm", "left hemidiaphragm"],
           "lateral": []}
# (class, landmark): (label, text offset in points)
LABELS = {"frontal": {("tracheal bifurcation", "carina"): ("Carina", (14, -4)),
                      ("aortic arch", "knob"): ("Aortic knob", (14, -2)),
                      ("heart", "right_cardiac_border"): ("R cardiac border", (-12, 10)),
                      ("heart", "left_cardiac_border"): ("L cardiac border", (-10, 16)),
                      ("right lung", "costophrenic_pulmonary"): ("R CP angle", (8, 14)),
                      ("left lung", "costophrenic_pulmonary"): ("L CP angle", (-8, 14))},
          "lateral": {("sternum", "jugular_notch"): ("Jugular notch", (12, -6)),
                      ("sternum", "sternal_angle"): ("Sternal angle", (12, 0)),
                      ("diaphragm", "posterior_costophrenic"): ("Posterior costophrenic angle", (-10, -14))}}


def level_colors(n):
    cmap = matplotlib.colormaps["Spectral"]
    return [matplotlib.colors.to_hex(cmap(0.05 + 0.9 * i / max(n - 1, 1))) for i in range(n)]


def figure_colors():
    c = {"heart": "#ef8a62", "ascending aorta": "#b2abd2", "aortic arch": "#8073ac", "descending aorta": "#542788",
         "pulmonary artery": "#4393c3", "trachea": "#fee08b", "esophagus": "#66bd63", "sternum": "#80cdc1",
         "diaphragm": "#92c5de", "clavicle right": "#35978f", "clavicle left": "#35978f",
         "scapula right": "#80cdc1", "scapula left": "#80cdc1"}
    rib = level_colors(12)
    for i, o in enumerate(ORD):
        for side in ("right", "left"):
            c[f"posterior {o} rib {side}"] = rib[i]
    verts = [f"vertebrae {p}{i}" for p, n in (("C", 7), ("T", 12), ("L", 5)) for i in range(1, n + 1)]
    for v, col in zip(verts, level_colors(len(verts))):
        c[v] = col
    zone = dict(zip(("apical zone lung", "upper zone lung", "mid zone lung", "lung base"),
                    ("#fdae61", "#fee08b", "#abdda4", "#66c2a5")))
    for z, col in zone.items():
        c[z] = col
        for side in ("right", "left"):
            c[f"{side} {z}"] = col
    return c


def landmark_region():
    import cxas.landmarks as L
    rows = csv.DictReader(open(Path(L.__file__).parent / "data" / "landmarks.csv"))
    out = {}
    for r in rows:
        for col in ("classes_frontal", "classes_lateral"):
            for c in [x.strip() for x in r[col].replace("|", ";").split(";") if x.strip()]:
                out[(c, r["landmark"])] = GROUP_REGION.get(r["group"], "Meta")
    return out


def pad_square(a, fill=0):
    """Pad the short side symmetrically so the whole image stays visible; returns (array, x offset, y offset)."""
    H, W = a.shape[:2]
    n = max(H, W)
    oy, ox = (n - H) // 2, (n - W) // 2
    out = np.full((n, n) + a.shape[2:], fill, dtype=a.dtype)
    out[oy:oy + H, ox:ox + W] = a
    return out, ox, oy


def draw_masks(ax, masks, names, colors, size):
    for n in names:
        m = masks.get(n)
        if m is None or not m.any():
            continue
        m = cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        rgb = matplotlib.colors.to_rgb(colors[n])
        layer = np.zeros((size, size, 4)); layer[m > 0] = (*rgb, 0.42)
        ax.imshow(layer, interpolation="nearest")
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for ct in cs:
            if len(ct) > 2:
                ct = ct[:, 0]
                ax.plot(np.r_[ct[:, 0], ct[0, 0]], np.r_[ct[:, 1], ct[0, 1]], color="white", lw=0.7, alpha=0.9)


def draw_outlines(ax, masks, names, size):
    for n in names:
        m = masks.get(n)
        if m is None or not m.any():
            continue
        m = cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for ct in cs:
            if len(ct) > 2:
                ct = ct[:, 0]
                ax.plot(np.r_[ct[:, 0], ct[0, 0]], np.r_[ct[:, 1], ct[0, 1]], color="white", lw=0.7, alpha=0.35)


def draw_landmarks(ax, lms, region, view, sx, sy, ox, oy, size, labelled=True):
    stroke = [pe.withStroke(linewidth=2.6, foreground="black")]
    for lm in lms:
        x, y = (lm["x"] - ox) * sx, (lm["y"] - oy) * sy
        if not (0 <= x < size and 0 <= y < size):
            continue
        col = REGION_COLOR[region.get((lm["class"], lm["landmark"]), "Meta")]
        key = (lm["class"], lm["landmark"])
        big = labelled and key in LABELS[view]
        ax.plot(x, y, "o", ms=7 if big else 3.2, mfc=col, mec="white", mew=1.0 if big else 0.4,
                alpha=1.0 if big else 0.85, zorder=5 if big else 4)
        if big:
            text, (dx, dy) = LABELS[view][key]
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color="white", fontsize=8,
                        va="center", ha="left" if dx >= 0 else "right", path_effects=stroke, zorder=6,
                        arrowprops=dict(arrowstyle="-", color="white", lw=0.6, alpha=0.8, shrinkA=0, shrinkB=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontal", required=True)
    ap.add_argument("--lateral", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()

    from cxas import CXAS
    from cxas.landmarks import LandmarkDetector

    model = CXAS("M2F_anatomy_v2", gpus=a.gpus)
    det = LandmarkDetector(device="cpu" if a.gpus == "cpu" else f"cuda:{a.gpus}")
    region = landmark_region()
    S = a.size
    panels = []
    for path, view in ((a.frontal, "frontal"), (a.lateral, "lateral")):
        model.model.view_override = view
        p = model.process_file(path)
        img = np.ascontiguousarray(p["orig_data"].transpose(1, 2, 0))
        H, W = img.shape[:2]
        names = model.model.cxas_names
        masks = {n: p["segmentation_preds"][0][i].numpy() for i, n in enumerate(names)}
        masks.update(p["extra_segmentation"][0])
        lms = [l for l in det.predict(img, view=view) if l["confidence"] >= 0.1]
        sq, ox, oy = pad_square(img[..., 0])
        crop = cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA)
        cmasks = {n: pad_square(m.astype(np.uint8))[0] for n, m in masks.items()}
        panels.append(dict(view=view, img=crop, masks=cmasks, lms=lms, s=S / sq.shape[0], ox=-ox, oy=-oy))

    fr, la = panels
    cols = figure_colors()
    titles = ["Heart & great vessels", "Bones", "Lung zones", "Named landmarks", "Lateral: anatomy + landmarks"]
    fig = plt.figure(figsize=(5 * 3.2 + 0.4, 3.2 + 0.55), facecolor=BG)
    gap = 0.004
    w = (1 - 6 * gap) / 5
    for i, t in enumerate(titles):
        ax = fig.add_axes([gap + i * (w + gap), 0.135, w, 0.855])
        ax.set_axis_off()
        pnl = la if i == 4 else fr
        ax.imshow(pnl["img"], cmap="gray", vmin=0, vmax=255)
        if i == 0:
            draw_masks(ax, pnl["masks"], VESSELS, cols, S)
        elif i == 1:
            draw_masks(ax, pnl["masks"], BONES, cols, S)
        elif i == 2:
            draw_masks(ax, pnl["masks"], ZONES, cols, S)
        elif i == 3:
            draw_outlines(ax, pnl["masks"], CONTEXT["frontal"], S)
            draw_landmarks(ax, pnl["lms"], region, "frontal", pnl["s"], pnl["s"], pnl["ox"], pnl["oy"], S)
        else:
            draw_masks(ax, pnl["masks"], LATERAL, cols, S)
            draw_landmarks(ax, pnl["lms"], region, "lateral", pnl["s"], pnl["s"], pnl["ox"], pnl["oy"], S)
        ax.set_xlim(0, S); ax.set_ylim(S, 0)
        fig.text(gap + i * (w + gap) + w / 2, 0.06, t, color=INK, ha="center", va="center", fontsize=11)
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
