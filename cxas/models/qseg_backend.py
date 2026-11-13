"""Query-based (Mask2Former) anatomy segmentation backend.

A model is a set of *bundles*, one per view (``frontal`` / ``lateral``), each exported from a qseg training run
by ``scripts/export_qseg_bundle.py``. Per image the backend keeps the best-scoring instance of every class above the
bundle's score floor, applies the bundle's label adapter, and returns masks at the image's native resolution:

* ``segmentation_preds[i]``: bool tensor (159, H, W) in the CXAS label space (``cxas/data/paxray_labels.json``),
  filled by class name, so every CXAS feature, saver and the registration keep working unchanged;
* ``extra_segmentation[i]``: {class name: bool ndarray (H, W)} for the model's classes outside the CXAS label space
  (e.g. sub-diaphragmatic regions, image quadrants, humeri, head);
* ``instances[i]``: {class name: score} of every kept class, and ``view[i]``.

The model code is in ``cxas.models.m2f`` (plain PyTorch + transformers); no qseg or detectron2 needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..label_mapper import id2label_dict

VIEWS = ("frontal", "lateral")
FRONTAL_POSITIONS = {"PA", "AP", "AP AXIAL", "PA LLD", "AP LLD", "AP RLD", "PA RLD"}
LATERAL_POSITIONS = {"LATERAL", "LL", "RL", "LAO", "RAO", "LPO", "RPO", "XTABLE LATERAL", "LEFT LATERAL", "RIGHT LATERAL"}


def view_from_position(view_position: str | None, default: str = "frontal") -> str:
    """DICOM ViewPosition (0018,5101) -> 'frontal' | 'lateral'; unknown or empty -> default."""
    vp = (view_position or "").strip().upper()
    if vp in LATERAL_POSITIONS or vp.endswith("LATERAL"):
        return "lateral"
    if vp in FRONTAL_POSITIONS:
        return "frontal"
    return default


class _Bundle:
    """One view's model: the Mask2Former port in ``cxas.models.m2f`` built from a bundle directory."""

    def __init__(self, path: Path, device: torch.device):
        from .m2f.model import Mask2FormerAnatomy

        self.path = Path(path)
        self.meta = json.load(open(self.path / "meta.json"))
        self.adapter = json.load(open(self.path / "adapter.json"))
        self.model, self.class_names = Mask2FormerAnatomy.from_bundle(self.path)
        self.model = self.model.to(device)
        self.device = device
        self.short_edge = int(self.meta.get("input_short_edge", 512))
        self.score_floor = float(self.meta.get("score_floor", 0.05))

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Shortest edge to the training size, longest edge at most twice that (PIL bilinear, as in training)."""
        from PIL import Image

        h, w = image.shape[:2]
        scale = self.short_edge / min(h, w)
        nh, nw = h * scale, w * scale
        if max(nh, nw) > 2 * self.short_edge:
            s = 2 * self.short_edge / max(nh, nw)
            nh, nw = nh * s, nw * s
        nh, nw = int(nh + 0.5), int(nw + 0.5)
        if (nh, nw) == (h, w):
            return image
        return np.asarray(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))

    @torch.no_grad()
    def predict(self, image: np.ndarray, score_floor: float | None = None) -> dict:
        """image: (H, W, 3) uint8 RGB -> {class name: (score, bool mask (H, W) on CPU)}, adapter applied.

        Inference runs at the training resolution; only the kept masks' logits are upsampled to (H, W)."""
        H, W = image.shape[:2]
        arr = self.resize(image)
        x = torch.as_tensor(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        floor = self.score_floor if score_floor is None else float(score_floor)
        out = {}
        for c, (s, logits) in self.model.best_per_class(x, floor).items():
            m = F.interpolate(logits[None, None].float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0] > 0
            if m.any():
                out[self.class_names[c]] = (s, m.cpu().numpy())
        return apply_adapter(out, self.adapter)


def apply_adapter(masks: dict, adapter: dict) -> dict:
    """{raw name: (score, mask)} -> {true name: (score, mask)}; relabel, drop_raw, then derive (score = min of inputs)."""
    rel = adapter.get("relabel", {})
    drop = set(adapter.get("drop_raw", []))
    out = {rel.get(n, n): v for n, v in masks.items() if n not in drop}
    for name, spec in adapter.get("derive", {}).items():
        if spec.get("only_if_missing") and name in out:
            continue
        if spec["op"] == "minus":
            if spec["from"] not in out or not all(m in out for m in spec["minus"]):
                continue
            m = out[spec["from"]][1].copy()
            for sub in spec["minus"]:
                m &= ~out[sub][1]
            if spec.get("keep") == "largest_component":
                from scipy import ndimage
                lab, k = ndimage.label(m)
                if k == 0:
                    continue
                sizes = ndimage.sum(m, lab, range(1, k + 1))
                m = lab == int(np.argmax(sizes)) + 1
            score = min(out[spec["from"]][0], *[out[s][0] for s in spec["minus"]])
        elif spec["op"] == "union":
            parts = [out[p] for p in spec["parts"] if p in out]
            if not parts:
                continue
            m = np.zeros_like(parts[0][1])
            for _, pm in parts:
                m |= pm
            score = min(s for s, _ in parts)
        else:
            raise ValueError(f"unknown adapter op {spec['op']!r}")
        out[name] = (score, m)
    return out


def looks_frontal(pred: dict, width: int, min_sep: float = 0.2, max_iou: float = 0.2) -> bool:
    """View check on the FRONTAL model's output: on a frontal radiograph its left and right lung are two separate,
    side-by-side masks; on a lateral they overlap or one is missing."""
    if "left lung" not in pred or "right lung" not in pred:
        return False
    l, r = pred["left lung"][1], pred["right lung"][1]
    inter = np.logical_and(l, r).sum()
    union = np.logical_or(l, r).sum()
    if union == 0 or inter / union > max_iou:
        return False
    cx = lambda m: np.nonzero(m)[1].mean()
    return abs(cx(l) - cx(r)) > min_sep * width


class QsegAnatomy(nn.Module):
    """View-routed Mask2Former anatomy segmentor with the CXAS output contract."""

    def __init__(self, bundle_root: str | Path, device: str | torch.device = "cpu", default_view: str = "frontal"):
        super().__init__()
        self.device_ = torch.device(device)
        self.bundles = {}
        for v in VIEWS:
            p = Path(bundle_root) / v
            if (p / "meta.json").exists():
                self.bundles[v] = _Bundle(p, self.device_)
        if not self.bundles:
            raise FileNotFoundError(f"no model bundle under {bundle_root}")
        self.default_view = default_view
        # None: DICOM ViewPosition if present, else automatic; "frontal" / "lateral": force for every image
        self.view_override = None
        self.cxas_names = [id2label_dict[str(i)].strip() for i in range(len(id2label_dict))]
        self.cxas_index = {n: i for i, n in enumerate(self.cxas_names)}
        self.score_floor = None

    def _bundle_for(self, view: str) -> _Bundle:
        return self.bundles.get(view) or self.bundles.get(self.default_view) or next(iter(self.bundles.values()))

    def forward(self, file_dict: dict) -> dict:
        origs = file_dict["orig_data"]
        if isinstance(origs, np.ndarray) and origs.ndim == 3:
            origs = [origs]
        views = file_dict.get("view") or [None] * len(origs)
        if isinstance(views, str):
            views = [views]
        seg, extra, scores, used_views = [], [], [], []
        for orig, v in zip(origs, views):
            image = np.ascontiguousarray(np.asarray(orig).transpose(1, 2, 0))
            H, W = image.shape[:2]
            v = v or self.view_override
            if v in VIEWS:
                pred = self._bundle_for(v).predict(image, self.score_floor)
            else:
                # view unknown (no / unusable ViewPosition): decide from the frontal model's own output
                pred = self._bundle_for("frontal").predict(image, self.score_floor)
                v = "frontal"
                if "lateral" in self.bundles and not looks_frontal(pred, W):
                    v = "lateral"
                    pred = self.bundles["lateral"].predict(image, self.score_floor)
            t = torch.zeros(len(self.cxas_names), H, W, dtype=torch.bool)
            ex = {}
            for name, (s, m) in pred.items():
                if name in self.cxas_index:
                    t[self.cxas_index[name]] = torch.from_numpy(m)
                else:
                    ex[name] = m
            seg.append(t)
            extra.append(ex)
            scores.append({n: round(float(s), 4) for n, (s, _) in pred.items()})
            used_views.append(v)
        return {**file_dict, "segmentation_preds": seg, "extra_segmentation": extra,
                "instances": scores, "view": used_views}
