"""Landmark-based registration: a similarity transform fitted to many named landmarks.

Each predicted landmark that also has a reference position contributes one correspondence, weighted by its
confidence. The fit is a weighted least-squares similarity transform (rotation, uniform scale, translation;
Umeyama 1991) that is refitted after dropping points whose residual exceeds `outlier_factor` times the median
residual, so a single misplaced landmark cannot pull the result. With two points (the T4/T10 method) there is
nothing to cross-check; with dozens there is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REFERENCE = Path(__file__).resolve().parent.parent / "data" / "landmark_reference_v1.json"


def load_landmark_reference(path: Optional[str] = None) -> Dict[str, Dict[Tuple[str, str], np.ndarray]]:
    """{view: {(class, landmark): (x, y) in the 512 x 512 reference frame}}."""
    ref = json.load(open(path or REFERENCE))
    return {view: {(l["class"], l["landmark"]): np.array([l["x"], l["y"]], float) for l in v["landmarks"]}
            for view, v in ref["views"].items()}


def weighted_similarity(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """2x3 matrix M minimising sum_i w_i |M [src_i, 1] - dst_i|^2 over similarity transforms."""
    w = w / w.sum()
    mu_s, mu_d = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    s, d = src - mu_s, dst - mu_d
    cov = (w[:, None, None] * d[:, :, None] * s[:, None, :]).sum(0)
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    var_s = (w * (s ** 2).sum(1)).sum()
    scale = float(np.trace(np.diag(S) @ D) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]])


def fit_landmarks(landmarks: List[dict], reference: Dict[Tuple[str, str], np.ndarray], image_hw: Tuple[int, int],
                  min_confidence: float = 0.3, outlier_factor: float = 3.0, iterations: int = 3,
                  min_points: int = 6, allow_mirror: bool = False) -> Tuple[Optional[np.ndarray], dict]:
    """Similarity transform (2x3, image pixels -> image pixels) taking the landmarks onto the reference, with the
    reference frame (512 x 512) scaled to the image size as in the T4/T10 method. Returns (matrix or None, info).

    allow_mirror: also try the horizontally mirrored image and keep the better fit. For laterals, which are taken
    facing either way; never for frontals, where a mirror would swap left and right."""
    if allow_mirror:
        facing = facing_right(landmarks)
        if facing is not None:
            # decided by the anatomy (sternum in front of the spine): the reference laterals face right
            if facing:
                M, info = _fit(landmarks, reference, image_hw, min_confidence, outlier_factor, iterations, min_points)
                return M, {**info, "mirrored": False} if M is not None else info
            W = image_hw[1]
            flip = np.array([[-1.0, 0.0, W - 1.0], [0.0, 1.0, 0.0]])
            M, info = _fit([dict(lm, x=W - 1.0 - float(lm["x"])) for lm in landmarks], reference, image_hw,
                           min_confidence, outlier_factor, iterations, min_points)
            if M is None:
                return M, info
            return np.hstack([M[:, :2] @ flip[:, :2], (M[:, :2] @ flip[:, 2] + M[:, 2])[:, None]]), {**info, "mirrored": True}
    best = _fit(landmarks, reference, image_hw, min_confidence, outlier_factor, iterations, min_points)
    if allow_mirror:
        W = image_hw[1]
        flip = np.array([[-1.0, 0.0, W - 1.0], [0.0, 1.0, 0.0]])
        mirrored = [dict(lm, x=W - 1.0 - float(lm["x"])) for lm in landmarks]
        alt = _fit(mirrored, reference, image_hw, min_confidence, outlier_factor, iterations, min_points)
        if alt[0] is not None and (best[0] is None or alt[1]["rms_residual_px"] < best[1]["rms_residual_px"]):
            M = alt[0]
            composed = M[:, :2] @ flip[:, :2]
            alt = (np.hstack([composed, (M[:, :2] @ flip[:, 2] + M[:, 2])[:, None]]), {**alt[1], "mirrored": True})
            return alt
    if best[0] is not None and allow_mirror:
        best[1]["mirrored"] = False
    return best


def facing_right(landmarks, min_confidence: float = 0.1) -> Optional[bool]:
    """Lateral view: True if the patient faces right (sternum right of the spine), None if undecidable."""
    st = [float(l["x"]) for l in landmarks if l["class"].strip() == "sternum" and l.get("confidence", 1) >= min_confidence]
    vt = [float(l["x"]) for l in landmarks if l["class"].startswith("vertebrae") and l.get("confidence", 1) >= min_confidence]
    if len(st) < 2 or len(vt) < 4:
        return None
    return bool(np.mean(st) > np.mean(vt))


def _fit(landmarks, reference, image_hw, min_confidence, outlier_factor, iterations, min_points):
    H, W = image_hw
    ref_scale = np.array([W / 512.0, H / 512.0])
    src, dst, w, keys = [], [], [], []
    for lm in landmarks:
        key = (lm["class"].strip(), lm["landmark"])
        if key in reference and float(lm.get("confidence", 1.0)) >= min_confidence:
            src.append((float(lm["x"]), float(lm["y"])))
            dst.append(reference[key] * ref_scale)
            w.append(float(lm.get("confidence", 1.0)))
            keys.append(key)
    info = {"n_points": len(src)}
    if len(src) < min_points:
        info["error"] = f"only {len(src)} usable landmarks (need {min_points})"
        return None, info
    src, dst, w = np.array(src), np.array(dst), np.array(w)
    keep = np.ones(len(src), bool)
    for _ in range(iterations):
        M = weighted_similarity(src[keep], dst[keep], w[keep])
        res = np.linalg.norm(src @ M[:, :2].T + M[:, 2] - dst, axis=1)
        new_keep = res <= outlier_factor * max(np.median(res[keep]), 1e-6)
        if new_keep.sum() < min_points or (new_keep == keep).all():
            break
        keep = new_keep
    M = weighted_similarity(src[keep], dst[keep], w[keep])
    res = np.linalg.norm(src @ M[:, :2].T + M[:, 2] - dst, axis=1)
    info.update(n_inliers=int(keep.sum()), rms_residual_px=float(np.sqrt(np.mean(res[keep] ** 2))),
                outliers=[f"{c}: {n}" for (c, n), k in zip(keys, keep) if not k])
    return M, info
