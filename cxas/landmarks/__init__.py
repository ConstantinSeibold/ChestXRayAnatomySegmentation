"""Named anatomical landmarks (132 definitions, 311 view-specific class-landmark pairs) for chest radiographs.

The joint dense RAD-DINO model of "Semantic landmark detection in chest radiographs" (Seibold, Havemann, Vinu,
Fink). The inference code is vendored from https://github.com/ConstantinSeibold/cxr-landmarks (MIT); the
vocabulary tables (definitions, clinical groups, literature) are in ``cxas/landmarks/data``. Coordinates are
returned in the input image's pixels.

    from cxas.landmarks import LandmarkDetector
    det = LandmarkDetector(device="cuda:0")
    lms = det.predict(image_rgb_uint8, view="frontal")   # [{class, landmark, x, y, confidence}, ...]
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

LANDMARK_MODEL = "landmarks_v1"
LANDMARK_HF_REPO = os.environ.get("CXAS_LANDMARK_REPO", "cmseibold/cxr-landmarks")
LANDMARK_HF_REVISION = os.environ.get("CXAS_LANDMARK_REVISION", "v1.0.0")


def _local_weights() -> Optional[str]:
    from ..models import weights_dir

    p = os.path.join(weights_dir(), LANDMARK_MODEL)
    return p if os.path.isfile(os.path.join(p, "model.safetensors")) else None


class LandmarkDetector:
    def __init__(self, device: Optional[str] = None, source: Optional[str] = None):
        from .predictor import LandmarkPredictor

        self.predictor = LandmarkPredictor.from_pretrained(source or _local_weights() or LANDMARK_HF_REPO,
                                                           device=device, revision=LANDMARK_HF_REVISION)

    def predict(self, image: np.ndarray, view: str = "frontal", min_confidence: float = 0.0) -> List[dict]:
        """image: (H, W, 3) uint8 RGB or (H, W) uint8; view: 'frontal' | 'lateral'."""
        out = self.predictor.predict(image, view=view)
        return [lm for lm in out["landmarks"] if lm["confidence"] >= min_confidence]

    def predict_with_masks(self, image: np.ndarray, view: str = "frontal") -> dict:
        """Landmarks plus the model's own per-structure masks (COCO RLE, one per structure, with a score)."""
        return self.predictor.predict(image, view=view)
