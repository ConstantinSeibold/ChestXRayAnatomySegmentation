"""Vendored from cxr-landmarks (https://github.com/ConstantinSeibold/cxr-landmarks, MIT) @ baa9bcc: inference only.

Inference with the joint dense model. Preprocessing equals the evaluation pipeline of the paper: RGB, shortest
edge resized to 512 with the longest edge capped at 1024 (bilinear), zero padding to a multiple of 32; bf16
autocast on CUDA.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_util

from .decode import decode_dark
from .channels import VIEWS, Channels, view_of
from .model import STRIDE, JointDenseModel, pad_batch

HF_REPO = "cmseibold/cxr-landmarks"
SHORT_EDGE, MAX_EDGE = 512, 1024


def load_image(src) -> np.ndarray:
    """Path or array -> (H, W, 3) uint8 RGB."""
    if isinstance(src, (str, Path)):
        return np.asarray(Image.open(src).convert("RGB"))
    a = np.asarray(src)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.dtype != np.uint8:
        raise ValueError("expected a uint8 image")
    return a


def resize_for_model(img: np.ndarray) -> np.ndarray:
    """detectron2 ResizeShortestEdge(512, max_size=1024) with PIL bilinear, as in the paper."""
    h, w = img.shape[:2]
    scale = SHORT_EDGE / min(h, w)
    nh, nw = h * scale, w * scale
    if max(nh, nw) > MAX_EDGE:
        s = MAX_EDGE / max(nh, nw)
        nh, nw = nh * s, nw * s
    nh, nw = int(nh + 0.5), int(nw + 0.5)
    if (nh, nw) == (h, w):
        return img
    return np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))


class LandmarkPredictor:
    def __init__(self, model: JointDenseModel, channels: Channels, device: str | torch.device | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.ch = channels

    @classmethod
    def from_pretrained(cls, source: str = HF_REPO, device=None, revision: str | None = None) -> "LandmarkPredictor":
        """`source`: a local directory with model.safetensors + config.json, or a Hugging Face repo id (downloaded
        at `revision` and checked against its SHA256SUMS)."""
        from safetensors.torch import load_file
        p = Path(source)
        if not p.is_dir():
            from huggingface_hub import snapshot_download
            from ..models import verify_sha256
            p = Path(snapshot_download(source, revision=revision))
            if (p / "SHA256SUMS").is_file():
                verify_sha256(str(p), str(p / "SHA256SUMS"))
        cfg = json.load(open(p / "config.json"))
        ch = Channels()
        assert (cfg["n_landmarks"], cfg["n_classes"]) == (ch.n, ch.n_classes), "vocabulary / weights mismatch"
        model = JointDenseModel(ch.n, ch.n_classes, pretrained_encoder=False)
        state = load_file(str(p / "model.safetensors"))
        # The encoder goes through transformers' own loader, which maps checkpoint parameter names onto the installed
        # version's (transformers >= 5.10 renamed the DINOv2 attention parameters); the heads load directly.
        import tempfile
        from safetensors.torch import save_file
        from transformers import AutoModel
        enc = {k[len("enc."):]: v.contiguous() for k, v in state.items() if k.startswith("enc.")}
        with tempfile.TemporaryDirectory(prefix="cxas_lm_") as tmp:
            model.enc.config.save_pretrained(tmp)
            save_file(enc, str(Path(tmp) / "model.safetensors"), metadata={"format": "pt"})
            model.enc = AutoModel.from_pretrained(tmp)
        missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if not k.startswith("enc.")},
                                                    strict=False)
        missing = [k for k in missing if not k.startswith("enc.")]
        if missing or unexpected:
            raise RuntimeError(f"landmark weights do not match the model: missing={missing[:5]} unexpected={unexpected[:5]}")
        return cls(model, ch, device)

    @torch.no_grad()
    def raw(self, imgs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        x = pad_batch([torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))) for im in imgs]).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            hm, sl = self.model(x)
        return hm.float(), sl.float(), [im.shape[:2] for im in imgs]

    def _decode(self, hm, sl, inp_hw, orig_hw, view):
        ih, iw = inp_hw
        H, W = orig_hw
        h4, w4 = math.ceil(ih / STRIDE), math.ceil(iw / STRIDE)
        xy, peak = decode_dark(hm[:, :h4, :w4])
        xy = xy * STRIDE + (STRIDE - 1) / 2
        xy[:, 0] *= W / iw
        xy[:, 1] *= H / ih
        p = F.interpolate(sl[None, :, :h4, :w4], size=(ih, iw), mode="bilinear", align_corners=False)
        p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0].sigmoid()
        m = p > 0.5
        area = m.flatten(1).sum(1)
        mscore = (p * m).flatten(1).sum(1) / area.clamp(min=1)
        return xy.cpu().numpy(), peak.cpu().numpy(), m.cpu().numpy(), area.cpu().numpy(), mscore.cpu().numpy()

    def predict_batch(self, images: list, views: list[str]) -> list[dict]:
        imgs = [load_image(s) for s in images]
        hm, sl, inp_hw = self.raw([resize_for_model(im) for im in imgs])
        out = []
        for b, (im, v) in enumerate(zip(imgs, views)):
            if v not in VIEWS:
                raise ValueError(f"view must be one of {VIEWS}")
            xy, peak, m, area, mscore = self._decode(hm[b], sl[b], inp_hw[b], im.shape[:2], v)
            lms = []
            for c, slots in sorted(self.ch.by_class[v].items()):
                for k, chn in sorted(slots):
                    lms.append({"class": self.ch.class_names[c], "landmark": self.ch.kp_names[c][k],
                                "x": float(xy[chn, 0]), "y": float(xy[chn, 1]), "confidence": float(peak[chn])})
            structs = []
            for c in np.nonzero(area)[0]:
                rle = mask_util.encode(np.asfortranarray(m[c].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode()
                structs.append({"class": self.ch.class_names[c], "score": float(mscore[c]), "mask_rle": rle})
            out.append({"height": im.shape[0], "width": im.shape[1], "view": v, "landmarks": lms,
                        "structures": structs, "_xy": xy, "_peak": peak, "_m": m, "_area": area, "_ms": mscore})
        return out

    def predict(self, image, view: str = "frontal") -> dict:
        r = self.predict_batch([image], [view])[0]
        return {k: v for k, v in r.items() if not k.startswith("_")}

    def coco_results(self, r: dict, image_id: int) -> tuple[list[dict], list[dict]]:
        """COCO keypoint results (one per class with landmarks in the view; score = mean peak) and
        COCO segmentation results (one per class with a non-empty mask; score = mean in-mask probability)."""
        v, xy, peak = r["view"], r["_xy"], r["_peak"]
        kps = []
        for c, slots in self.ch.by_class[v].items():
            K = len(self.ch.kp_names[c])
            k3 = [0.0] * (3 * K)
            for k, chn in slots:
                k3[3 * k:3 * k + 3] = [float(xy[chn, 0]), float(xy[chn, 1]), 2.0]
            kps.append({"image_id": image_id, "category_id": int(self.ch.cat_ids[c]), "keypoints": k3,
                        "score": float(np.mean([peak[chn] for _, chn in slots]))})
        segs = []
        for c in np.nonzero(r["_area"])[0]:
            rle = mask_util.encode(np.asfortranarray(r["_m"][c].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            segs.append({"image_id": image_id, "category_id": int(self.ch.cat_ids[c]), "segmentation": rle,
                         "score": float(r["_ms"][c]), "bbox": [float(t) for t in mask_util.toBbox(rle)]})
        return kps, segs
