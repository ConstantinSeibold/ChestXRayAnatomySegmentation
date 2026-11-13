"""Mask2Former anatomy segmentor (qseg architecture), inference only, in plain PyTorch + transformers.

Architecture: ViT encoder (DINOv3 or RAD-DINO, Hugging Face format) -> ViT-CoMer pyramid {p4, p8, p16, p32}
-> MSDeformAttn pixel decoder -> 6-layer masked-attention decoder (200 queries) -> per query class logits
(classes + void) and a stride-4 mask. Built from a CXAS model bundle (``config.yaml`` = the qseg training config,
``backbone/`` = the encoder, ``model.safetensors`` = everything else), so the qseg / detectron2 training stack is not
needed at inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .decoder import MultiScaleMaskedTransformerDecoder
from .pixel_decoder import MSDeformAttnPixelDecoder, ShapeSpec
from .vit_comer import ViTCoMerFPN


class ViTBackbone(nn.Module):
    """ViT encoder + ViT-CoMer pyramid (qseg DINOv3D2Backbone, vit_comer variant)."""

    def __init__(self, encoder: nn.Module, patch_size: int, embed_dim: int, out_strides=(4, 8, 16, 32),
                 out_channels: int = 256, num_fuse: int = 4, cnn_dims=(96, 192, 384, 384)):
        super().__init__()
        self.encoder = encoder
        self.patch_size = int(patch_size)
        n_blocks = int(encoder.config.num_hidden_layers)
        self.fuse_layers = [max(0, (n_blocks * (i + 1)) // num_fuse - 1) for i in range(num_fuse)]
        self.fpn = ViTCoMerFPN(in_channels_vit=embed_dim, out_channels=out_channels, out_strides=out_strides,
                               cnn_dims=tuple(cnn_dims), num_fuse_levels=num_fuse)
        self.out_strides = tuple(out_strides)
        self.out_channels = out_channels

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {f"p{s}": ShapeSpec(self.out_channels, s) for s in self.out_strides}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        p = self.patch_size
        H, W = x.shape[-2:]
        if H % p or W % p:
            x = F.pad(x, (0, (p - W % p) % p, 0, (p - H % p) % p))
        hidden = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True).hidden_states
        hp, wp = x.shape[-2] // p, x.shape[-1] // p
        feats = []
        for i in self.fuse_layers:
            t = hidden[i + 1]
            feats.append(t[:, t.shape[1] - hp * wp:].transpose(1, 2).reshape(t.shape[0], t.shape[2], hp, wp))
        return self.fpn(x, feats)


class MaskFormerHead(nn.Module):
    def __init__(self, pixel_decoder: nn.Module, predictor: nn.Module):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = predictor

    def forward(self, features):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        return self.predictor(multi_scale_features, mask_features)


class Mask2FormerAnatomy(nn.Module):
    def __init__(self, backbone: ViTBackbone, sem_seg_head: MaskFormerHead, num_classes: int,
                 pixel_mean, pixel_std, size_divisibility: int = 32, test_topk_per_image: int = 200):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.num_classes = num_classes
        self.size_divisibility = size_divisibility
        self.test_topk_per_image = test_topk_per_image
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1), False)

    @classmethod
    def from_bundle(cls, path: str | Path) -> Tuple["Mask2FormerAnatomy", List[str]]:
        """(model on CPU in eval mode, class names in the model's output order)."""
        import json

        import yaml
        from safetensors.torch import load_file
        from transformers import AutoModel

        path = Path(path)
        cfg = yaml.safe_load(open(path / "config.yaml"))
        m, pyr, head = cfg["model"], cfg["model"]["pyramid"], cfg["model"]["mask2former_head"]
        if m.get("arch") != "mask2former" or pyr.get("type") != "vit_comer":
            raise ValueError(f"unsupported bundle architecture: arch={m.get('arch')} pyramid={pyr.get('type')}")
        for key in ("mask_upsample_factor", "mask_refine_s2"):
            if head.get(key) not in (None, 0, 1, False):
                raise ValueError(f"bundle uses {key}={head[key]}, which this inference port does not implement")
        cats = [c for c in json.load(open(path / "categories.json"))["categories"] if c.get("name") != "__ignore__"]
        names = [c["name"].strip() for c in sorted(cats, key=lambda c: c["id"])]

        encoder = AutoModel.from_pretrained(str(path / "backbone"))
        out_strides = [int(s) for s in pyr["out_strides"]]
        out_ch = int(pyr["out_channels"])
        backbone = ViTBackbone(encoder, int(encoder.config.patch_size), int(encoder.config.hidden_size),
                               out_strides, out_ch, int(pyr.get("vit_comer_num_fuse_levels", 4)),
                               pyr.get("vit_comer_cnn_dims", (96, 192, 384, 384)))
        feature_names = [f"p{s}" for s in out_strides]
        shapes = backbone.output_shape()
        pixel_decoder = MSDeformAttnPixelDecoder(
            {k: shapes[k] for k in feature_names}, transformer_dropout=0.0, transformer_nheads=8,
            transformer_dim_feedforward=1024, transformer_enc_layers=6, conv_dim=out_ch, mask_dim=out_ch,
            transformer_in_features=feature_names[-3:], common_stride=min(out_strides))
        # enforce_input_project=False: qseg builds the decoder that way regardless of the config value
        predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=out_ch, num_classes=len(names), hidden_dim=int(head["hidden_dim"]),
            num_queries=int(m["num_queries"]), nheads=8, dim_feedforward=2048,
            dec_layers=int(head["transformer_decoder_layers"]), pre_norm=False, mask_dim=out_ch,
            enforce_input_project=False)
        topk = int(cfg.get("eval", {}).get("detections_per_image", 0) or 0) or 100
        model = cls(backbone, MaskFormerHead(pixel_decoder, predictor), len(names),
                    m["backbone"]["pixel_mean"], m["backbone"]["pixel_std"], max(out_strides), topk)

        state = load_file(str(path / "model.safetensors"))
        state = {k: v for k, v in state.items() if not k.startswith(("criterion.", "kpt_head."))}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("backbone.encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"bundle {path} does not match the architecture: missing={missing[:5]} "
                               f"unexpected={unexpected[:5]}")
        return model.eval(), names

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """image: (3, h, w) float 0..255 RGB, already at working resolution ->
        {pred_logits (Q, C+1), pred_masks (Q, h, w) logits at working resolution}."""
        x = (image.to(self.pixel_mean.device).float() - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        d = self.size_divisibility
        x = F.pad(x, (0, (d - w % d) % d, 0, (d - h % d) % d))[None]
        out = self.sem_seg_head(self.backbone(x))
        masks = F.interpolate(out["pred_masks"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"pred_logits": out["pred_logits"][0], "pred_masks": masks[0, :, :h, :w]}

    @torch.no_grad()
    def best_per_class(self, image: torch.Tensor, score_floor: float = 0.05) -> Dict[int, Tuple[float, torch.Tensor]]:
        """Top-k (query, class) pairs scored as softmax class probability x mean in-mask probability, then the
        best-scoring pair of every class above score_floor: {class index: (score, mask logits (h, w))}."""
        out = self.forward(image)
        mask_pred = out["pred_masks"]
        scores = F.softmax(out["pred_logits"], dim=-1)[:, :-1]
        Q, C = scores.shape
        labels = torch.arange(C, device=scores.device).unsqueeze(0).repeat(Q, 1).flatten()
        top, idx = scores.flatten().topk(self.test_topk_per_image, sorted=False)
        classes, queries = labels[idx], idx // C
        m = mask_pred[queries]
        binm = (m > 0).float()
        maskness = (m.sigmoid().flatten(1) * binm.flatten(1)).sum(1) / (binm.flatten(1).sum(1) + 1e-6)
        final = top * maskness
        best = {}
        for j in torch.argsort(final, descending=True).tolist():
            s = float(final[j])
            if s <= score_floor:
                break
            c = int(classes[j])
            if c not in best:
                best[c] = (s, m[j])
        return best
