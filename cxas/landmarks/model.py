"""Vendored from cxr-landmarks (https://github.com/ConstantinSeibold/cxr-landmarks, MIT) @ baa9bcc: inference only.

Joint dense model: a RAD-DINO ViT-B/14 encoder shared by a landmark-heatmap decoder (one channel
per view-specific class-landmark pair, stride 4) and a mask decoder (one sigmoid channel per class).

Module names (`enc`, `head`, `seg`) match the training code, so the paper checkpoint loads as is.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4                      # heatmap stride w.r.t. the input
PAD_MULTIPLE = 32               # inputs are zero-padded to a multiple of this
RAD_DINO = "microsoft/rad-dino"
MEAN, STD = 124.7, 64.4         # RAD-DINO intensity statistics (0..255 scale, all channels)


class DeconvHead(nn.Module):
    """ViTPose 'classic' decoder: two 4x4 stride-2 deconvolutions (BN, ReLU) and a 1x1 conv."""

    def __init__(self, cin: int, cout: int, width: int = 512):
        super().__init__()
        layers, c = [], cin
        for _ in range(2):
            layers += [nn.ConvTranspose2d(c, width, 4, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            c = width
        self.up = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, cout, 1)
        nn.init.normal_(self.out.weight, std=0.001)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.up(x))


class JointDenseModel(nn.Module):
    """forward(x) -> (heatmaps (B, n_landmarks, H/4, W/4), mask logits (B, n_classes, H/4, W/4)).

    x: (B, 3, H, W) float in 0..255 with H, W multiples of 32. The image is resized by 14/16 before
    the encoder so that RAD-DINO's patch-14 token grid equals a patch-16 grid of the input.
    """

    def __init__(self, n_landmarks: int, n_classes: int, pretrained_encoder: bool = True):
        super().__init__()
        from transformers import AutoModel, Dinov2Config
        if pretrained_encoder:
            self.enc = AutoModel.from_pretrained(RAD_DINO)
        else:
            # architecture only, from the config shipped with CXAS: the weights come from the checkpoint
            cfg = Dinov2Config.from_json_file(str(Path(__file__).parent / "data" / "rad_dino_config.json"))
            self.enc = AutoModel.from_config(cfg)
        self.head = DeconvHead(768, n_landmarks)
        self.seg = DeconvHead(768, n_classes)
        nn.init.constant_(self.seg.out.bias, -4.6)                   # sigmoid prior 0.01
        self.register_buffer("mean", torch.full((1, 3, 1, 1), MEAN), persistent=False)
        self.register_buffer("std", torch.full((1, 3, 1, 1), STD), persistent=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x = (x - self.mean) / self.std
        xs = F.interpolate(x, size=(H // 16 * 14, W // 16 * 14), mode="bilinear", align_corners=False)
        t = self.enc(pixel_values=xs).last_hidden_state[:, 1:]
        return t.transpose(1, 2).reshape(x.shape[0], -1, H // 16, W // 16)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.features(x)
        return self.head(g), self.seg(g)

    def param_groups(self, lr: float, wd: float, layer_decay: float) -> list[dict]:
        """AdamW groups with layer-wise lr decay on the encoder (ViTPose); no decay on norms/biases."""
        n_layers = 12
        groups: dict = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            scale = 1.0
            if name.startswith("enc."):
                lid = n_layers + 1
                if "encoder.layer." in name:
                    lid = int(name.split("encoder.layer.")[1].split(".")[0]) + 1
                elif "embeddings" in name:
                    lid = 0
                scale = layer_decay ** (n_layers + 1 - lid)
            no_wd = (p.ndim == 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name
                     or "position_embeddings" in name)
            g = groups.setdefault((scale, no_wd), {"params": [], "lr": lr * scale,
                                                   "weight_decay": 0.0 if no_wd else wd, "lr_scale": scale})
            g["params"].append(p)
        return list(groups.values())


def pad_batch(imgs: list[torch.Tensor], mult: int = PAD_MULTIPLE) -> torch.Tensor:
    """List of (3, h, w) images -> zero-padded float batch (B, 3, H, W), H and W multiples of `mult`."""
    H = -(-max(i.shape[1] for i in imgs) // mult) * mult
    W = -(-max(i.shape[2] for i in imgs) // mult) * mult
    out = imgs[0].new_zeros((len(imgs), 3, H, W), dtype=torch.float32)
    for b, im in enumerate(imgs):
        out[b, :, :im.shape[1], :im.shape[2]] = im.float()
    return out
