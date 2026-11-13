"""Vendored from qseg for inference (no detectron2 / MaskDINO / fvcore). Module and parameter names unchanged,
so qseg checkpoints load by name.

ViT-CoMer-style multi-scale pyramid.

A drop-in replacement for SimpleFeaturePyramid that:

- Maintains a CNN side-branch (stem + 3 stride-2 downsample stages) producing
  features at strides {4, 8, 16, 32}, giving spatial inductive bias the plain
  ViT lacks.
- Fuses ``num_fuse_levels`` evenly-spaced ViT hidden-state snapshots into the
  CNN features via a per-level CTI (CNN-Transformer Interaction) block. The
  CTI block bilinearly resizes the ViT feature to the CNN scale, projects it
  with a 1×1 conv, and adds it as a residual on the CNN feature. A small
  3×3 GroupNorm+GELU conv mixes the result.

This is a simplified take on the ViT-CoMer paper (full version uses bidirectional
cross-attention CTIs and an MSAF block). The minimal version here captures the
main structural benefit — a true multi-scale CNN pyramid jointly with multi-layer
ViT signal — at a fraction of the compute cost. It can be extended to full
cross-attention CTI later by swapping ``_CTIBlock`` for an attention module.

Output spec matches SimpleFeaturePyramid so MaskDINO's pixel decoder is unchanged.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    # Stick to 32-group GN whenever channels allow; fall back to fewer groups
    # for tiny channel counts (64, 128) so we don't crash on group divisibility.
    g = 32
    while c % g != 0:
        g //= 2
    return nn.GroupNorm(g, c)


class _CNNStage(nn.Module):
    """Stride-2 CNN stage: downsample + refine."""
    def __init__(self, in_dim: int, out_dim: int, stride: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
            _gn(out_dim), nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            _gn(out_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CTIBlock(nn.Module):
    """CNN-Transformer Interaction: project a ViT feature to the CNN channel
    count and add as a residual, followed by a small refinement conv.

    Lightweight stand-in for the cross-attention CTI in the full ViT-CoMer
    paper. Empirically captures most of the gain at a fraction of the cost.
    """
    def __init__(self, vit_dim: int, cnn_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(vit_dim, cnn_dim, kernel_size=1)
        self.proj_norm = _gn(cnn_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            _gn(cnn_dim), nn.GELU(),
        )

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        # cnn_feat: (B, C, h, w). vit_feat: (B, C_v, h_v, w_v).
        # Bilinearly resize ViT feature to CNN spatial size.
        if vit_feat.shape[-2:] != cnn_feat.shape[-2:]:
            vit_feat = F.interpolate(
                vit_feat, size=cnn_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        residual = self.proj_norm(self.proj(vit_feat))
        return self.refine(cnn_feat + residual)


class ViTCoMerFPN(nn.Module):
    """Multi-scale pyramid that fuses a CNN side-branch with multi-layer ViT.

    Parameters
    ----------
    in_channels_vit : int
        Channel count of each ViT hidden-state slot (= embed_dim).
    out_channels : int
        Output channels per pyramid level (MaskDINO pixel decoder expects 256).
    out_strides : tuple[int, ...]
        Pyramid strides; default (4, 8, 16, 32).
    cnn_dims : tuple[int, ...]
        CNN-branch channel counts per stage. Length must equal len(out_strides).
    num_fuse_levels : int
        How many ViT hidden-state snapshots will be passed at forward time;
        one CTI block per level is built.
    """

    def __init__(
        self,
        in_channels_vit: int,
        out_channels: int = 256,
        out_strides: Sequence[int] = (4, 8, 16, 32),
        cnn_dims: Sequence[int] = (96, 192, 384, 384),
        num_fuse_levels: int = 4,
        export_s2: bool = False,
    ):
        super().__init__()
        # export_s2: also return the stem's first (stride-2) activation as "s2", for the
        # mask head's stride-2 refinement (model.mask2former_head.mask_refine_s2)
        self.export_s2 = bool(export_s2)
        assert len(cnn_dims) == len(out_strides)
        assert num_fuse_levels >= 1
        self.out_strides = tuple(out_strides)
        self.num_fuse_levels = int(num_fuse_levels)

        # Stem: image → finest stride (= min(out_strides)). The number of stride-2
        # convs is log2(finest): 2 for the default stride-4 base, 1 for a stride-2
        # base. The CNN branch then genuinely starts at the finest level — so
        # out_strides=(2,4,8,16,32) builds a real stride-2 pyramid (not a bolted-on
        # p2 head). For the default (4,8,16,32) this is byte-identical to before.
        finest = min(self.out_strides)
        n_down = max(1, finest.bit_length() - 1)        # log2(finest) for powers of 2
        d0 = cnn_dims[0]
        stem_layers: list[nn.Module] = []
        if n_down == 1:
            stem_layers += [nn.Conv2d(3, d0, kernel_size=3, stride=2, padding=1), _gn(d0), nn.GELU()]
        else:
            c = d0 // 2
            stem_layers += [nn.Conv2d(3, c, kernel_size=3, stride=2, padding=1), _gn(c), nn.GELU()]
            for _ in range(n_down - 1):
                stem_layers += [nn.Conv2d(c, d0, kernel_size=3, stride=2, padding=1), _gn(d0), nn.GELU()]
                c = d0
        stem_layers += [nn.Conv2d(d0, d0, kernel_size=3, padding=1), _gn(d0), nn.GELU()]
        self.stem = nn.Sequential(*stem_layers)
        self.s2_channels = stem_layers[0].out_channels        # Downsample stages cascade by 2 over the remaining levels (len(cnn_dims)-1).
        self.stages = nn.ModuleList()
        for i in range(1, len(cnn_dims)):
            self.stages.append(_CNNStage(cnn_dims[i - 1], cnn_dims[i], stride=2))

        # One CTI per pyramid level. ViT level → CNN level is round-robin if
        # ``num_fuse_levels < len(out_strides)``.
        self.ctis = nn.ModuleList([
            _CTIBlock(in_channels_vit, cnn_dim) for cnn_dim in cnn_dims
        ])

        # 1×1 projection to a common out_channels per level.
        self.out_proj = nn.ModuleList([
            nn.Conv2d(cnn_dim, out_channels, kernel_size=1) for cnn_dim in cnn_dims
        ])

        # Optional last refinement at each output level — keeps GroupNorm running
        # stats consistent under DDP.
        self.out_norm = nn.ModuleList([_gn(out_channels) for _ in cnn_dims])

    def _level_to_vit(self, level_i: int, n_vit: int) -> int:
        """Round-robin: pyramid level i ↔ ViT slot i (clamped to n_vit-1).

        With 4 pyramid levels and 4 ViT slots, this maps 1-1. With fewer ViT
        slots than levels, deeper pyramid levels share the deepest ViT slot.
        """
        return min(level_i, n_vit - 1)

    def forward(
        self,
        image: torch.Tensor,
        vit_feats: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        image : (B, 3, H, W) input pixel image (after pixel-mean/std handling).
        vit_feats : list of (B, C_v, h_v, w_v) — ViT hidden states at the ViT's
            patch grid, one per fuse level (shallow → deep).

        Returns
        -------
        dict ``{f"p{s}": (B, out_channels, H/s, W/s)}`` for s in out_strides.
        """
        assert len(vit_feats) >= 1, "vit_feats must be non-empty"

        # CNN forward.
        if self.export_s2:
            x2 = self.stem[:3](image)               # stride 2 (conv-GN-GELU)
            x = self.stem[3:](x2)                   # stride 4
        else:
            x = self.stem(image)                    # stride 4
        cnn_feats = [x]
        for stage in self.stages:
            cnn_feats.append(stage(cnn_feats[-1]))

        # Fuse and project.
        out: dict[str, torch.Tensor] = {}
        for i, (cf, s) in enumerate(zip(cnn_feats, self.out_strides)):
            vit_f = vit_feats[self._level_to_vit(i, len(vit_feats))]
            fused = self.ctis[i](cf, vit_f)
            out[f"p{s}"] = self.out_norm[i](self.out_proj[i](fused))
        if self.export_s2:
            out["s2"] = x2
        return out
