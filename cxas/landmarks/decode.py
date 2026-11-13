"""Vendored from cxr-landmarks (https://github.com/ConstantinSeibold/cxr-landmarks, MIT) @ baa9bcc: inference only.

Heatmap decoding: argmax + DARK refinement (Zhang et al., CVPR 2020)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _extend_log(lm: torch.Tensor, n: int, fit: int = 5) -> torch.Tensor:
    """Extend a log-heatmap by n px per side with a least-squares quadratic over the `fit` edge px
    (exact for a log-Gaussian), so blurring and derivatives at the border see the continuation of a
    truncated peak instead of replicated edge values."""
    t = torch.arange(fit, device=lm.device, dtype=lm.dtype)
    A = torch.stack([torch.ones_like(t), t, t * t], 1)
    j = -torch.arange(n, 0, -1, device=lm.device, dtype=lm.dtype)
    W = torch.stack([torch.ones_like(j), j, j * j], 1) @ torch.linalg.pinv(A)
    for dim in (-1, -2):
        x = lm.movedim(dim, -1)
        lo = x[..., :fit] @ W.T
        hi = (x.flip(-1)[..., :fit] @ W.T).flip(-1)
        lm = torch.cat([lo, x, hi], -1).movedim(-1, dim)
    return lm


def decode_dark(hm: torch.Tensor, blur_k: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """hm (C, h, w) -> xy in heatmap coordinates (C, 2) and peak value (C,).

    Gaussian-blur the heatmap, then take a second-order Taylor step on its log at the argmax.
    """
    C, h, w = hm.shape
    hm = hm.float()
    peak, i = hm.reshape(C, -1).max(dim=1)
    y, x = (i // w).long(), (i % w).long()
    pad = blur_k // 2 + 2
    m = torch.exp(_extend_log(torch.log(hm.clamp(min=1e-10)), pad))
    m = torch.minimum(m, peak.clamp(min=1e-10).view(C, 1, 1))
    sig = 0.3 * ((blur_k - 1) * 0.5 - 1) + 0.8
    r = torch.arange(blur_k, device=hm.device, dtype=torch.float32) - blur_k // 2
    k1 = torch.exp(-r ** 2 / (2 * sig ** 2))
    k1 = k1 / k1.sum()
    m = F.conv2d(F.conv2d(m.unsqueeze(1), k1.view(1, 1, 1, -1)), k1.view(1, 1, -1, 1)).squeeze(1)
    lm = torch.log(m.clamp(min=1e-10))                                  # (C, h+4, w+4)
    yy, xx = y + 2, x + 2
    ar = torch.arange(C, device=hm.device)

    def at(dy: int, dx: int) -> torch.Tensor:
        return lm[ar, yy + dy, xx + dx]

    dx = 0.5 * (at(0, 1) - at(0, -1))
    dy = 0.5 * (at(1, 0) - at(-1, 0))
    dxx = 0.25 * (at(0, 2) - 2 * at(0, 0) + at(0, -2))
    dyy = 0.25 * (at(2, 0) - 2 * at(0, 0) + at(-2, 0))
    dxy = 0.25 * (at(1, 1) - at(-1, 1) - at(1, -1) + at(-1, -1))
    det = dxx * dyy - dxy ** 2
    ok = (det > 1e-6) & (dxx < 0)                                       # proper local maximum
    sd = torch.where(ok, det, torch.ones_like(det))
    ox = torch.where(ok, -(dyy * dx - dxy * dy) / sd, torch.zeros_like(dx)).clamp(-2, 2)
    oy = torch.where(ok, -(dxx * dy - dxy * dx) / sd, torch.zeros_like(dy)).clamp(-2, 2)
    return torch.stack([x.float() + ox, y.float() + oy], 1), peak
