"""Vendored from qseg for inference (no detectron2 / MaskDINO / fvcore). Module and parameter names unchanged,
so qseg checkpoints load by name.

Mask2Former MSDeformAttn pixel decoder, with Multi-Scale Deformable Attention in pure PyTorch (grid_sample). Checked
against MaskDINO's compiled CUDA kernel: identical kept classes, mask IoU >= 0.9994, same speed on GPU.
"""
from __future__ import annotations

import math
from collections import namedtuple
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import PositionEmbeddingSine, _get_activation_fn, _get_clones

ShapeSpec = namedtuple("ShapeSpec", ["channels", "stride"])


class Conv2d(nn.Conv2d):
    """detectron2.layers.Conv2d: convolution, then optional norm and activation (same parameter names)."""

    def __init__(self, *args, norm=None, activation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([int(H_) * int(W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, int(H_), int(W_))
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros",
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = (reference_points[:, :, None, :, None, :]
                              + sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout3(src2))

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes,
                              level_start_index, padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing="ij")
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_feature_levels=4, enc_n_points=4):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, d_model))

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_W.float() / W, valid_H.float() / H], -1)

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            lvl_pos_embed_flatten.append(pos_embed.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1))
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)
        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, input_shape: Dict[str, ShapeSpec], *, transformer_dropout: float = 0.0,
                 transformer_nheads: int = 8, transformer_dim_feedforward: int = 1024,
                 transformer_enc_layers: int = 6, conv_dim: int = 256, mask_dim: int = 256,
                 transformer_in_features: List[str] = ("p8", "p16", "p32"), common_stride: int = 4):
        super().__init__()
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))
            for c in transformer_in_channels[::-1]])
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim, dropout=transformer_dropout, nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward, num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels)
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs, output_convs = [], []
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = Conv2d(in_channels, conv_dim, kernel_size=1, bias=False, norm=nn.GroupNorm(32, conv_dim))
            output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False,
                                 norm=nn.GroupNorm(32, conv_dim), activation=F.relu)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward_features(self, features):
        with torch.autocast(device_type=next(iter(features.values())).device.type, enabled=False):
            srcs, pos = [], []
            for idx, f in enumerate(self.transformer_in_features[::-1]):
                x = features[f].float()
                srcs.append(self.input_proj[idx](x))
                pos.append(self.pe_layer(x))
            y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
            bs = y.shape[0]
            split = []
            for i in range(self.transformer_num_feature_levels):
                if i < self.transformer_num_feature_levels - 1:
                    split.append(int(level_start_index[i + 1] - level_start_index[i]))
                else:
                    split.append(int(y.shape[1] - level_start_index[i]))
            y = torch.split(y, split, dim=1)
            out = [z.transpose(1, 2).view(bs, -1, int(spatial_shapes[i][0]), int(spatial_shapes[i][1]))
                   for i, z in enumerate(y)]
            for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
                cur_fpn = self.lateral_convs[idx](features[f].float())
                y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
                out.append(self.output_convs[idx](y))
            multi_scale_features = out[:self.maskformer_num_feature_levels]
            return self.mask_features(out[-1]), out[0], multi_scale_features
