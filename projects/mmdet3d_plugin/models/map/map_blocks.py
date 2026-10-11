from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class SDMapCrossAttnEncoder(BaseModule):
    """
    - Q: instance_feature.detach()
    - K/V: SD 
    - K = content + pos, V = content
    - only in Layer 0, not in Layer 1-5
    - instance_feature = instance_feature + cross_attn_out

    """

    POINT_FEAT_DIM = 6

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_decoder: int = 6,
        dropout: float = 0.1,
        ema_alpha: float = 0.6,
    ):
        super(SDMapCrossAttnEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_sample = num_sample
        self.coords_dim = coords_dim
        self.num_decoder = num_decoder
        self.ema_alpha = ema_alpha

        pfd = self.POINT_FEAT_DIM

        self.point_content_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, pfd)
        )

        self.point_pos_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, pfd)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )

        self.out_proj = nn.Linear(embed_dims, embed_dims)

        # EMA only in Layer 0
        self.cached_cross_out = None


    PATCH_HALF_SIZE = 30.0

    def _extract_point_features(self, sd_map_data):
        """
        Args:
            sd_map_data: [bs, N, 42](num_sample*coords_dim + oneway + maxspeed)
        Returns:
            point_feats: [bs, N, num_sample, 6]
        """
        bs, N, _ = sd_map_data.shape
        S = self.num_sample

        coords = sd_map_data[:, :, :S * self.coords_dim]
        oneway = sd_map_data[:, :, -2:-1]
        maxspeed = sd_map_data[:, :, -1:]

        pts = coords.reshape(bs, N, S, self.coords_dim)
        pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. 
        tangent = torch.zeros_like(pts)
        tangent[:, :, 1:-1] = pts[:, :, 2:] - pts[:, :, :-2]
        tangent[:, :, 0] = pts[:, :, 1] - pts[:, :, 0]
        tangent[:, :, -1] = pts[:, :, -1] - pts[:, :, -2]
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # 2. 
        seg_len = (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1)   # [bs, N, S-1]
        cumlen = torch.cat([
            seg_len.new_zeros(bs, N, 1), seg_len.cumsum(dim=-1)
        ], dim=-1)                                                  # [bs, N, S]
        cumlen_norm = (cumlen / self.PATCH_HALF_SIZE).unsqueeze(-1) # [bs, N, S, 1]

        # 3.
        curvature = pts.new_zeros(bs, N, S, 1)
        curvature[:, :, 1:-1] = (
            pts[:, :, 2:] + pts[:, :, :-2] - 2 * pts[:, :, 1:-1]
        ).norm(dim=-1, keepdim=True)

        # 4-5. 
        oneway_exp = oneway.unsqueeze(2).expand(-1, -1, S, -1)
        maxspeed_exp = (
            torch.clamp(maxspeed, 0.0, 150.0) / 70.0
        ).unsqueeze(2).expand(-1, -1, S, -1)

        point_feats = torch.cat([
            tangent, cumlen_norm, curvature,
            oneway_exp, maxspeed_exp,
        ], dim=-1)
        return torch.nan_to_num(point_feats, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    #  
    # ------------------------------------------------------------------
    def encode_sd(self, sd_map_data, sd_map_mask=None):
        """
        Returns:
            sd_k:       [bs, N*num_sample, embed_dims]
            sd_v:       [bs, N*num_sample, embed_dims]
            point_mask: [bs, N*num_sample]  True=padding
        """
        sd_map_data = torch.nan_to_num(sd_map_data, nan=0.0, posinf=0.0, neginf=0.0)
        bs, N, _ = sd_map_data.shape
        S = self.num_sample

        point_feats = self._extract_point_features(sd_map_data)
        point_feats = point_feats.reshape(bs, N * S, -1)

        content = self.point_content_encoder(point_feats)
        key_pos = self.point_pos_encoder(point_feats)

        sd_k = content + key_pos
        sd_v = content

        if sd_map_mask is not None:
            point_mask = (
                sd_map_mask.unsqueeze(-1)
                .expand(-1, -1, S)
                .reshape(bs, N * S)
            )
        else:
            point_mask = None

        return sd_k, sd_v, point_mask

    # ------------------------------------------------------------------
    #  EMA 
    # ------------------------------------------------------------------
    def reset_ema_cache(self):

        self.cached_cross_out = None

    # ------------------------------------------------------------------
    #  （only in Layer 0 ）
    # ------------------------------------------------------------------
    def cross_attn_forward(self, instance_feature_detached, sd_k, sd_v,
                           sd_key_pad_mask):
        """
        Args:
            instance_feature_detached: [bs, num_query, embed_dims]( detach)
            sd_k, sd_v: [bs, N*S, embed_dims]
            sd_key_pad_mask: [bs, N*S]  True=padding
        Returns:
            out: [bs, num_query, embed_dims]
        """
        zero_out_all = False
        empty_samples = None
        key_pad_mask = sd_key_pad_mask

        if (not torch.isfinite(instance_feature_detached).all()
                or not torch.isfinite(sd_k).all()
                or not torch.isfinite(sd_v).all()):
            zero_out_all = True
            key_pad_mask = sd_key_pad_mask.clone()
            key_pad_mask[:, 0] = False
        elif sd_key_pad_mask is not None:
            num_valid = (~sd_key_pad_mask).sum(dim=-1)
            empty_samples = (num_valid == 0)
            if empty_samples.all():
                zero_out_all = True
                key_pad_mask = sd_key_pad_mask.clone()
                key_pad_mask[:, 0] = False
            elif empty_samples.any():
                key_pad_mask = sd_key_pad_mask.clone()
                key_pad_mask[empty_samples, 0] = False

        with torch.cuda.amp.autocast(enabled=False):
            q = instance_feature_detached.float()       #   .float() FP32  MHA
            k = sd_k.float()
            v = sd_v.float()
            if key_pad_mask is not None:
                attn_out, _ = self.cross_attn(
                    q, k, v, key_padding_mask=key_pad_mask
                )
            else:
                attn_out, _ = self.cross_attn(q, k, v)
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)

        out = self.out_proj(attn_out.to(instance_feature_detached.dtype))

        if zero_out_all:
            out = out * 0.0
        elif empty_samples is not None and empty_samples.any():
            out = out * (~empty_samples).float().view(-1, 1, 1)

        # EMA 
        cached = self.cached_cross_out
        if cached is not None and cached.shape == out.shape and not zero_out_all:
            out = self.ema_alpha * out + (1.0 - self.ema_alpha) * cached
        if not zero_out_all:
            self.cached_cross_out = out.detach().clone()

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@PLUGIN_LAYERS.register_module()
class SDMapFiLMEncoder(BaseModule):

    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_decoder: int = 6,
    ):
        super(SDMapFiLMEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.coords_dim = coords_dim
        self.num_decoder = num_decoder
        
        per_line_dim = 8
        
        self.line_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, per_line_dim)
        )
        
        self.global_proj = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2)
        )
        
        self.gamma_generators = nn.ModuleList()
        self.beta_generators = nn.ModuleList()
        for _ in range(num_decoder):
            self.gamma_generators.append(
                nn.Linear(embed_dims, embed_dims)
            )
            self.beta_generators.append(
                nn.Linear(embed_dims, embed_dims)
            )
        
        self._init_film_weights()
    
    def _init_film_weights(self):
        for gamma_gen in self.gamma_generators:
            nn.init.normal_(gamma_gen.weight, mean=0.0, std=0.001)
            nn.init.ones_(gamma_gen.bias)       # γ init ≈ 1
        for beta_gen in self.beta_generators:
            nn.init.normal_(beta_gen.weight, mean=0.0, std=0.001)
            nn.init.zeros_(beta_gen.bias)       # β init ≈ 0
    
    def _extract_features(self, sd_map_data, sd_map_mask=None):
        """
        Args:
            sd_map_data: [bs, N, 42](20*2 + oneway + maxspeed)
            sd_map_mask: [bs, N], True=padding
        Returns:
            line_feats: [bs, N, 8]
        """
        bs, N, _ = sd_map_data.shape
        
        coords = sd_map_data[:, :, :self.num_sample * self.coords_dim]  # [bs, N, 40]
        oneway = sd_map_data[:, :, -2:-1]   # [bs, N, 1]
        maxspeed = sd_map_data[:, :, -1:]   # [bs, N, 1]
        
        pts = coords.reshape(bs, N, self.num_sample, self.coords_dim)   # [bs, N, 20, 2]
        pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. 
        direction = pts[:, :, -1] - pts[:, :, 0]           # [bs, N, 2]
        dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction_normalized = direction / dir_norm         # [bs, N, 2]
        
        # 2. 
        dp = pts[:, :, 1:, :] - pts[:, :, :-1, :]          # [bs, N, 19, 2]
        ddp = dp[:, :, 1:, :] - dp[:, :, :-1, :]           # [bs, N, 18, 2]
        curvature = ddp.norm(dim=-1).mean(dim=-1, keepdim=True)  # [bs, N, 1]
        
        # 3. 
        seg_lengths = dp.norm(dim=-1)                       # [bs, N, 19]
        total_length = seg_lengths.sum(dim=-1, keepdim=True).clamp(min=1e-6) / 20.0  # [bs, N, 1]
        
        # 4. 
        dp_norm = dp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dp_dir = dp / dp_norm                               # [bs, N, 19, 2]
        dir_std = dp_dir.std(dim=-2)                        # [bs, N, 2]
        dir_std = torch.nan_to_num(dir_std, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 5. maxspeed 
        maxspeed_safe = torch.clamp(maxspeed, 0.0, 150.0)
        maxspeed_normalized = maxspeed_safe / 70.0          # [bs, N, 1]
        
        line_feats = torch.cat([
            direction_normalized, curvature, total_length,
            dir_std, oneway, maxspeed_normalized,
        ], dim=-1)                                          # [bs, N, 8]
        line_feats = torch.nan_to_num(line_feats, nan=0.0, posinf=0.0, neginf=0.0)
        
        if sd_map_mask is not None:
            line_feats = line_feats.masked_fill(sd_map_mask.unsqueeze(-1), 0.0)
        
        return line_feats
    
    def encode_global(self, sd_map_data, sd_map_mask=None):
        """
        Args:
            sd_map_data: [bs, N, 42]
            sd_map_mask: [bs, N], True=padding
        Returns:
            sd_global: [bs, embed_dims]
        """
        sd_map_data = torch.nan_to_num(sd_map_data, nan=0.0, posinf=0.0, neginf=0.0)
        line_feats = self._extract_features(sd_map_data, sd_map_mask)   # [bs, N, 8]
        line_embed = self.line_encoder(line_feats)                      # [bs, N, embed_dims]
        
        # masked mean pool
        if sd_map_mask is not None:
            line_embed = line_embed.masked_fill(sd_map_mask.unsqueeze(-1), 0.0)
            num_valid = (~sd_map_mask).sum(dim=-1, keepdim=True).float().clamp(min=1)
            sd_global = line_embed.sum(dim=1) / num_valid               # [bs, embed_dims]
        else:
            sd_global = line_embed.mean(dim=1)                          # [bs, embed_dims]
        
        sd_global = self.global_proj(sd_global)                         # [bs, embed_dims]
        sd_global = torch.nan_to_num(sd_global, nan=0.0, posinf=0.0, neginf=0.0)
        return sd_global
    
    def film_modulate(self, instance_feature, sd_global, decoder_idx):
        """
        Args:
            instance_feature: [bs, num_query, embed_dims]
            sd_global: [bs, embed_dims],
            decoder_idx: int, decoder(0-indexed)
        Returns:
            modulated_feature: [bs, num_query, embed_dims]
        """
        gamma = self.gamma_generators[decoder_idx](sd_global)   # [bs, embed_dims]
        beta = self.beta_generators[decoder_idx](sd_global)     # [bs, embed_dims]
        gamma = torch.nan_to_num(gamma, nan=1.0, posinf=1.0, neginf=1.0)  # identity 
        beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        
        # [bs, 1, embed_dims]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # FiLM ：γ≈1, β≈0 ≈ identity；NaN/Inf retreat
        out = gamma * instance_feature + beta
        return torch.where(torch.isfinite(out), out, instance_feature)


@POSITIONAL_ENCODING.register_module()
class SparsePoint3DEncoder(BaseModule):
    def __init__(
        self, 
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
    ):
        super(SparsePoint3DEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.input_dims = num_sample * coords_dim
        def embedding_layer(input_dims):    # embed_dims, in_loops=1, out_loops=2, input_dims
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(self.input_dims)

    def forward(self, anchor: torch.Tensor):
        pos_feat = self.pos_fc(anchor)  
        return pos_feat


@PLUGIN_LAYERS.register_module()
class SparsePoint3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
    ):
        super(SparsePoint3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.output_dim = num_sample * coords_dim
        self.num_cls = num_cls

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        output = self.layers(instance_feature + anchor_embed)
        output = output + anchor
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)  ## NOTE anchor embed?
        else:
            cls = None
        qt = None
        return output, cls, qt


@PLUGIN_LAYERS.register_module()
class SparsePoint3DKeyPointsGenerator(BaseModule): 
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        num_learnable_pts: int = 0,
        fix_height: Tuple = (0,),
        ground_height: int = 0,
        adaptive_offset: bool = False,
        ref_distance: float = 15.0,
        scale_min: float = 0.5,
        scale_max: float = 3.0,
    ):
        super(SparsePoint3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        if self.num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, self.num_pts * 2)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height
        self.adaptive_offset = adaptive_offset
        self.ref_distance = ref_distance
        self.scale_min = scale_min
        self.scale_max = scale_max

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        assert self.num_learnable_pts > 0, 'No learnable pts'
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        offset = (
            self.learnable_fc(instance_feature)
            .reshape(bs, num_anchor, self.num_sample, len(self.fix_height), self.num_learnable_pts, 2)
        )
        if self.adaptive_offset:
            per_pt_dist = torch.norm(key_points, dim=-1, keepdim=True)  # [bs, na, 20, 1]
            scale = (per_pt_dist / self.ref_distance).clamp(
                min=self.scale_min, max=self.scale_max
            )
            # broadcast to [bs, na, 20, num_heights, num_learnable_pts, 2]
            offset = offset * scale.unsqueeze(-2).unsqueeze(-2)
        key_points = offset + key_points[..., None, None, :]
        key_points = torch.cat(
            [
                key_points,
                key_points.new_full(key_points.shape[:-1]+(1,), fill_value=self.ground_height),
            ],
            dim=-1,
        )
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:,None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]
        key_points = key_points.flatten(2, 4)
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        for i, t_time in enumerate(temp_timestamps):
            temp_key_points = key_points
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    # @staticmethod
    def anchor_projection(
        self,
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            dst_anchor = anchor.clone()
            bs, num_anchor, _ = anchor.shape
            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(1, 2)
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            dst_anchor = (
                torch.matmul(
                    T_src2dst[..., :2, :2], dst_anchor[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :2, 3]
            )

            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(2, 3)
            dst_anchors.append(dst_anchor)
        return dst_anchors