from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


@HEADS.register_module()
class MotionPlanningHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        cmd_ego_attn_model=None,

        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        col_loss_weight=0.1,
    ):
        super(MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "cmd_ego_attn": [cmd_ego_attn_model, ATTENTION],

            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims
        self.num_refine = sum(1 for op in operation_order if op == "refine")
        self.aux_loss_weights = [0.1, 0.2, 1.0][:self.num_refine]

        if self.decouple_attn:              # cfg: True
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        

        self.cmd_embedding = nn.Embedding(9, embed_dims)    # 9 = 3 L1 X 3 L2

        self.num_det = num_det
        self.num_map = num_map
        self.col_loss_weight = col_loss_weight

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)     # [N, num_anchors, num_classes] -> [N, num_anchors]
        motion_anchor = self.motion_anchor[cls_ids] # [N,900,6,12,2]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)       # agent(local ——> lidar(global
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:              # cfg: rue
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):   
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]   # [bs, num_anchors900, embed_dims256]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid() # [bs, num_anchors900, num_classes10]
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values      # [bs, num_anchors900]
        _, (instance_feature_selected, anchor_embed_selected, det_anchors_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed, det_anchors
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid() # [bs, num_anchors100, num_classes-3]
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected, map_anchors_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed, map_anchors
        )       # topk10

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)# [bs,900,6,12,2]
        # plan_anchor = torch.tile(                               # [bs,3,6,6,2]
        #     self.plan_anchor[None], (bs, 1, 1, 1, 1)
        # )
        ################## new anchor ##################
        plan_anchor = torch.tile(                               # plan_anchor: (3,3,6,T,2) -> (bs,3,3,6,T,2)
            self.plan_anchor[None], (bs, 1, 1, 1, 1, 1)
        )
        # =========== L1 coarse command: select 18 anchors from 54 before refinement ===========
        cmd_l1 = metas['gt_ego_fut_cmd_l1l2'][:, 0, :].argmax(dim=-1)  # (bs,)
        cmd_l2 = metas['gt_ego_fut_cmd_l1l2'][:, 1, :].argmax(dim=-1)  # (bs,)
        bs_indices = torch.arange(bs, device=plan_anchor.device)
        plan_anchor = plan_anchor[bs_indices, cmd_l1]               # (bs,3,3,6,T,2) -> (bs,3,6,T,2)

        #
        cmd_idx = cmd_l1 * 3 + cmd_l2                              # l1l2，(bs,)  0~8
        cmd_embed = self.cmd_embedding(cmd_idx).unsqueeze(1)           # (bs, 1, dim)
        # l1    l2    l1*3+l2
        # 0     0     0
        # 0     1     1
        # 0     2     2
        # 1     0     3
        # 1     1     4
        # 1     2     5
        # 2     0     6
        # 2     1     7
        # 2     2     8

        # =========== cmd_ego_attn: corridor mask based on plan_anchor ===========
        # corridor_points = plan_anchor.reshape(bs, -1, 2)   # (bs, 18*T, 2)
        anchor_corridor_points = plan_anchor.reshape(bs, -1, 2)   # (bs, 18*T, 2)
        corridor_points = anchor_corridor_points
        # agent distance: each agent has 1 position (x,y)
        agent_pos = det_anchors_selected[..., :2].detach()                    # (bs, 50, 2)
        agent_dist = torch.cdist(agent_pos, corridor_points)          # (bs, 50, 18*T)
        agent_min_dist = agent_dist.min(dim=-1).values                # (bs, 50)
        # map distance: each map element has 20 points
        map_pos = map_anchors_selected.reshape(bs, self.num_map, 20, 2)  # (bs, 10, 20, 2)
        map_pos_flat = map_pos.reshape(bs, self.num_map * 20, 2)      # (bs, 200, 2)
        map_dist_flat = torch.cdist(map_pos_flat, corridor_points)    # (bs, 200, 18*T)
        map_dist = map_dist_flat.reshape(bs, self.num_map, 20, -1)    # (bs, 10, 20, 18*T)
        map_min_dist = map_dist.min(dim=-1).values.min(dim=-1).values # (bs, 10)
        map_kv_mask = map_min_dist >= 3.0
        all_map_kv_mask = map_kv_mask.all(dim=-1, keepdim=True)
        map_kv_mask = map_kv_mask & ~all_map_kv_mask
        # # mask: True = ignore (distance >= threshold)
        # cmd_kv_mask = torch.cat([agent_min_dist >= 3.0, map_min_dist >= 3.0], dim=1)  # (bs, 60)
        # # fallback: if all masked for a sample, unmask all to avoid empty attention
        # all_masked = cmd_kv_mask.all(dim=-1, keepdim=True)            # (bs, 1)
        # cmd_kv_mask = cmd_kv_mask & ~all_masked

        agent_kv_mask = agent_min_dist >= 3.0
        all_agent_kv_mask = agent_kv_mask.all(dim=-1, keepdim=True)
        agent_kv_mask = agent_kv_mask & ~all_agent_kv_mask

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))   # [bs,900,6,256]
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])      # [bs,3,6,256]
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1) # [bs,1,18,256]

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        refine_count = 0
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,                                      #（bs,901,256)->(bs,51,256)
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":                     # agent-map
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "cmd_ego_attn":               # L1 command-guided ego-agent
                ego_feat = instance_feature[:, num_anchor:]      # (bs, 1, dim)
                ego_pos = anchor_embed[:, num_anchor:]           # (bs, 1, dim)
                kv_features = instance_feature_selected[:, :self.num_det]       # (bs, 50, dim)
                kv_pos = anchor_embed_selected[:, :self.num_det]                # (bs, 50, dim)
                ego_feat = self.layers[i](
                    ego_feat,
                    key=kv_features,
                    query_pos=ego_pos + cmd_embed,
                    key_pos=kv_pos,
                    key_padding_mask=agent_kv_mask,
                )
                instance_feature = torch.cat(
                    [instance_feature[:, :num_anchor], ego_feat], dim=1
                )
            # elif op == "cmd_ego_attn_map":           # ego-map 
            #     ego_feat = instance_feature[:, num_anchor:]      # (bs, 1, dim)
            #     ego_pos = anchor_embed[:, num_anchor:]           # (bs, 1, dim)
            #     ego_feat = self.layers[i](
            #         ego_feat,
            #         key=map_instance_feature_selected,           # (bs, 10, dim)
            #         query_pos=ego_pos + cmd_embed,
            #         key_pos=map_anchor_embed_selected,           # (bs, 10, dim)
            #         key_padding_mask=map_kv_mask,
            #     )
            #     instance_feature = torch.cat(
            #         [instance_feature[:, :num_anchor], ego_feat], dim=1
            #     )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)   # [bs,900,6,256]+ （[bs,901,256]->[bs,900,256].unsqueeze(2)）
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)       # [bs,1,18,256]
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,                #（bs,1,10）
                ) = self.layers[i](
                    motion_query,                   # [bs,900,6,256]
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                motion_classification.append(motion_cls)    # [bs,900,6]
                motion_prediction.append(motion_reg)        # [bs,900,6,12,2]
                planning_classification.append(plan_cls)    # [bs,1,18]
                planning_prediction.append(plan_reg)        # [bs,1,18,6,2]
                planning_status.append(plan_status)         # [bs,1,10]

                refine_count += 1
                if refine_count < self.num_refine:
                    pred_traj_cum = plan_reg.detach().cumsum(dim=-2) # (bs,1,18,T,2)
                    # corridor_points = pred_traj_cum.reshape(bs, -1, 2)
                    pred_corridor_points = pred_traj_cum.reshape(bs, -1, 2)   # (bs, 18*T, 2)
                    corridor_points = torch.cat(
                        [anchor_corridor_points, pred_corridor_points], dim=1
                    )                                                         # (bs, 36*T, 2)
                    # agent_dist = torch.cdist(agent_pos, corridor_points)
                    # agent_min_dist = agent_dist.min(dim=-1).values
                    agent_dist_anchor = torch.cdist(agent_pos, anchor_corridor_points)   # (bs, 50, 18*T)
                    agent_dist_pred   = torch.cdist(agent_pos, pred_corridor_points)     # (bs, 50, 18*T)
                    agent_min_dist = torch.min(
                        agent_dist_anchor.min(dim=-1).values,
                        agent_dist_pred.min(dim=-1).values,
                    )
                    agent_kv_mask = agent_min_dist >= 3.0
                    all_agent_kv_mask = agent_kv_mask.all(dim=-1, keepdim=True)
                    agent_kv_mask = agent_kv_mask & ~all_agent_kv_mask

                    map_dist_flat = torch.cdist(map_pos_flat, corridor_points)
                    map_dist = map_dist_flat.reshape(bs, self.num_map, 20, -1)
                    map_min_dist = map_dist.min(dim=-1).values.min(dim=-1).values
                    map_kv_mask = map_min_dist >= 3.0
                    all_map_kv_mask = map_kv_mask.all(dim=-1, keepdim=True)
                    map_kv_mask = map_kv_mask & ~all_map_kv_mask
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,                   # [bs,900]
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,       # [bs,1]
            "anchor_queue": self.instance_queue.ego_anchor_queue,   # [bs,1,11]
        }
        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache,
        det_output=None,
    ):
        loss = {}
        motion_loss, agent_reg_pred = self.loss_motion(
            motion_model_outs, data, motion_loss_cache,
        )
        loss.update(motion_loss)
        planning_loss = self.loss_planning(
            planning_model_outs, data, det_output, agent_reg_pred,
        )
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]   # [bs,900,6]
        reg_preds = model_outs["prediction"]        # [bs,900,6,12,2]
        output = {}
        agent_reg_pred = None
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target,             # [bs,900]
                cls_weight, 
                reg_pred,              # [bs,900,12,2]
                reg_target,
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            agent_reg_pred = reg_pred
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)                    # [bs*900,6]
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            aux_w = self.aux_loss_weights[decoder_idx] if decoder_idx < len(self.aux_loss_weights) else 1.0
            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss * aux_w,
                    f"motion_loss_reg_{decoder_idx}": reg_loss * aux_w,
                }
            )

        return output, agent_reg_pred

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(
        self, model_outs, data, det_output=None, agent_reg_pred=None,
    ):
        cls_scores = model_outs["classification"]       # [bs,1,18]
        reg_preds = model_outs["prediction"]            # [bs,1,18,6,2]
        status_preds = model_outs["status"]             # [bs,1,10]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,                    # [bs,1,18]—>[bs,3,1,6]—>[bs,1,6]
                cls_target,             # [bs,1]
                cls_weight,             # [bs,1]
                reg_pred,               # [bs,1,6,2]
                reg_target,             # [bs,1,6,2]
                reg_weight,             # [bs,6]->[bs,1,6]  
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],           # [bs,6,2]
                data['gt_ego_fut_masks'],           # [bs,6]
                data,
            )
            cls = cls.flatten(end_dim=1)        # [bs,1,6]——>[bs,6]
            cls_target = cls_target.flatten(end_dim=1)      # [bs,1]——>[bs]
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)
                # 0.1066    0.0349
            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )   # 0.7338    0.7374
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            aux_w = self.aux_loss_weights[decoder_idx] if decoder_idx < len(self.aux_loss_weights) else 1.0
            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss * aux_w,
                    f"planning_loss_reg_{decoder_idx}": reg_loss * aux_w,
                    f"planning_loss_status_{decoder_idx}": status_loss * aux_w,
                }
            )

            # if agent_reg_pred is not None and det_output is not None:
            if agent_reg_pred is not None and det_output is not None and decoder_idx == len(cls_scores) - 1:
                gt_traj = data['gt_ego_fut_trajs']  # (bs, T, 2) incremental
                col_loss = self.collision_loss(
                    reg_pred, agent_reg_pred, det_output, gt_traj,
                )
                # output[f"planning_loss_col_{decoder_idx}"] = col_loss * self.col_loss_weight * aux_w
                output[f"planning_loss_col_{decoder_idx}"] = col_loss * self.col_loss_weight
                # if decoder_idx == len(cls_scores) - 1 and getattr(self, '_col_monitor', None):
                if getattr(self, '_col_monitor', None):
                    m = self._col_monitor
                    output['col_gap_mean'] = torch.tensor(m['gap_mean'], device=col_loss.device)
                    output['col_gap_min'] = torch.tensor(m['gap_min'], device=col_loss.device)
                    output['col_gap_q25'] = torch.tensor(m['gap_q25'], device=col_loss.device)
                    output['col_gap_median'] = torch.tensor(m['gap_median'], device=col_loss.device)
                    output['col_gated_ratio'] = torch.tensor(m['gated_ratio'], device=col_loss.device)

        return output

    def collision_loss(self, plan_reg, agent_reg_pred, det_output, gt_ego_traj):
        """GT-gated box-aware differentiable collision proxy.

        Only penalizes when predicted trajectory is MORE dangerous than GT.
        If GT itself passes close to an agent (car-following, adjacent lane,
        intersection waiting), the model is NOT punished for doing the same.
        This prevents collision loss from conflicting with regression loss.
        """
        bs = plan_reg.shape[0]
        T = self.ego_fut_ts

        # ===== Ego trajectory and heading =====
        ego_traj = plan_reg.cumsum(dim=-2)  # (bs, T, 2)
        ego_traj_full = torch.cat([
            ego_traj.new_zeros(bs, 1, 2), ego_traj
        ], dim=1)  # (bs, T+1, 2) prepend origin
        # Per-timestep heading (detached: frame rotation only, no gradient)
        ego_heading = torch.atan2(
            ego_traj_full[:, 1:, 1] - ego_traj_full[:, :-1, 1],
            ego_traj_full[:, 1:, 0] - ego_traj_full[:, :-1, 0],
        ).detach()  # (bs, T)
        ego_cos = torch.cos(ego_heading)
        ego_sin = torch.sin(ego_heading)

        # ===== GT ego trajectory (detached, used as safety baseline) =====
        gt_traj = gt_ego_traj.detach().cumsum(dim=-2)  # (bs, T, 2)

        # Ego half-extents (dim_scale=1.1 consistent with rescore)
        ego_half_along = 4.08 / 2 * 1.1
        ego_half_cross = 1.73 / 2 * 1.1

        # ===== Agent info (all detached — gradient only flows through ego_traj) =====
        det_anchors = det_output["prediction"][-1].detach()
        det_cls = det_output["classification"][-1].detach().sigmoid()
        det_confidence = det_cls.max(dim=-1).values  # (bs, 900)

        agent_half_along = det_anchors[:, :, W].exp() / 2  # (bs, 900)
        agent_half_cross = det_anchors[:, :, L].exp() / 2
        agent_yaw = torch.atan2(
            det_anchors[:, :, SIN_YAW], det_anchors[:, :, COS_YAW]
        )

        # GT-best matched agent trajectory → absolute positions (same as loss_motion)
        agent_traj = agent_reg_pred.detach()[:, :, :T, :].cumsum(dim=-2)
        agent_traj = agent_traj + det_anchors[:, :, None, :2]  # (bs, 900, T, 2)

        # ===== GT baseline: min distance from GT to each agent =====
        gt_dist = torch.norm(
            gt_traj[:, None, :, :] - agent_traj, dim=-1
        )  # (bs, 900, T)
        gt_min_dist = gt_dist.min(dim=-1).values  # (bs, 900)

        # ===== SAT-based overlap in ego local frame =====
        dx = agent_traj[..., 0] - ego_traj[:, None, :, 0]  # (bs, 900, T)
        dy = agent_traj[..., 1] - ego_traj[:, None, :, 1]

        local_x = dx * ego_cos[:, None, :] + dy * ego_sin[:, None, :]
        local_y = -dx * ego_sin[:, None, :] + dy * ego_cos[:, None, :]

        # Agent box projection onto ego's axes (depends on relative heading)
        delta_yaw = agent_yaw[:, :, None] - ego_heading[:, None, :]
        cos_dy = torch.cos(delta_yaw).abs()
        sin_dy = torch.sin(delta_yaw).abs()
        agent_proj_along = (agent_half_along[:, :, None] * cos_dy
                            + agent_half_cross[:, :, None] * sin_dy)
        agent_proj_cross = (agent_half_along[:, :, None] * sin_dy
                            + agent_half_cross[:, :, None] * cos_dy)

        # Separation gaps (negative → overlap on that axis) sep > 0：gap . sep < 0：cover  crash
        approach_margin = 0.3
        sep_along = local_x.abs() - (ego_half_along + agent_proj_along + approach_margin)
        sep_cross = local_y.abs() - (ego_half_cross + agent_proj_cross + approach_margin)

        # Differentiable overlap: non-zero only when BOTH axes overlap
        overlap = torch.relu(-sep_along) * torch.relu(-sep_cross)  # (bs, 900, T)
        # max_overlap = overlap.max(dim=-1).values  # (bs, 900)
        # Time-weighted sum: emphasize far-future where collisions concentrate
        time_weight = torch.tensor([0.1, 0.1, 0.2, 1.0, 2.0, 3.0], device=overlap.device)
        max_overlap = (overlap * time_weight[None, None, :]).sum(dim=-1) / time_weight.sum()  # (bs, 900)

        # ===== GT-gating: suppress penalty where GT is already close =====
        # If GT trajectory itself passes close to an agent, suppress the
        # collision penalty for that agent — the model should not be punished
        # for following the GT behavior (car-following, adjacent lane, etc.)
        #
        # contact_dist: worst-case center-to-center distance at physical
        # collision = ego half-diagonal + agent half-diagonal.
        ego_half_diag = 2.22   # sqrt((4.08/2)^2 + (1.73/2)^2), fixed for ego
        agent_half_diag = torch.sqrt(agent_half_along ** 2 + agent_half_cross ** 2)
        contact_dist = ego_half_diag + agent_half_diag  # (bs, 900)
        # Gate threshold = contact distance + 1.0m margin for normal passing
        gt_safe_passing = (gt_min_dist < contact_dist + 0.5).float()
        max_overlap = max_overlap * (1.0 - gt_safe_passing)

        collision_cost = max_overlap * det_confidence
        collision_cost = collision_cost * (det_confidence > 0.3).float()

        # ===== Monitor gt_min_dist relative to contact_dist =====
        confident_mask = (det_confidence > 0.3)  # (bs, 900)
        if confident_mask.any():
            # "gap" = gt_min_dist - contact_dist for confident agents
            gap = (gt_min_dist - contact_dist)[confident_mask]  # (N,)
            self._col_monitor = {
                'gap_mean': gap.mean().item(),
                'gap_min': gap.min().item(),
                'gap_q25': gap.quantile(0.25).item(),
                'gap_median': gap.median().item(),
                'gated_ratio': gt_safe_passing[confident_mask].mean().item(),
            }
        else:
            self._col_monitor = None

        return collision_cost.sum(dim=-1).mean()
    
    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output,
            data,
        )

        return motion_result, planning_result