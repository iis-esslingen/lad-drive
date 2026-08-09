from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from typing import Any, List, Dict, Optional, Union, Tuple
import math
from torch import Tensor, permute


def norm_odo(odo_info_fut):
    odo_info_fut_x = odo_info_fut[..., 0:1]  
    odo_info_fut_y = odo_info_fut[..., 1:2]  
    
    x_min, x_max = -2.5, 0.2    
    y_min, y_max = -11.5, 0.5
    
    odo_info_fut_x = 2 * (odo_info_fut_x - x_min) / (x_max - x_min) - 1
    odo_info_fut_y = 2 * (odo_info_fut_y - y_min) / (y_max - y_min) - 1
    
    return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

def denorm_odo(odo_info_fut): 
    odo_info_fut_x = odo_info_fut[..., 0:1]  
    odo_info_fut_y = odo_info_fut[..., 1:2]  
    
    x_min, x_max = -2.5, 0.2
    y_min, y_max = -11.5, 0.5
    
    odo_info_fut_x = (odo_info_fut_x + 1) / 2 * (x_max - x_min) + x_min
    odo_info_fut_y = (odo_info_fut_y + 1) / 2 * (y_max - y_min) + y_min
    
    return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class GridSampleCrossBEVAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=8):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.attention_weights = nn.Linear(embed_dims,num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)

        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """

        bs, num_queries, num_points, _ = traj_points.shape
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / 35 # TODO check
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / 30

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y TODO check
        # we might need to swap this, dont know how grid_sample works exactly with respect to x,y and H,W
        
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        
        original_dtype = value.dtype
        if original_dtype == torch.bfloat16:
            value = value.float()
            grid = grid.float()

        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # bs, C, num_queries, num_points

        if original_dtype == torch.bfloat16:
            sampled_features = sampled_features.to(original_dtype)
        # Sample features
        #sampled_features = torch.nn.functional.grid_sample(
        #    value, 
        #    grid, 
        #    mode='bilinear', 
        #    padding_mode='zeros', 
        #    align_corners=False
        #) # bs, C, num_queries, num_points

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C
        out = self.output_proj(out)

        return self.dropout(out) + queries
    
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 num_head,
                 use_ego_status=False,
                 use_spatial_bev=False
                 ):
        super().__init__()
        if use_spatial_bev:
            self.cross_bev_attention = GridSampleCrossBEVAttention(
                d_model,
                num_head,
                num_points=num_poses,
                in_bev_dims=256,
            )
        self.cross_plan_attention = nn.MultiheadAttention(
            d_model,
            num_head,
            0.0,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.dropout1 = nn.Dropout(0.1)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        if use_ego_status:
            self.cross_ego_attention = nn.MultiheadAttention(
                d_model,
                num_head,
                0.0,
                batch_first=True,
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(0.1)
            
        self.time_modulation = ModulationLayer(d_model,d_model)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                target_hidden_states, 
                time_embed, 
                ego_status_embed=None,
                bev_maps=None,
                global_img=None):
        if bev_maps is not None:
            #print("Using BEV spatial attention")
            traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_maps, spatial_shape=bev_maps.shape[2:])
        

        traj_feature = traj_feature + self.dropout1(self.cross_plan_attention(traj_feature, target_hidden_states, target_hidden_states)[0])
        traj_feature = self.norm1(traj_feature)
        if ego_status_embed is not None:
            #print("Using ego status attention")
            traj_feature = traj_feature + self.dropout2(self.cross_ego_attention(traj_feature, ego_status_embed, ego_status_embed)[0])
            traj_feature = self.norm2(traj_feature)
        
        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None, global_img=global_img)
        
        poses_reg, poses_cls = self.task_decoder(traj_feature) 
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points

        return poses_reg, poses_cls


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                ego_query, 
                time_embed, 
                ego_status_embed=None,
                bev_maps=None,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, ego_query, time_embed, ego_status_embed, bev_maps, global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list
    
def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=6,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 2)

        return plan_reg, plan_cls
        
