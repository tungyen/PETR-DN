import torch
from torch import nn as nn
import math

d_model = 256
n_heads = 8
dropout = 0.0

def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    # Inputs:
    #     pos - The input query with shape (B, queryNum, 3)
    #     num_pos_feats - The length of temperature
    #     temperature - The value of temperature variable
    # Outputs:
    #     posemb - The resulting 3D position embedding with shape (B, queryNum, 3 * num_pos_feats)
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_x, pos_y, pos_z), dim=-1)
    return posemb
dim_model=256
position_encoder = nn.Sequential(
            nn.Linear(384, dim_model), 
            nn.ReLU(inplace=True),
            nn.Linear(dim_model, dim_model), 
        )
    
feat2d = torch.randn(2, 294, 256)
point3dPE = torch.randn(2, 294, 256)
point2dFE = torch.randn(2, 294, 256)
position = nn.Embedding(900, 3)
refs = position.weight.unsqueeze(0).repeat(2, 1, 1)
query = position_encoder(pos2posemb3d(refs)) # BxNxC
tgt = torch.zeros_like(query)
pos3d_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
tgt0 = pos3d_attn(with_pos_embed(tgt, query).transpose(0, 1),
                                with_pos_embed(point3dPE, point2dFE).transpose(0,1),
                                point3dPE.transpose(0, 1))[0].transpose(0,1)

print(tgt0.shape)