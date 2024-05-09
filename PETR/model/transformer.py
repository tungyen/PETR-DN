import torch
import torch.nn as nn
import math

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

class 

class Transformer(nn.Module):
    def __init__(self, dim_model=256, headNum=8, decoderLayerNum=6, 
                 dimBackbone=2048, dropout=0.1, queryNum=900):
        super(Transformer, self).__init__()
        self.dim_model = dim_model
        self.headNum = headNum
        self.decoderLayerNum = decoderLayerNum
        self.dimBackbone = dimBackbone
        self.dropout = dropout
        self.queryNum = queryNum
        
        self.position = nn.Embedding(self.queryNum, 3)
        nn.init.uniform_(self.position.weight.data, 0, 1)
        
        self.position_encoder = nn.Sequential(
            nn.Linear(384, self.dim_model), 
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_model, self.dim_model), 
        )
        
    def forward(self, feat2d, point3dPE, point2dFE):
        B, L, C = feat2d.size()
        refs = self.position.weight.unsqueeze(0).repeat(B, 1, 1)
        query = self.position_encoder(pos2posemb3d(refs)) # BxNxC
        print(query.shape)

        outputs_feats = []
        outputs_refs = []
        output = torch.zeros_like(query)
        return query
        
        
if __name__ == '__main__':
    feat2d = torch.randn(2, 294, 256)
    point3dPE = torch.randn(2, 294, 256)
    point2dFE = torch.randn(2, 294, 256)
    model = Transformer()
    output = model(feat2d, point3dPE, point2dFE)