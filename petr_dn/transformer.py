import torch
import torch.nn as nn
import math
import copy

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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class resBlock(nn.Module):
    def __init__(self, inputC=256, outputC=1024, dropout=0.0):
        super(resBlock, self).__init__()
        self.linear1 = nn.Linear(inputC, outputC)
        self.relu = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(outputC, inputC)
        self.LN = nn.LayerNorm(inputC)
        
    def forward(self, x):
        out = self.linear2(self.dropout1(self.linear1(x)))
        out = self.LN(x+self.dropout2(out))
        return out

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_emb=256, dim_resblock=1024, dropout=0.0, numHead=8):
        super(TransformerDecoderLayer, self).__init__()
        self.iniAttn = nn.MultiheadAttention(dim_emb, numHead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.LN1 = nn.LayerNorm(dim_emb)
        
        self.crossAttn = nn.MultiheadAttention(dim_emb, numHead, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.LN2 = nn.LayerNorm(dim_emb)
        
        self.selfAttn = nn.MultiheadAttention(dim_emb, numHead, dropout=dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.LN3 = nn.LayerNorm(dim_emb)
        
        self.ffn = resBlock(dim_emb, dim_resblock, dropout=dropout)
        
    def forward(self, tgt, feat2d, query_pos, point3dPE, point2dFE):
        tgt0 = self.iniAttn((tgt+query_pos).transpose(0, 1),
                                (point3dPE+point2dFE).transpose(0,1),
                                point3dPE.transpose(0, 1))[0].transpose(0,1)
        tgt = tgt + self.dropout1(tgt0)
        tgt = self.LN1(tgt)
        # self attention
        q = k = tgt + query_pos
        tgt2 = self.selfAttn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.LN3(tgt)

        tgt2 = self.crossAttn((tgt+query_pos).transpose(0, 1),
                                (feat2d+point3dPE).transpose(0,1),
                                feat2d.transpose(0, 1))[0].transpose(0,1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.LN2(tgt)
        tgt = self.ffn(tgt)
        return tgt

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
        
        decoder_layer = TransformerDecoderLayer(dim_resblock=dimBackbone, dropout=dropout)
        self.decoder_layers = _get_clones(decoder_layer, decoderLayerNum)
        
    def forward(self, feat2d, point3dPE, point2dFE):
        B, L, C = feat2d.size()
        refs = self.position.weight.unsqueeze(0).repeat(B, 1, 1)
        query = self.position_encoder(pos2posemb3d(refs)) # BxNxC

        outputFeats = []
        outputRefs = []
        output = torch.zeros_like(query)
        
        for layer in self.decoder_layers:
            output = layer(output, feat2d, query, point3dPE, point2dFE)
            output = torch.nan_to_num(output)
            outputFeats.append(output)
            outputRefs.append(refs.clone())
            
        return outputFeats, outputRefs
        
        
if __name__ == '__main__':
    feat2d = torch.randn(2, 98, 256)
    point3dPE = torch.randn(2, 98, 256)
    point2dFE = torch.randn(2, 98, 256)
    model = Transformer(decoderLayerNum=2)
    feats, refs = model(feat2d, point3dPE, point2dFE)
    print(len(feats))