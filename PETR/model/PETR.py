import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .cameraEncoder import CamEncoder
from .transformer import *

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def sin_positional_encoding3D(size, device='cpu', num_feats=128, offset=0.0, temperature=10000):
    # Inputs:
    #     size - The input size with info [Batch_size, Camera_num, Hf, Wf]
    #     num_feats - The length of temperature
    #     temperature - The value of temperature variable
    # Outputs:
    #     pos - The resulting 3D position embedding with shape (Batch_size, Camera_num, Hf, Wf, 3 * num_feats)
    scale = 2 * math.pi
    eps = 1e-6
    mask = torch.ones(size, dtype=torch.float32, device=device)
    n_embed = mask.cumsum(1, dtype=torch.float32)
    y_embed = mask.cumsum(2, dtype=torch.float32)
    x_embed = mask.cumsum(3, dtype=torch.float32)
    n_embed = (n_embed + offset) / (n_embed[:, -1:, :, :] + eps) * scale
    y_embed = (y_embed + offset) / (y_embed[:, :, -1:, :] + eps) * scale
    x_embed = (x_embed + offset) / (x_embed[:, :, :, -1:] + eps) * scale
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=device)
    dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    pos_n = n_embed[:, :, :, :, None] / dim_t
    pos_x = x_embed[:, :, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, :, None] / dim_t
    B, N, H, W = mask.size()
    pos_n = torch.stack((pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
    pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
    pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
    pos = torch.cat((pos_n, pos_y, pos_x), dim=4)
    return pos

class posEncoder3d(nn.Module):
    def __init__(self, camNum, D, grid, camC, LID=True, depthStart=1):
        super(posEncoder3d, self).__init__()
        self.camNum = camNum
        self.D = D
        self.camC = camC
        self.xBound = grid['xbound']
        self.yBound = grid['ybound']
        self.zBound = grid['zbound']
        self.LID = LID
        self.depthStart = depthStart
        
        self.CamEncoder = CamEncoder()
        self.feat_conv = nn.Conv2d(self.camC, 256, 1, 1)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(self.camC, 256, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, self.D, 1, 1),
        )
        self.pos_conv = nn.Sequential(
            nn.Conv2d(self.D * 3, 1024, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 256, 1, 1),
        )
        self.adapt_pos2d = nn.Sequential(
            nn.Conv2d(384, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
        )
    
    # def pointGenerator(self, Hf, Wf, D, I, rots, translations):
    def pointGenerator(self, Hf, Wf, D, K, rectRot=None):
        # Inputs:
        #     Hf - The height of 2D feature after backbone model
        #     Wf - The width of 2D feature after backbone model
        #     D - The maximum depth of the frustum space
        #     I - The intrinsic matrix of each camera with shape (B, N, 3, 3) and torch device
        #     rots - The rotation matrix of each camera with shape (B, N, 3, 3)
        #     translations - The translation vector of each camera with shape (B, N, 3)
        # Outputs:
        #     points - The resulted 3D coordinates points with shape (B, N, D, Hf, Wf, 3)
        
        # frustum = self.getFrustum(Hf, Wf, D)
        # frustum = frustum.to(I.device)
        # B, N, _ = translations.shape
        # points = frustum.view(1, 1, D, Hf, Wf, 3).repeat(B, N, 1, 1, 1, 1)
        # combine = rots.matmul(torch.inverse(I))
        # points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        # points += translations.view(B, N, 1, 1, 1, 3)
        # points = self.posNorm(points)
        # return points
        frustum = self.getFrustum(Hf, Wf, D)
        frustum = frustum.to(K.device)
        B, N, _, _ = K.shape
        points = frustum.view(1, 1, D, Hf, Wf, 3).repeat(B, N, 1, 1, 1, 1)
        
        if rectRot != None:
            inv_rectRot = torch.inverse(rectRot).view(B, 1, 1, 1, 3, 3)[:, None, ...]
            points = inv_rectRot.matmul(points.unsqueeze(-1)).squeeze(-1)
        
        homo_tensor_points = torch.ones((B, N, D, Hf, Wf, 1))
        points = torch.cat((points, homo_tensor_points), dim=-1) # (B, N, D, Hf, Wf, 4)

        homo_tensor_K = torch.zeros((B, N, 1, 4))
        homo_tensor_K[:, :, 0, 3] = 1
        K = torch.cat((K, homo_tensor_K), dim=2) # (B, N, 4, 4)
        inv_K = torch.inverse(K).view(B, N, 1, 1, 1, 4, 4)
        points = inv_K.matmul(points.unsqueeze(-1)).squeeze(-1)
        
        points = points / points[...,-1].unsqueeze(-1)
        points = points[...,:-1]
        points = self.posNorm(points)
        print(points.shape)
        return points
    
    def posNorm(self, points):
        # Inputs:
        #     points - The 3D world coordinates with shape (B, N, D, Hf, Wf, 3)
        # Outputs:
        #     pointsNorm - The normalized 3D world coordinates with shape (B, N, D, Hf, Wf, 3)
        x = (points[..., 0] - self.xBound[0]) / (self.xBound[1] - self.xBound[0])
        y = (points[..., 1] - self.yBound[0]) / (self.yBound[1] - self.yBound[0])
        z = (points[..., 2] - self.zBound[0]) / (self.zBound[1] - self.zBound[0])
        pointsNorm = torch.stack([x,y,z], dim=-1)
        pointsNorm = pointsNorm.clamp(min=0.0, max=1.0)
        return pointsNorm
    
    def getFrustum(self, Hf, Wf, D):
        # Inputs:
        #     Hf - The height of 2D feature after backbone model
        #     Wf - The width of 2D feature after backbone model
        #     D - The maximum depth of the frustum space
        # Outputs:
        #     frustum - The frustum space with shape (D, Hf, Wf, 3)
        ys, xs = torch.meshgrid([torch.arange(Hf), torch.arange(Wf)])
        ones = torch.ones((Hf, Wf))
        
        if self.LID:
            index  = torch.arange(start=0, end=self.D, step=1).float().view(-1, 1, 1, 1)
            bin_size = (self.xBound[1] - self.depthStart) / (self.D * (1 + self.D))
            ds = self.depthStart + bin_size * index * (index+1)
        else:
            ds = torch.arange(D, dtype=torch.float).view(-1, 1, 1, 1) + 0.5
        xys = torch.stack([xs+0.5, ys+0.5, ones], dim=-1).view(1, Hf, Wf, 3)
        frustum = xys * ds
        return frustum
    
    def forward(self, input):
        B, N, _, H, W = input['image'].size()
        x = self.CamEncoder(input['image'])
        B, N, Cf, Hf, Wf = x.size()
        x = x.view(-1, Cf, Hf, Wf)
        BN = x.shape[0]

        feat = self.feat_conv(x)
        feat2d = feat.permute(0, 2, 3, 1) # BN x H x W x C
        feat2d = feat2d.reshape(B, -1, 256)

        # depth_dist = self.depth_conv(x).sigmoid()

        intrins = input['intrins'].clone()
        intrins[:, :, 0, :] *= ( Wf / W)
        intrins[:, :, 1, :] *= ( Hf/ H)
        # point3d = self.pointGenerator(Wf, Hf, self.D, intrins, input['rots'], input['trans'])
        point3d = self.pointGenerator(Hf, Wf, self.D, intrins, input['rectRots'])
        # if self.bev_aug and self.training:
        #     bev_rot = input['bev_rot'].view(B, N, 1, 1, 1, 3, 3)
        #     img_pos = bev_rot.matmul(img_pos.unsqueeze(-1)).squeeze(-1)

        point3dNorm = self.posNorm(point3d) # BxNxDxHxWx3
        point3dNorm = point3dNorm.permute(0, 1, 2, 5, 3, 4).contiguous().view(BN, -1, Hf, Wf)        
        point3dPE = self.pos_conv(point3dNorm)
        # pos_emb = pos_emb * depth_dist
        point3dPE = point3dPE.permute(0,2,3,1) # BN x H x W x C
        point3dPE = point3dPE.reshape(B, -1, 256)

        point2dFE = sin_positional_encoding3D((B, N, Hf, Wf))
        point2dFE = point2dFE.permute(0, 1, 4, 2, 3).contiguous().view(BN, -1, Hf, Wf)
        point2dFE = self.adapt_pos2d(point2dFE)
        point2dFE = point2dFE.permute(0,2,3,1)
        point2dFE = point2dFE.reshape(B, -1, 256)
        return feat2d, point3dPE, point2dFE
    
    
class PETR(nn.Module):
    def __init__(self, grid, camNum, camC, D, clsNum, decoderLayerNum=6, 
                 wCls=1, wBox=1, auxLoss=False):
        super(PETR, self).__init__()
        self.clsNum = clsNum
        self.decoderLayerNum = decoderLayerNum
        self.auxLoss = auxLoss
        self.wCls = wCls
        self.wBox = wBox
        
        self.xBound = grid['xbound']
        self.yBound = grid['ybound']
        self.zBound = grid['zbound']
        
        self.positionEncoder3D = posEncoder3d(camNum, D, grid, camC)
        self.transformer = Transformer(decoderLayerNum=decoderLayerNum)
        
        self.clsMLP = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, clsNum)
        )
        
        self.bboxMLP = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8)
        )
        
        self.init_parameters()
        empty_weight = torch.ones(self.clsNum + 1)
        empty_weight[-1] = 0.1 # empty weight
        self.register_buffer('empty_weight', empty_weight)
    
    def init_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.clsMLP[-1].bias.data = torch.ones(self.clsNum) * bias_value
        nn.init.constant_(self.bboxMLP[-1].bias.data, 0)
        if self.auxLoss:
            self.clsMLP = nn.ModuleList([self.clsMLP for _ in range(self.decoderLayerNum)])
            self.bboxMLP = nn.ModuleList([self.bboxMLP for _ in range(self.decoderLayerNum)])
            
    def decode_boxes(self, bboxes, ref):
        ref = inverse_sigmoid(ref)
        xyz = (bboxes[..., 0:3] + ref).sigmoid()
        x = xyz[..., 0] * (self.xBound[1] - self.xBound[0]) + self.xBound[0]
        y = xyz[..., 1] * (self.yBound[1] - self.yBound[0]) + self.yBound[0]
        z = xyz[..., 2] * (self.zBound[1] - self.zBound[0]) + self.zBound[0]
        xyz = torch.stack([x, y, z], dim=-1)
        bboxes = torch.cat([xyz, bboxes[..., 3:]], dim=2)
        return  bboxes
        
    def forward(self, input):
        feat2d, point3dPE, point2dFE = self.positionEncoder3D(input)
        outputFeats, outputRefs = self.transformer(feat2d, point3dPE, point2dFE)
        cls_score = self.clsMLP(outputFeats[-1]).float()
        bboxes = self.bboxMLP(outputFeats[-1])
        bboxes = self.decode_boxes(bboxes, outputRefs[-1]).float()
        pred = {'pred_logits': cls_score, 'pred_boxes': bboxes}
        return pred
        
        
if __name__ == '__main__':
    num_views = 2
    input_channels = 3
    grid = {}
    grid['xbound'] = [-61.2, 61.2]
    grid['ybound'] = [-61.2, 61.2]
    grid['zbound'] = [-10, 10]
    xBound = grid['xbound']
    yBound = grid['ybound']
    zBound = grid['zbound']
    input_images = torch.randn(2, num_views, input_channels, 224, 224)
    model = PETR(grid=grid, camNum=num_views, camC=2048, D=64, clsNum=10, decoderLayerNum=1)
    
    input = {}
    input['image'] = input_images
    input['rots'] = torch.randn(2, num_views, 3, 3)
    input['rectRots'] = torch.randn(2, 3, 3)
    input['intrins'] = torch.randn(2, num_views, 3, 4)
    input['trans'] = torch.randn(2, num_views, 3)
    pred = model(input)
    print(pred['pred_logits'].shape) # (B, 900, 10)
    print(pred['pred_boxes'].shape) # (B, 900, 8)
    
