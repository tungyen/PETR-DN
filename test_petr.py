import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
import argparse
import time
from tqdm import tqdm
import os

from Kitti import KittiDataset
from model import petr

color_list = [(0, 255, 255), (255, 255, 0), (0, 200, 255), (127, 0, 255), (0, 127, 255), \
    (127, 0, 127), (127, 100, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), ]

def decode(pred, thres=0.1):
    res = []
    for i in range(pred['pred_logits'].shape[0]):
        cls_prob, cls_idx = torch.topk(pred['pred_logits'][i].sigmoid(), k=1, dim=1)
        mask = cls_prob.squeeze() > thres
        pred_cls = cls_idx[mask]
        pred_prob = cls_prob[mask]
        boxes = pred['pred_boxes'][i]
        boxes_xyz = boxes[:, 0:3]
        boxes_wlh = boxes[:, 3:6].exp()
        boxes_ang = boxes[..., -1].view(-1, 1)
        boxes_dec = torch.cat([boxes_xyz, boxes_wlh, boxes_ang], dim=-1).clone()
        pred_box = boxes_dec[mask, :].clone()
        result = torch.cat([pred_cls.float(), pred_prob, pred_box], dim=1).detach().cpu()
        res.append(result)
    return res

def xyzwhl2Corners(boxes):
    # boxes: (M, 7)
    res = []
    for i in range(boxes.shape[0]):
        x, y, z = boxes[i, :3]
        h, w, l = boxes[i, 3:6]
        theta = boxes[i, -1]
        
        local_corners = np.array([
            [-w/2, -h/2, -l/2],
            [ w/2, -h/2, -l/2],
            [ w/2,  h/2, -l/2],
            [-w/2,  h/2, -l/2],
            [-w/2, -h/2,  l/2],
            [ w/2, -h/2,  l/2],
            [ w/2,  h/2,  l/2],
            [-w/2,  h/2,  l/2]
        ])
        
        R = np.array([
            [np.sin(theta), 0, np.cos(theta)],
            [0,  1, 0],
            [-np.cos(theta), 0, np.sin(theta)]
        ])
        
        rotated_corners = local_corners.dot(R.T)
        world_corners = rotated_corners + np.array([x, y-h/2, z])
        res.append(world_corners)
    return np.array(res)

def test_petr():
    # Prepare for the dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kitti_path = "Kitti"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    testDataset = KittiDataset(1280, 640, root=kitti_path, transform=transform, mode="test")
    batchSize = 1
    nw = min([os.cpu_count(), batchSize if batchSize > 1 else 0, 8])
    trainDataloader = DataLoader(testDataset, batch_size=batchSize, shuffle=True,
                                 pin_memory=True, num_workers=nw)

    
    num_cls = 8
    grid = {}
    grid['xbound'] = [-61.2, 61.2]
    grid['ybound'] = [-61.2, 61.2]
    grid['zbound'] = [-10, 10]
    
    model = petr(grid=grid, camNum=2, camC=2048, D=64, clsNum=num_cls, decoderLayerNum=6).to(device)

    # test
    for data in trainDataloader:
        filename = data['filename']
        pred = model(data)
        res = decode(pred)
        
        boxCoord = xyzwhl2Corners(res) # list[(M, 8)]
        rectRot = data['rectRots'] # (B, 3, 3)
        K = data['intrins'][:, 0, :, :] # Use left-camera
        
        
    
if __name__ == '__main__':
    test_petr()