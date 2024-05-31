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

connections = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

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

def seperateData(data, device):
    # Inputs:
    #     data - Dict including the input data from the dataset with different tensors
    #     device - cuda() or cpu()
    # Outputs:
    #     inputData - The required data fro model
    imgs = data['image'].to(device)
    intrins = data['intrins'].to(device)
    rectRots = data['rectRots'].to(device)
    
    inputData = {"image":imgs, "intrins":intrins, "rectRots":rectRots}
    return inputData

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
    testDataloader = DataLoader(testDataset, batch_size=batchSize, shuffle=False,
                                 pin_memory=True, num_workers=nw)

    
    num_cls = 8
    grid = {}
    grid['xbound'] = [-61.2, 61.2]
    grid['ybound'] = [-61.2, 61.2]
    grid['zbound'] = [-10, 10]
    
    model = petr(grid=grid, camNum=2, camC=2048, D=64, clsNum=num_cls, decoderLayerNum=6).to(device)
    weightPath = "weight/petr.pth"
    model.load_state_dict(torch.load(weightPath, map_location=device))
    
    # test
    for data in testDataloader:
        data_cuda = seperateData(data, device)
        filename = data['filename']

        pred, cls_score_aux, bboxes_aux = model(data_cuda)
        res = decode(pred, thres=0.2)
        
        # for each image in a batch
        for i in range(len(res)):
            boxCoord = xyzwhl2Corners(res[i]) # (M, 8, 3)
            M, _, _ = boxCoord.shape

            rectRot = data['rectRots'][i, :, :] # (3, 3)
            K = data['intrins'][i, 0, :, :] # Use left-camera
            K_homo = np.zeros((1, 4))
            K_homo[0, 3] = 1
            K = np.concatenate((K, K_homo), axis=0)
            imgName = filename[i]
            
            boxCoord = boxCoord.dot(rectRot.T)
            boxCoord = np.concatenate((boxCoord, np.ones((M, 8, 1))), axis=2)
            boxCoord = boxCoord.dot(K.T)[:, :, :-1]
            boxCoord = boxCoord / boxCoord[:, :, -1][..., None]
            boxCoord = boxCoord[:, :, :-1].astype(np.uint8)
            
            imgPath = "Kitti/image_left/testing/image_2/" + imgName + ".png"
            img = cv2.imread(imgPath)
            
            for i in range(boxCoord.shape[0]):
                for j in range(8):
                    cv2.circle(img, tuple(boxCoord[i, j, :]), 5, (0, 255, 0), -1)
                
                for start, end in connections:
                    cv2.line(img, tuple(boxCoord[i, start, :]), tuple(boxCoord[i, end, :]), (0, 255, 0), 2)
            cv2.imwrite('test.png', img)
        break
        
        
    
if __name__ == '__main__':
    test_petr()