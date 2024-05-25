import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import os
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
import numpy as np

from Kitti import KittiDataset
from PETR import petr_loss
from PETR import petr
from PETR import matcher
# from metrics import *

lr0 = 0.005
weight_decay = 0.01
lrf = 0.05
max_epoch = 12
momentum = 0.9
steps = [9, 11]
lrdecay = 0.1

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
            
def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
            
def make_optimizer(model):
    pg0, pg1, pg2, pg3 = [], [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.LayerNorm):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            if hasattr(v, 'img_encoder'):
                pg3.append(v.weight)
            else:
                pg1.append(v.weight)  # apply decay

    params = [{'params': pg1, 'lr':lr0, 'weight_decay': weight_decay},
            {'params': pg3, 'lr':(lr0 * 0.1), 'weight_decay': weight_decay},
            {'params': pg0, 'lr':lr0, 'weight_decay': 0.},
            {'params': pg2, 'lr':lr0, 'weight_decay': 0.}]

    optimizer = optim.AdamW(params, betas=(momentum, 0.999))
    scheduler = 'lambda'
    if scheduler == 'lambda':
        lf = one_cycle(1, lrf, max_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lrdecay)
    return optimizer, scheduler

def seperateData(data, device):
    # Inputs:
    #     data - Dict including the input data from the dataset with different tensors
    #     device - cuda() or cpu()
    # Outputs:
    #     inputData - The required data fro model
    #     tgt - The ground truth tensor
    imgs = data['image'].to(device)
    intrins = data['intrins'].to(device)
    rectRots = data['rectRots'].to(device)
    bboxes = data['box3d'].to(device)
    labels = data['labels'].to(device)
    
    inputData = {"image":imgs, "intrins":intrins, "rectRots":rectRots}
    tgt = {"gt_boxes":bboxes, "labels":labels}
    return inputData, tgt

def PETR_train():
    # Prepare for the dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kitti_path = "Kitti"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainDataset = KittiDataset(1280, 640, root=kitti_path, transform=transform)
    valDataset = KittiDataset(1280, 640, root=kitti_path, transform=transform, mode='eval')
    
    batchSize = 4
    nw = min([os.cpu_count(), batchSize if batchSize > 1 else 0, 8])
    trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True,
                                 pin_memory=True, num_workers=nw)
    evalDataloader = DataLoader(valDataset, batch_size=batchSize, shuffle=False,
                                 pin_memory=True, num_workers=nw)
    
    num_cls = 8
    grid = {}
    grid['xbound'] = [-61.2, 61.2]
    grid['ybound'] = [-61.2, 61.2]
    grid['zbound'] = [-10, 10]
    
    model = petr(grid=grid, camNum=2, camC=2048, D=64, clsNum=num_cls, decoderLayerNum=6).to(device)
    optimizer, scheduler = make_optimizer(model)
    criterion = petr_loss(num_cls=num_cls)
    
    for epoch in tqdm(range(max_epoch)):
        print("Epoch {} start now!".format(epoch+1))
        
        # train
        for data in trainDataloader:
            data, tgt = seperateData(data)
            pred = model(data)
            loss, _ = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        print("Epoch {}-training loss===>{}".format(epoch+1, loss.item()))
        
        # Validation
        with torch.no_grad():
            AP_list = {}
            count = {}
            ATE = 0
            ate_count = 0
            for batch in evalDataloader:
                # Get prediction and ground truth
                data, tgt = seperateData(batch)
                pred = model(data)
                batch_tgt = []
                for i in range(tgt['labels'].shape[0]):
                    mask = tgt['gt_boxes'][i, :, 0] != -1
                    masked_boxes = tgt['gt_boxes'][i, mask, :]
                    masked_boxes = petr_loss.gt_boxes_process(masked_boxes)
                    masked_labels = tgt['labels'][i, mask]
                    batch_tgt.append({"labels":masked_labels, "gt_boxes":masked_boxes})
                
                # Process the batch case by case
                batch_pred_logits, batch_pred_boxes = pred['pred_logits'], pred['pred_boxes']
                B = batch_pred_logits.shape[0]
                for i in range(B):
                    pred_labels = torch.max(batch_pred_logits[i, :, :], dim=1)[1] # (900, )
                    pred_boxes = batch_pred_boxes[i, :, :] # (900, 7)
                    gt_labels = batch_tgt[i]['labels'] # (obj_num, )
                    gt_boxes = batch_tgt[i]['gt_boxes'] # (obj_num, 7)
                    
                    uniques = torch.unique(torch.cat((pred_labels, gt_labels)))
                    for cls in uniques:
                        if cls not in count:
                            count[cls] = 0.0
                            AP_list[cls] = 0.0
                        
                        pred_boxes_cls = pred_boxes[pred_labels==cls, :]
                        gt_boxes_cls = gt_boxes[gt_labels==cls, :]
                        
                        if pred_boxes_cls.shape[0] == 0 or gt_boxes_cls.shape[0] == 0:
                            continue
                        
                        cost_boxes = torch.cdist(pred_boxes_cls, gt_boxes_cls, p=1)
                        cost_boxes = cost_boxes / torch.max(cost_boxes)
                        rows, cols = linear_sum_assignment(cost_boxes)
                        
                        # AP
                        y_true = np.ones(gt_boxes_cls.shape[0])
                        y_scores = cost_boxes[rows, cols].numpy()
                        ap = average_precision_score(y_true, y_scores)
                        AP_list[cls] += ap
                        count[cls] += 1
                        
                        # ATE
                        pred_boxes_cls = pred_boxes_cls[rows]
                        gt_boxes_cls = gt_boxes_cls[cols]
                        ate = torch.sum((pred_boxes_cls[:, :3]-gt_boxes_cls[:, :3])**2) / pred_boxes_cls.shape[0]
                        ATE += ate
                        ate_count += 1
                        
            ATE = ATE / ate_count
            for cls in AP_list:
                AP_list[cls] = AP_list[cls] / count[cls]
            mAP = np.sum(AP_list.item().values()) / len(AP_list)
            
            print("mAP of epoch {} is {}.".format(epoch+1, mAP))
            print("ATE of epoch {} is {}.".format(epoch+1, ATE))
                        
                        
                        
                        
                    
                    
                    
                    
                

    
if __name__ == '__main__':
    PETR_train()