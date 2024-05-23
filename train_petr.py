import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import os

from Kitti import KittiDataset
from PETR import petr_loss as petr_loss
from PETR import petr as petr


def PETR_train():
    # Prepare for the dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    kitti_path = "Kitti"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainDataset = KittiDataset(1280, 640, root=kitti_path, transform=transform)
    batchSize = 4
    nw = min([os.cpu_count(), batchSize if batchSize > 1 else 0, 8])
    trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True,
                                 pin_memory=True, num_workers=nw)
    
    num_cls = 8
    grid = {}
    grid['xbound'] = [-61.2, 61.2]
    grid['ybound'] = [-61.2, 61.2]
    grid['zbound'] = [-10, 10]
    
    model = petr(grid=grid, camNum=2, camC=2048, D=64, clsNum=num_cls, decoderLayerNum=6).float()
    for input in trainDataloader:
        pred = model(input)
        print(pred['pred_logits'].shape) # (B, 900, 8)
        print(pred['pred_boxes'].shape) # (B, 900, 7)
        break

    
if __name__ == '__main__':
    PETR_train()