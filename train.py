import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
from Kitti.kittiDataset import KittiDataset
from PETR.loss import PETR_loss
from PETR.model import PETR

import os


def PETR_train():
    # Prepare for the dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    
if __name__ == '__main__':
    PETR_train()