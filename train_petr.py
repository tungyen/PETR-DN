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
from PETR import petr_loss
from PETR import petr
from PETR import matcher

lr0 = 0.005
weight_decay = 0.01
scheduler = 'lambda'
lrf = 0.05
max_epoch = 12
momentum: 0.9
steps: [9, 11]
lrdecay: 0.1

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

    if scheduler == 'lamda':
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

def metrics_calculate(pred, tgt):
    # Inputs:
    #     preds - Dictionary including:
    #          "pred_logits": Tensor with shape (B, N_query, num_cls)
    #          "pred_boxes": Tensor with shape (B, N_query, 7)
    #     tgts - A list of dictionary including:
    #          "labels": Tensor with shape (obj_num, ), where obj_num is different for each batch
    #          "gt_boxes": Tensor with shape (obj_num, 7)
    association = matcher()
    
    tgt_list = []
    for i in range(tgt['labels'].shape[0]):
        mask = tgt['gt_boxes'][i, :, 0] != -1
        masked_boxes = tgt['gt_boxes'][i, mask, :]
        masked_boxes = self.gt_boxes_process(masked_boxes)
        masked_labels = tgt['labels'][i, mask]
        tgt_list.append({"labels":masked_labels, "gt_boxes":masked_boxes})
    
    indices, _ = association(pred, tgt_list)
    
    pred_logits, pred_boxes = pred['pred_logits'], pred['pred_boxes']
    B = pred_logits.shape[0]
    for i in range(B):
        pred_logit, pred_box = pred_logits[i, :, :], pred_bboxes[i, :, :] # (900, 8), (900, 7)
        queryIndex, objectIndex = indices[i] # (900, ), (obj_num, )
        gt_label, gt_box = tgt_list[i]['labels'], tgt_list[i]['gt_boxes'] # (obj_num, ), (obj_num, 7)
        
        match_pred_logit, match_pred_box = pred_logit[queryIndex, :], pred_box[queryIndex, :] # (obj_num, 8), # (obj_num, 7)
        match_gt_label, match_gt_box = gt_label[objectIndex], gt_box[objectIndex, :] # (obj_num, ), # (obj_num, 7)
        match_pred_cls = torch.max(match_pred_logit, dim=1)[1]
        
    


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
    evalDataloader = DataLoader(evalDataset, batch_size=batchSize, shuffle=False,
                                 pin_memory=True, num_workers=nw)
    
    num_cls = 8
    grid = {}
    grid['xbound'] = [-61.2, 61.2]
    grid['ybound'] = [-61.2, 61.2]
    grid['zbound'] = [-10, 10]
    
    model = petr(grid=grid, camNum=2, camC=2048, D=64, clsNum=num_cls, decoderLayerNum=6).to(device)
    optimizer, scheduler = make_optimizer(model)
    criterion = petr_loss(num_cls=num_cls)
    
    for epoch in tqdm(range(epoch)):
        print("Epoch {} start now!".format(epoch+1))
        
        # train
        for data in trainDataloader:
            data, tgt = seperateData(data)
            pred = model(data)
            # print(pred['pred_logits'].shape) # (B, 900, 8)
            # print(pred['pred_boxes'].shape) # (B, 900, 7)
            loss, loss_tags = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        print("Epoch {}-training loss===>{}".format(epoch+1, loss.item()))
        
        # Validation
        with torch.no_grad():
            for data in evalDataloader:
                data, tgt = seperateData(data)
                pred = model(data)

    
if __name__ == '__main__':
    PETR_train()