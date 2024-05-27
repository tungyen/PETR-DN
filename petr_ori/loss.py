import torch
import torch.nn as nn
from torch.nn import functional as F 
from petr.matcher import HungarianMatcher

from . import iou3d_nms_cuda

class PETR_loss(nn.Module):
    def __init__(self, num_cls):
        super(PETR_loss, self).__init__()
        self.matcher = HungarianMatcher()
        self.num_cls = num_cls
        
    def forward(self, pred, tgt):
        # Inputs:
        #     pred - Include "pred_logits" with shape (B, N_query, num_cls) and "pred_boxes" with shape (B, N_query, 7)
        #     tgt - Include "labels" with shape (B, object_max_num) and "gt_boxes" with shape (B, object_max_num, 7)
        # Outputs:
        #     loss - The sum of class loss and boxes loss
        #     loss_tag - Include loss_cls, boxes_loss, and total_loss
        
        # Filter all valid object labels and bounding boxes from the dataset output
        tgt_list = []
        for i in range(tgt['labels'].shape[0]):
            mask = tgt['gt_boxes'][i, :, 0] != -1
            masked_boxes = tgt['gt_boxes'][i, mask, :]
            masked_boxes = self.gt_boxes_process(masked_boxes)
            masked_labels = tgt['labels'][i, mask]
            tgt_list.append({"labels":masked_labels, "gt_boxes":masked_boxes})
        indices, cost_tags = self.matcher(pred, tgt_list)
        totalBox = sum([t['labels'].shape[0] for t in tgt_list])
        
        loss_cls = self.loss_cls(pred, tgt_list, indices, totalBox)
        loss_box = self.loss_box(pred, tgt_list, indices, totalBox)
        
        loss = loss_cls + loss_box
        loss_tags = {
            'cls': loss_cls.clone().detach(),
            'box': loss_box.clone().detach(),
            'loss': loss.clone().detach()
        }
        loss_tags.update(cost_tags)
        return loss, loss_tags
         
    def gt_boxes_process(self, gt_boxes):
        # Inputs:
        #     gt_boxes - The input ground truth boxes with shape (M, 7), where B is batch size and M is object number
        # Outputs:
        #     gt_boxes_processed - The processed gt boxes with shape (M, 7)
        xyz = gt_boxes[:, 0:3]
        wlh = gt_boxes[:, 3:6].log()
        ang = gt_boxes[:, -1]
        gt_boxes_processed = torch.cat([xyz, wlh, ang], dim=-1)
        return gt_boxes_processed
    
    def loss_cls(self, pred, tgt_list, indices, totalBox):
        # Inputs:
        #     pred - Include "pred_logits" with shape (B, N_query, num_cls) and "pred_boxes" with shape (B, N_query, 7)
        #     tgt_list - List include label and ground truth bounding box for each case
        #     indices - The list include matching result for each case with tuple (query_index, gt_label_index)
        #     totalBox - The total number of object in the whole batch
        # Outputs:
        #     loss - The mean classification loss
        cls_scores = pred['pred_logits']
        idx = self.getQueryIndex(indices)
        target_labels = torch.cat([t['labels'][col] for t, (_, col) in zip(tgt_list, indices)])
        target_labels_table = torch.full(cls_scores.shape[:2], self.num_cls, dtype=torch.int64, device=cls_scores.device)
        target_labels_table[idx] = target_labels
        target_labels_onehot = torch.zeros([cls_scores.shape[0], cls_scores.shape[1], cls_scores.shape[2]+1],
                                            dtype=cls_scores.dtype, layout=cls_scores.layout, device=cls_scores.device)
        target_labels_onehot.scatter_(2, target_labels_table.unsqueeze(-1), 1)

        target_labels_onehot = target_labels_onehot[:, :, :-1]
        loss = self.sigmoid_focal_loss(cls_scores, target_labels_onehot, totalBox)
        return loss
    
    def loss_box(self, pred, tgt_list, indices, totalBox):
        # Inputs:
        #     pred - Include "pred_logits" with shape (B, N_query, num_cls) and "pred_boxes" with shape (B, N_query, 7)
        #     tgt_list - List include label and ground truth bounding box for each case
        #     indices - The list include matching result for each case with tuple (query_index, gt_label_index)
        #     totalBox - The total number of object in the whole batch
        # Outputs:
        #     loss - The mean bounding boxes loss
        idx = self.getObjectIndex(indices)
        pred_boxes = pred['pred_boxes'][idx]
        target_boxes = torch.cat([t['gt_boxes'][col] for t, (_, col) in zip(tgt_list, indices)], dim=0)
        loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction='none').sum()
        loss_giou = torch.sum(1 - torch.diag(boxes_iou3d_gpu(pred_boxes, target_boxes)))
        loss = (loss_bbox + loss_giou) /totalBox
        return loss
        
    def getQueryIndex(self, indices):
        # Inputs:
        #     indices - The list include matching result for each case with tuple (query_index, gt_label_index)
        # Outputs:
        #     batchIndex - The batch index for each query index with shape (total_boxes, )
        #     queryIndex - The index of the query with shape (total_boxes, )
        batchIndex = torch.cat([torch.full_like(row, i) for i, (row, _) in enumerate(indices)])
        queryIndex = torch.cat([row for (row, _) in indices])
        return batchIndex, queryIndex
    
    def getObjectIndex(self, indices):
        # Inputs:
        #     indices - The list include matching result for each case with tuple (query_index, gt_label_index)
        # Outputs:
        #     batchIndex - The batch index for each query index with shape (total_boxes, )
        #     objectIndex - The index of the query with shape (total_boxes, )
        batchIndex = torch.cat([torch.full_like(col, i) for i, (_, col) in enumerate(indices)])
        objectIndex = torch.cat([col for (_, col) in indices])
        return batchIndex, objectIndex
        
        
    def sigmoid_focal_loss(self, cls_scores, target_labels_onehot, totalBox, alpha=0.25, gamma=2):
        prob = cls_scores.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(cls_scores, target_labels_onehot, reduction="none")
        p_t = prob * target_labels_onehot + (1 - prob) * (1 - target_labels_onehot)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * target_labels_onehot + (1 - alpha) * (1 - target_labels_onehot)
            loss = alpha_t * loss
        return loss.sum() / totalBox
    

def boxes_iou3d_gpu(boxes_a, boxes_b):
    # Inputs:
    #     boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    #     boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]
    # Outputs:
    #     ans_iou: (N, N)


    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d
