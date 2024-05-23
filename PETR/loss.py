import torch
import torch.nn as nn
from torch.nn import functional as F 
from PETR.matcher import HungarianMatcher

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
        loss = F.l1_loss(pred_boxes, target_boxes, reduction='none')
        loss = loss.sum() /totalBox
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