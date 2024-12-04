import torch
import torch.nn as nn
from torch.nn import functional as F 
from .matcher import HungarianMatcher

from .OpenPCDet.pcdet.ops.iou3d_nms import iou3d_nms_cuda

class PETR_loss(nn.Module):
    def __init__(self, num_cls):
        super(PETR_loss, self).__init__()
        self.matcher = HungarianMatcher()
        self.num_cls = num_cls
        
    def forward(self, pred, tgt, cls_score_aux, bboxes_aux):
        # Inputs:
        #     pred - Include "pred_logits" with shape (B, N_query, num_cls) and "pred_boxes" with shape (B, N_query, 7)
        #     tgt - Include "labels" with shape (B, object_max_num) and "gt_boxes" with shape (B, object_max_num, 7)
        #     cls_score_aux - The list including label outputs from each transformer layer 
        #     bboxes_aux - The list including bounding box outputs from each transformer layer
        # Outputs:
        #     loss - The sum of class loss and boxes loss
        #     loss_tag - Include loss_cls, boxes_loss, and total_loss
        
        # Compute the aux loss first
        loss_aux = 0
        if cls_score_aux is not None and bboxes_aux is not None:
            for i in range(len(cls_score_aux)):
                pred_aux = {'pred_logits': cls_score_aux[i], 'pred_boxes': bboxes_aux[i]}
                with torch.cuda.amp.autocast(enabled=False):
                    loss_aux_i, _ = self.get_loss(pred_aux, tgt)
                loss_aux = loss_aux + loss_aux_i
        loss, loss_tags = self.get_loss(pred, tgt)
        if cls_score_aux is not None:
            loss += loss_aux
            loss_tags['aux_loss'] = loss_aux.clone().detach()
        return loss, loss_tags     

    def get_loss(self, pred, tgt):
        # Filter all valid object labels and bounding boxes from the dataset output
        tgt_list = []
        for i in range(tgt['labels'].shape[0]):
            mask = tgt['gt_boxes'][i, :, 0] != -1
            masked_boxes = tgt['gt_boxes'][i, mask, :]
            masked_boxes = self.gt_boxes_process(masked_boxes)
            masked_labels = tgt['labels'][i, mask]
            tgt_list.append({"labels":masked_labels.long(), "gt_boxes":masked_boxes.float()})
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
        xyz = gt_boxes[:, 3:6]
        wlh = gt_boxes[:, 0:3].log()
        ang = gt_boxes[:, -1].view(-1, 1)
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
        loss = sigmoid_focal_loss(cls_scores, target_labels_onehot, totalBox)
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
        # loss = (loss_bbox + loss_giou) /totalBox
        loss = (loss_bbox) /totalBox
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
    
class DN_loss(nn.Module):
    def __init__(self, training, aux_num=5, focal_alpha=0.25):
        super(DN_loss, self).__init__()
        self.training = training
        self.aux_num = aux_num
        self.focal_alpha = focal_alpha
        
    def forward(self, mask_dict):
        losses = {}
        if self.training and 'output_known_lbs_bboxes' in mask_dict:
            known_labels, known_bboxs, output_known_class, output_known_coord, \
            num_tgt = self.prepare_loss(mask_dict)
            losses.update(self.label_loss(output_known_class[-1], known_labels, num_tgt))
            losses.update(self.box_loss(output_known_coord[-1], known_bboxs, num_tgt))
        else:
            losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
            
        for i in range(self.aux_num):
            if self.training and 'output_known_lbs_bboxes' in mask_dict:
                l_dict = self.label_loss(output_known_class[i], known_labels, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = self.box_loss(output_known_coord[i], known_bboxs, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
            
    def prepare_loss(self, mask_dict):
        # Prepare needy variables
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice']
        known_indice = mask_dict['known_indice']
        batch_idx = mask_dict['batch_idx']
        bid = batch_idx[known_indice]
        
        # Get corresponding box from the transformer output
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt
        
    def box_loss(self, src_boxes, tgt_boxes, num_tgt):
        if len(tgt_boxes) == 0:
            return {
                'tgt_loss_bbox': torch.as_tensor(0.).to('cuda'),
                'tgt_loss_giou': torch.as_tensor(0.).to('cuda'),
            }

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        losses = {}
        losses['tgt_loss_bbox'] = loss_bbox.sum() / num_tgt

        loss_giou = 1 - torch.diag(boxes_iou3d_gpu(src_boxes, tgt_boxes))
        losses['tgt_loss_giou'] = loss_giou.sum() / num_tgt
        return losses
    
    def label_loss(self, src_logits_, tgt_labels_, num_tgt):
        if len(tgt_labels_) == 0:
            return {
                'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
                'tgt_class_error': torch.as_tensor(0.).to('cuda'),
            }
            
        src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {'tgt_loss_ce': loss_ce}
        return losses
            
        
def sigmoid_focal_loss(cls_scores, target_labels_onehot, totalBox, alpha=0.25, gamma=2):
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

def dn_post_process(outputs_class, outputs_coord, mask_dict):
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_bboxes']=(output_known_class,output_known_coord)
        
    return outputs_class, outputs_coord
