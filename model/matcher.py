import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):

    def __init__(self, lambda_cls=2.0, lambda_bbox=0.25, alpha = 0.25, gamma = 2.0):

        super().__init__()
        self.cost_class = lambda_cls
        self.cost_bbox = lambda_bbox
        self.alpha=alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, preds, tgts):
        # Inputs:
        #     preds - Dictionary including:
        #          "pred_logits": Tensor with shape (B, N_query, num_cls)
        #          "pred_boxes": Tensor with shape (B, N_query, 7)
        #     tgts - A list of dictionary including:
        #          "labels": Tensor with shape (obj_num, ), where obj_num is different for each batch
        #          "gt_boxes": Tensor with shape (obj_num, 7)
        # Outputs:
        B, N_query, _ = preds["pred_logits"].shape
        pred_scores = preds["pred_logits"].flatten(0, 1).sigmoid() # (B * N_query, num_cls)
        pred_bboxes = preds["pred_boxes"].flatten(0, 1)  # (B * N_query, 7)

        gt_labels = torch.cat([t["labels"] for t in tgts]) # (total, )
        gt_boxes = torch.cat([t["gt_boxes"] for t in tgts]) # (total, 7)
        # print("Fuck gt boxes: ", gt_boxes)

       # Compute the classification cost.
        neg_cost_class = (1 - self.alpha) * (pred_scores ** self.gamma) * (-(1 - pred_scores + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
        cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels] # (B * N_query, total)
        cost_bbox = torch.cdist(pred_bboxes, gt_boxes) # (B * N_query, total)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class # (B * N_query, total)
        C = C.view(B, N_query, -1).cpu() # (B, N_query, total)
        
        sizes = [t["gt_boxes"].shape[0] for t in tgts]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        # i corresponds to N_query, and j corresponds to all objects in a batch
        out_ind = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # vis match cost
        cost_cls = self.calc_match_cost(cost_class, indices, B, N_query, sizes)
        cost_box = self.calc_match_cost(cost_bbox, indices, B, N_query, sizes)
        cost_tags = {
            'matcher_cost_cls': cost_cls.clone().detach(),
            'matcher_cost_box': cost_box.clone().detach(),
        }
        return out_ind, cost_tags

    @torch.no_grad()
    def calc_match_cost(self, cost, indices, bs, num_queries, sizes):
        cost_ = cost.view(bs, num_queries, -1)
        cost_b_list = []
        for b, c in enumerate(cost_.split(sizes, -1)):
            i, j = indices[b]
            cost_b = c[b][i, j].mean()
            cost_b_list.append(cost_b)
        return torch.stack(cost_b_list).mean()