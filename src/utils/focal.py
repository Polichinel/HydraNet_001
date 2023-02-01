import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            gamma=2.0,
            reduction='mean'
    ):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = torch.sigmoid(input_tensor)
        prob = torch.exp(log_prob)
        
        term = ((1.0 - prob) ** self.gamma) * log_prob
        return F.nll_loss(term, target_tensor)

# class FocalLoss(nn.Module):
#     def __init__(
#             self,
#             weight=None,
#             gamma=2.,
#             reduction='mean'
#     ):
#         nn.Module.__init__(self)
#         self.weight = weight
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, input_tensor, target_tensor):
#         log_prob = F.log_softmax(input_tensor, dim=-1)
#         prob = torch.exp(log_prob)
#         return F.nll_loss(
#                 ((1 - prob) ** self.gamma) * log_prob,
#                 target_tensor,
#                 weight=self.weight,
#                 reduction=self.reduction
#         )
