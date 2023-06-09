import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss_new(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss_new, self).__init__()
        self.alpha = alpha  # Focal loss balancing parameter
        self.gamma = gamma  # Focal loss focusing parameter
        self.reduction = reduction  # Loss reduction method

    def forward(self, logits, targets):

        # input, target = input.unsqueeze(0), target.unsqueeze(0)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # Calculate the cross-entropy loss
        pt = torch.exp(-ce_loss)  # Calculate the probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Compute the focal loss

        if self.reduction == 'mean':
            return focal_loss.mean()  # Average the loss if reduction is set to 'mean'
        elif self.reduction == 'sum':
            return focal_loss.sum()  # Sum the loss if reduction is set to 'sum'
        else:
            return focal_loss  # Return the focal loss without reduction
