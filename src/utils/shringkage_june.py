import torch
import torch.nn as nn
import torch.nn.functional as F

class ShrinkageLoss_new(nn.Module):
    def __init__(self, a=10, c=0.2, size_average=True):
        super(ShrinkageLoss_new, self).__init__()
        self.a = a  # Shrinkage factor
        self.c = c  # Threshold
        self.size_average = size_average

    def forward(self, input, target):

        # input, target = input.unsqueeze(0), target.unsqueeze(0) 

        l = torch.abs(target - input)  # Absolute difference between target and input
        exp_term = torch.exp(self.a * (self.c - l))  # Exponential term to ensure numerical stability
        loss = (l ** 2) / (1 + exp_term)  # Shrinkage loss calculation

        if self.size_average:
            return loss.mean()  # Average the loss if size_average is True
        else:
            return loss.sum()  # Sum the loss if size_average is False
