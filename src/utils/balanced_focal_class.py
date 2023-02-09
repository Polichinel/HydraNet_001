####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BalancedFocalLossClass(nn.Module):

    def __init__(self, gamma=0, alpha=0.5, size_average=True):
        super(BalancedFocalLossClass, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):

        input, target = input.unsqueeze(0), target.unsqueeze(0)
        input = torch.clamp(input, min = torch.exp(torch.tensor(-100).to(device))) # so we do not log(0)

        pos = (-self.alpha * (1-input)**self.gamma * torch.log(input))
        neg = (-(1-self.alpha) * (1-1-input)**self.gamma *  torch.log(1-input))

        loss = (pos * target + neg * (1-target))

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()