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

class stabelBalancedFocalLossClass(nn.Module):

    def __init__(self, gamma=0, alpha=0.5, size_average=True):
        super(stabelBalancedFocalLossClass, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):

        input, target = input.unsqueeze(0), target.unsqueeze(0)
        

        #for logits
        # pos = (-self.alpha * (1-F.sigmoid(input))**self.gamma * F.logsigmoid(input))
        # neg = (-(1-self.alpha) * (-F.sigmoid(input))**self.gamma *  F.logsigmoid(1-input))
        # loss = (pos * target + neg * (1-target))

        # for probs
        min_ind = torch.exp(torch.tensor(-100).to(device)) # almost 0
        max_ind = torch.tensor(1.0).to(device) - torch.exp(torch.tensor(-10)).to(device) # almost 1
        input = torch.clamp(input, min = min_ind, max = max_ind) # so we do not log(0)
        
        pos = (-self.alpha * (1-input)**self.gamma * torch.log(input))
        neg = (-(1-self.alpha) * (1-1-input)**self.gamma *  torch.log(1-input))
        loss = (pos * target + neg * (1-target))

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()