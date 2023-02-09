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

class FocalLossClass(nn.Module):

    def __init__(self, gamma=0, alpha=1, size_average=True):
        super(FocalLossClass, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):

        input, target = input.unsqueeze(0), target.unsqueeze(0)
        
        #for logits
        pos = ((1-F.sigmoid(input))**self.gamma * F.logsigmoid(input))
        neg = ((-F.sigmoid(input))**self.gamma *  F.logsigmoid(1-input))

        loss = -(pos * target + neg * (1-target))
        
        #for preds 
        # input = torch.clamp(input, min = torch.exp(torch.tensor(-100).to(device))) # so we do not log(0)

        # logpt = (target * torch.log(input) + (1-target) * torch.log(1-input))
        # loss = -self.alpha * ((1-torch.exp(logpt))**self.gamma) * logpt # for gamma = 0 and alpha = 1 we get the BCELoss

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()