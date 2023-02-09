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

class FocalLossReg(nn.Module):

    def __init__(self, gamma=0, alpha=1, size_average=True):
        super(FocalLossReg, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):

        input, target = input.unsqueeze(0), target.unsqueeze(0)
        #input = torch.clamp(input, min = np.exp(torch.tensor(-100))) # could do this for no negatives???

        error = target - input
        exp_error = torch.exp(error)  #torch.clamp(se, min = np.exp(-100)))
        loss = exp_error.mean()

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()