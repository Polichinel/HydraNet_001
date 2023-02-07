####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLossReg(nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLossReg, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        #weight = Variable(self.weight) # why even here?
        #logpt = -F.cross_entropy(input, target) # why - ???
        l = F.L1_Loss(input, target)
        l2 = F.MSELoss(input, target)

        # compute the loss
        loss = (l**self.gamma) * l2

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()