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

class stableBalancedFocalLossClass(nn.Module):

    def __init__(self, gamma=0, alpha=0.5, size_average=True):
        super(stableBalancedFocalLossClass, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):

        input, target = input.unsqueeze(0), target.unsqueeze(0)
        
        # Numerical stabilityt pytorhc trick.
        log_input = torch.clamp(torch.log(input), -100, 100)
        log_input_rev = torch.clamp(torch.log(1-input), -100, 100)

        # for probs
        pos = (-self.alpha * (1-input)**self.gamma * log_input)
        neg = (-(1-self.alpha) * (1-1-input)**self.gamma * log_input_rev)
        
        loss = (pos * target + neg * (1-target))

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



        # input, target = input.unsqueeze(0), target.unsqueeze(0)

        # # fo   r probs
        # min_ind = torch.exp(torch.tensor(-100).to(device)) # almost 0
        # max_ind = torch.tensor(1.0).to(device)- torch.exp(torch.tensor(-10).to(device)) # almost 1
        # input = torch.clamp(input, min = min_ind, max = max_ind) # so we do not log(0) or log(1) due to under- or overflow

        # pos = (-self.alpha * (1-input)**self.gamma * torch.log(input))
        # neg = (-(1-self.alpha) * (1-1-input)**self.gamma *  torch.log(1-input))
        # loss = (pos * target + neg * (1-target))

        # # Seem pytorch have something like this.. The gradient clipping like nullifies this anyway...
        # if loss.mean() >= max_ind:
        #     multuplier = 10
        # else:
        #     multuplier = 1

        # loss =  loss * 2 * multuplier # *2 is just a constant to make it more like BCE

        # # averaging (or not) loss
        # if self.size_average:
        #     return loss.mean()
        # else:
        #     return loss.sum()
        
        # The same as above... 
        #pos = (-self.alpha * (1-input)**self.gamma * torch.log(input))
        #neg = ((1-self.alpha) * (-input)**self.gamma *  torch.log(1-input))