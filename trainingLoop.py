import numpy as np
import random
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from trainingLoopUtils import *
from recurrentUnet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

start_t = time.time()

print('loading data....')
#location = '/home/simon/Documents/Articles/ConflictNet/data/raw'
location = '/home/projects/ku_00017/data/raw/ConflictNet'
file_name = "/ucpd_vol.pkl"
pkl_file = open(location + file_name, 'rb')
ucpd_vol = pickle.load(pkl_file)
pkl_file.close()


# Hyper parameters.
hidden_channels = 64
input_channels = 1
output_channels = 1
dropout_rate = 0.5

unet = UNet(input_channels, hidden_channels, output_channels, dropout_rate).to(device)

learning_rate = 0.0001
weight_decay = 0.01
optimizer = torch.optim.Adam(unet.parameters(), lr = learning_rate, weight_decay = weight_decay)
criterion_reg = nn.MSELoss().to(device)
criterion_class = nn.CrossEntropyLoss().to(device)

transformer = transforms.Compose([transforms.RandomRotation((0,360)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])

draws = 1000 # same as draws..give other name... 
avg_losses = []

print('Training initiated...')

for i in range(draws):

    #input_tensor = torch.tensor(train_ucpd_vol[:, sub_images_y[i][0]:sub_images_y[i][1], sub_images_x[i][0]:sub_images_x[i][1], 4].reshape(1, seq_len, dim, dim)).float() #Why not do this in funciton?
    input_tensor = get_input_tensors(ucpd_vol)
    # data augmentation (can be turned of for final experiments)
    input_tensor = transformer(input_tensor) # rotations and flips

    avg_loss = train(unet, optimizer, criterion_reg, criterion_class, input_tensor, plot = False)
    avg_losses.append(avg_loss.cpu().detach().numpy())

    
    if i % 100 == 0: # print steps 100
        print(f'{i} {avg_loss:.4f}') # could plot ap instead...



print('Done training. Saving model...')

PATH = 'unet.pth'
torch.save(unet.state_dict(), PATH)