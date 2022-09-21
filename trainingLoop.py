import numpy as np
import random
import pickle
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

#import geomloss # New loss. also needs: pip install pykeops 

import wandb


from trainingLoopUtils import *
from recurrentUnet import *


def get_data():

    # Data
    print('loading data....')
    #location = '/home/simon/Documents/Articles/ConflictNet/data/raw'
    location = '/home/projects/ku_00017/data/raw/conflictNet'
    file_name = "/ucpd_vol.pkl"
    pkl_file = open(location + file_name, 'rb')
    ucpd_vol = pickle.load(pkl_file)
    pkl_file.close()

    return(ucpd_vol)


def choose_loss(config):

    if config.loss == 'a':
        PATH = 'unet_sinkhorn.pth'
        criterion_reg = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)
        criterion_class = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)

        # But scaling does a big difference so woth trying 0.3-0.7
        # set higer reach: ex 64
        # set highet scaling = 0.9
        # Scaling 0.1 worse, scaking 0.9 worse
        # try p = 1
        # Needs to set reach: "[...] if reach is None (balanced Optimal Transport), the resulting routine will expect measures whose total masses are equal with each other."
        # Needs to set backend explicitly: online or multiscale

    elif config.loss == 'b':
        PATH = 'unet.pth'
        criterion_reg = nn.MSELoss().to(device) # works
        criterion_class = nn.BCELoss().to(device) # works

    else:
        print('Wrong loss...')
        sys.exit()

    return(criterion_reg, criterion_class)


def make(config):

    unet = UNet(config.input_channels, config.hidden_channels, config.output_channels, config.dropout_rate).to(device)

    criterion = choose_loss(config) # this is a touple of the reg and the class criteria

    optimizer = torch.optim.Adam(unet.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
    # optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = config.betas)

    return(unet, criterion, optimizer) #, dataloaders, dataset_sizes)



def training_loop(config, unet, criterion, optimizer, ucpd_vol):
    # add spatail transformer
    transformer = transforms.Compose([transforms.RandomRotation((0,360)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
    avg_losses = []

    criterion_reg, criterion_class = criterion

    print('Training initiated...')

    for i in range(config.samples):

        print(f'{i+1}/{config.sample}', end = '\r')

        #input_tensor = torch.tensor(train_ucpd_vol[:, sub_images_y[i][0]:sub_images_y[i][1], sub_images_x[i][0]:sub_images_x[i][1], 4].reshape(1, seq_len, dim, dim)).float() #Why not do this in funciton?
        input_tensor, meta_tensor_dict = get_input_tensors(ucpd_vol)
        # data augmentation (can be turned of for final experiments)
        input_tensor = transformer(input_tensor) # rotations and flips

        train(unet, optimizer, criterion_reg, criterion_class, input_tensor, meta_tensor_dict, device, unet, plot = False)

        #avg_loss = train(unet, optimizer, criterion_reg, criterion_class, input_tensor, meta_tensor_dict, device, unet, plot = False)
        #avg_losses.append(avg_loss.cpu().detach().numpy())

        
        # if i % 100 == 0: # print steps 100
        #     print(f'{i} {avg_loss:.4f}') # could plot ap instead...


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="RUNET_experiments", entity="nornir", config=hyperparameters): #new projrct name!!!
        
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # get the data
        ucpd_vol = get_data()

        # make the model, data, and optimization problem
        unet, criterion, optimizer = make(config)
        print(unet)

        training_loop(config, unet, criterion, optimizer, ucpd_vol)
        print('Done training')

        return(unet)


if __name__ == "__main__":

    wandb.login()

    # Hyper parameters.
    hyperparameters = {
    "hidden_channels" : 64,
    "input_channels" : 1,
    "output_channels": 1,
    "dropout_rate" : 0.05,
    'learning_rate' :  0.0001,
    "weight_decay" :  0.01,
    "epochs": 2,
    "batch_size": 8,
    "samples" : 250}


    loss_arg = input(f'a) Sinkhorn \nb) BCE/MSE \n')

    # why you do not set the other hyper parameters this why idk..
    hyperparameters['loss'] = loss_arg

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    start_t = time.time()

    unet = model_pipeline(hyperparameters)

    print('Saving model...')

    if hyperparameters['loss'] == 'a':
        PATH = 'unet_sinkhorn.pth'

    elif hyperparameters['loss'] == 'b':
        PATH = 'unet.pth'

    torch.save(unet.state_dict(), PATH)

    end_t = time.time()
    minutes = (end_t - start_t)/60
    print(f'Done. Runtime: {minutes:.3f} minutes')
