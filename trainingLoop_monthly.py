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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss


import wandb


from trainingLoopUtils import *
# from testLoopUtils import *
from recurrentUnet import *


def get_data():

    # Data
    print('loading data....')
    #location = '/home/simon/Documents/Articles/ConflictNet/data/raw'
    location = '/home/projects/ku_00017/data/raw/conflictNet'
    file_name = "/ucpd_monthly_vol.pkl"
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

    #optimizer = torch.optim.Adam(unet.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = config.betas)

    return(unet, criterion, optimizer) #, dataloaders, dataset_sizes)



def training_loop(config, unet, criterion, optimizer, ucpd_vol):

    #wandb.watch(unet, [criterion_reg, criterion_class], log="all", log_freq=128)

    # add spatail transformer
    transformer = transforms.Compose([transforms.RandomRotation((0,360)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
    avg_losses = []

    criterion_reg, criterion_class = criterion

    #wandb.watch(unet, [criterion_reg, criterion_class], log="all", log_freq=128)


    print('Training initiated...')

    for sample in range(config.samples):

        print(f'{sample+1}/{config.samples}', end = '\r')

        #input_tensor = torch.tensor(train_ucpd_vol[:, sub_images_y[i][0]:sub_images_y[i][1], sub_images_x[i][0]:sub_images_x[i][1], 4].reshape(1, seq_len, dim, dim)).float() #Why not do this in funciton?
        input_tensor, meta_tensor_dict = get_input_tensors(ucpd_vol, config)
        # data augmentation (can be turned of for final experiments)
        input_tensor = transformer(input_tensor) # rotations and flips

        train(unet, optimizer, criterion_reg, criterion_class, input_tensor, meta_tensor_dict, device, unet, sample, plot = False)

        #avg_loss = train(unet, optimizer, criterion_reg, criterion_class, input_tensor, meta_tensor_dict, device, unet, plot = False)
        #avg_losses.append(avg_loss.cpu().detach().numpy())

        
        # if i % 100 == 0: # print steps 100
        #     print(f'{i} {avg_loss:.4f}') # could plot ap instead...

    print('training done...')

    # torch.onnx.export(unet, ucpd_vol, "RUnet.onnx")
    # wandb.save("RUnet.onnx")

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def test(model, input_tensor, device):
    model.eval() # remove to allow dropout to do its thing as a poor mans ensamble. but you need a high dropout..
    model.apply(apply_dropout)
  # but there was also something else that you neede to acount for when doing this..?

# ---------------------------------------------------------------------------------------------------------------------
#input_tensor = input_tensor[:,-48:,:,:] # b, c, h, w # just the last 4 years.
# ---------------------------------------------------------------------------------------------------------------------

    h_tt = model.init_hTtime(hidden_channels = model.base).float().to(device)
    seq_len = input_tensor.shape[1] # og nu køre eden bare helt til roden

    #print(f'seq_len: {seq_len}') #!!!!!!!!!!!!!!!!!!!!!!!!

    H = input_tensor.shape[2]
    W = input_tensor.shape[3] 

    for i in range(seq_len-1): # need to get hidden state... You are predicting one step ahead so the -1

        t0 = input_tensor[:, i, :, :].reshape(1, 1 , H , W).to(device) 
        # t1 = input_tensor[:, i+1, :, :].reshape(1, 1 , H, W).to(device) # you don't use this under test time...

        t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

        # running_ap = average_precision_score((t1.cpu().detach().numpy() > 0) * 1, t1_pred_class.cpu().detach().numpy()) #!!!!!!!!!!!!!!!!!!!!!!!!
        # print(f'ap: {running_ap}') #!!!!!!!!!!!!!!!!!!!!!!!!

  # You only want the last one
    tn_pred_np = t1_pred.cpu().detach().numpy() # so yuo take the final pred..
    tn_pred_class_np = t1_pred_class.cpu().detach().numpy() # so yuo take the final pred..

    return tn_pred_np, tn_pred_class_np


def get_posterior(unet, ucpd_vol, device, n):
    print('Testing initiated...')

  #ttime_tensor = torch.tensor(ucpd_vol[:, :, : , 4].reshape(1, 31, 360, 720)).float().to(device) #Why not do this in funciton?
#   ttime_tensor = torch.tensor(ucpd_vol[:, :, : , 7].reshape(1, 31, 360, 720)).float().to(device) #log best is 7 not 4 when you do sinkhorn or just have coords.
    ttime_tensor = torch.tensor(ucpd_vol[:, :, : , 7].reshape(1, -1, 360, 720)).float().to(device) #log best is 7 not 4 when you do sinkhorn or just have coords.
    # And you reshape to get a batch dim

    pred_list = []
    pred_list_class = []

    for i in range(n):
        t31_pred_np, tn_pred_class_np = test(unet, ttime_tensor, device)
        pred_list.append(t31_pred_np)
        pred_list_class.append(tn_pred_class_np)

        #if i % 10 == 0: # print steps 10
        print(f'{i}/{n}', end = '\r')

    # reg statistics
    t31_pred_np = np.array(pred_list)
    t31_pred_np_mean = t31_pred_np.mean(axis=0)
    t31_pred_np_std = t31_pred_np.std(axis=0)

    # Class statistics - right noe this does not get updated through backprob..
    t31_pred_class_np = np.array(pred_list_class)
    t31_pred_class_np_mean = t31_pred_class_np.mean(axis=0)
    t31_pred_class_np_std = t31_pred_class_np.std(axis=0)

    # Classification results
    y_var = t31_pred_np_std.reshape(360*720)
    y_score = t31_pred_np_mean.reshape(360*720)

    # HERE
    #y_score_prob = torch.sigmoid(torch.tensor(y_score)) # old trick..
    y_score_prob = t31_pred_class_np_mean.reshape(360*720) # way better brier!

    # y_true = ucpd_vol[30,:,:,4].reshape(360*720) # 7 not 4 when you do sinkhorn and have coords 
    y_true = ucpd_vol[-1,:,:,7].reshape(360*720)

    y_true_binary = (y_true > 0) * 1

    #print('Unet')

    #loss = nn.MSELoss()
    #mse = loss(y_true, y_score)

    # mean_se = mse(y_true, y_score) #just a dummy..
    # area_uc = auc(y_score_prob, y_true_binary)

    mean_se = mean_squared_error(y_true, y_score)
    ap = average_precision_score(y_true_binary, y_score_prob)
    area_uc = roc_auc_score(y_true_binary, y_score_prob)
    brier = brier_score_loss(y_true_binary, y_score_prob)

    wandb.log({"mean_squared_error": mean_se})
    wandb.log({"average_precision_score": ap})
    wandb.log({"roc_auc_score": area_uc})
    wandb.log({"brier_score_loss": brier})


  #return pred_list, pred_list_class



# def end_test(unet, ucpd_vol, config):

#     print('Testing initiated...')

#     pred_list, pred_list_class = get_posterior(unet, ucpd_vol, device, n=config.test_samples)

#     # reg statistics
#     t31_pred_np = np.array(pred_list)
#     t31_pred_np_mean = t31_pred_np.mean(axis=0)
#     t31_pred_np_std = t31_pred_np.std(axis=0)

#     # Class statistics - right noe this does not get updated through backprob..
#     t31_pred_class_np = np.array(pred_list_class)
#     t31_pred_class_np_mean = t31_pred_class_np.mean(axis=0)
#     t31_pred_class_np_std = t31_pred_class_np.std(axis=0)

#     # Classification results
#     y_var = t31_pred_np_std.reshape(360*720)
#     y_score = t31_pred_np_mean.reshape(360*720)

#     # HERE
#     #y_score_prob = torch.sigmoid(torch.tensor(y_score)) # old trick..
#     y_score_prob = t31_pred_class_np_mean.reshape(360*720) # way better brier!

#     # y_true = ucpd_vol[30,:,:,4].reshape(360*720) # 7 not 4 when you do sinkhorn and have coords 
#     y_true = ucpd_vol[-1,:,:,7].reshape(360*720)

#     y_true_binary = (y_true > 0) * 1

#     #print('Unet')

#     #loss = nn.MSELoss()
#     #mse = loss(y_true, y_score)

#     # mean_se = mse(y_true, y_score) #just a dummy..
#     # area_uc = auc(y_score_prob, y_true_binary)

#     mean_se = mean_squared_error(y_true, y_score)
#     ap = average_precision_score(y_true_binary, y_score_prob)
#     area_uc = roc_auc_score(y_true_binary, y_score_prob)
#     brier = brier_score_loss(y_true_binary, y_score_prob)

#     wandb.log({"mean_squared_error": mean_se})
#     wandb.log({"average_precision_score": ap})
#     wandb.log({"roc_auc_score": area_uc})
#     wandb.log({"brier_score_loss": brier})



def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="RUNET_monthly_experiments", entity="nornir", config=hyperparameters): 
        
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # get the data
        ucpd_vol = get_data()

        # make the model, data, and optimization problem
        unet, criterion, optimizer = make(config)
        #print(unet)

        training_loop(config, unet, criterion, optimizer, ucpd_vol)
        print('Done training')
        
        get_posterior(unet, ucpd_vol, device, n=config.test_samples)
        #end_test(unet, ucpd_vol, config)
        print('Done testing')

        return(unet)


if __name__ == "__main__":

    wandb.login()

    # Hyper parameters.
    hyperparameters = {
    "hidden_channels" : 10, # 10 is max if you do full timeline in test.. might nee to be smaller for monthly # you like do not have mem for more than 64
    "input_channels" : 1,
    "output_channels": 1,
    "dropout_rate" : 0.05, #0.05
    'learning_rate' :  0.0001,
    "weight_decay" :  0.05,
    'betas' : (0.9, 0.999),
    "epochs": 2, # as it is now, this is samples...
    "batch_size": 8,
    "samples" : 100,
    "test_samples": 128,
    "min_events": 18}


    loss_arg = input(f'a) Sinkhorn \nb) BCE/MSE \n')

    # why you do not set the other hyper parameters this why idk..
    hyperparameters['loss'] = loss_arg

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    start_t = time.time()

    unet = model_pipeline(hyperparameters)

    print('Saving model...')

    if hyperparameters['loss'] == 'a':
        PATH = 'unet_monthly_sinkhorn.pth'

    elif hyperparameters['loss'] == 'b':
        PATH = 'unet_monthly.pth'

    torch.save(unet.state_dict(), PATH)

    end_t = time.time()
    minutes = (end_t - start_t)/60
    print(f'Done. Runtime: {minutes:.3f} minutes')





# -------------------------

