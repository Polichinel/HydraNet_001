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

        print(f'Sample: {sample+1}/{config.samples}', end = '\r')

        #input_tensor = torch.tensor(train_ucpd_vol[:, sub_images_y[i][0]:sub_images_y[i][1], sub_images_x[i][0]:sub_images_x[i][1], 4].reshape(1, seq_len, dim, dim)).float() #Why not do this in funciton?
        train_tensor, meta_tensor_dict = get_train_tensors(ucpd_vol, config, sample)
        # data augmentation (can be turned of for final experiments)
        train_tensor = transformer(train_tensor) # rotations and flips

        train(unet, optimizer, criterion_reg, criterion_class, train_tensor, meta_tensor_dict, device, unet, sample, plot = False)

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

def test(model, test_tensor, device):
    model.eval() # remove to allow dropout to do its thing as a poor mans ensamble. but you need a high dropout..
    model.apply(apply_dropout)

    # wait until you know if this work as usually
    pred_np_list = []
    pred_class_np_list = []
    out_of_sampel = 0


    h_tt = model.init_hTtime(hidden_channels = model.base).float().to(device)
    seq_len = test_tensor.shape[1] # og nu køre eden bare helt til roden
    print(f'\t\t\t\t sequence length: {seq_len}', end= '\r')

    #print(f'seq_len: {seq_len}') #!!!!!!!!!!!!!!!!!!!!!!!!

    H = test_tensor.shape[2]
    W = test_tensor.shape[3]

    for i in range(seq_len-1): # need to get hidden state... You are predicting one step ahead so the -1

        # HERE - IF WE GO BEYOUND -36 THEN USE t1_pred AS t0

        if i < seq_len-1-36: # take form the test set
            print(f'\t\t\t\t\t\t\t in sample. month: {i+1}', end= '\r')

            t0 = test_tensor[:, i, :, :].reshape(1, 1 , H , W).to(device)  # YOU ACTUALLY PUT IT TO DEVICE HERE SO YOU CAN JUST NOT DO IT EARLIER FOR THE FULL VOL!!!!!!!!!!!!!!!!!!!!!
            # t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

            #t1_pred, t1_pred_class, h_tt = model(t0, h_tt)


        else: # take the last t1_pred
            print(f'\t\t\t\t\t\t\t Out of sample. month: {i+1}', end= '\r')
            t0 = t1_pred.detach()

            out_of_sampel = 1
            # t1_pred, t1_pred_class, h_tt = model(t0, h_tt)
            # But teh nyou also need to store results for all 36 months here.
            # You only want the last one
            # tn_pred_np = t1_pred.cpu().detach().numpy() # so yuo take the final pred..
            # tn_pred_class_np = t1_pred_class.cpu().detach().numpy

        t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

        if out_of_sampel == 1:

            pred_np_list.append(t1_pred.cpu().detach().numpy().squeeze())
            pred_class_np_list.append(t1_pred_class.cpu().detach().numpy().squeeze())


        # running_ap = average_precision_score((t1.cpu().detach().numpy() > 0) * 1, t1_pred_class.cpu().detach().numpy()) #!!!!!!!!!!!!!!!!!!!!!!!!
        # print(f'ap: {running_ap}') #!!!!!!!!!!!!!!!!!!!!!!!!

        # THIS NEEDS TO BE WORSE

#   # You only want the last one
#     tn_pred_np = t1_pred.cpu().detach().numpy() # so yuo take the final pred..
#     tn_pred_class_np = t1_pred_class.cpu().detach().numpy() # so yuo take the final pred..

#     return tn_pred_np, tn_pred_class_np
    return pred_np_list, pred_class_np_list



def get_posterior(unet, ucpd_vol, device, n):
    print('Testing initiated...')

    # SIZE NEED TO CHANGE WITH VIEWS
    test_tensor = torch.tensor(ucpd_vol[:, :, : , 7].reshape(1, -1, 360, 720)).float()#.to(device) #log best is 7 not 4 when you do sinkhorn or just have coords.
    print(test_tensor.shape)

    out_of_sample_tensor = test_tensor[:,-36:,:,:]
    print(out_of_sample_tensor.shape)

    posterior_list = []
    posterior_list_class = []

    for i in range(n):
        pred_np_list, pred_class_np_list = test(unet, test_tensor, device)
        posterior_list.append(pred_np_list)
        posterior_list_class.append(pred_class_np_list)

        #if i % 10 == 0: # print steps 10
        print(f'Posterior sample: {i}/{n}', end = '\r')


    mean_array = np.array(posterior_list).mean(axis = 0) # get mean for each month!
    std_array = np.array(posterior_list).std(axis = 0)

    mean_class_array = np.array(posterior_list_class).mean(axis = 0) # get mean for each month!
    std_class_array = np.array(posterior_list_class).std(axis = 0)

    ap_list = []
    mse_list = []
    auc_list = []
    brier_list = []

    for i in range(mean_array.shape[0]): #  0 of mean array is the temporal dim

        y_score = mean_array[i].reshape(-1) # make it 1d  #  360*720
        y_score_prob = mean_class_array[i].reshape(-1) #  360*720

        # do not really know what to do with these yet.
        y_var = std_array[i].reshape(-1)  #  360*720
        y_var_prob = std_class_array[i].reshape(-1)  #  360*720

        y_true = out_of_sample_tensor[:,i].reshape(-1)  #  360*720. dim 0 is time
        y_true_binary = (y_true > 0) * 1

        #print(y_true.shape)
        #print(y_score.shape)

        mse = mean_squared_error(y_true, y_score)
        ap = average_precision_score(y_true_binary, y_score_prob)
        auc = roc_auc_score(y_true_binary, y_score_prob)
        brier = brier_score_loss(y_true_binary, y_score_prob)

        # Works?
        log_dict = ({"monthly/out_sample_month": i,
                     "monthly/mean_squared_error": mse,
                     "monthly/average_precision_score": ap,
                     "monthly/roc_auc_score": auc,
                     "monthly/brier_score_loss":brier})

        wandb.log(log_dict)

        mse_list.append(mse)
        ap_list.append(ap) # add to list.
        auc_list.append(auc)
        brier_list.append(brier)

    # Works
    # wandb.log({"monthly_mean_squared_error": mse_list})
    # wandb.log({"monthly_average_precision_score": ap_list})
    # wandb.log({"monthly_roc_auc_score": auc_list})
    # wandb.log({"monthly_brier_score_loss":brier_list})

    

    wandb.log({"36month_mean_squared_error": np.mean(mse_list)})
    wandb.log({"36month_average_precision_score": np.mean(ap_list)})
    wandb.log({"36month_roc_auc_score": np.mean(auc_list)})
    wandb.log({"36month_brier_score_loss":np.mean(brier_list)})


# -----------------------------------


    # # reg statistics
    # t31_pred_np = np.array(____pred_list)
    # t31_pred_np_mean = t31_pred_np.mean(axis=0)
    # t31_pred_np_std = t31_pred_np.std(axis=0)

    # # Class statistics - right noe this does not get updated through backprob..
    # t31_pred_class_np = np.array(___pred_list_class)
    # t31_pred_class_np_mean = t31_pred_class_np.mean(axis=0)
    # t31_pred_class_np_std = t31_pred_class_np.std(axis=0)

    # # Classification results
    # y_var = t31_pred_np_std.reshape(360*720)
    # y_score = t31_pred_np_mean.reshape(360*720)

    # y_score_prob = t31_pred_class_np_mean.reshape(360*720) # way better brier!

    # y_true = ucpd_vol[-1,:,:,7].reshape(360*720)

    # y_true_binary = (y_true > 0) * 1

    # mean_se = mean_squared_error(y_true, y_score)
    # ap = average_precision_score(y_true_binary, y_score_prob)
    # area_uc = roc_auc_score(y_true_binary, y_score_prob)
    # brier = brier_score_loss(y_true_binary, y_score_prob)

    # wandb.log({"mean_squared_error": mean_se})
    # wandb.log({"average_precision_score": ap})
    # wandb.log({"roc_auc_score": area_uc})
    # wandb.log({"brier_score_loss": brier})


  #return pred_list, pred_list_class


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="RUNET_monthly_experiments36", entity="nornir", config=hyperparameters): #monthly36 when you get there--

        # NEW ------------------------------------------------------------------
        wandb.define_metric("monthly/out_sample_month")
        wandb.define_metric("monthly/*", step_metric="monthly/out_sample_month")
        # -----------------------------------------------------------------------


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
    "batch_size": 8, # this also you do not ues
    "samples" : 150,
    "test_samples": 128, # go 128, but this is tj́sut to see is there is a collaps
    "min_events": 20}


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

