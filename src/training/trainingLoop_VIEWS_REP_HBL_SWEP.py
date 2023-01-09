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

sys.path.insert(0, "/home/projects/ku_00017/people/simpol/scripts/conflictNet/src/networks")
sys.path.insert(0, "/home/projects/ku_00017/people/simpol/scripts/conflictNet/src/configs")
sys.path.insert(0, "/home/projects/ku_00017/people/simpol/scripts/conflictNet/src/utils")

#from trainingLoopUtils import *
# from testLoopUtils import *
from recurrentUnet import UNet
from swep_config import *
from utils import *

# def get_data():

#     """Function to load the volumes of conflict history. Currently load two volumes. 
#     One which is based of the views 2019 replication data (views_monthly_REP_vol.pkl).
#     And one based direct of UCDP and PRIO data, covering the whole globe (views_world_monthly_vol.pkl).
#     The volumes should be changes to be based on data from viewser."""

#     # Data
#     print('loading data....')
#     location = '/home/projects/ku_00017/data/raw/conflictNet' # data dir in computerome.

#     # The views replication data only covering africa
#     file_name = "/views_monthly_REP_vol.pkl"

#     pkl_file = open(location + file_name, 'rb')
#     views_vol = pickle.load(pkl_file)
#     pkl_file.close()


#     # Even for the views sub suet you want to train on the whole world.
#     file_name2 = "/views_world_monthly_vol.pkl" 

#     pkl_file2 = open(location + file_name2, 'rb')
#     world_vol = pickle.load(pkl_file2)
#     pkl_file2.close()

#     return(views_vol, world_vol)


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
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = (0.9, 0.999))

    return(unet, criterion, optimizer) #, dataloaders, dataset_sizes)



# ----------------------------------------------------------------------------------------------------------------------------------



def train(model, optimizer, criterion_reg, criterion_class, train_tensor, meta_tensor_dict, device, unet, sample, plot = False):
    
    wandb.watch(unet, [criterion_reg, criterion_class], log= None, log_freq=2048)

    avg_loss_reg_list = []
    avg_loss_class_list = []
    avg_loss_list = []

    #pred_list = []
    #observed_list = []

    model.train()  # train mode
    
    seq_len = train_tensor.shape[1] 
    window_dim = train_tensor.shape[2]
    
    # initialize a hidden state
#    h = unet.init_h(hidden_channels = model.base, dim = window_dim).float().to(device)
    h = unet.init_h(hidden_channels = model.base, dim = window_dim, train_tensor = train_tensor).float().to(device)

    #for i in range(seq_len-1): # so your sequnce is the full time len - last month.
    for i in range(seq_len-1): # so your sequnce is the full time len - last month.
        print(f'\t\t month: {i+1}/{seq_len}...', end='\r')
     

        # AGIAN YOU DO PUT THE INPUT TENSOR TO DEVICE HERE SO YOU MIGHT NOT NEED TO DO THE WHOLE VOL BEFORE!!!!!!!!! 
        # ACTUALLY I DO NOT THINK YOU DO HERE!!! IT IS ONLY FOR TESTING...... STOP THAT
        t0 = train_tensor[:, i, :, :].reshape(1, 1 , window_dim , window_dim).to(device)  # this is the real x and y
        t1 = train_tensor[:, i+1, :, :].reshape(1, 1 , window_dim, window_dim).to(device)

        # is this the right dime?

        t1_binary = (t1.clone().detach().requires_grad_(True) > 0) * 1.0 # 1.0 to ensure float. Should avoid cloning warning now.
        
        # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True),rather than torch.tensor(sourceTensor).

        # forward
        t1_pred, t1_pred_class, h = model(t0, h.detach())

        # debug
        # print(t1_pred_class.dtype)
        # print(t1_binary.dtype)

        # backwards    
        optimizer.zero_grad()


        # SHOULD THIS BE criterion_reg(t1_pred, t1) !!!!!?
        # loss_reg = criterion_reg(t1, t1_pred)  # forward-pass 
        loss_reg = criterion_reg(t1_pred, t1)  # forward-pass. # correct and working!!!

        # NEEDS DEBUGGING
#        loss_class = criterion_class(t1_binary, t1_pred_class)  # forward-pass
        loss_class = criterion_class(t1_pred_class, t1_binary)  # forward-pass # correct and working!!!
        # ---------------------------------------------------

        loss = loss_reg + loss_class # naive no weights und so weider


        loss.backward()  # backward-pass
        optimizer.step()  # update weights

        avg_loss_reg_list.append(loss_reg.detach().cpu().numpy().item())
        avg_loss_class_list.append(loss_class.detach().cpu().numpy().item())
        avg_loss_list.append(loss.detach().cpu().numpy().item())

    train_log(avg_loss_reg_list, avg_loss_class_list, avg_loss_list)


# ----------------------------------------------------------------------------------------------------------------------------------




def training_loop(config, unet, criterion, optimizer, views_vol):


    # add spatail transformer
    transformer = transforms.Compose([transforms.RandomRotation((0,360)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
    avg_losses = []

    criterion_reg, criterion_class = criterion


    print('Training initiated...')

    for sample in range(config.samples):

        print(f'Sample: {sample+1}/{config.samples}', end = '\r')

        #input_tensor = torch.tensor(train_views_vol[:, sub_images_y[i][0]:sub_images_y[i][1], sub_images_x[i][0]:sub_images_x[i][1], 4].reshape(1, seq_len, dim, dim)).float() #Why not do this in funciton?
        train_tensor, meta_tensor_dict = ALT_get_train_tensors(views_vol, config, sample)
        # data augmentation (can be turned of for final experiments)
        train_tensor = transformer(train_tensor) # rotations and flips

        train(unet, optimizer, criterion_reg, criterion_class, train_tensor, meta_tensor_dict, device, unet, sample, plot = False)


    print('training done...')

    # torch.onnx.export(unet, views_vol, "RUnet.onnx")
    # wandb.save("RUnet.onnx")

# def apply_dropout(m):
#     if type(m) == nn.Dropout:
#         m.train()



# these need to stay, but it must be validatiion nad not test as such.
def test(model, test_tensor, device):
    model.eval() # remove to allow dropout to do its thing as a poor mans ensamble. but you need a high dropout..
    model.apply(apply_dropout)

    # wait until you know if this work as usually
    pred_np_list = []
    pred_class_np_list = []
    out_of_sampel = 0

    #!!!!!!!
    h_tt = model.init_hTtime(hidden_channels = model.base, H = 180, W  = 180, test_tensor = test_tensor).float().to(device) # coul auto the...
    seq_len = test_tensor.shape[1] # og nu køre eden bare helt til roden
    print(f'\t\t\t\t sequence length: {seq_len}', end= '\r')


    H = test_tensor.shape[2]
    W = test_tensor.shape[3]

    for i in range(seq_len-1): # need to get hidden state... You are predicting one step ahead so the -1

        # HERE - IF WE GO BEYOUND -36 THEN USE t1_pred AS t0

        if i < seq_len-1-36: # take form the test set
            print(f'\t\t\t\t\t\t\t in sample. month: {i+1}', end= '\r')

            t0 = test_tensor[:, i, :, :].reshape(1, 1 , H , W).to(device)  # YOU ACTUALLY PUT IT TO DEVICE HERE SO YOU CAN JUST NOT DO IT EARLIER FOR THE FULL VOL!!!!!!!!!!!!!!!!!!!!!
            t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

        else: # take the last t1_pred
            print(f'\t\t\t\t\t\t\t Out of sample. month: {i+1}', end= '\r')
            t0 = t1_pred.detach()

            out_of_sampel = 1

        #t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

        #if out_of_sampel == 1:
            t1_pred, t1_pred_class, _ = model(t0, h_tt)
            pred_np_list.append(t1_pred.cpu().detach().numpy().squeeze())
            pred_class_np_list.append(t1_pred_class.cpu().detach().numpy().squeeze())
            t1_pred, t1_pred_class, _ = model(t0, h_tt)

    return pred_np_list, pred_class_np_list



def get_posterior(unet, views_vol, device, n):
    print('Testing initiated...')

    # SIZE NEED TO CHANGE WITH VIEWS
    test_tensor = torch.tensor(views_vol[:, :, : , 5].reshape(1, -1, 180, 180)).float()#  nu 180x180     175, 184 views dim .to(device) #log best is 7 not 4 when you do sinkhorn or just have coords.
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

        y_score = mean_array[i].reshape(-1) # make it 1d  # nu 180x180 
        y_score_prob = mean_class_array[i].reshape(-1) # nu 180x180 
        # do not really know what to do with these yet.
        y_var = std_array[i].reshape(-1)  # nu 180x180  
        y_var_prob = std_class_array[i].reshape(-1)  # nu 180x180 

        y_true = out_of_sample_tensor[:,i].reshape(-1)  # nu 180x180 . dim 0 is time
        y_true_binary = (y_true > 0) * 1


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
    

    wandb.log({"36month_mean_squared_error": np.mean(mse_list)})
    wandb.log({"36month_average_precision_score": np.mean(ap_list)})
    wandb.log({"36month_roc_auc_score": np.mean(auc_list)})
    wandb.log({"36month_brier_score_loss":np.mean(brier_list)})



def model_pipeline(config=None):

    # tell wandb to get started
    with wandb.init(project="RUNET_VIEWS_REP_experiments36_HBL", entity="nornir", config=config): #monthly36 when you get there--

        # NEW ------------------------------------------------------------------
        wandb.define_metric("monthly/out_sample_month")
        wandb.define_metric("monthly/*", step_metric="monthly/out_sample_month")
        # -----------------------------------------------------------------------


        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # get the data
        views_vol, world_vol = get_data()

        # make the model, data, and optimization problem
        unet, criterion, optimizer = make(config)
        #print(unet)

        # RIGHT NOW YOU JUST TRAIN ON AFRICA
        training_loop(config, unet, criterion, optimizer, world_vol) # TRAIN ON WHOLE WORLD

        #training_loop(config, unet, criterion, optimizer, views_vol) # TRAIN ON WHOLE WORLD
        print('Done training')

        # GET POSTERIOR CAN GET THE AFRICA ONE

        get_posterior(unet, views_vol, device, n=config.test_samples) # TEST ON AFRICA (VIEWS SUBSET)
        #end_test(unet, views_vol, config)
        print('Done testing')

        #return(unet)


# def get_swep_config():
#     sweep_config = {
#     'method': 'grid'
#     }

#     metric = {
#         'name': '36month_mean_squared_error',
#         'goal': 'minimize'   
#         }

#     sweep_config['metric'] = metric


#     parameters_dict = {
#         'hidden_channels': {'values': [28, 30]},
#         'min_events': {'values': [20, 22]},
#         'samples': {'values': [450, 550]},
#         "dropout_rate" : {'values' : [0.05, 0.1]},
#         'learning_rate': {'values' : [0.00001, 0.00005]},
#         "weight_decay" : {'values' : [0.1, 0.05]},
#         'input_channels' : {'value' : 1},
#         'output_channels': { 'value' : 1},
#         'loss' : { 'value' : 'b'},
#         'test_samples': { 'value' : 128}       
#         }


#     sweep_config['parameters'] = parameters_dict

#     return sweep_config


if __name__ == "__main__":

    wandb.login()

    sweep_config = get_swep_config()
    sweep_id = wandb.sweep(sweep_config, project="RUNET_VIEWS_REP_experiments36_HBL") # and then you put in the right project name

    # Hyper parameters.
    # hyperparameters = {
    # "hidden_channels" : 4, # 10 is max if you do full timeline in test.. might nee to be smaller for monthly # you like do not have mem for more than 64
    # "input_channels" : 1,
    # "output_channels": 1,
    # "dropout_rate" : 0.05, #0.05
    # 'learning_rate' :  0.00005,
    # "weight_decay" :  0.05,
    # 'betas' : (0.9, 0.999),
    # "epochs": 2, # as it is now, this is samples...
    # "batch_size": 8, # this also you do not ues
    # "samples" : 140,
    # "test_samples": 128, # go 128, but this is tj́sut to see is there is a collaps
    # "min_events": 22}


    #loss_arg = input(f'a) Sinkhorn \nb) BCE/MSE \n')

    # why you do not set the other hyper parameters this why idk..
    #hyperparameters['loss'] = loss_arg

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    start_t = time.time()

    wandb.agent(sweep_id, model_pipeline)

    # unet = model_pipeline(hyperparameters)

    # print('Saving model...')

    # if hyperparameters['loss'] == 'a':
    #     PATH = 'unet_monthly_sinkhorn.pth'

    # elif hyperparameters['loss'] == 'b':
    #     PATH = 'unet_monthly.pth'

    # torch.save(unet.state_dict(), PATH)

    end_t = time.time()
    minutes = (end_t - start_t)/60
    print(f'Done. Runtime: {minutes:.3f} minutes')


