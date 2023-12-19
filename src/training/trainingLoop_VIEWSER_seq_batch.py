import numpy as np
#import random
import pickle
import time
import sys
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LinearLR, OneCycleLR, CyclicLR
#from torch.optim.lr_scheduler import ChainedScheduler

#from torchvision import transforms
#import geomloss # New loss. also needs: pip install pykeops

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss

import wandb

sys.path.insert(0, "/home/projects/ku_00017/people/simpol/scripts/conflictNet/src/networks")
sys.path.insert(0, "/home/projects/ku_00017/people/simpol/scripts/conflictNet/src/configs")
# sys.path.insert(0, "/home/projects/ku_00017/people/simpol/scripts/conflictNet/src/utils")
sys.path.insert(0, "/home/projects/ku_00017/people/simpol/scripts/conflictNet/src/utils")


#from trainingLoopUtils import *
# from testLoopUtils import *
from recurrentUnet import UNet
from gatedrecurrentUnet_v01 import GUNet_v01
from gatedrecurrentUnet_v02 import GUNet_v02
from gatedrecurrentUnet_v03 import GUNet_v03
from HydraBNrecurrentUnet_01 import HydraBNUNet01
from HydraBNrecurrentUnet_02 import HydraBNUNet02
from HydraBNrecurrentUnet_03 import HydraBNUNet03
from HydraBNrecurrentUnet_04 import HydraBNUNet04
from HydraBNrecurrentUnet_05 import HydraBNUNet05
from HydraBNrecurrentUnet_06 import HydraBNUNet06
from HydraBNrecurrentUnet_06_LSTM import HydraBNUNet06_LSTM
from HydraBNrecurrentUnet_07 import HydraBNUNet07
from HydraBNrecurrentUnet_06_LSTM2 import HydraBNUNet06_LSTM2
from HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4


from BNrecurrentUnet import BNUNet
#from focal import FocalLoss
from focal_class import FocalLossClass
#from focal_reg import FocalLossReg
from balanced_focal_class import BalancedFocalLossClass
from shrinkage import ShrinkageLoss
from stable_balanced_focal_class import stableBalancedFocalLossClass

from shringkage_june import ShrinkageLoss_new
from focal_june import FocalLoss_new
from warmup_decay_lr_scheduler import WarmupDecayLearningRateScheduler


#from rmsle import RMSLELoss

#from utils import *
from mtloss import *
from utils_sbnsos import *
from swep_config import *
from hyperparameters_config import *

def choose_loss(config):

    if config.loss_reg == 'a':
        criterion_reg = nn.MSELoss().to(device)

    elif config.loss_reg == 'b':  # IN USE!!!!!!!!!!!!!!!
        criterion_reg = ShrinkageLoss(a=config.loss_reg_a, c=config.loss_reg_c).to(device)

    elif config.loss_reg == 'c': # should change to this and I might need violence specific a and c....
        criterion_reg = ShrinkageLoss_new(a=config.loss_reg_a, c=config.loss_reg_c, size_average = True).to(device)

    else:
        print('Wrong reg loss...')
        sys.exit()

    if config.loss_class == 'a':
        criterion_class = nn.BCELoss().to(device)

    # elif config.loss_class == 'b':
    #     criterion_class =  FocalLossClass(gamma=config.loss_class_gamma, alpha = 1).to(device)

    elif config.loss_class == 'b':
        criterion_class =  BalancedFocalLossClass(alpha = config.loss_class_alpha, gamma=config.loss_class_gamma).to(device)

    elif config.loss_class == 'c': # works.. but not right and for probs
        criterion_class =  stableBalancedFocalLossClass(alpha = config.loss_class_alpha, gamma=config.loss_class_gamma).to(device)

    elif config.loss_class == 'd': # works and w/ logits. But I might need violence specific gamma and alpha....
        criterion_class =  FocalLoss_new(alpha = config.loss_class_alpha, gamma=config.loss_class_gamma).to(device) # THIS IS IN USE

    else:
        print('Wrong class loss...')
        sys.exit()


    # if config.loss == 'a': #3 not currently implemented
    #     PATH = 'unet_sinkhorn.pth'
    #     criterion_reg = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)
    #     criterion_class = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)


    print(f'Regression loss: {criterion_reg}\n classification loss: {criterion_class}')

    is_regression = torch.Tensor([True, True, True, False, False, False])   # for vea you can just have 1 extre False (classifcation) in the end for the kl... Or should it really be seen as a reg?
    multitaskloss_instance = MultiTaskLoss(is_regression, reduction = 'sum') # also try mean

    return(criterion_reg, criterion_class, multitaskloss_instance)


def make(config):

    # unet = UNet(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

# ------------------------------------------------------------------------------------------------------ COULD BE A FUNCTION IN utils_sbnsos.py
    if config.model == 'UNet':
        unet = UNet(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'GUNet_v01':
        unet = GUNet_v01(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'GUNet_v02':
        unet = GUNet_v02(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'GUNet_v03':
        unet = GUNet_v03(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet01':
        unet = HydraBNUNet01(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet02':
        unet = HydraBNUNet02(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet03':
        unet = HydraBNUNet03(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet04':
        unet = HydraBNUNet04(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet05':
        unet = HydraBNUNet05(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet06':
        unet = HydraBNUNet06(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet06_LSTM':
        unet = HydraBNUNet06_LSTM(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet06_LSTM2':
        unet = HydraBNUNet06_LSTM2(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet06_LSTM4':
        unet = HydraBNUNet06_LSTM4(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'HydraBNUNet07':
        unet = HydraBNUNet07(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'BNUNet':
        unet = BNUNet(config.input_channels, config.total_hidden_channels, config.output_channels, config.dropout_rate).to(device)

    else:
        print('no model...')
    # ------------------------------------------------------------------------------------------------------DEBUG

    # Create a partial function with the initialization function and the config parameter
    init_fn = functools.partial(init_weights, config=config)

    # Apply the initialization function to the model
    unet.apply(init_fn)

    # ------------------------------------------------------------------------------------------------------COULD BE A FUNCTION IN utils_sbnsos.py


    criterion = choose_loss(config) # this is a touple of the reg and the class criteria
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999)) # no weight decay when using scheduler
    #optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = (0.9, 0.999))

    if config.scheduler == 'plateau':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer)

    elif config.scheduler == 'step': # seems to be an DEPRECATION issue
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = StepLR(optimizer, step_size= 60)

    elif config.scheduler == 'linear':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = LinearLR(optimizer)

    elif config.scheduler == 'CosineAnnealingLR1':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.samples, eta_min = 0.00005) # you should try with config.samples * 0.2, 0,33 and 0.5

    elif config.scheduler == 'CosineAnnealingLR02':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.samples * 0.2, eta_min = 0.00005)

    elif config.scheduler == 'CosineAnnealingLR033':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.samples * 0.33, eta_min = 0.00005)

    elif config.scheduler == 'CosineAnnealingLR05':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.samples * 0.5, eta_min = 0.00005)

    elif config.scheduler == 'CosineAnnealingLR004':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.samples * 0.04, eta_min = 0.00005)


    elif config.scheduler == 'OneCycleLR':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = OneCycleLR(optimizer,
                       total_steps=32, 
                       max_lr = config.learning_rate, # Upper learning rate boundaries in the cycle for each parameter group
                       anneal_strategy = 'cos') # Specifies the annealing strategy

    elif config.scheduler == 'CyclicLR':

        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        scheduler = CyclicLR(optimizer,
                       step_size_up=200,
                       base_lr = config.learning_rate * 0.1,
                       max_lr = config.learning_rate, # Upper learning rate boundaries in the cycle for each parameter group
                       mode = 'triangular2') # Specifies the annealing strategy
        
    elif config.scheduler == 'WarmupDecay':
        
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas = (0.9, 0.999))
        d = config.window_dim * config.window_dim * config.input_channels # this is the dimension of the input window
        scheduler = WarmupDecayLearningRateScheduler(optimizer, d = d, warmup_steps = config.warmup_steps)


    else:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = (0.9, 0.999))
        scheduler = [] # could set to None...

    return(unet, criterion, optimizer, scheduler) #, dataloaders, dataset_sizes)



def train(model, optimizer, scheduler, criterion_reg, criterion_class, multitaskloss_instance, views_vol, sample, config, device): # views vol and sample

    wandb.watch(model, [criterion_reg, criterion_class], log= None, log_freq=2048)

    avg_loss_reg_list = []
    avg_loss_class_list = []
    avg_loss_list = []
    total_loss = 0

    model.train()  # train mode
    multitaskloss_instance.train() # meybe another place...


    # Batch loops:# -----------------------------------------------------------------------------------------------------------
    for batch in range(config.batch_size):

        # Getting the train_tensor
        train_tensor = get_train_tensors(views_vol, sample, config, device)
        seq_len = train_tensor.shape[1]
        window_dim = train_tensor.shape[-1] # the last dim should always be a spatial dim (H or W)

        # initialize a hidden state
        h = model.init_h(hidden_channels = model.base, dim = window_dim, train_tensor = train_tensor).float().to(device)

        # Sequens loop rnn style
        for i in range(seq_len-1): # so your sequnce is the full time len - last month.
            print(f'\t\t month: {i+1}/{seq_len}...', end='\r')

            t0 = train_tensor[:, i, :, :, :]

            t1 = train_tensor[:, i+1, :, :, :]
            t1_binary = (t1.clone().detach().requires_grad_(True) > 0) * 1.0 # 1.0 to ensure float. Should avoid cloning warning now.

            # forward-pass
            t1_pred, t1_pred_class, h = model(t0, h.detach())
        
            losses_list = []

            for j in range(t1_pred.shape[1]): # first each reggression loss. Should be 1 channel, as I conccat the reg heads on dim = 1

                losses_list.append(criterion_reg(t1_pred[:,j,:,:], t1[:,j,:,:])) # index 0 is batch dim, 1 is channel dim (here pred), 2 is H dim, 3 is W dim

            for j in range(t1_pred_class.shape[1]): # then each classification loss. Should be 1 channel, as I conccat the class heads on dim = 1

                losses_list.append(criterion_class(t1_pred_class[:,j,:,:], t1_binary[:,j,:,:])) # index 0 is batch dim, 1 is channel dim (here pred), 2 is H dim, 3 is W dim

            losses = torch.stack(losses_list)
            loss = multitaskloss_instance(losses)

            total_loss += loss

            # traning output
            loss_reg = losses[:t1_pred.shape[1]].sum() # sum the reg losses
            loss_class = losses[-t1_pred.shape[1]:].sum() # assuming 

            avg_loss_reg_list.append(loss_reg.detach().cpu().numpy().item())
            avg_loss_class_list.append(loss_class.detach().cpu().numpy().item())
            avg_loss_list.append(loss.detach().cpu().numpy().item())


    # ---------------------------------------------------------------------------------

    # log each sequence/timeline/batch
    train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list) # FIX!!!

    # Backpropagation and optimization - after a full sequence... 
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient Clipping
    if config.clip_grad_norm == True:
        clip_value = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

    else:
        pass

    # optimize
    optimizer.step()

    # Adjust learning rate based on the loss
    scheduler.step()

    # -----------------------------------------------------------------------------------------------------------


def training_loop(config, model, criterion, optimizer, scheduler, views_vol):

    # # add spatail transformer

    criterion_reg, criterion_class, multitaskloss_instance = criterion

    np.random.seed(config.np_seed)
    torch.manual_seed(config.torch_seed)
    print(f'Training initiated...')

    for sample in range(config.samples):

        print(f'Sample: {sample+1}/{config.samples}', end = '\r')

        train(model, optimizer, scheduler , criterion_reg, criterion_class, multitaskloss_instance, views_vol, sample, config, device)

    print('training done...')



def test(model, test_tensor, time_steps, config, device): # should be called eval/validation
    model.eval() # remove to allow dropout to do its thing as a poor mans ensamble. but you need a high dropout..
    model.apply(apply_dropout)

    # Set the STN to evaluation mode (disable it) for inference
    #model.stn.is_training = False #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEWNEWNENWNEWNENW

    # wait until you know if this work as usually
    pred_np_list = []
    pred_class_np_list = []

    h_tt = model.init_hTtime(hidden_channels = model.base, H = 180, W  = 180, test_tensor = test_tensor).float().to(device) # coul auto the...
    seq_len = test_tensor.shape[1] # og nu køre eden bare helt til roden
    print(f'\t\t\t\t sequence length: {seq_len}', end= '\r')


    for i in range(seq_len-1): # need to get hidden state... You are predicting one step ahead so the -1

        if i < seq_len-1-time_steps: # take form the test set

            print(f'\t\t\t\t\t\t\t in sample. month: {i+1}', end= '\r')

            t0 = test_tensor[:, i, :, :, :]
            t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

        else: # take the last t1_pred
            print(f'\t\t\t\t\t\t\t Out of sample. month: {i+1}', end= '\r')
            t0 = t1_pred.detach()


# NEW-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            if config.freeze_h == "hl": # freeze the long term memory
                
                split = int(h_tt.shape[1]/2) # split h_tt into hs_tt and hl_tt and save hl_tt as the forzen cell state/long term memory. Call it hl_frozen. Half of the second dimension which is channels.
                _, hl_frozen = torch.split(h_tt, split, dim=1)
                t1_pred, t1_pred_class, h_tt = model(t0, h_tt) 
                hs, _ = torch.split(h_tt, split, dim=1) # Again split the h_tt into hs_tt and hl_tt. But discard the hl_tt
                h_tt = torch.cat((hs, hl_frozen), dim=1) # Concatenate the frozen cell state/long term memory (hl_frozen) with the new hidden state/short term memory. this is the new h_tt

            elif config.freeze_h == "hs": # freeze the short term memory

                split = int(h_tt.shape[1]/2) 
                hs_frozen, _ = torch.split(h_tt, split, dim=1)
                t1_pred, t1_pred_class, h_tt = model(t0, h_tt)
                _, hl = torch.split(h_tt, split, dim=1)
                h_tt = torch.cat((hs_frozen, hl), dim=1) 

            elif config.freeze_h == "all": # freeze both h_l and h_s

                t1_pred, t1_pred_class, _ = model(t0, h_tt) 


            elif config.freeze_h == "none": # dont freeze
                t1_pred, t1_pred_class, h_tt = model(t0, h_tt) # dont freeze anything.


            elif config.freeze_h == "random": # random pick between what tho freeze of hs1, hs2, hl1, and hl2

                t1_pred, t1_pred_class, h_tt_new = model(t0, h_tt)

                split_four_ways = int(h_tt.shape[1] / 8) # spltting the tensor four ways along dim 1 to get hs1, hs2, hl1, and hl2

                hs_1_frozen, hs_2_frozen, hs_3_frozen, hs_4_frozen, hl_1_frozen, hl_2_frozen, hl_3_frozen, hl_4_frozen = torch.split(h_tt, split_four_ways, dim=1) # split the h_tt from the last step
                hs_1_new, hs_2_new, hs_3_new, hs_4_new, hl_1_new, hl_2_new, hl_3_new, hl_4_new = torch.split(h_tt_new, split_four_ways, dim=1) # split the h_tt from the current step

                pairs = [(hs_1_frozen, hs_1_new), (hs_2_frozen, hs_2_new), (hs_3_frozen, hs_3_new), (hs_4_frozen, hs_4_new), (hl_1_frozen, hl_1_new), (hl_2_frozen, hl_2_new), (hl_3_frozen, hl_3_new), (hl_4_frozen, hl_4_new)] # make pairs of the frozen and new hidden states
                h_tt = torch.cat([pair[0] if torch.rand(1) < 0.5 else pair[1] for pair in pairs], dim=1) # concatenate the frozen and new hidden states. Randomly pick between the frozen and new hidden states for each pair.

            else:
                print('Wrong freez option...')
                sys.exit()

# NEW -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            t1_pred_class = torch.sigmoid(t1_pred_class) # there is no sigmoid in the model (the loss takes logits) so you need to do it here.
            pred_np_list.append(t1_pred.cpu().detach().numpy().squeeze()) # squeeze to remove the batch dim. So this is a list of 3x180x180 arrays
            pred_class_np_list.append(t1_pred_class.cpu().detach().numpy().squeeze()) # squeeze to remove the batch dim. So this is a list of 3x180x180 arrays

    return pred_np_list, pred_class_np_list


def get_posterior(model, views_vol, config, device, n): 
    print('Testing initiated...')

    test_tensor = get_test_tensor(views_vol, config, device) # better cal thiis evel tensor

    out_of_sample_vol = test_tensor[:,-config.time_steps:,:,:,:].cpu().numpy() # From the 

    posterior_list = []
    posterior_list_class = []

    for i in range(n):
        pred_np_list, pred_class_np_list = test(model, test_tensor, config.time_steps, config, device) # --------------------------------------------------------------
        posterior_list.append(pred_np_list)
        posterior_list_class.append(pred_class_np_list)

        #if i % 10 == 0: # print steps 10
        print(f'Posterior sample: {i}/{n}', end = '\r')

    if config.sweep == False: # should ne in config...
        dump_location = '/home/projects/ku_00017/data/generated/conflictNet/'
        posterior_dict = {'posterior_list' : posterior_list, 'posterior_list_class': posterior_list_class, 'out_of_sample_vol' : out_of_sample_vol}
        with open(f'{dump_location}posterior_dict_{config.time_steps}_{config.run_type}.pkl', 'wb') as file:
            pickle.dump(posterior_dict, file)

        print("Posterior pickle dumped!")

    else:
        print('Running sweep. no posterior pickle dumped')


    # YOU ARE MISSING SOMETHING ABOUT FEATURES HERE WHICH IS WHY YOU REPORTED AP ON WandB IS BIASED DOWNWARDS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!RYRYRYRYERYERYR
    # Get mean and std
    mean_array = np.array(posterior_list).mean(axis = 0) # get mean for each month!
    std_array = np.array(posterior_list).std(axis = 0)

    mean_class_array = np.array(posterior_list_class).mean(axis = 0) # get mean for each month!
    std_class_array = np.array(posterior_list_class).std(axis = 0)

    out_sample_month_list = [] # only used for pickle...
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

        y_true = out_of_sample_vol[:,i].reshape(-1)  # nu 180x180 . dim 0 is time
        y_true_binary = (y_true > 0) * 1

        mse = mean_squared_error(y_true, y_score)  
        ap = average_precision_score(y_true_binary, y_score_prob)
        auc = roc_auc_score(y_true_binary, y_score_prob)
        brier = brier_score_loss(y_true_binary, y_score_prob)

        log_dict = get_log_dict(i, mean_array, mean_class_array, std_array, std_class_array, out_of_sample_vol, config)# so at least it gets reported sep.

        wandb.log(log_dict)

        out_sample_month_list.append(i) # only used for pickle...
        mse_list.append(mse)
        ap_list.append(ap) # add to list.
        auc_list.append(auc)
        brier_list.append(brier)

    if config.sweep == False:

    # DUMP
        metric_dict = {'out_sample_month_list' : out_sample_month_list, 'mse_list': mse_list,
                    'ap_list' : ap_list, 'auc_list': auc_list, 'brier_list' : brier_list}

        with open(f'{dump_location}metric_dict_{config.time_steps}_{config.run_type}.pkl', 'wb') as file:
            pickle.dump(metric_dict, file)

        with open(f'{dump_location}test_vol_{config.time_steps}_{config.run_type}.pkl', 'wb') as file: # make it numpy
            pickle.dump(test_tensor.cpu().numpy(), file)

        print('Metric and test pickle dumped!')

    else:
        print('Running sweep. no metric or test pickle dumped')

    # ------------------------------------------------------------------------------------
    wandb.log({f"{config.time_steps}month_mean_squared_error": np.mean(mse_list)})
    wandb.log({f"{config.time_steps}month_average_precision_score": np.mean(ap_list)})
    wandb.log({f"{config.time_steps}month_roc_auc_score": np.mean(auc_list)})
    wandb.log({f"{config.time_steps}month_brier_score_loss":np.mean(brier_list)})

def model_pipeline(config=None, project=None):

    # tell wandb to get started
    with wandb.init(project=project, entity="nornir", config=config): # project and config ignored when runnig a sweep

        wandb.define_metric("monthly/out_sample_month")
        wandb.define_metric("monthly/*", step_metric="monthly/out_sample_month")

        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        views_vol = get_data(config)


        # make the model, data, and optimization problem
        unet, criterion, optimizer, scheduler = make(config)

        training_loop(config, unet, criterion, optimizer, scheduler, views_vol)
        print('Done training')

        get_posterior(unet, views_vol, config, device, n=config.test_samples) # actually since you give config now you do not need: time_steps, run_type, is_sweep,
        print('Done testing')

        if config.sweep == False: # if it is not a sweep
            return(unet)


if __name__ == "__main__":

    wandb.login()

    time_steps_dict = {'a':12,
                       'b':24,
                       'c':36,
                       'd':48,}

    time_steps = time_steps_dict[input('a) 12 months\nb) 24 months\nc) 36 months\nd) 48 months\nNote: 48 is the current VIEWS standard.\n')]


    runtype_dict = {'a' : 'calib', 'b' : 'test'}
    run_type = runtype_dict[input("a) Calibration\nb) Testing\n")]
    print(f'Run type: {run_type}\n')

    do_sweep = input(f'a) Do sweep \nb) Do one run and pickle results \n')

    if do_sweep == 'a':

        print('Doing a sweep!')

        project = f"RUNET_VIEWSER_{time_steps}_{run_type}_experiments_016_sbnsos" # 4 is without h freeze... See if you have all the outputs now???

        sweep_config = get_swep_config()
        sweep_config['parameters']['time_steps'] = {'value' : time_steps}
        sweep_config['parameters']['run_type'] = {'value' : run_type}
        sweep_config['parameters']['sweep'] = {'value' : True}

        sweep_id = wandb.sweep(sweep_config, project=project) # and then you put in the right project name

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        start_t = time.time()
        wandb.agent(sweep_id, model_pipeline)

    elif do_sweep == 'b':

        print(f'One run and pickle!')

        project = f"RUNET_VIEWS_{time_steps}_{run_type}_pickled_sbnsos"

        hyperparameters = get_hp_config()
        hyperparameters['time_steps'] = time_steps
        hyperparameters['run_type'] = run_type
        hyperparameters['sweep'] = False

        print(f"using: {hyperparameters['model']}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        start_t = time.time()

        unet = model_pipeline(config = hyperparameters, project = project)

    end_t = time.time()
    minutes = (end_t - start_t)/60
    print(f'Done. Runtime: {minutes:.3f} minutes')


