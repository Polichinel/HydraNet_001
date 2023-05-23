import numpy as np
import random
import pickle
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss

import wandb


def get_data(run_type):

    # Data
    location = '/home/projects/ku_00017/data/raw/conflictNet' # data dir in computerome.

    # The viewser data

    if run_type == 'calib':
        file_name = "/viewser_monthly_vol_calib_sbnsos.pkl"

    elif run_type == 'test':
        file_name = "/viewser_monthly_vol_test_sbnsos.pkl"

    print('loading data....')
    pkl_file = open(location + file_name, 'rb')
    views_vol = pickle.load(pkl_file)
    pkl_file.close()

    return(views_vol)


def norm(x, a = 0, b = 1):

    """Return a normalized x in range [a:b]. Default is [0:1]"""
    x_norm = (b-a)*(x - x.min())/(x.max()-x.min())+a
    return(x_norm)


def unit_norm(x, noise = False):

    """Return a normalized x (unit vector)"""
    x_unit_norm = x / torch.linalg.norm(x)

    if noise == True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_unit_norm += torch.randn(len(x_unit_norm), dtype=torch.float, requires_grad=False, device = device) * x_unit_norm.std()

    return(x_unit_norm)


def standard(x, noise = False):

    """Return a standardnized x """
    x_standard = (x - x.mean()) / x.std()

    if noise == True:
        x_unit_norm += np.random.normal(loc = 0, scale = x_standard.std(), size = len(x_standard))

    return(x_standard)

# def my_decay(draw, min_events, max_events):
    
#     k = 0.01
#     adj_min_events = max_events/(1+np.exp( k *(draw - 256)))
#     adj_min_events += min_events
#     adj_min_events = adj_min_events.astype('int')

#     return(adj_min_events)

# def my_decay(draw, min_events, max_events):
    
#     k = 0.01
#     adj_min_events = max_events/(1+np.exp( k *(draw - 256)))
#     adj_min_events += min_events
#     adj_min_events = adj_min_events.astype('int')

#     return(adj_min_events)

# def my_decay(sample, min_events, max_events): # ---------------------------------------------- works!

#     k = 0.01
#     adj_min_events = max_events/(1 + np.exp(k*sample))
#     adj_min_events += min_events
#     adj_min_events = adj_min_events.astype('int')

#     #adj_min_events = max(adj_min_events.astype('int'), min_events)

#     return(adj_min_events)

# def my_decay(sample, min_events, max_events):# ---------------------------------------------- works!


#     k = 0.01
#     adj_min_events = max_events/(1 + np.exp(k*sample))
#     adj_min_events += min_events
#     adj_min_events = adj_min_events.astype('int')

    #adj_min_events = max(adj_min_events.astype('int'), min_events)

    return(adj_min_events)

# def my_decay(sample, min_events, max_events):

#     k = 0.1
#     adj_min_events = (max_events/(1 + np.exp(k*sample))) * 2
#     adj_min_events = max(adj_min_events.astype('int'), min_events)

#     return(adj_min_events)


# def my_decay(sample, min_events, max_events):

#     k = 0.1
#     adj_min_events = ((max_events/(1 + np.exp(k*sample))) + min_events).astype('int')
    
#     return(adj_min_events)



def my_decay(sample, samples, min_events, max_events):

    if min_events == 10:
        coef = 6

    elif min_events == 15:
        coef = 4

    elif min_events == 20:
        coef = 3

    elif min_events == 30:
        coef = 2

    else:
        print('wrong min_events. Must be either 10, 15, 20 or 30. Now set to 10 as default..')
        coef = 6

    k = np.log(coef)/samples #now, with 6, min_events will alway be 10. 4 is 15, 2 is 30 
    adj_min_events =((max_events/(np.exp(k*sample)))).astype('int')
    
    return(adj_min_events)

# def draw_window(views_vol, config, sample): 

#     """Draw/sample a window/patch from the traning tensor.
#     The dimensions of the windows are HxWxD, 
#     where H=D in {16,32,64} and D is the number of months in the training data.
#     The windows are constrained to be sampled from an area with some
#     minimum number of log_best events (min_events)."""


#     # BY NOW THIS IS PRETTY HACKY... SHOULD BE MADE MORE ELEGANT AT SOME POINT..

#     ln_best_sb_idx = 5 # 5 = ln_best_sb 
#     last_feature_idx = ln_best_sb_idx + config.input_channels - 1 # 5 + 3 - 1 = 7 which is os
#     min_events = config.min_events

#     # so you get more dens observations in the beginning..
#     min_events = my_decay(sample, min_events) # ----------------------------------------------------------------------------------------------------------------------------------wrong!!! Sample is not month!!!

#     if sample == 0: # bisically, give me the index of the cells which saw the most violence the 4 first months...  # TEST -----------------------------------------------------------------------------------------------------------wrong!!! Sample is not month!!!
#         views_vol_count = np.count_nonzero(views_vol[:,:,:,0:3], axis = 0).sum(axis=2)
#         min_events_index = np.where(views_vol_count >= views_vol_count.max()) # the observation with most events.

#     else: # TEST --------------------------------------------
#         views_vol_count = np.count_nonzero(views_vol[:,:,:,ln_best_sb_idx:last_feature_idx], axis = 0).sum(axis=2) #for either sb, ns, os
#         min_events_index = np.where(views_vol_count >= min_events) # number of events so >= 1 or > 0 is the same as np.nonzero

#     min_events_row = min_events_index[0]
#     min_events_col = min_events_index[1]

#     # it is index... Not lat long.
#     min_events_indx = [(row, col) for row, col in zip(min_events_row, min_events_col)] 

#     #indx = random.choice(min_events_indx)
#     indx = min_events_indx[np.random.choice(len(min_events_indx))] # dumb but working solution of np.random instead of random
#     dim = np.random.choice([16, 32, 64]) 

#     # if you wnat a random temporal window, it is here.
#     window_dict = {'lat_indx':indx[0], 'long_indx':indx[1], 'dim' : dim} 

#     return(window_dict)


def draw_window(views_vol, config, sample): 

    """Draw/sample a window/patch from the traning tensor.
    The dimensions of the windows are HxWxD, 
    where H=D in {16,32,64} and D is the number of months in the training data.
    The windows are constrained to be sampled from an area with some
    minimum number of log_best events (min_events)."""


    # BY NOW THIS IS PRETTY HACKY... SHOULD BE MADE MORE ELEGANT AT SOME POINT..

    ln_best_sb_idx = 5 # 5 = ln_best_sb 
    last_feature_idx = ln_best_sb_idx + config.input_channels - 1 # 5 + 3 - 1 = 7 which is os
    min_events = config.min_events
    samples = config.samples

    # so you get more dens observations in the beginning..
    #min_events = my_decay(sample, min_events) # ----------------------------------------------------------------------------------------------------------------------------------wrong!!! Sample is not month!!!



    # WITH THE NEW DECAY FUNCTION THIS IF STATEMENT SHOULD NOT MATTER!!!!
    # if sample == 0: # bisically, give me the index of the cells which saw the most violence the 4 first months...  # TEST -----------------------------------------------------------------------------------------------------------wrong!!! Sample is not month!!!
    #     views_vol_count = np.count_nonzero(views_vol[:,:,:,0:3], axis = 0).sum(axis=2)
    #     max_events = views_vol_count.max()
    #     min_events_index = np.where(views_vol_count == max_events) # the observation with most events.

    # else: # TEST --------------------------------------------
    #     views_vol_count = np.count_nonzero(views_vol[:,:,:,ln_best_sb_idx:last_feature_idx], axis = 0).sum(axis=2) #for either sb, ns, os
    #     max_events = views_vol_count.max()
    #     min_events = my_decay(sample, min_events, max_events)
    #     min_events_index = np.where(views_vol_count >= min_events) # number of events so >= 1 or > 0 is the same as np.nonzero

    views_vol_count = np.count_nonzero(views_vol[:,:,:,ln_best_sb_idx:last_feature_idx], axis = 0).sum(axis=2) #for either sb, ns, os
    max_events = views_vol_count.max()
    min_events = my_decay(sample, samples, min_events, max_events)
    min_events_index = np.where(views_vol_count >= min_events) # number of events so >= 1 or > 0 is the same as np.nonzero

    min_events_row = min_events_index[0]
    min_events_col = min_events_index[1]

    # it is index... Not lat long.
    min_events_indx = [(row, col) for row, col in zip(min_events_row, min_events_col)] 

    #indx = random.choice(min_events_indx)
    indx = min_events_indx[np.random.choice(len(min_events_indx))] # dumb but working solution of np.random instead of random


    # more deterministik solution:
    # if sample <= int(config.samples/2):
    #     dim = 32

    # else:
    #     dim = 16

    #dim = 32 # just if this is more consistent..... 

    # if sample <= 4: # a bit more infor in the beginning
    #     dim = 32

    # else:
    #     dim = np.random.choice([16, 32]) 

    #dim = config.dim

    dim = np.random.choice([16, 32]) 

    # if you wnat a random temporal window, it is here.
    window_dict = {'lat_indx':indx[0], 'long_indx':indx[1], 'dim' : dim} 

    return(window_dict)



def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list):

    avg_loss = np.mean(avg_loss_list)
    avg_loss_reg = np.mean(avg_loss_reg_list)
    avg_loss_class = np.mean(avg_loss_class_list)
    # # Where the magic happens
    
    # this can be put inyo one statment!
    # also log maps...
    wandb.log({"avg_loss": avg_loss, "avg_loss_reg": avg_loss_reg, "avg_loss_class": avg_loss_class})

    # wandb.log({"avg_loss": avg_loss})
    # wandb.log({"avg_loss_reg": avg_loss_reg})
    # wandb.log({"avg_loss_class": avg_loss_class})



def get_train_tensors(views_vol, sample, config, device):

    train_views_vol = views_vol[:-config.time_steps] # not tha last 36 months - these ar for test set

    shift = config.seed  # TEST -------------------------------------------- try REMOVE!

    # To handle "edge windows"
    while True:

        np.random.seed(sample + shift)   # TEST -------------------------------------------- ALRIGHT THIS WORKS; BUT FOR WIERD REASONS... I think it simply discurage the sampler from sampling the same... try REMOVE!

        try:
            window_dict = draw_window(views_vol = views_vol, config = config, sample = sample)
            #print(window_dict)

            min_lat_indx = int(window_dict['lat_indx'] - (window_dict['dim']/2)) 
            max_lat_indx = int(window_dict['lat_indx'] + (window_dict['dim']/2))
            min_long_indx = int(window_dict['long_indx'] - (window_dict['dim']/2))
            max_long_indx = int(window_dict['long_indx'] + (window_dict['dim']/2))

            input_window = train_views_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, :]
            assert input_window.shape[1] == window_dict['dim'] and input_window.shape[2] == window_dict['dim']
            break

        except:
            print('Resample edge...', end= '\r') # if you don't like this, simply pad to whol volume from 180x180 to 192x192. But there is a point to a avoide edges that might have wierd artifacts.
            shift += 1   # TEST --------------------------------------------
            continue

    ln_best_sb_idx = 5
    last_feature_idx = ln_best_sb_idx + config.input_channels
    train_tensor = torch.tensor(input_window).float().to(device).unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, ln_best_sb_idx:last_feature_idx, :, :]


    #wandb.log({"index": })

    #print(f'train_tensor: {train_tensor.shape}')  # debug
    return(train_tensor)


def get_test_tensor(views_vol, config, device):

    ln_best_sb_idx = 5
    last_feature_idx = ln_best_sb_idx + config.input_channels
    test_tensor = torch.tensor(views_vol).float().to(device).unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, ln_best_sb_idx:last_feature_idx, :, :]

    #print(f'test_tensor: {test_tensor.shape}') # debug
    return test_tensor


def get_log_dict(i, mean_array, mean_class_array, std_array, std_class_array, out_of_sample_vol, config):

    log_dict = {}
    log_dict["monthly/out_sample_month"] = i

    for j in range(config.output_channels):

        y_score = mean_array[i,j,:,:].reshape(-1) # make it 1d  # nu 180x180 
        y_score_prob = mean_class_array[i,j,:,:].reshape(-1) # nu 180x180 
        
        # do not really know what to do with these yet.
        y_var = std_array[i,j,:,:].reshape(-1)  # nu 180x180  
        y_var_prob = std_class_array[i,j,:,:].reshape(-1)  # nu 180x180 

        y_true = out_of_sample_vol[:,i,j,:,:].reshape(-1)  # nu 180x180 . dim 0 is time
        y_true_binary = (y_true > 0) * 1


        mse = mean_squared_error(y_true, y_score)
        ap = average_precision_score(y_true_binary, y_score_prob)
        auc = roc_auc_score(y_true_binary, y_score_prob)
        brier = brier_score_loss(y_true_binary, y_score_prob)

        log_dict[f"monthly/mean_squared_error{j}"] = mse
        log_dict[f"monthly/average_precision_score{j}"] = ap
        log_dict[f"monthly/roc_auc_score{j}"] = auc
        log_dict[f"monthly/brier_score_loss{j}"] = brier

    return (log_dict)
