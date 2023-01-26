import numpy as np
import random
import pickle
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import wandb


def get_data(run_type):

    """Function to load the volumes of conflict history. Currently load two volumes. 
    One which is based of the views 2019 replication data (views_monthly_REP_vol.pkl).
    And one based direct of UCDP and PRIO data, covering the whole globe (views_world_monthly_vol.pkl).
    The volumes should be changes to be based on data from viewser."""

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


def draw_window(ucpd_vol, min_events, sample):

    """Draw/sample a window/patch from the traning tensor.
    The dimensions of the windows are HxWxD, 
    where H=D in {16,32,64} and D is the number of months in the training data.
    The windows are constrained to be sampled from an area with some
    minimum number of log_best events (min_events)."""

    # with coordinates in vol, log best was 7 #CHANGED FOR VIEWSER
#    ucpd_vol_count = np.count_nonzero(ucpd_vol[:,:,:,5], axis = 0) #CHANGED FOR VIEWSER
    ucpd_vol_count = np.count_nonzero(ucpd_vol[:,:,:,5:8], axis = 0).sum(axis=2) #for either sb, ns, os


    # number of events so >= 1 or > 0 is the same as np.nonzero
    min_events_index = np.where(ucpd_vol_count >= min_events) 

    min_events_row = min_events_index[0]
    min_events_col = min_events_index[1]

    # it is index... Not lat long.
    min_events_indx = [(row, col) for row, col in zip(min_events_row, min_events_col)] 
    
    indx = random.choice(min_events_indx)

    dim = np.random.choice([16, 32, 64]) 

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
    
    wandb.log({"avg_loss": avg_loss})
    wandb.log({"avg_loss_reg": avg_loss_reg})
    wandb.log({"avg_loss_class": avg_loss_class})



def get_train_tensors(ucpd_vol, config, sample):

    train_ucpd_vol = ucpd_vol[:-config.time_steps] # not tha last 36 months - these ar for test set

    # The lenght of a whole time lime.
    seq_len = train_ucpd_vol.shape[0]

    # To handle "edge windows"
    while True:
        try:
            window_dict = draw_window(ucpd_vol = ucpd_vol, min_events = config.min_events, sample= sample)
            #window_dict = draw_window(vol_t2, 22, sample)

            min_lat_indx = int(window_dict['lat_indx'] - (window_dict['dim']/2)) 
            max_lat_indx = int(window_dict['lat_indx'] + (window_dict['dim']/2))
            min_long_indx = int(window_dict['long_indx'] - (window_dict['dim']/2))
            max_long_indx = int(window_dict['long_indx'] + (window_dict['dim']/2))

            input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 5:8].reshape(1, seq_len, window_dict['dim'], window_dict['dim'], config.input_channels)
            #input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 5:8].reshape(1, seq_len, window_dict['dim'], window_dict['dim'], 3)
            break

        except:
            print('Resample edge', end= '\r') # if you don't like this, simply pad to whol volume from 180x180 to 192x192. But there is a point to a avoide edges that might have wierd artifacts.
            continue

    # sinkhorn stuff...        
    # 0 since this is constant across years. 1 dim for batch and one dim for time.
    # gids = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 0].reshape(1, 1, window_dict['dim'], window_dict['dim'])
    # longitudes = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 1].reshape(1, 1, window_dict['dim'], window_dict['dim'])
    # latitudes = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 2].reshape(1, 1, window_dict['dim'], window_dict['dim']) 

    # gids_tensor = torch.tensor(gids, dtype=torch.int) # must be int. You don't use it any more.
    # longitudes_tensor = torch.tensor(longitudes, dtype=torch.float)
    # latitudes_tensor = torch.tensor(latitudes, dtype=torch.float)

    #meta_tensor_dict = {'gids' : gids_tensor, 'longitudes' : longitudes_tensor, 'latitudes' : latitudes_tensor }
    
    meta_tensor_dict = {} # right now just a place holder.. Delete if you end up not using it.
    train_tensor = torch.tensor(input_window).float()

    return(train_tensor, meta_tensor_dict)



# OLD

# def get_train_tensors(ucpd_vol, config, sample):
  
#     # This need to change to be validation, but it depends on the viewser setup
#     # train_ucpd_vol = ucpd_vol[:-36] # not tha last 36 months - these ar for test set
#     train_ucpd_vol = ucpd_vol[:-config.time_steps] # not tha last 36 months - these ar for test set


#     # The lenght of a whole time lime.
#     seq_len = train_ucpd_vol.shape[0]

#     # To handle "edge windows"
#     while True:
#         try:
#             window_dict = draw_window(ucpd_vol = ucpd_vol, min_events = config.min_events, sample= sample)
            
#             min_lat_indx = int(window_dict['lat_indx'] - (window_dict['dim']/2)) 
#             max_lat_indx = int(window_dict['lat_indx'] + (window_dict['dim']/2))
#             min_long_indx = int(window_dict['long_indx'] - (window_dict['dim']/2))
#             max_long_indx = int(window_dict['long_indx'] + (window_dict['dim']/2))

#             #HBL = np.random.randint(7,10,1).item()#CHANGED FOR VIEWSER
#             HBL = 5 # right now just the log_best #CHANGED FOR VIEWSER

#             input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, HBL].reshape(1, seq_len, window_dict['dim'], window_dict['dim'])
#             break

#         except:
#             print('Resample edge', end= '\r') # if you don't like this, simply pad to whol volume from 180x180 to 192x192. But there is a point to a avoide edges that might have wierd artifacts.
#             continue

#     # 0 since this is constant across years. 1 dim for batch and one dim for time.
#     gids = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 0].reshape(1, 1, window_dict['dim'], window_dict['dim'])
#     longitudes = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 1].reshape(1, 1, window_dict['dim'], window_dict['dim'])
#     latitudes = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 2].reshape(1, 1, window_dict['dim'], window_dict['dim']) 

#     gids_tensor = torch.tensor(gids, dtype=torch.int) # must be int. You don't use it any more.
#     longitudes_tensor = torch.tensor(longitudes, dtype=torch.float)
#     latitudes_tensor = torch.tensor(latitudes, dtype=torch.float)

#     meta_tensor_dict = {'gids' : gids_tensor, 'longitudes' : longitudes_tensor, 'latitudes' : latitudes_tensor }
#     train_tensor = torch.tensor(input_window).float()

#     return(train_tensor, meta_tensor_dict)