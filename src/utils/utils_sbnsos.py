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



def my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio):

    """Return a number of events (y) sampled from a linear decay function. 
    The decay function is defined by the slope_ratio and the number of samples.
    It has a roof at roof_ratio*max_events and a floor at min_events"""

    b = ((-max_events + min_events)/(samples*slope_ratio))
    y = (max_events + b * sample)
    
    y = min(y, max_events*roof_ratio)
    y = max(y, min_events)
    
    return(int(y))


def get_window_index(views_vol, config, sample): 

    """Draw/sample a cell which serves as the ancor for the sampeled window/patch drawn from the traning tensor.
    The dimensions of the windows are HxWxD, 
    where H=D in {16,32,64} and D is the number of months in the training data.
    The windows are constrained to be sampled from an area with some
    minimum number of log_best events (min_events)."""


    # BY NOW THIS IS PRETTY HACKY... SHOULD BE MADE MORE ELEGANT AT SOME POINT..

    ln_best_sb_idx = 5 # 5 = ln_best_sb 
    last_feature_idx = ln_best_sb_idx + config.input_channels - 1 # 5 + 3 - 1 = 7 which is os
    min_events = config.min_events
    samples = config.samples
    slope_ratio = config.slope_ratio
    roof_ratio = config.roof_ratio

    # NEW----------------------------------------------------------------------------------------------------------------------------
    fatcats = np.arange(ln_best_sb_idx, last_feature_idx, 1)
    n_fatcats = len(fatcats)

    fatcat = fatcats[sample % n_fatcats]
    views_vol_count = np.count_nonzero(views_vol[:,:,:,fatcat], axis = 0) #.sum(axis=2) #for either sb, ns, os
    
    # --------------------------------------------------------------------------------------------------------------------------------

    max_events = views_vol_count.max()
    min_events = my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio)
    
    min_events_index = np.where(views_vol_count >= min_events) # number of events so >= 1 or > 0 is the same as np.nonzero

    min_events_row = min_events_index[0]
    min_events_col = min_events_index[1]

    # it is index... Not lat long.
    min_events_indx = [(row, col) for row, col in zip(min_events_row, min_events_col)] 

    #indx = random.choice(min_events_indx) RANDOMENESS!!!!
    indx = min_events_indx[np.random.choice(len(min_events_indx))] # dumb but working solution of np.random instead of random

    # if you want a random temporal window, it is here.
    window_index = {'row_indx':indx[0], 'col_indx':indx[1]} 

    return(window_index)


# def get_window_coords(window_index, config):

#     # you can change this back to random if you want
#     window_dim = config.window_dim

#     # Randomly select the position of lat_indx and long_indx within the sampled tensor
#     min_lat = np.random.randint(0, window_dim)
#     max_lat =  window_dim - min_lat
#     min_long = np.random.randint(0, window_dim)
#     max_long =  window_dim - min_long

#     min_lat_indx = int(window_index['lat_indx'] - min_lat) 
#     max_lat_indx = int(window_index['lat_indx'] + max_lat)
#     min_long_indx = int(window_index['long_indx'] - min_long)
#     max_long_indx = int(window_index['long_indx'] + max_long)

#     window_coords = {'min_lat_indx':min_lat_indx, 'max_lat_indx':max_lat_indx, 'min_long_indx':min_long_indx, 'max_long_indx':max_long_indx, 'dim':window_dim}

#     return(window_coords)


# ----------------------------------------------------------------------------------------------------
def get_window_coords(window_index, config):
    """Return the coordinates of the window around the sampled index. 
    This implementaions ensures that the window does never go out of bounds.
    (Thus no need for sampling until a window is found that does not go out of bounds)."""

    # you can change this back to random if you want
    window_dim = config.window_dim

    # Randomly select a window around the sampled index. np.clip is used to ensure that the window does not go out of bounds
    min_row_indx = np.clip(int(window_index['row_indx'] - np.random.randint(0, window_dim)), 0, 180 - window_dim)
    max_row_indx = min_row_indx + window_dim
    min_col_indx = np.clip(int(window_index['col_indx'] - np.random.randint(0, window_dim)), 0, 180 - window_dim)
    max_col_indx = min_col_indx + window_dim

    # make dict of window coords to return
    window_coords = {
        'min_row_indx':min_row_indx, 
        'max_row_indx':max_row_indx, 
        'min_col_indx':min_col_indx, 
        'max_col_indx':max_col_indx, 
        'dim':window_dim}

    return(window_coords)

# ----------------------------------------------------------------------------------------------------


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list):

    avg_loss = np.mean(avg_loss_list)
    avg_loss_reg = np.mean(avg_loss_reg_list)
    avg_loss_class = np.mean(avg_loss_class_list)
    
    # also log maps...
    wandb.log({"avg_loss": avg_loss, "avg_loss_reg": avg_loss_reg, "avg_loss_class": avg_loss_class})


def get_train_tensors(views_vol, sample, config, device):

    """Uses the get_window_index and get_window_coords functions to sample a window from the training tensor. 
    The window is returned as a tensor of size 1 x config.time_steps x config.input_channels x 180 x 180.
    A few spatial transformations are applied to the tensor at the end."""

    # Not using the last 36 months - these ar for test set
    train_views_vol = views_vol[:-config.time_steps] 

    window_index = get_window_index(views_vol = views_vol, config = config, sample = sample) # you should try and take this out of the loop - so you keep the index but changes the window_coords!!!
    window_coords = get_window_coords(window_index = window_index, config = config)

    input_window = train_views_vol[ : , window_coords['min_row_indx'] : window_coords['max_row_indx'] , window_coords['min_col_indx'] : window_coords['max_col_indx'], :]

    # Keep trying until we get a valid sampled - i.e. not "edge windows"
    # while True:

        # try:
        #     # note that both functions below, get_window_index and get_window_coords induce randomness... 
        #     window_index = get_window_index(views_vol = views_vol, config = config, sample = sample) # you should try and take this out of the loop - so you keep the index but changes the window_coords!!!
        #     window_coords = get_window_coords(window_index = window_index, config = config)

        #     input_window = train_views_vol[ : , window_coords['min_row_indx'] : window_coords['max_row_indx'] , window_coords['min_col_indx'] : window_coords['max_col_indx'], :]
        #     assert input_window.shape[1] == window_coords['dim'] and input_window.shape[2] == window_coords['dim']
        #     break

        # except:
        #     print('Resample edge...', end= '\r') # if you don't like this, simply pad to whol volume from 180x180 to 192x192. But there is a point to a avoide edges that might have wierd artifacts.
        #     continue

    ln_best_sb_idx = 5
    last_feature_idx = ln_best_sb_idx + config.input_channels
    train_tensor = torch.tensor(input_window).float().to(device).unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, ln_best_sb_idx:last_feature_idx, :, :]

    # Reshape
    N = train_tensor.shape[0] # batch size. Always 1 - you do batch a different way here
    C = train_tensor.shape[1] # months
    D = config.input_channels # features
    H = train_tensor.shape[3] # height
    W =  train_tensor.shape[4] # width

    # add spatial transformer
    transformer = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])

    # data augmentation (can be turned of for final experiments)
    train_tensor = train_tensor.reshape(N, C*D, H, W)
    train_tensor = transformer(train_tensor[:,:,:,:])
    train_tensor = train_tensor.reshape(N, C, D, H, W)


    return(train_tensor)


def get_test_tensor(views_vol, config, device):

    """Uses to get the features for the test tensor. The test tensor is of size 1 x config.time_steps x config.input_channels x 180 x 180."""

    ln_best_sb_idx = 5
    last_feature_idx = ln_best_sb_idx + config.input_channels

    # THIS MIGHT BE WHERE TO CHANGE THE TEST SET TO NOT BE NORMALIZED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    test_tensor = torch.tensor(views_vol).float().to(device).unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, ln_best_sb_idx:last_feature_idx, :, :]

    return test_tensor


def get_log_dict(i, mean_array, mean_class_array, std_array, std_class_array, out_of_sample_vol, config):

    """Return a dictionary of metrics for the monthly out-of-sample predictions for W&B."""

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


