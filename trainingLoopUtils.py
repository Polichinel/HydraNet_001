import numpy as np
import torch
import random
#import geomloss
import wandb

from recurrentUnet import *

def norm(x, a = 0, b = 1):

    """Return a normalized x in range [a:b]. Default is [0:1]"""
    x_norm = (b-a)*(x - x.min())/(x.max()-x.min())+a
    return(x_norm)

def unit_norm(x, noise = False):

    """Return a normalized x (unit vector)"""
    x_unit_norm = x / torch.linalg.norm(x)

    if noise == True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #x_unit_norm += torch.tensor(np.random.normal(1, 2*x_unit_norm.std(), len(x_unit_norm)), dtype = torch.float).to(device)
        x_unit_norm += torch.randn(len(x_unit_norm), dtype=torch.float, requires_grad=False, device = device) * x_unit_norm.std()

    return(x_unit_norm)

def standard(x, noise = False):

    """Return a standardnized x """
    x_standard = (x - x.mean()) / x.std()

    if noise == True:
        x_unit_norm += np.random.normal(loc = 0, scale = x_standard.std(), size = len(x_standard))

    return(x_standard)

def draw_window(ucpd_vol, min_events):
# dim should be in some range and not fixed to 16..
# Make sure you do not go over the edge..

    #ucpd_vol_count = np.count_nonzero(ucpd_vol[:,:,:,4], axis = 0) # with coordinates in vol, log best is 7
    ucpd_vol_count = np.count_nonzero(ucpd_vol[:,:,:,7], axis = 0) # with coordinates in vol, log best is 7

    min_events_index = np.where(ucpd_vol_count >= min_events) # number of events so >= 1 or > 0 is the same as np.nonzero

    min_events_row = min_events_index[0]
    min_events_col = min_events_index[1]

    min_events_indx = [(row, col) for row, col in zip(min_events_row, min_events_col)] # is is index... Not lat long.
    
    indx = random.choice(min_events_indx)
    #dim = 16 # if truble, start by hard coding this to 16
    dim = np.random.choice([8, 16, 32, 64]) #

    temporal_dim = np.random.choice([6, 12, 24, 48, 96, 192]) 

    window_dict = {'lat_indx':indx[0], 'long_indx':indx[1], 'dim' : dim, 'temporal_dim': temporal_dim} # if you wnat a random temporal window, it is here.

    return(window_dict)


def get_input_tensors(ucpd_vol, config):
  
    # ...
    train_ucpd_vol = ucpd_vol[:-1] # all except the last year
    #print(f'train data shape: {train_ucpd_vol.shape}') # debug.

    # The lenght of a whole time lime.
    seq_len = train_ucpd_vol.shape[0]

    # ...
    window_dict = draw_window(ucpd_vol = ucpd_vol, min_events = config.min_events)
    
    min_lat_indx = int(window_dict['lat_indx'] - (window_dict['dim']/2)) 
    max_lat_indx = int(window_dict['lat_indx'] + (window_dict['dim']/2))
    min_long_indx = int(window_dict['long_indx'] - (window_dict['dim']/2))
    max_long_indx = int(window_dict['long_indx'] + (window_dict['dim']/2))


    # if you want a fixe temporal window, this is here.
    # It is now 7, not 4, since you keep coords.
#    input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx , 4].reshape(1, seq_len, window_dict['dim'], window_dict['dim'])
#    input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 7].reshape(1, seq_len, window_dict['dim'], window_dict['dim']) 
    
    # id it is monthly
    if train_ucpd_vol[:,:,:,:].shape[0] > 32:

        input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 7].reshape(1, seq_len, window_dict['dim'], window_dict['dim'])

        # --------------------------------------------
        # min_temp_idx = 0
        # max_temp_idx = train_ucpd_vol[:,:,:,:].shape[0] - window_dict['temporal_dim']

        # # dump hack to make sure there is somthing in here
        
        # input_window = torch.zeros(train_ucpd_vol[:,:,:,:].shape[0], dtype=torch.int32, requires_grad=False)
        
        # while input_window.sum() < 1:

        #     start_temp = np.random.choice(np.arange(min_temp_idx, max_temp_idx+1, 1))
        #     end_temp = start_temp + window_dict['temporal_dim']
        #     input_window = train_ucpd_vol[start_temp: end_temp , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 7].reshape(1, -1, window_dict['dim'], window_dict['dim'])
        # # -------------------------------------------------


    else: # it must be yearly
        input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 7].reshape(1, seq_len, window_dict['dim'], window_dict['dim'])

    # 0 since this is constant across years. 1 dim for batch and one dim for time.
    gids = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 0].reshape(1, 1, window_dict['dim'], window_dict['dim'])
    longitudes = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 1].reshape(1, 1, window_dict['dim'], window_dict['dim'])
    latitudes = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 2].reshape(1, 1, window_dict['dim'], window_dict['dim']) 

    gids_tensor = torch.tensor(gids, dtype=torch.int) # must be int. You don't use it any more.
    longitudes_tensor = torch.tensor(longitudes, dtype=torch.float)
    latitudes_tensor = torch.tensor(latitudes, dtype=torch.float)

    meta_tensor_dict = {'gids' : gids_tensor, 'longitudes' : longitudes_tensor, 'latitudes' : latitudes_tensor }
    input_tensor = torch.tensor(input_window).float()

    return(input_tensor, meta_tensor_dict)


def train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list):

    avg_loss = np.mean(avg_loss_list)
    avg_loss_reg = np.mean(avg_loss_reg_list)
    avg_loss_class = np.mean(avg_loss_class_list)
    # # Where the magic happens
    
    # wandb.log({"epoch": sample, "avg_loss": avg_loss}, step= sequence_step)
    # wandb.log({"epoch": sample, "avg_loss_reg": avg_loss_reg}, step = sequence_step)
    # wandb.log({"epoch": sample, "avg_loss_class": avg_loss_class}, step = sequence_step)

    wandb.log({"avg_loss": avg_loss})
    wandb.log({"avg_loss_reg": avg_loss_reg})
    wandb.log({"avg_loss_class": avg_loss_class})


def train(model, optimizer, criterion_reg, criterion_class, input_tensor, meta_tensor_dict, device, unet, sample, plot = False):

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    #wandb.watch(model, [criterion_reg, criterion_class], log="all", log_freq=128)
    
    wandb.watch(unet, [criterion_reg, criterion_class], log= None, log_freq=2048)# 128 need to change this for monthly!!!!!!!

    # TRY THIS TO TEST:
    train_tensor = input_tensor[:, :-1, :, :]
    test_tensor  = input_tensor[:, -1:, :, :] # just to see the shape
    print(f'shape input tensor: {input_tensor.shape}')
    print(f'shape train tensor: {train_tensor.shape}')
    print(f'shape test tensor: {test_tensor.shape}')

    # avg_loss_reg = 0
    # avg_loss_class = 0
    # avg_loss = 0

    avg_loss_reg_list = []
    avg_loss_class_list = []
    avg_loss_list = []

    #pred_list = []
    #observed_list = []

    model.train()  # train mode
    
    seq_len = train_tensor.shape[1] 
    window_dim = train_tensor.shape[2]
    
    # initialize a hidden state
    h = unet.init_h(hidden_channels = model.base, dim = window_dim).float().to(device)

    #for i in range(seq_len-1): # so your sequnce is the full time len - last month.
    for i in range(seq_len): # so your sequnce is the full time len - last month.
     

        # AGIAN YOU DO PUT THE INPUT TENSOR TO DEVICE HERE SO YOU MIGHT NOT NEED TO DO THE WHOLE VOL BEFORE!!!!!!!!! 
        # ACTUALLY I DO NOT THINK YOU DO HERE!!! IT IS ONLY FOR TESTING...... STOP THAT
        t0 = train_tensor[:, i, :, :].reshape(1, 1 , window_dim , window_dim).to(device)  # this is the real x and y
        t1 = train_tensor[:, i+1, :, :].reshape(1, 1 , window_dim, window_dim).to(device)

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

# for debuf and testing..
        # print(f'reg: {loss_reg}')
        # print(f'class: {loss_class}')

        # loss = loss_reg # naive no weights und so weider

        loss.backward()  # backward-pass
        optimizer.step()  # update weights
        
        # Reporting 
        # avg_loss += loss / (seq_len-1)
        # avg_loss_reg += loss_reg / (seq_len-1)
        # avg_loss_class += loss_class / (seq_len-1)

        avg_loss_reg_list.append(loss_reg.detach().cpu().numpy().item())
        avg_loss_class_list.append(loss_class.detach().cpu().numpy().item())
        avg_loss_list.append(loss.detach().cpu().numpy().item())

    train_log(avg_loss_reg_list, avg_loss_class_list, avg_loss_list)

        #pred_list.append(t1_pred)
        #observed_list.append(t1)

        # wandb.log({"avg_loss": avg_loss})
        # wandb.log({"avg_loss_reg": avg_loss_reg})
        # wandb.log({"avg_loss_class": avg_loss_class})

    #return(pred_list, observed_list)    