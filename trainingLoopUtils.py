import numpy as np
import torch
import random
from recurrentUnet import *

def draw_window(ucpd_vol, min_events = 10):
# dim should be in some range and not fixed to 16..
# Make sure you do not go over the edge..

    ucpd_vol_count = np.count_nonzero(ucpd_vol[:,:,:,4], axis = 0)
    min_events_index = np.where(ucpd_vol_count >= min_events) # number of events so >= 1 or >0 is the same as np.nonzero

    min_events_row = min_events_index[0]
    min_events_col = min_events_index[1]

    min_events_coord = [(row, col) for row, col in zip(min_events_row, min_events_col)]
    
    coord = random.choice(min_events_coord)
    #dim = 16 # if truble, start by hard coding this to 16
    dim = np.random.choice([8, 16, 32, 64]) # 8, 64

    window_dict = {'lat':coord[0], 'long':coord[1], 'dim' : dim}

    return(window_dict)


def get_input_tensors(ucpd_vol):
  
    # ...
    train_ucpd_vol = ucpd_vol[:-1] # all except the last year
    #print(f'train data shape: {train_ucpd_vol.shape}') # debug.

    # ...
    seq_len = train_ucpd_vol.shape[0]

    # ...
    window_dict = draw_window(ucpd_vol = ucpd_vol, min_events = 5)
    
    min_lat = int(window_dict['lat'] - (window_dict['dim']/2)) 
    max_lat = int(window_dict['lat'] + (window_dict['dim']/2))
    min_long = int(window_dict['long'] - (window_dict['dim']/2))
    max_long = int(window_dict['long'] + (window_dict['dim']/2))

    input_window = train_ucpd_vol[ : , min_lat : max_lat , min_long : max_long , 4].reshape(1, seq_len, window_dict['dim'], window_dict['dim'])
    input_tensor = torch.tensor(input_window).float()

    return(input_tensor)


def train(model, optimizer, criterion_reg, criterion_class, input_tensor, device, unet, plot = False):
    
    avg_loss = 0

    model.train()  # train mode
    
    seq_len = input_tensor.shape[1] 
    window_dim = input_tensor.shape[2]
    
    h = unet.init_h(hidden_channels = model.base, dim = window_dim).float().to(device)

    for i in range(seq_len-1): # so your sequnce is the full time len
         
        t0 = input_tensor[:, i, :, :].reshape(1, 1 , window_dim , window_dim).to(device)  # this is the real x and y
        t1 = input_tensor[:, i+1, :, :].reshape(1, 1 , window_dim, window_dim).to(device)

        # NEW THING!
        # t1_binary = (input_tensor[:, i+1, :, :] > 0).float().reshape(1, 1 , window_dim, window_dim).to(device)
        t1_binary = torch.tensor((input_tensor[:, i+1, :, :] > 0).float().reshape(1, 1 , window_dim, window_dim), dtype=torch.float32, device=device)


        # forward
        t1_pred, t1_pred_class, h = model(t0, h.detach())
        t1_pred_class = torch.tensor(t1_pred_class, dtype=torch.float32)

        # backwards    
        optimizer.zero_grad()
        loss_reg = criterion_reg(t1, t1_pred)  # forward-pass 
        
        # NEEDS DEBUGGING
        loss_class = criterion_class(t1_binary, t1_pred_class)  # forward-pass 
        loss = loss_reg + loss_class # naive no weights und so weider

        # loss = loss_reg # naive no weights und so weider

        loss.backward()  # backward-pass
        optimizer.step()  # update weights
        
        # Reporting 
        avg_loss += loss / (seq_len-1)

    return(avg_loss)    