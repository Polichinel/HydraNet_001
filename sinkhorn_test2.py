import numpy as np
import torch
import geomloss # also needs: pip install pykeops
import time
import os 
import pickle
import random


def get_data():
    print('loading data....')
    #location = '/home/simon/Documents/Articles/ConflictNet/data/raw'
    location = '/home/projects/ku_00017/data/raw/conflictNet'
    file_name = "/ucpd_vol.pkl"
    pkl_file = open(location + file_name, 'rb')
    ucpd_vol = pickle.load(pkl_file)
    pkl_file.close()

    return(ucpd_vol)


def draw_window(ucpd_vol, min_events = 10):
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
    dim = 64 # np.random.choice([8, 16, 32, 64]) # 8, 64

    window_dict = {'lat_indx':indx[0], 'long_indx':indx[1], 'dim' : dim}

    return(window_dict)


def get_input_tensors():
  
    ucpd_vol = get_data()

    # ...
    train_ucpd_vol = ucpd_vol[:-1] # all except the last year
    #print(f'train data shape: {train_ucpd_vol.shape}') # debug.

    # The lenght of a whole time lime.
    seq_len = train_ucpd_vol.shape[0]

    # ...
    window_dict = draw_window(ucpd_vol = ucpd_vol, min_events = 5)
    
    min_lat_indx = int(window_dict['lat_indx'] - (window_dict['dim']/2)) 
    max_lat_indx = int(window_dict['lat_indx'] + (window_dict['dim']/2))
    min_long_indx = int(window_dict['long_indx'] - (window_dict['dim']/2))
    max_long_indx = int(window_dict['long_indx'] + (window_dict['dim']/2))

    # It is now 7, not 4, since you keep coords.
#    input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx , 4].reshape(1, seq_len, window_dict['dim'], window_dict['dim'])
    input_window = train_ucpd_vol[ : , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 7].reshape(1, seq_len, window_dict['dim'], window_dict['dim']) 
    
    # 0 since this is constant across years. 1 dim for batch and one dim for time.
    gids = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 0].reshape(1, 1, window_dict['dim'], window_dict['dim'])
    longitudes = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 1].reshape(1, 1, window_dict['dim'], window_dict['dim'])
    latitudes = train_ucpd_vol[0 , min_lat_indx : max_lat_indx , min_long_indx : max_long_indx, 2].reshape(1, 1, window_dict['dim'], window_dict['dim']) 

    gids_tensor = torch.tensor(gids, dtype=torch.int) # must be int.
    longitudes_tensor = torch.tensor(longitudes, dtype=torch.float)
    latitudes_tensor = torch.tensor(latitudes, dtype=torch.float)

    meta_tensor_dict = {'gids' : gids_tensor, 'longitudes' : longitudes_tensor, 'latitudes' : latitudes_tensor }
    input_tensor = torch.tensor(input_window).float()

    return(input_tensor, meta_tensor_dict)


def test_sinkhorn_time():

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    
    device = torch.device(dev) 
    start = time.time()

    #coords0, coords1 = np.random.rand(2, 64*64, 2)
    # weights0, weights1 = np.random.rand(2, M) 

    input_tensor, meta_tensor_dict = get_input_tensors()
    
    gids0 = meta_tensor_dict['gids'].to(device).reshape(-1).detach().clone()
    gids1 = meta_tensor_dict['gids'].to(device).reshape(-1).detach().clone()
    longitudes = meta_tensor_dict['longitudes'].to(device).reshape(-1).detach().clone()
    latitudes= meta_tensor_dict['latitudes'].to(device).reshape(-1).detach().clone()

    coords = torch.column_stack([longitudes, latitudes])

    # just comparing tow years. fixes dim at 64
    t0 = input_tensor[:, 0, :, :].reshape(1, 1 , 64 , 64).to(device).reshape(-1)
    t1 = input_tensor[:, 1, :, :].reshape(1, 1 , 64 , 64).to(device).reshape(-1)

    #weights0, weights1 = np.random.rand(2, 64*64) 

    #t0 = torch.tensor(weights0, dtype=torch.float).to(device)
    #t1 = torch.tensor(weights1, dtype=torch.float).to(device)

    loss = geomloss.SamplesLoss(loss='sinkhorn', p = 1, blur= 0.05, verbose=False)


    #labels0t = torch.tensor(np.arange(0, coords0.shape[0], 1), dtype=torch.int).to(device)
    #labels1t = torch.tensor(np.arange(0, coords1.shape[0], 1), dtype=torch.int).to(device)

    #coords0t = torch.tensor(coords0, dtype=torch.float).to(device)
    #coords1t = torch.tensor(coords1, dtype=torch.float).to(device)

    # weights0t = torch.tensor(weights0, dtype=torch.float).to(device)
    # weights1t = torch.tensor(weights1, dtype=torch.float).to(device)


    print(gids0.shape)
    print(gids1.shape)
    # print(coords0t.shape)
    # print(coords1t.shape)
    print(t0.shape)
    print(t1.shape)

    #sinkhornLoss = loss(labels0t, weights0t, coords0t, labels1t, weights1t, coords1t)
    sinkhornLoss = loss(gids0, t0, coords, gids1, t1, coords)
    #sinkhornLoss = 0

    end = time.time()
    run_time = (end - start)

    print(f'Runtime: {run_time:.1f} sec')
    print(f'Distance: {sinkhornLoss.item():.3f}')

def main():

    #os.environ['CXX'] = 'g++-8' # does not appear to make a difference

    # M = input('Input number of cells (e.g. 259200 for full prio grid):')
    # M = int(M)
    
    test_sinkhorn_time()

#full prio grid is 360×720 = 259200 cells.
#M = 4096 # =64x64 #10000 # 100000 = 57m 0.2 sec on laptop cpu

if __name__ == '__main__':
    main()
