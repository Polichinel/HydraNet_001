import numpy as np
import random
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss

from testLoopUtils import *
from recurrentUnet import *

import geomloss

start_t = time.time()


loss_arg = input(f'a) Sinkhorn \nb) BCE/MSE \n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print('loading data....')
#location = '/home/simon/Documents/Articles/ConflictNet/data/raw'
location = '/home/projects/ku_00017/data/raw/conflictNet'
file_name = "/ucpd_vol.pkl"
pkl_file = open(location + file_name, 'rb')
ucpd_vol = pickle.load(pkl_file)
pkl_file.close()


print('loading model....')

hidden_channels = 64
input_channels = 1
output_channels = 1
dropout_rate = 0.5

unet = UNet(input_channels, hidden_channels, output_channels, dropout_rate).to(device)

if loss_arg == 'a':
    PATH = 'unet_sinkhorn.pth'

elif loss_arg == 'b':
    PATH = 'unet.pth'

unet.load_state_dict(torch.load(PATH))
#unet.eval() # you do this some other place right?


pred_list, pred_list_class = get_posterior(unet, ucpd_vol, device, n=100)

# reg statistics
print('reg')
t31_pred_np = np.array(pred_list)

t31_pred_np_mean = t31_pred_np.mean(axis=0)
print(t31_pred_np.min(), t31_pred_np.max())

t31_pred_np_std = t31_pred_np.std(axis=0)
print(t31_pred_np_std.min(), t31_pred_np_std.max())

# Class statistics - right noe this does not get updated through backprob..
print('class')
t31_pred_class_np = np.array(pred_list_class)

t31_pred_class_np_mean = t31_pred_class_np.mean(axis=0)
print(t31_pred_class_np.min(), t31_pred_class_np.max()) # the minimun is very high!

t31_pred_class_np_std = t31_pred_class_np.std(axis=0)
print(t31_pred_class_np_std.min(), t31_pred_class_np_std.max())



# Classification results
y_var = t31_pred_np_std.reshape(360*720)
y_score = t31_pred_np_mean.reshape(360*720)

# HERE
#y_score_prob = torch.sigmoid(torch.tensor(y_score)) # old trick..
y_score_prob = t31_pred_class_np_mean.reshape(360*720) # way better brier!


# y_true = ucpd_vol[30,:,:,4].reshape(360*720) # 7 not 4 when you do sinkhorn and have coords 
y_true = ucpd_vol[30,:,:,7].reshape(360*720)

y_true_binary = (y_true > 0) * 1

print('Unet')

print(mean_squared_error(y_true, y_score))
print(average_precision_score(y_true_binary, y_score_prob))
print(roc_auc_score(y_true_binary, y_score_prob))
print(brier_score_loss(y_true_binary, y_score_prob))

# ------------------------------------------------------------------
criterion_reg = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.9, reach = None, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)
criterion_class = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.9, reach = None, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)

#criterion_reg = geomloss.ImagesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)
#criterion_class = geomloss.ImagesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)


longitudes = ucpd_vol[0 ,  :  ,  : , 1].reshape(-1)
latitudes = ucpd_vol[0 ,  :  ,  : , 2].reshape(-1) 

# norm to between 0 and 1 - does another norm change the result?
#longitudes_norm = torch.tensor((longitudes - longitudes.min())/(longitudes.max()-longitudes.min()), dtype = torch.float).to(device)#.detach()
#latitudes_norm = torch.tensor((latitudes - latitudes.min())/(latitudes.max()-latitudes.min()), dtype = torch.float).to(device)#.detach()

# longitudes_norm = torch.tensor(norm(longitudes, 0 ,1), dtype = torch.float).to(device)#.detach()
# latitudes_norm = torch.tensor(norm(latitudes, 0 ,1), dtype = torch.float).to(device)#.detach()

# longitudes_norm = torch.tensor(unit_norm(longitudes), dtype = torch.float).to(device)#.detach()
# latitudes_norm = torch.tensor(unit_norm(latitudes), dtype = torch.float).to(device)#.detach()

#longitudes_norm = torch.tensor(standard(longitudes), dtype = torch.float).to(device)#.detach()
#latitudes_norm = torch.tensor(standard(latitudes), dtype = torch.float).to(device)#.detach()

longitudes_norm = torch.tensor(unit_norm(longitudes, noise= True), dtype = torch.float).to(device)#.detach()
latitudes_norm = torch.tensor(unit_norm(latitudes, noise= True), dtype = torch.float).to(device)#.detach()


# NxD
coords = torch.column_stack([longitudes_norm, latitudes_norm])

# weights
y_true_t = torch.tensor(y_true, dtype = torch.float).to(device) 
y_score_t = torch.tensor(y_score, dtype = torch.float).to(device) 
y_true_binary_t = torch.tensor(y_true_binary, dtype = torch.float).to(device) 
y_score_prob_t = torch.tensor(y_score_prob, dtype = torch.float).to(device)

sinkhorn_reg = criterion_reg(y_true_t, coords, y_score_t, coords)
sinkhorn_class = criterion_class(y_true_binary_t, coords, y_score_prob_t, coords)

# softmax to get prob dens TEST!
#softmax = torch.nn.Softmax(dim = 0)

#sinkhorn_reg = criterion_reg(softmax(y_true_t), coords, softmax(y_score_t), coords)
#sinkhorn_class = criterion_class(softmax(y_true_binary_t), coords, softmax(y_score_prob_t), coords)
# -----------------------------------------------------------------------

print(np.sqrt(sinkhorn_reg.item()))
print(np.sqrt(sinkhorn_class.item()))

#print(sinkhorn_reg.item())
#print(sinkhorn_class.item())

end_t = time.time()
minutes = (end_t - start_t)/60

print(f'Done. Runtime: {minutes:.3f} minutes')


#TODO:
# Ppersitance model should go back in and yo should output in a .txt fil.