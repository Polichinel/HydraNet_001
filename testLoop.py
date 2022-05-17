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


end_t = time.time()
minutes = (end_t - start_t)/60

print(f'Done. Runtime: {minutes:.3f} minutes')


#TODO:
# Ppersitance model should go back in and yo should output in a .txt fil.