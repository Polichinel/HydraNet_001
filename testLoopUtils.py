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

def norm(x, a = 0, b = 1):

    """Return a normalized x in range [a:b]. Default is [0:1]. Used for coordinates"""
    x_norm = (b-a)*(x - x.min())/(x.max()-x.min())+a
    return(x_norm)

def unit_norm(x, noise = False):

    """Return a normalized x (unit vector). Used for coordinates"""
    x_unit_norm = x / np.linalg.norm(x)

    if noise == True:
        x_unit_norm += np.random.normal(loc = 0, scale = x_unit_norm.std(), size = len(x_unit_norm))

    return(x_unit_norm)

def standard(x, noise = False):

    """Return a standardnized x. Used for coordinates"""
    x_standard = (x - x.mean()) / x.std()

    if noise == True:
        x_unit_norm += np.random.normal(loc = 0, scale = x_standard.std(), size = len(x_standard))

    return(x_standard)

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def test(model, input_tensor, device):
  model.eval() # remove to allow dropout to do its thing as a poor mans ensamble. but you need a high dropout..
  model.apply(apply_dropout)
  # but there was also something else that you neede to acount for when doing this..?

  h_tt = model.init_hTtime(hidden_channels = model.base).float().to(device)
  seq_len = input_tensor.shape[1] 
  H = input_tensor.shape[2]
  W = input_tensor.shape[3] 

  for i in range(seq_len-1): # need to get hidden state...

    t0 = input_tensor[:, i, :, :].reshape(1, 1 , H , W).to(device) 
    t1 = input_tensor[:, i+1, :, :].reshape(1, 1 , H, W).to(device)

    t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

  # You only want the last one
  tn_pred_np = t1_pred.cpu().detach().numpy() # so yuo take the final pred..
  tn_pred_class_np = t1_pred_class.cpu().detach().numpy() # so yuo take the final pred..

  return tn_pred_np, tn_pred_class_np

def get_posterior(model, ucpd_vol, device, n=100):

  #ttime_tensor = torch.tensor(ucpd_vol[:, :, : , 4].reshape(1, 31, 360, 720)).float().to(device) #Why not do this in funciton?
  ttime_tensor = torch.tensor(ucpd_vol[:, :, : , 7].reshape(1, 31, 360, 720)).float().to(device) #7 not 4 when you do sinkhorn
  
  pred_list = []
  pred_list_class = []

  for i in range(n):
    t31_pred_np, tn_pred_class_np = test(model, ttime_tensor, device)
    pred_list.append(t31_pred_np)
    pred_list_class.append(tn_pred_class_np)

    if i % 10 == 0: # print steps 10
        print(f'{i}')

  return pred_list, pred_list_class