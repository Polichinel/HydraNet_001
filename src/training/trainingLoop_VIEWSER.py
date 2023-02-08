import numpy as np
import random
import pickle
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LinearLR
from torch.optim.lr_scheduler import ChainedScheduler

from torchvision import transforms
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
from BNrecurrentUnet import BNUNet
#from focal import FocalLoss
from focal_class import FocalLossClass
from focal_reg import FocalLossReg
from shrinkage import ShrinkageLoss


from rmsle import RMSLELoss

#from utils import *
from mtloss import *
from utils_sbnsos import *
from swep_config import *
from hyperparameters_config import *


def choose_loss(config):

    if config.loss_reg == 'a':
        criterion_reg = nn.MSELoss().to(device)

    elif config.loss_reg == 'b':
        criterion_reg = FocalLossReg(gamma=5).to(device)

    elif config.loss_reg == 'c':
        criterion_reg = ShrinkageLoss(a=10, c=0.2).to(device)

    else:
        print('Wrong reg loss...')
        sys.exit()


    if config.loss_class == 'a':
        criterion_class = nn.BCELoss().to(device)

    elif config.loss_class == 'b':
        criterion_class =  FocalLossClass(gamma=2, alpha = 1).to(device)

    else:
        print('Wrong class loss...')
        sys.exit()


    # if config.loss == 'a': #3 not currently implemented
    #     PATH = 'unet_sinkhorn.pth'
    #     criterion_reg = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)
    #     criterion_class = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)


    print(f'Regression loss: {criterion_reg}\n classification loss: {criterion_class}')

    is_regression = torch.Tensor([True, True, True, False, False, False])
    multitaskloss_instance = MultiTaskLoss(is_regression, reduction = 'sum') # also try mean

    return(criterion_reg, criterion_class, multitaskloss_instance)


def make(config):

    # unet = UNet(config.input_channels, config.hidden_channels, config.output_channels, config.dropout_rate).to(device)

    # ------------------------------------------------------------------------------------------------------DEBUG
    if config.model == 'UNet':
        unet = UNet(config.input_channels, config.hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'GUNet_v01':
        unet = GUNet_v01(config.input_channels, config.hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'GUNet_v02':
        unet = GUNet_v02(config.input_channels, config.hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'GUNet_v03':
        unet = GUNet_v03(config.input_channels, config.hidden_channels, config.output_channels, config.dropout_rate).to(device)

    elif config.model == 'BNUNet':
        unet = BNUNet(config.input_channels, config.hidden_channels, config.output_channels, config.dropout_rate).to(device)

    else:
        print('no model...')
    # ------------------------------------------------------------------------------------------------------DEBUG

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

    else:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = (0.9, 0.999))
        scheduler = [] # coul set to None...

    return(unet, criterion, optimizer, scheduler) #, dataloaders, dataset_sizes)


def train(model, optimizer, scheduler, criterion_reg, criterion_class, multitaskloss_instance, train_tensor, config, device):

    wandb.watch(model, [criterion_reg, criterion_class], log= None, log_freq=2048)

    avg_loss_reg_list = []
    avg_loss_class_list = []
    avg_loss_list = []

    model.train()  # train mode
    multitaskloss_instance.train() # meybe another place...

    seq_len = train_tensor.shape[1]
    window_dim = train_tensor.shape[-1] # the last dim should always be a spatial dim (H or W)

    # initialize a hidden state
    h = model.init_h(hidden_channels = model.base, dim = window_dim, train_tensor = train_tensor).float().to(device)

    for i in range(seq_len-1): # so your sequnce is the full time len - last month.
        print(f'\t\t month: {i+1}/{seq_len}...', end='\r')

        t0 = train_tensor[:, i, :, :, :]

        t1 = train_tensor[:, i+1, :, :, :]
        t1_binary = (t1.clone().detach().requires_grad_(True) > 0) * 1.0 # 1.0 to ensure float. Should avoid cloning warning now.

        # forward
        t1_pred, t1_pred_class, h = model(t0, h.detach())

        #print(t1_pred.shape) # debug


        optimizer.zero_grad()

        # forward-pass

        losses_list = []

        for i in range(config.output_channels):

            losses_list.append(criterion_reg(t1_pred[:,i,:,:], t1[:,i,:,:])) #  works


        for i in range(config.output_channels):

            losses_list.append(criterion_class(t1_pred_class[:,i,:,:], t1_binary[:,i,:,:]))


        losses = torch.stack(losses_list)
        loss = multitaskloss_instance(losses)

        # backward-pass
        loss.backward()

        if config.clip_grad_norm == True:
        #     nn.utils.clip_grad_norm_(model.parameters(), 1)  # you cen try this also... --------------------------------------------------------------------------------------
        #    nn.utils.clip_grad_value_(model.parameters(), 0.1)
            for p in model.parameters():
                p.grad.data.clamp_(max=1)

        # else:
        #     pass

        optimizer.step()  # update weights

        if type(scheduler) != list: # becaus you use an empty list
            scheduler.step(loss)

        else:
            pass

        # ------------------------------------------------------------------------------------------------------DEBUG
        loss_reg = losses[:config.output_channels].sum()
        loss_class = losses[-config.output_channels:].sum()
        # ------------------------------------------------------------------------------------------------------DEBUG

        avg_loss_reg_list.append(loss_reg.detach().cpu().numpy().item())
        avg_loss_class_list.append(loss_class.detach().cpu().numpy().item())
        avg_loss_list.append(loss.detach().cpu().numpy().item())

    train_log(avg_loss_reg_list, avg_loss_class_list, avg_loss_list)


def training_loop(config, model, criterion, optimizer, scheduler, views_vol):

    # add spatail transformer
    transformer = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])

    avg_losses = []

    criterion_reg, criterion_class, multitaskloss_instance = criterion

    print('Training initiated...')

    for sample in range(config.samples):

        print(f'Sample: {sample+1}/{config.samples}', end = '\r')

        train_tensor = get_train_tensors(views_vol, sample, config, device)


        # -------------------------------------------------------------------
        # Could be in get train tnesor

        # Should really be N x C x D x H x W. Rigth now you do N x D x C x H x W (in your head, but it might bit really relevant)

        N = train_tensor.shape[0] # batch size. Always 1
        C = train_tensor.shape[1] # months
        D = config.input_channels # features
        H = train_tensor.shape[3] # height
        W =  train_tensor.shape[4] # width

        # data augmentation (can be turned of for final experiments)
        train_tensor = train_tensor.reshape(N, C*D, H, W)
        train_tensor = transformer(train_tensor[:,:,:,:])
        train_tensor = train_tensor.reshape(N, C, D, H, W)

        # -------------------------------------------------------------------


        # Should be an assert thing here..

        train(model, optimizer, scheduler , criterion_reg, criterion_class, multitaskloss_instance, train_tensor, config, device)

    print('training done...')



def test(model, test_tensor, time_steps, config, device): # should be called eval/validation
    model.eval() # remove to allow dropout to do its thing as a poor mans ensamble. but you need a high dropout..
    model.apply(apply_dropout)

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
            #t0 = torch.cat([t0, t0, t0], 1) # VERY MUCH A DEBUG HACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #if out_of_sampel == 1:
            t1_pred, t1_pred_class, _ = model(t0, h_tt) # freeze
            #t1_pred, t1_pred_class, h_tt = model(t0, h_tt) # or dont freez !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            pred_np_list.append(t1_pred.cpu().detach().numpy().squeeze())
            pred_class_np_list.append(t1_pred_class.cpu().detach().numpy().squeeze())

    return pred_np_list, pred_class_np_list


def get_posterior(model, views_vol, config, device, n):
    print('Testing initiated...')

    test_tensor = get_test_tensor(views_vol, config, device) # better cal thiis evel tensor
    #print(test_tensor.shape) # RIGHT NOW 1,324,3,180,180)


    # out_of_sample_vol = test_tensor[:,-time_steps:,0,:,:].cpu().numpy() # not really a tensor now.. # 0 is TEMP HACK unitl real dynasim !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    out_of_sample_vol = test_tensor[:,-config.time_steps:,:,:,:].cpu().numpy() # not really a tensor now.. # 0 is TEMP HACK unitl real dynasim !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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


        mse = mean_squared_error(y_true, y_score)  # SHOULD THIS THEN ALSO BE SHRINKAGE OR SOMETHING???
        ap = average_precision_score(y_true_binary, y_score_prob)
        auc = roc_auc_score(y_true_binary, y_score_prob)
        brier = brier_score_loss(y_true_binary, y_score_prob)

        # Works?
        # log_dict = ({"monthly/out_sample_month": i,
        #              "monthly/mean_squared_error": mse,
        #              "monthly/average_precision_score": ap,
        #              "monthly/roc_auc_score": auc,
        #              "monthly/brier_score_loss":brier})

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

        views_vol = get_data(config.run_type)

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

        project = f"RUNET_VIEWSER_{time_steps}_{run_type}_experiments_011_sbnsos" # 4 is without h freeze... See if you have all the outputs now???

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
        #hyperparameters['loss'] = 'b' # change this or implement sinkhorn correctly also in sweeps.
        hyperparameters['time_steps'] = time_steps
        hyperparameters['run_type'] = run_type
        hyperparameters['sweep'] = False

        print(f"using: {hyperparameters['model']}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        start_t = time.time()

        unet = model_pipeline(config = hyperparameters, project = project)

        # print('Saving model...') # this should be an opiton wen not sweeping

        # if hyperparameters['loss'] == 'a':
        #     PATH = 'unet_monthly_sinkhorn.pth'

        # elif hyperparameters['loss'] == 'b':
        #     PATH = 'unet_monthly.pth'

        # torch.save(unet.state_dict(), PATH)

    end_t = time.time()
    minutes = (end_t - start_t)/60
    print(f'Done. Runtime: {minutes:.3f} minutes')


