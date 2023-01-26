import numpy as np
import random
import pickle
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
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
#from utils import *
from utils_sbnsos import *
from swep_config import *
from hyperparameters_config import *


def choose_loss(config):

    if config.loss == 'a':
        PATH = 'unet_sinkhorn.pth'
        criterion_reg = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)
        criterion_class = geomloss.SamplesLoss(loss='sinkhorn', scaling = 0.5, reach = 64, backend = 'multiscale', p = 2, blur= 0.05, verbose=False).to(device)

        # But scaling does a big difference so woth trying 0.3-0.7
        # set higer reach: ex 64
        # set highet scaling = 0.9
        # Scaling 0.1 worse, scaking 0.9 worse
        # try p = 1
        # Needs to set reach: "[...] if reach is None (balanced Optimal Transport), the resulting routine will expect measures whose total masses are equal with each other."
        # Needs to set backend explicitly: online or multiscale

    elif config.loss == 'b':
        PATH = 'unet.pth'
        criterion_reg = nn.MSELoss().to(device) # works
        criterion_class = nn.BCELoss().to(device) # works

    else:
        print('Wrong loss...')
        sys.exit()

    return(criterion_reg, criterion_class)


def make(config):

    unet = UNet(config.input_channels, config.hidden_channels, config.output_channels, config.dropout_rate).to(device)

    criterion = choose_loss(config) # this is a touple of the reg and the class criteria

    #optimizer = torch.optim.Adam(unet.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = (0.9, 0.999))

    return(unet, criterion, optimizer) #, dataloaders, dataset_sizes)



# ----------------------------------------------------------------------------------------------------------------------------------



def train(model, optimizer, criterion_reg, criterion_class, train_tensor, meta_tensor_dict, config, device, sample, plot = False):
    
    wandb.watch(model, [criterion_reg, criterion_class], log= None, log_freq=2048)

    avg_loss_reg_list = []
    avg_loss_class_list = []
    avg_loss_list = []

    #pred_list = []
    #observed_list = []

    model.train()  # train mode
    
    seq_len = train_tensor.shape[1] 
    window_dim = train_tensor.shape[2]
    
    # initialize a hidden state
#    h = unet.init_h(hidden_channels = model.base, dim = window_dim).float().to(device)
    h = model.init_h(hidden_channels = model.base, dim = window_dim, train_tensor = train_tensor).float().to(device)

    #for i in range(seq_len-1): # so your sequnce is the full time len - last month.
    for i in range(seq_len-1): # so your sequnce is the full time len - last month.
        print(f'\t\t month: {i+1}/{seq_len}...', end='\r')
     

        # AGIAN YOU DO PUT THE INPUT TENSOR TO DEVICE HERE SO YOU MIGHT NOT NEED TO DO THE WHOLE VOL BEFORE!!!!!!!!! 
        # ACTUALLY I DO NOT THINK YOU DO HERE!!! IT IS ONLY FOR TESTING...... STOP THAT
        # t0 = train_tensor[:, i, :, :].reshape(1, 1 , window_dim , window_dim).to(device)  # this is the real x and y
        # t1 = train_tensor[:, i+1, :, :].reshape(1, 1 , window_dim, window_dim).to(device)


        # NOT SURE IF YOU GET STUF RIGHT HERE!!! Meybe use permute first?
        t0 = train_tensor[:, i, :, :, :].reshape(1, config.input_channels, window_dim , window_dim).to(device)  # So three channels feauture ''''''''''''''''''''''''''''''''''''''''
        t1 = train_tensor[:, i+1, :, :, 0].reshape(1, 1 , window_dim, window_dim).to(device) # but one channel (sb) taget. For now. '''''''''''''''''''''''''''''''''''''''''''''''''''''


        # is this the right dime?

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


        loss.backward()  # backward-pass
        optimizer.step()  # update weights

        avg_loss_reg_list.append(loss_reg.detach().cpu().numpy().item())
        avg_loss_class_list.append(loss_class.detach().cpu().numpy().item())
        avg_loss_list.append(loss.detach().cpu().numpy().item())

    train_log(avg_loss_reg_list, avg_loss_class_list, avg_loss_list)


# ----------------------------------------------------------------------------------------------------------------------------------


def training_loop(config, model, criterion, optimizer, views_vol):


    # add spatail transformer
    transformer = transforms.Compose([transforms.RandomRotation((0,360)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
    avg_losses = []

    criterion_reg, criterion_class = criterion


    print('Training initiated...')

    for sample in range(config.samples):

        print(f'Sample: {sample+1}/{config.samples}', end = '\r')

        #input_tensor = torch.tensor(train_views_vol[:, sub_images_y[i][0]:sub_images_y[i][1], sub_images_x[i][0]:sub_images_x[i][1], 4].reshape(1, seq_len, dim, dim)).float() #Why not do this in funciton?
        train_tensor, meta_tensor_dict = get_train_tensors(views_vol, config, sample)
        # data augmentation (can be turned of for final experiments)
        
        
        #train_tensor = train_tensor.permute(0,1,4,2,3) # just for debugging
        #train_tensor = transformer(train_tensor[:,:,:,:,0]) # rotations and flips # skip for now... '''''''''''''''''''''''''''''''''''''''''''''''''''''' bug only take 4 dims.. could just squezze the batrhc dom and then give it again afterwards?
        #train_tensor = train_tensor.permute(0,1,2,4,3) # just for debugging


        train(model, optimizer, criterion_reg, criterion_class, train_tensor, meta_tensor_dict, config, device, sample, plot = False)


    print('training done...')


# these need to stay, but it must be validatiion nad not test as such.
def test(model, test_tensor, time_steps, config, device):
    model.eval() # remove to allow dropout to do its thing as a poor mans ensamble. but you need a high dropout..
    model.apply(apply_dropout)

    # wait until you know if this work as usually
    pred_np_list = []
    pred_class_np_list = []
    out_of_sampel = 0

    #!!!!!!!
    h_tt = model.init_hTtime(hidden_channels = model.base, H = 180, W  = 180, test_tensor = test_tensor).float().to(device) # coul auto the...
    seq_len = test_tensor.shape[1] # og nu køre eden bare helt til roden
    print(f'\t\t\t\t sequence length: {seq_len}', end= '\r')


    H = test_tensor.shape[2]
    W = test_tensor.shape[3]

    for i in range(seq_len-1): # need to get hidden state... You are predicting one step ahead so the -1

        # HERE - IF WE GO BEYOUND -36 (or time steps) THEN USE t1_pred AS t0

        # if i < seq_len-1-36: # take form the test set
        if i < seq_len-1-time_steps: # take form the test set

            print(f'\t\t\t\t\t\t\t in sample. month: {i+1}', end= '\r')

            t0 = test_tensor[:, i, :, :, :].reshape(1,  config.input_channels , H , W).to(device)  # YOU ACTUALLY PUT IT TO DEVICE HERE SO YOU CAN JUST NOT DO IT EARLIER FOR THE FULL VOL!!!!!!!!!!!!!!!!!!!!!
            t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

        else: # take the last t1_pred
            print(f'\t\t\t\t\t\t\t Out of sample. month: {i+1}', end= '\r')
            t0 = t1_pred.detach()

            t0 = torch.cat([t0, t0, t0], 1) # VERY MUCH A DEBUG HACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # issue -> this needs to be three now also....

            out_of_sampel = 1

        #t1_pred, t1_pred_class, h_tt = model(t0, h_tt)

        #if out_of_sampel == 1:
            t1_pred, t1_pred_class, _ = model(t0, h_tt)
            pred_np_list.append(t1_pred.cpu().detach().numpy().squeeze())
            pred_class_np_list.append(t1_pred_class.cpu().detach().numpy().squeeze())
            t1_pred, t1_pred_class, _ = model(t0, h_tt)

    return pred_np_list, pred_class_np_list


def get_posterior(model, views_vol, time_steps, run_type, is_sweep, config, device, n):
    print('Testing initiated...')

    # SIZE NEED TO CHANGE WITH VIEWS
    test_tensor = torch.tensor(views_vol[:, :, : , 5:8].reshape(1, -1, 180, 180, config.input_channels)).float()#  nu 180x180     175, 184 views dim .to(device) #log best is 7 not 4 when you do sinkhorn or just have coords.
    print(test_tensor.shape)

    # out_of_sample_tensor = test_tensor[:,-36:,:,:]
    out_of_sample_tensor = test_tensor[:,-time_steps:,:,:]

    print(out_of_sample_tensor.shape)

    posterior_list = []
    posterior_list_class = []

    for i in range(n):
        pred_np_list, pred_class_np_list = test(model, test_tensor, time_steps, config, device) # --------------------------------------------------------------
        posterior_list.append(pred_np_list)
        posterior_list_class.append(pred_class_np_list)

        #if i % 10 == 0: # print steps 10
        print(f'Posterior sample: {i}/{n}', end = '\r')

    if is_sweep == False:
        dump_location = '/home/projects/ku_00017/data/generated/conflictNet/' 
        posterior_dict = {'posterior_list' : posterior_list, 'posterior_list_class': posterior_list_class, 'out_of_sample_tensor' : out_of_sample_tensor}
        with open(f'{dump_location}posterior_dict_{time_steps}_{run_type}.pkl', 'wb') as file: 
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

        y_true = out_of_sample_tensor[:,i].reshape(-1)  # nu 180x180 . dim 0 is time
        y_true_binary = (y_true > 0) * 1


        mse = mean_squared_error(y_true, y_score)
        ap = average_precision_score(y_true_binary, y_score_prob)
        auc = roc_auc_score(y_true_binary, y_score_prob)
        brier = brier_score_loss(y_true_binary, y_score_prob)

        # Works?
        log_dict = ({"monthly/out_sample_month": i,
                     "monthly/mean_squared_error": mse,
                     "monthly/average_precision_score": ap,
                     "monthly/roc_auc_score": auc,
                     "monthly/brier_score_loss":brier})

        wandb.log(log_dict)

        out_sample_month_list.append(i) # only used for pickle...
        mse_list.append(mse)
        ap_list.append(ap) # add to list.
        auc_list.append(auc)
        brier_list.append(brier)
    
    if is_sweep == False: 
    
    # DUMP 
        metric_dict = {'out_sample_month_list' : out_sample_month_list, 'mse_list': mse_list, 
                    'ap_list' : ap_list, 'auc_list': auc_list, 'brier_list' : brier_list}

        with open(f'{dump_location}metric_dict_{time_steps}_{run_type}.pkl', 'wb') as file:
            pickle.dump(metric_dict, file)

        with open(f'{dump_location}test_tensor_{time_steps}_{run_type}.pkl', 'wb') as file: # make it numpy
            pickle.dump(test_tensor, file)

        print('Metric and test pickle dumped!')

    else:
        print('Running sweep. no metric or test pickle dumped')

    # ------------------------------------------------------------------------------------
    wandb.log({f"{time_steps}month_mean_squared_error": np.mean(mse_list)})
    wandb.log({f"{time_steps}month_average_precision_score": np.mean(ap_list)})
    wandb.log({f"{time_steps}month_roc_auc_score": np.mean(auc_list)})
    wandb.log({f"{time_steps}month_brier_score_loss":np.mean(brier_list)})

def model_pipeline(config=None, project=None):

    # # This is a proxy for wheter it is a sweep
    # if config == None:
    #     # project = f"RUNET_VIEWSER_{run_type}_experiments_001" # this gets ignorede if you do the sweep anyway
    #     is_sweep = True
    
    # # Or not a sweep.
    # else:
    #     is_sweep = False

    # tell wandb to get started
    with wandb.init(project=project, entity="nornir", config=config): # project and config ignored when runnig a sweep

        wandb.define_metric("monthly/out_sample_month")
        wandb.define_metric("monthly/*", step_metric="monthly/out_sample_month")
                
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        time_steps = config['time_steps']
        run_type = config['run_type']
        is_sweep = config['sweep']

        views_vol = get_data(run_type)

        # make the model, data, and optimization problem
        unet, criterion, optimizer = make(config)

        training_loop(config, unet, criterion, optimizer, views_vol) 
        print('Done training')

        get_posterior(unet, views_vol, time_steps, run_type, is_sweep, config, device, n=config.test_samples) # actually since you give config now you do not need: time_steps, run_type, is_sweep,
        print('Done testing')

        if is_sweep == False: # if it is not a sweep
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

        project = f"RUNET_VIEWSER_{time_steps}_{run_type}_experiments_001_sbnsos"

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

        print('One run and pickle!')

        project = f"RUNET_VIEWS_{time_steps}_{run_type}_pickeled_sbnsos"

        hyperparameters = get_hp_config()
        hyperparameters['loss'] = 'b' # change this or implement sinkhorn correctly also in sweeps.
        hyperparameters['time_steps'] = time_steps
        hyperparameters['run_type'] = run_type
        hyperparameters['sweep'] = False

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


