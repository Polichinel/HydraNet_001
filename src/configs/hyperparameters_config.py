    
def get_hp_config():
    
    hyperparameters = {
    'model' : 'HydraBNUNet04', #'BNUNet',
    'clip_grad_norm' : True,
    'scheduler' : 'linear',
    'hidden_channels' : 32,
    'min_events' : 5,
    'samples': 512, # 10 just for debug
    'batch_size': 6, 
    'dropout_rate' : 0.05,
    'learning_rate' :  0.001,
    'weight_decay' :  0.1,
    'slope_ratio' : 0.75,
    'roof_ratio' : 0.7,
    'input_channels' : 3,
    'output_channels' : 3,
    'loss_class': 'c',  # band c are is still unstable... 
    'loss_class_gamma' : 0, # 2 works and 0 works. , # try with 1 here and 0.75 below # # gamma=0 no wieght. Between 0 and 1 seems very unstable... In general, this here above 0 seems unstable... At least the the reg values below 
    'loss_class_alpha' : 0.5, # try with 0.75 # alpha=0.5 even.
    'loss_reg': 'b',
    'loss_reg_a' : 14, 
    'loss_reg_c' :  0.01, # 0.05 works...
    'test_samples': 128,
    'np_seed' : 4,
    'torch_seed' : 4,
    'window_dim' : 32,
    'h_init' : 'abs_rand_exp-100' # right now this is just as a note to self. Can't change it here}
    }

    return hyperparameters

