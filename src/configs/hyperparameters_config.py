    
def get_hp_config():
    
    hyperparameters = {
    "model" : 'BNUNet',
    "clip_grad_norm" : True,
    "scheduler" : 'step',
    "hidden_channels" : 32,
    "min_events" : 10,
    "samples": 1200, # 10 just for debug n
    "dropout_rate" : 0.05,
    'learning_rate' :  0.0001,
    "weight_decay" :  0.1,
    'input_channels' : 3,
    "output_channels" : 3,
    "loss_class": 'b',
    'loss_class_gamma' : 2, # try with 1 here and 0.75 below
    'loss_class_alpha' : 0.5,
    "loss_reg": 'b',
    'loss_reg_a' : 5,
    'loss_reg_c' :  0.1,
    "test_samples": 128}

    return hyperparameters

