    
def get_hp_config():
    
    hyperparameters = {
    "model" : 'BNUNet',
    "clip_grad_norm" : True,
    "scheduler" : 'step',
    "hidden_channels" : 16,
    "min_events" : 20,
    "samples": 300, # 10 just for debug n
    "dropout_rate" : 0.05,
    'learning_rate' :  0.0001,
    "weight_decay" :  0.1,
    'input_channels' : 3,
    "output_channels" : 3,
    "loss_reg": 'b',
    "loss_class": 'b', 
    "test_samples": 128}

    return hyperparameters

