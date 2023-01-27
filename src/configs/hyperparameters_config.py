    
def get_hp_config():
    
    hyperparameters = {
    "hidden_channels" : 32,
    "min_events" : 16,
    "samples": 360, # 10 just for debug n
    "dropout_rate" : 0.05,
    'learning_rate' :  0.00005,
    "weight_decay" :  0.1,
    'input_channels' : 3,
    "output_channels" : 1,
    "loss": 'b', 
    "test_samples": 128}

    return hyperparameters

