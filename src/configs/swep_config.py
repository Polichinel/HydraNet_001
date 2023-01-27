def get_swep_config():
    sweep_config = {
    'method': 'grid'
    }

    metric = {
        'name': '36month_mean_squared_error',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric


    parameters_dict = {
        'hidden_channels': {'values': [16, 24, 32]},
        'min_events': {'values': [10, 15, 20]},
        'samples': {'values': [300, 400, 500, 600, 700, 800]},
        "dropout_rate" : {'values' : [0.05, 0.1]},
        'learning_rate': {'values' : [0.00001, 0.00005]},
        "weight_decay" : {'values' : [0.1, 0.05]},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 3},
        'loss' : { 'value' : 'b'},
        'test_samples': { 'value' : 128}       
        }

    sweep_config['parameters'] = parameters_dict

    return sweep_config