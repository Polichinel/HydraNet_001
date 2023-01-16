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
        'hidden_channels': {'value' : 32}, #{'values': [24, 32]},
        'min_events': {'value' : 22},#{'values': [20, 22]},
        'samples': {'value' : 800}, #{'values': [800, 850]},
        "dropout_rate" : {'value' : 0.1}, #{'values' : [0.05, 0.1]},
        'learning_rate': {'values' : [0.00001, 0.00005]},
        "weight_decay" : {'value' : 0.05}, #{'values' : [0.1, 0.05]},
        'input_channels' : {'value' : 1},
        'output_channels': { 'value' : 1},
        'loss' : { 'value' : 'b'},
        'test_samples': { 'value' : 128}       
        }


    sweep_config['parameters'] = parameters_dict

    return sweep_config