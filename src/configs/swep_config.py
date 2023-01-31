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
        'model' : {'values' : 'GUNet'},
        'clip_grad_norm' : {'value': True},
        'hidden_channels': {'value': 32},
        'min_events': {'value': 16},
        'samples': {'value': 300},
        "dropout_rate" : {'value' : 0.05},
        'learning_rate': {'values' : [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
        "weight_decay" : {'value' : 0.1},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 3},
        'loss' : { 'value' : 'b'},
        'test_samples': { 'value' : 128}       
        }

    # parameters_dict = {
    #     'model' : {'values' : ['UNet', 'GUNet']},
    #     'clip_grad_norm' : {'values': [True, False]},
    #     'hidden_channels': {'values': [16, 24, 32]},
    #     'min_events': {'values': [10, 15, 20]},
    #     'samples': {'values': [300, 400, 500, 600, 700, 800]},
    #     "dropout_rate" : {'values' : [0.05, 0.1]},
    #     'learning_rate': {'values' : [0.00001, 0.00005]},
    #     "weight_decay" : {'values' : [0.1, 0.05]},
    #     'input_channels' : {'value' : 3},
    #     'output_channels': {'value' : 3},
    #     'loss' : { 'value' : 'b'},
    #     'test_samples': { 'value' : 128}       
    #     }

    sweep_config['parameters'] = parameters_dict

    return sweep_config