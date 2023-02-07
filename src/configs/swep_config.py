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
        'model' : {'value' : 'UNet'},
        'clip_grad_norm' : {'value': True},
        'scheduler' : {'values': ['step', 'linear', None]},
        'hidden_channels': {'value': 16},
        'min_events': {'value': 20},
        'samples': {'value': 800},
        "dropout_rate" : {'value' : 0.05},
        'learning_rate': {'values' : [0.0001, 0.00001]},
        "weight_decay" : {'value' : 0.1},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 3},
        'loss_class' : { 'values' : ['a', 'b']},
        'loss_reg' : { 'values' : ['a', 'b', 'c']},
        'test_samples': { 'value' : 128}       
        }


    # parameters_dict = {
    #     'model' : {'values' : ['BNUNet', 'UNet']},
    #     'clip_grad_norm' : {'value': True},
    #     'scheduler' : {'values': ['step', 'linear', None]},
    #     'hidden_channels': {'values': [16, 32]},
    #     'min_events': {'values': [10, 15, 20]},
    #     'samples': {'values': [500, 600, 700, 800]},
    #     "dropout_rate" : {'value' : 0.05},
    #     'learning_rate': {'values' : [0.0001, 0.00005, 0.00001]},
    #     "weight_decay" : {'value' : 0.1},
    #     'input_channels' : {'value' : 3},
    #     'output_channels': {'value' : 3},
    #     'loss' : { 'values' : ['c', 'b']},
    #     'test_samples': { 'value' : 128}       
    #     }

    # parameters_dict = {
    #     'model' : {'values' : ['UNet', 'BNUNet', 'GUNet_v01', 'GUNet_v02', 'GUNet_v03']},
    #     'clip_grad_norm' : {'value': True},
    #     'scheduler' : {'values': ['plateau', 'step', 'linear', None]},
    #     'hidden_channels': {'values': [16, 24, 32]},
    #     'min_events': {'values': [16, 12, 32]},
    #     'samples': {'values': [300, 400, 500, 600, 700, 800]},
    #     "dropout_rate" : {'values' : [0.2, 0.1, 0.05]},
    #     'learning_rate': {'values' : [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
    #     "weight_decay" : {'value' : 0.1},
    #     'input_channels' : {'value' : 3},
    #     'output_channels': {'value' : 3},
    #     'loss' : { 'value' : 'b'},
    #     'test_samples': { 'value' : 128}       
    #     }

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