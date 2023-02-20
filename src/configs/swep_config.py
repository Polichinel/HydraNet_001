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
        'model' : {'value' : 'BNUNet'},
        'clip_grad_norm' : {'value': True},
        'scheduler' : {'value': 'step'},
        'hidden_channels': {'values': [16, 32]},
        'min_events': {'value': 10},
        'samples': {'value': 800},
        "dropout_rate" : {'value' : 0.05},
        'learning_rate': {'value' : 0.0001},
        "weight_decay" : {'value' : 0.1},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 3},
        'loss_class' : { 'value' :  'b'}, # det nytter jo ikke noget at du køre over gamma og alpha for loss-class a...
        'loss_class_gamma' : { 'values' : [0, 0.5, 1, 2, 5]},
        'loss_class_alpha' : { 'values' : [0.5, 0.75, 0.95]}, # should be between 0.5 and 0.95...
        'loss_reg' : { 'value' : 'b'},
        'loss_reg_a' : { 'values' : [0.5, 1, 1.5]},
        'loss_reg_c' : { 'values' : [0.05, 0.1]},
        'test_samples': { 'value' : 128},
        'h_init' : {'value' : 'zero'}  # right now this is just as a note to self. Can't change it here     
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