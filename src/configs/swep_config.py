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
        'model' : {'value' :'HydraBNUNet03'},
        'clip_grad_norm' : {'value': True},
        'scheduler' : {'value': 'linear'}, # 'OneCycleLR'
        'hidden_channels': {'value': 32}, # you like need 32, it seems from qualitative results
        'min_events': {'value': 5},
        'samples': {'values':   [512, 777]}, # should be a function of batches becaus batch 3 and sample 1000 = 3000....
        'batch_size': {'values':  [3, 6, 9]}, # just speed running here..
        "dropout_rate" : {'value' : 0.05},
        'learning_rate': {'values' : [0.005, 0.001]},
        "weight_decay" : {'value' : 0.1},
        "slope_ratio" : {'value' : 0.75},
        "roof_ratio" : {'value' :  0.7},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 3},
        'loss_class' : { 'value' : 'c'}, # det nytter jo ikke noget at du køre over gamma og alpha for loss-class a...
        'loss_class_gamma' : {'value' : 2},
        'loss_class_alpha' : {'value' : 0.5}, # should be between 0.5 and 0.95...
        'loss_reg' : { 'value' : 'b', },
        'loss_reg_a' : { 'value' : 14},
        'loss_reg_c' : { 'value' : 0.01},
        'test_samples': { 'value' : 128},
        #'start_months' :{'values' : [1,2,4,6,8,12]},
        'np_seed' : {'values' : [3, 4]},
        'torch_seed' : {'values' : [3, 4]},
        #'dim' : {'values' : [32,16]},
        'h_init' : {'value' : 'abs_rand_exp-100'}  # right now this is just as a note to self. Can't change it here     
        }

    sweep_config['parameters'] = parameters_dict

    return sweep_config