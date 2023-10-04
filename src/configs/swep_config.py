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
        'model' : {'value' :'HydraBNUNet06'},
        'weight_init' : {'value' : 'xavier_norm'}, # ['xavier_uni', 'xavier_norm', 'kaiming_uni', 'kaiming_normal']
        'clip_grad_norm' : {'value': True},
        'scheduler' : {'values': ['CosineAnnealingLR', 'linear']}, # 'OneCycleLR'
        'hidden_channels': {'value': 32}, # you like need 32, it seems from qualitative results
        'min_events': {'value': 5},
        'samples': {'values':   [300, 400, 500]}, # should be a function of batches becaus batch 3 and sample 1000 = 3000....
        'batch_size': {'value':  3}, # just speed running here..
        "dropout_rate" : {'values' : [0.125, 0.25]},
        'learning_rate': {'values' :  [0.001, 0.0005]}, #0.001 default, but 0.005 might be better
        "weight_decay" : {'value' : 0.1},
        "slope_ratio" : {'value' : 0.75},
        "roof_ratio" : {'value' :  0.7},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 3},
        'loss_class' : { 'value' : 'd'}, # det nytter jo ikke noget at du køre over gamma og alpha for loss-class a...
        'loss_class_gamma' : {'value' : 1.5},
        'loss_class_alpha' : {'value' : 0.75}, # should be between 0.5 and 0.95...
        'loss_reg' : { 'value' : 'b', },
        'loss_reg_a' : { 'value' : 14},
        'loss_reg_c' : { 'value' : 0.01},
        'test_samples': { 'value' : 128},
        #'start_months' :{'values' : [1,2,4,6,8,12]},
        'np_seed' : {'values' : [3,4]},
        'torch_seed' : {'values' : [3,4]},
        'window_dim' : {'value' : 32},
        'h_init' : {'value' : 'abs_rand_exp-100'}  # right now this is just as a note to self. Can't change it here     
        }

    sweep_config['parameters'] = parameters_dict

    return sweep_config