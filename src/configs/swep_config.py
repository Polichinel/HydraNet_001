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
        'model' : {'value' :'HydraBNUNet06_LSTM2'},
        'weight_init' : {'value' : 'xavier_norm'}, # ['xavier_uni', 'xavier_norm', 'kaiming_uni', 'kaiming_normal']
        'clip_grad_norm' : {'value': True},
        'scheduler' : {'value': 'WarmupDecay'}, #CosineAnnealingLR004  'CosineAnnealingLR' 'OneCycleLR'
        'hidden_channels': {'value': 32}, # you like need 32, it seems from qualitative results
        'min_events': {'value': 5},
        'samples': {'value':   600}, # should be a function of batches becaus batch 3 and sample 1000 = 3000....
        'batch_size': {'value':  3}, # just speed running here..
        "dropout_rate" : {'value' : 0.125},
        'learning_rate': {'value' :  0.0005}, #0.001 default, but 0.005 might be better
        "weight_decay" : {'value' : 0.1},
        "slope_ratio" : {'value' : 0.75},
        "roof_ratio" : {'value' :  0.7},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 3},
        'loss_class' : { 'value' : 'd'}, # det nytter jo ikke noget at du køre over gamma og alpha for loss-class a...
        'loss_class_gamma' : {'value' : 1.5},
        'loss_class_alpha' : {'value' : 0.75}, # should be between 0.5 and 0.95...
        'loss_reg' : { 'value' : 'b', },
        'loss_reg_a' : { 'value' : 28 },
        'loss_reg_c' : { 'value' : 0.005},
        'test_samples': { 'value' : 128},
        #'start_months' :{'values' : [1,2,4,6,8,12]},
        'np_seed' : {'values' : [3,4]},
        'torch_seed' : {'values' : [3,4]},
        'window_dim' : {'value' : 32},
        'h_init' : {'value' : 'abs_rand_exp-100'},
        'un_log' : {'value' : False},
        'warmup_steps' : {'value' : 100},
        'first_feature_idx' : {'value' : 5},
        'norm_target' : {'value' : False},
        'freeze_h' : {'values' : ["all", "random"]},
        #'loss_distance_scale' : {'values' : [1, 0.5, 0.1]}  # right now this is just as a note to self. Can't change it here     
        }

    sweep_config['parameters'] = parameters_dict

    return sweep_config
