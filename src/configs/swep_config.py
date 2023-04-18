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
        'scheduler' : {'value': 'step'},
        'hidden_channels': {'value': 32}, # you like need 32, it seems from qualitative results
        'min_events': {'value': 10},
        'samples': {'value': 512}, # just speed running here..
        "dropout_rate" : {'value' : 0.05},
        'learning_rate': {'value' : 0.0001},
        "weight_decay" : {'value' : 0.1},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 3},
        'loss_class' : { 'value' : 'c'}, # det nytter jo ikke noget at du køre over gamma og alpha for loss-class a...
        'loss_class_gamma' : {'value' : 0},
        'loss_class_alpha' : {'value' : 0.5}, # should be between 0.5 and 0.95...
        'loss_reg' : { 'value' : 'b'},
        'loss_reg_a' : { 'values' : [10, 12, 14, 16, 18]},
        'loss_reg_c' : { 'value' : 0.01},
        'test_samples': { 'value' : 128},
        'h_init' : {'value' : 'abs_rand_exp-100'}  # right now this is just as a note to self. Can't change it here     
        }

    sweep_config['parameters'] = parameters_dict

    return sweep_config