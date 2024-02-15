import sys
import os

# Set the base path relative to the current script location
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(base_path)

# Add the required directories to the system path
sys.path.insert(0, os.path.join(base_path, "networks"))
sys.path.insert(0, os.path.join(base_path, "configs"))
sys.path.insert(0, os.path.join(base_path, "utils"))  # Assuming "__utils" is the correct directory name

# Your code continues...
from utils import choose_model, choose_loss, choose_sheduler, get_train_tensors, get_test_tensor, apply_dropout, execute_freeze_h_option, get_log_dict, train_log, init_weights, get_data
#from config_sweep import get_swep_config
print('Imports done...')