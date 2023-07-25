
# main.py
from spinup import sac_pytorch as sac
from MMP.my_env import my_env 
from spinup.utils.test_policy import load_policy_and_env, run_policy
import torch
import os

TRAIN = 1  # 

# env1 = lambda : env() #
# env1 = env()

log_path = os.path.join(os.getcwd(),'log_7_24')

if TRAIN:
    ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU) # 
    logger_kwargs = dict(output_dir=log_path, exp_name='task_modulate')# 
    print(1)
    sac(lambda : my_env(), ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs,
        steps_per_epoch=5000, epochs=4000,save_freq=1,max_ep_len=5000)

else:
    _, get_action = load_policy_and_env(log_path) # 
    env_test = my_env()
    run_policy(env_test, get_action)

