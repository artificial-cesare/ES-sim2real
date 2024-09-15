import gymnasium as gym
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

from train_test_fetch import train_model, test_model

env_name = "FetchSlideDense-v2"
seed = 42

train_steps = 30_000_000 

#Set to True to train or test the model
TRAIN = True
TEST = False 

n_envs = 16
n_steps = 32 #256 
batch_size = 64 #256 
n_epochs = 3
gamma= 0.975 #0.995 0.97 o 0.98
ent_coef = 0.0
gae_lambda = 0.9
sde_sample_freq=4
#use_sde = True
#activation_fn = nn.Tanh #nn.ReLU

configs = { "conf1" : [ True, nn.Tanh], #3539
            "conf2" : [ True, nn.ReLU], #1809
            "conf3" : [ False, nn.Tanh], #3636
            "conf4" : [ False, nn.ReLU], #3301
            }

if __name__ == "__main__":
    if TRAIN:
        for config in configs.keys():
            use_sde = configs[config][0]
            activation_fn = configs[config][1]
            policy_kwargs = dict(log_std_init=-2,  
                    ortho_init=True, #False
                    activation_fn=activation_fn,
                    net_arch=dict(pi=[256, 128], vf=[256, 128])
                    )
            model_path, env_path = train_model(env_name, seed, train_steps, policy_kwargs, n_envs, n_steps, batch_size, n_epochs, gamma, ent_coef, gae_lambda, use_sde, config)
            print(f'Model saved at {model_path}')
            print(f'Env saved at {env_path}')

    if TEST:
        #for seed in [42, 142, 242, 342, 442]:
        #for use_sde in [False, True]:
        for config in configs.keys():
            model_path = f'{env_name}_{seed}_logs/{env_name}_{seed}_{config}.ckpt'
            env_path = f'{env_name}_{seed}_logs/{env_name}_vecnormalize_{seed}_{config}.pkl'
            perf_path = test_model(model_path, env_path, env_name, seed, n_envs, config)
            print(f'Performance saved at {perf_path}')