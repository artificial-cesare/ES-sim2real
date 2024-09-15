import gymnasium as gym
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize #, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

from train_test import train_model, test_model


## Hopper settings
env_name = 'Hopper-v4'
seed = 42

#Set to True to train or test the model
TRAIN = True #False
TEST = False #True

train_steps = 10_000_000 #30_000_000

configs = { "conf1" : [ 256, 256, nn.Tanh], #3539
            "conf2" : [ 128, 512, nn.ReLU], #1809
            "conf3" : [ 256, 512, nn.ReLU], #3636
            "conf4" : [ 128, 256, nn.Tanh], #3301
            "conf5" : [ 256, 256, nn.ReLU], #1021
            "conf6" : [ 128, 512, nn.Tanh], #3075
            "conf7" : [ 256, 512, nn.Tanh], #3283
            "conf8" : [ 128, 256, nn.ReLU]  #1476
            } 

n_envs = 16
#n_steps = 256 #128
#batch_size = 256 #512
n_epochs = 3
gamma = 0.995 #0.999
ent_coef = 0.0
gae_lambda = 0.9
#sde_sample_freq = 4
#use_sde = False #True
#activation_fn = nn.Tanh #nn.ReLU

if __name__ == "__main__":
    if TRAIN:
        for config in configs.keys():
            n_steps = configs[config][0]
            batch_size = configs[config][1]
            activation_fn = configs[config][2]
        
            policy_kwargs = dict(log_std_init=-2,  # smaller log_std -> smaller std -> more deterministic
                        ortho_init=True, #False
                        activation_fn=activation_fn,
                        net_arch=dict(pi=[256, 128], vf=[256, 128])
                        )
            print(f'Running config {config} with n_steps={n_steps}, batch_size={batch_size}, activation_fn={activation_fn}')
            model_path, env_path = train_model(env_name, seed, train_steps, policy_kwargs, n_envs, n_steps, batch_size, n_epochs, gamma, ent_coef, gae_lambda, config)
            print(f'Model saved at {model_path}')
            print(f'Env saved at {env_path}')

    if TEST:
        for config in configs.keys():
            n_steps = configs[config][0]
            batch_size = configs[config][1]
            activation_fn = configs[config][2]
        
            policy_kwargs = dict(log_std_init=-2,  # smaller log_std -> smaller std -> more deterministic
                        ortho_init=True, #False
                        activation_fn=activation_fn,
                        net_arch=dict(pi=[256, 128], vf=[256, 128])
                        )
            
            model_path = f'{env_name}_{seed}_logs/{env_name}_{seed}_{config}.ckpt'
            env_path = f'{env_name}_{seed}_logs/{env_name}_vecnormalize_{seed}_{config}.pkl'
            perf_path = test_model(model_path, env_path, env_name, seed, n_envs, config)
            print(f'Performance saved at {perf_path}')