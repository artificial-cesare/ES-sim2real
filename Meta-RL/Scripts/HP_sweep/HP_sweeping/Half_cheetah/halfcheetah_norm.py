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
env_name = "HalfCheetah-v4"
#seed = 42

#Set to True to train or test the model
TRAIN = False
TEST = True

train_steps = 10_000_000 #30_000_000

n_envs = 16
n_steps = 256 #512
batch_size = 256 #64 #512
n_epochs = 3
gamma=0.995 #0.999
ent_coef = 0.0
gae_lambda = 0.9
sde_sample_freq=4
#use_sde = False #True
activation_fn = nn.Tanh #nn.ReLU

policy_kwargs = dict(log_std_init=-2,  # smaller log_std -> smaller std -> more deterministic
                    ortho_init=True, #False
                    activation_fn=activation_fn,
                    net_arch=dict(pi=[256, 128], vf=[256, 128])
                    )

if __name__ == "__main__":
    if TRAIN:
        for seed in [42, 142, 242, 342, 442, 542]:
            model_path, env_path = train_model(env_name, seed, train_steps, policy_kwargs, n_envs, n_steps, batch_size, n_epochs, gamma, ent_coef, gae_lambda, sde_sample_freq)
            print(f'Model saved at {model_path}')
            print(f'Env saved at {env_path}')

    if TEST:
        for seed in [42, 142, 242, 342, 442, 542]:
            model_path = f'{env_name}_{seed}_logs/{env_name}_{seed}.ckpt'
            env_path = f'{env_name}_{seed}_logs/{env_name}_vecnormalize_{seed}.pkl'
            perf_path = test_model(model_path, env_path, env_name, seed, n_envs)
            print(f'Performance saved at {perf_path}')
