import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.utils import set_random_seed

from train_test import train_model, test_model
from dr_wrapper import DR_Wrapper, Auto_DR_Wrapper

# Set to True to train or test the model
TRAIN = True
TEST = False

# Load settings from JSON file
with open('hopper_settings.json', 'r') as f:
    settings = json.load(f)

# try diff vec_env_cls = DummyVecEnv, SubprocVecEnv
# try diff activation functions: nn.ReLU, nn.Tanh
## !!! when training with SetParams remember to provide sample_task as argument to train_model

# Extract settings
env_name = settings["env_name"]
rand_class = settings["rand_class"]
train_steps = settings["train_steps"]
n_envs = settings["n_envs"]
n_steps = settings["n_steps"]
batch_size = settings["batch_size"]
n_epochs = settings["n_epochs"]
gamma = settings["gamma"]
ent_coef = settings["ent_coef"]
gae_lambda = settings["gae_lambda"]
sde_sample_freq = settings["sde_sample_freq"]
activation_fn = getattr(nn, settings["activation_fn"])
policy_kwargs = settings["policy_kwargs"]
policy_kwargs["activation_fn"] = activation_fn
dr_specs = settings["dr_specs"]

if __name__ == "__main__":
    
    policy_kwargs = dict(
                log_std_init=-2,
                activation_fn=nn.Tanh,
                net_arch=dict(pi=[256], vf=[256]),
                lstm_hidden_size=256,
                )

    def make_env(rank, seed):
        def _init():
            env = gym.make(env_name)#, healthy_reward=0, reset_noise_scale=5e-2)
            env = Auto_DR_Wrapper(env, randomization_specs=dr_specs_hopper)
            env.reset(seed=seed+rank)
            return env
        return _init 
    
    env = DummyVecEnv([make_env(i, seed) for i in range(16)]) #
    env = VecMonitor(env)

    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env.reset()

    if TRAIN:
        #for seed in [42, 142, 242, 342, 442]:
        seed = 100
        model_path, env_path = train_model(
            env_name, rand_class, seed, train_steps, policy_kwargs, dr_specs, n_envs, n_steps, batch_size, n_epochs, gamma, ent_coef, gae_lambda, sde_sample_freq
        )
        print(f'Model saved at {model_path}')
        print(f'Env saved at {env_path}')

    if TEST:
        #for seed in [42, 142, 242, 342, 442]:
        seed = 100
        model_path = f'{env_name}_{seed}_logs/{env_name}_{seed}.ckpt'
        env_path = f'{env_name}_{seed}_logs/{env_name}_vecnormalize_{seed}.pkl'
        perf_path = test_model(model_path, env_path, env_name, rand_class, seed, n_envs, dr_specs=dr_specs)
        print(f'Performance saved at {perf_path}')