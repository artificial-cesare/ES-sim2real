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

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

from dr_wrapper import DR_Wrapper, Auto_DR_Wrapper
from train_test_fetch import train_model, test_model

# Set to True to train or test the model
TRAIN = True
TEST = False

# Load settings from JSON file
with open('push_settings.json', 'r') as f:
    settings = json.load(f)

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
    if TRAIN:
        #for seed in [42, 142, 242, 342, 442]:
        seed = 100
        model_path, env_path = train_model(
            env_name, rand_class, seed, train_steps, policy_kwargs, 
            dr_specs, n_envs, n_steps, batch_size, n_epochs, gamma, 
            ent_coef, gae_lambda, sde_sample_freq
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