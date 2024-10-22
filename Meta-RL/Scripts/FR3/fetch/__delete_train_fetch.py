# test env to see how to import classes to use in gym and with sb3

import sys
import os
import gymnasium as gym

import torch
import torch.nn as nn
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fetch')))

from reach import MujocoFetchReachEnv

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
use_sde = True
activation_fn = nn.Tanh #nn.ReLU

policy_kwargs = dict(log_std_init=-2,  
                    ortho_init=True, #False
                    activation_fn=activation_fn,
                    net_arch=dict(pi=[256, 128], vf=[256, 128])
                    )

set_random_seed(seed, using_cuda=True)

env = make_vec_env(gym.make(MujocoFetchReachEnv(render_mode='human'), max_episode_steps=100), n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device.type.upper()}')

model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, ent_coef=ent_coef, gae_lambda=gae_lambda, tensorboard_log="tensorboard/", use_sde=use_sde, sde_sample_freq=sde_sample_freq,seed=seed, device=device)
model.learn(total_timesteps=train_steps, tb_log_name='MujocoFetchReachEnv')

logs_folder = f"./MujocoFetchReachEnv_{seed}_logs/"

folder_path = os.path.join(os.getcwd(), logs_folder)

model_path = folder_path+f'MujocoFetchReachEnv_{seed}.ckpt'
env_path = folder_path+f'MujocoFetchReachEnv_vecnormalize_{seed}.pkl'

model.save(model_path)
env.save(env_path)

"""
import time
st = time.time()
i = 0
action = env.action_space.sample()
while time.time()-st < 10:
    if i%12==0:
        action = env.action_space.sample()
    env.step(action)
    i+=1
"""