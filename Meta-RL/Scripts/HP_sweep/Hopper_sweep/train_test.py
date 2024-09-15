import gymnasium as gym
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize #, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

#from stable_baselines3.common.callbacks import CheckpointCallback

class RandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _set_params(self, mass, friction):
        #to get parameters from env.model check mjModel on mujoco documentation 
        #to see the shape and the order of the parameters check the xml file
        #torso mass: mean 6.0, sd 1.5, low 3.0, hi 9.0
        #foot friction: mean 2.0, sd 0.25, lo 1.5, hi 2.5
        self.env.model.body_mass[1] = mass #body mass shape: (5,) the second element is the torso mass ()
        self.env.model.geom_friction[4][0] = friction # 0 is slide, 1 is spin, 2 is roll

def train_model(env_name, seed, train_steps, policy_kwargs, n_envs, n_steps, batch_size, n_epochs, gamma, ent_coef, gae_lambda, sde_sample_freq=4):
    """
    Train PPO model with given hyperparameters.
    """
   
    use_sde = True
    train_env_mass = 6
    train_env_friction = 2 

    set_random_seed(seed, using_cuda=True)
    
    env = make_vec_env(env_name, n_envs=n_envs, seed=seed,  vec_env_cls=DummyVecEnv, wrapper_class=RandomizedEnvWrapper)

    set_params_fn = env.get_attr('_set_params')
    for i in range(len(set_params_fn)): #for each env it sets the mass to train_env_mass and the friction to train_env_friction
        set_params_fn[i](train_env_mass, train_env_friction)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device.type.upper()}')

    logs_folder = f"./{env_name}_{seed}_logs/"

    folder_path = os.path.join(os.getcwd(), logs_folder)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{logs_folder}' created successfully.")
    else:
        print(f"Folder '{logs_folder}' already exists.")

    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, ent_coef=ent_coef, gae_lambda=gae_lambda, tensorboard_log="tensorboard/", use_sde=use_sde, sde_sample_freq=sde_sample_freq,seed=seed, device=device)
    model.learn(total_timesteps=train_steps, tb_log_name=f'{env_name}')

    model_path = folder_path+f'{env_name}_{seed}.ckpt'
    env_path = folder_path+f'{env_name}_vecnormalize_{seed}.pkl'
    
    model.save(model_path)
    env.save(env_path)
    
    return model_path, env_path

def test_model(model_path, env_path, env_name, seed, n_envs, n_eval_episodes=50):
    """
    Evaluate model on environment.
    """
    mass = 6
    friction = 2

    folder_path = f"{env_name}_{seed}_logs/"

    perf = pd.DataFrame(columns=["reward"])
    set_random_seed(seed, using_cuda=True)

    env = make_vec_env(env_name, n_envs, seed=seed, vec_env_cls=DummyVecEnv, wrapper_class=RandomizedEnvWrapper)
    env = VecNormalize.load(env_path, env)
    env.training = False
    env.norm_reward = False

    set_params_fn = env.get_attr('_set_params')

    for i in range(len(set_params_fn)):
        set_params_fn[i](mass, friction)

    model = PPO.load(model_path, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
    y = pd.DataFrame({'reward': [mean_reward]})
    perf = pd.concat([perf, y], ignore_index=True)

    print('Train levels: ', mean_reward, '+-', std_reward)
    
    perf_path = folder_path+f'perf_{env_name}_{seed}.csv'
    perf.to_csv(perf_path, index=False)

    return perf_path


