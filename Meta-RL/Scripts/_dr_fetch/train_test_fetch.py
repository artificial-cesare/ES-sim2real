import gymnasium as gym
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

from dr_wrapper import DR_Wrapper, Auto_DR_Wrapper

def train_model(env_name, rand_class, seed, train_steps, 
                policy_kwargs, dr_specs_dict, n_envs, n_steps, 
                batch_size, n_epochs, gamma, ent_coef, gae_lambda, 
                use_sde=False, sample_task=None, sde_sample_freq=-1):
    """
    Train PPO model with given hyperparameters.
    """

    set_random_seed(seed, using_cuda=True)

    if rand_class == "SetParams":
        wrapper_class = DR_Wrapper
        env = make_vec_env(env_name, n_envs=n_envs, seed=seed, 
                           vec_env_cls=SubprocVecEnv, env_kwargs= {'max_episode_steps':100},
                            wrapper_class=wrapper_class, wrapper_kwargs={'randomization_specs': dr_specs_dict}) 
        if sample_task is not None:
            for i in range(n_envs):
                env.env_method('set_task', sample_task)[i]
                
        elif sample_task is None:
            random_task = env.env_method('sample_task')[0]
            for i in range(n_envs):
                env.env_method('set_task', random_task)[i]
    
    elif rand_class == "Rand":
        wrapper_class = DR_Wrapper
        env = make_vec_env(env_name, n_envs=n_envs, seed=seed,
                           vec_env_cls=SubprocVecEnv, env_kwargs= {'max_episode_steps':100},
                           wrapper_class=wrapper_class, wrapper_kwargs={'randomization_specs': dr_specs_dict})
        random_task = env.env_method('sample_task') # params randomization
        for i in range(len(random_task)):
            env.env_method('set_task', random_task[i])[i]

    elif rand_class == "AutoRand":
        wrapper_class = Auto_DR_Wrapper
        env = make_vec_env(env_name, n_envs=n_envs, seed=seed,
                           vec_env_cls=SubprocVecEnv, env_kwargs= {'max_episode_steps':100},
                           wrapper_class=wrapper_class, wrapper_kwargs={'randomization_specs': dr_specs_dict})
        
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device.type.upper()}')

    logs_folder = f"./{env_name}_{seed}_{rand_class}_logs/"

    folder_path = os.path.join(os.getcwd(), logs_folder)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{logs_folder}' created successfully.")
    else:
        print(f"Folder '{logs_folder}' already exists.")

    model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma,
                ent_coef=ent_coef, gae_lambda=gae_lambda, tensorboard_log="tensorboard/",
                use_sde=use_sde, sde_sample_freq=sde_sample_freq,seed=seed, device=device)
    
    model.learn(total_timesteps=train_steps, tb_log_name=f'{env_name}')

    model_path = folder_path+f'{env_name}_{seed}_{rand_class}.ckpt'
    env_path = folder_path+f'{env_name}_vecnormalize_{seed}_{rand_class}.pkl'
    
    model.save(model_path)
    env.save(env_path)
    
    return model_path, env_path

def test_model(model_path, env_path, env_name, rand_class,seed, n_envs,
                n_eval_episodes=50):
    """
    Evaluate model on environment.
    """
    folder_path = f"{env_name}_{seed}_{rand_class}_logs/"

    perf = pd.DataFrame(columns=["reward"])
    set_random_seed(seed, using_cuda=True)

    env = make_vec_env(env_name, n_envs, seed=seed, vec_env_cls=DummyVecEnv)
    env = VecNormalize.load(env_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    y = pd.DataFrame({'reward': [mean_reward]})
    perf = pd.concat([perf, y], ignore_index=True)

    print('Train levels: ', mean_reward, '+-', std_reward)
    
    perf_path = folder_path+f'perf_{env_name}_{seed}_{rand_class}.csv'
    perf.to_csv(perf_path, index=False)

    return perf_path


