"""
Run as:
    mpirun -n (n_workers) python example_es.py

Important: if using a shared GPU on the same node, you should also run:
    # nvidia-cuda-mps-control -d

"""

import time

import torch
import torch as th
import torchvision
import numpy as np
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mpi4py import MPI

import gymnasium as gym
from gymnasium.wrappers import TimeLimit, ClipAction

from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, is_vecenv_wrapped, VecMonitor

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO


from esmpi_sim2real.esmpi import ESMPI, compute_centered_ranks, compute_identity_ranks
from esmpi_sim2real.metarl_envs import HalfCheetahDirEnv, HalfCheetahVelEnv
from esmpi_sim2real.mpi_vecnormalize import MPIVecNormalize

from esmpi_sim2real.hopper_wrappers import MLRandomizedEnvWrapper, MLAutoRandomizedEnvWrapper

# TODO: to fine-tune centered-ranks vs identity-ranks, lr, pop size (e.g., in prev paper they used 300)


# batch-size for the outer (meta) loop
population_size = 16*12 # make sure this is a multiple of the number of workers used

n_parallel_envs_per_worker = 1  # not strictly required, but it can be useful to speed-up
                                # the inner-loop operations; this is especially true for the
                                # evaluation of adapted policies


# TODO: it would be best to ppo.train 

num_inner_loop_ppo_iterations = 6

ppo_n_steps = 512
ppo_batch_size = 64
num_total_inner_loop_steps = num_inner_loop_ppo_iterations * (n_parallel_envs_per_worker * ppo_n_steps)
# TODO: check: num_total_inner_loop_steps should hopefully be a few episodes; but note: if using parallel environments, the total 
#       number of steps will be reached well before the end of an episode!
# TODO: auto balance all these parameters

print('PPO inner loop steps: ', num_total_inner_loop_steps)
if num_total_inner_loop_steps % ppo_batch_size != 0:
    print('\n\n*** WARNING: ',num_total_inner_loop_steps,'should be a multiple of ppo_batch_size ', ppo_batch_size, '\n\n')


outer_loop_lr = 0.003
inner_loop_lr = 3e-4 #0.003
es_sigma = 0.02

random_seed = 42

n_meta_iterations = 400



### [ CREATE ENV and PPO for each worker ]
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using GPU')
else:
    device = torch.device("cpu")
    print('Using CPU')

is_master_worker = (MPI.COMM_WORLD.Get_rank()==0)

# seed + mpi4py rank id
seed = random_seed + n_parallel_envs_per_worker*MPI.COMM_WORLD.Get_rank()
set_random_seed(seed, using_cuda=torch.cuda.is_available())


# """
## Hopper-v4 

policy_kwargs = dict(log_std_init=-2,
                    ortho_init=False,
                    activation_fn=nn.ReLU,
                    net_arch=dict(pi=[128, 128], vf=[128, 128])
                    )
#seeds = [seed + i for i in range(n_parallel_envs_per_worker)]



random_task = {'mass':6, 'friction':2}


def make_env(rank, seed, **kwargs):
    def _init():
        env = gym.make('Hopper-v4', **kwargs)
        env = MLRandomizedEnvWrapper(env)
        env.reset(seed=seed+rank)
        return env
    return _init
env = DummyVecEnv([make_env(i, seed) for i in range(n_parallel_envs_per_worker)])
env = VecMonitor(env)

# TODO:
env = MPIVecNormalize.load('out/env_stats.pkl', env)
env.training = False
env.norm_reward = False
env.env_method('set_task', random_task)
env.reset()


model_results = PPO.load('out/model', env)




mean_reward, _ = evaluate_policy(model_results, env, n_eval_episodes=5, deterministic=True)
print('Before adaptation: ', mean_reward)


model_results.learn(num_total_inner_loop_steps)

mean_reward, _ = evaluate_policy(model_results, env, n_eval_episodes=5, deterministic=True)
print('After adaptation: ', mean_reward)



env = DummyVecEnv([make_env(i, seed, render_mode='human') for i in range(n_parallel_envs_per_worker)])
env = VecMonitor(env)

# TODO:
env = MPIVecNormalize.load('out/env_stats.pkl', env)
env.training = False
env.norm_reward = False
env.env_method('set_task', random_task)
env.reset()


obs = env.reset()
for i in range(1000):
    action, _states = model_results.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render("human")




