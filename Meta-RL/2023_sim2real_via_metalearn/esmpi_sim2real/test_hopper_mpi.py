import numpy as np

import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

from itertools import product
import pandas as pd

from __train_hopper import RandomizedEnvWrapper, AutoRandomizedEnvWrapper

if __name__ == "__main__":
    seeds = [100, 200, 300, 400, 500] 
    train_classes = ["SetParams", "Rand", "AutoRand"]
    for train_class in train_classes:
        for seed in seeds:
            set_random_seed(seed, using_cuda=True)
    
            train_env_mass = 6
            train_env_friction = 2 
            
            perf = pd.DataFrame(columns=["mass", "friction", "reward"])
            masses = np.arange(0.5, 12+0.5, 0.5)
            frictions = np.arange(0.1, 4+0.1, 0.1) #note that are both outside the range
            #torso mass: mean 6.0, sd 1.5, low 3.0, hi 9.0
            #foot friction: mean 2.0, sd 0.25, lo 1.5, hi 2.5
            combinations = list(product(masses, frictions))

            for mass, friction in combinations:
                env = make_vec_env('Hopper-v4', n_envs=16, seed=seed, vec_env_cls=DummyVecEnv, wrapper_class=RandomizedEnvWrapper)
                env = VecNormalize.load('/home/u933585/R-Meta-Learning/Train_Data/Hopper-v4_vecnormalize_'+str(train_env_mass)+'_'+str(train_env_friction)+'_'+str(train_class)+'_'+str(seed)+'.pkl', env)
                env.training = False
                env.norm_reward = False

                set_params_fn = env.get_attr('_set_params')

                for i in range(len(set_params_fn)):
                    set_params_fn[i](mass, friction)

                model = PPO.load('/home/u933585/R-Meta-Learning/Train_Data/Hopper-v4_'+str(train_env_mass)+'_'+str(train_env_friction)+'_'+str(train_class)+'_'+str(seed)+'.ckpt', env=env)

                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
                y = pd.DataFrame({'mass': [mass], 'friction': [friction], 'reward': [mean_reward]})
                perf = pd.concat([perf, y], ignore_index=True)

                print(mass, 'kg \t', friction, 'friction \t', 'Train levels: ', mean_reward, '+-', std_reward)
    
            perf.to_csv('/home/u933585/R-Meta-Learning/Test_data/perf_{}_{}_{}_{}.csv'.format(train_env_mass, train_env_friction, seed, train_class), index=False)
    