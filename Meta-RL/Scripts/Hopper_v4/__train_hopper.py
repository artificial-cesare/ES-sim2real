import os
import gymnasium as gym
import numpy as np

import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed


class RandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def randomize(self):
        ## TMP/HACKY, only for Hopper
        min_mass, max_mass = (1,9)
        self.torso_mass = np.random.random()*(max_mass-min_mass) + min_mass

        min_friction, max_friction = (1.5, 2.5)
        self.foot_friction = np.random.random()*(max_friction-min_friction) + min_friction

        self._set_params(self.torso_mass, self.foot_friction)

    def _set_params(self, mass, friction):
        #to get parameters from env.model check mjModel on mujoco documentation 
        #to see the shape and the order of the parameters check the xml file
        #torso mass: mean 6.0, sd 1.5, low 3.0, hi 9.0
        #foot friction: mean 2.0, sd 0.25, lo 1.5, hi 2.5
        self.env.model.body_mass[1] = mass #body mass shape: (5,) the second element is the torso mass ()
        self.env.model.geom_friction[4][0] = friction # 0 is slide, 1 is spin, 2 is roll

class AutoRandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        #print("RESETTING")
        obs = self.env.reset(seed=seed)
        self.randomize()
        return obs

    def randomize(self):
        ## TMP/HACKY, only for Hopper
        #For random samples from the normal distribution with mean mu and standard deviation sigma, use:
        #sigma * np.random.randn(...) + mu

        while True: 
            self.torso_mass = np.random.randn()*1.5 + 6
            self.foot_friction = np.random.randn()*0.25 + 2
            if self.torso_mass >= 0.5 and self.torso_mass <= 9.5 and self.foot_friction >= 1.5 and self.foot_friction <= 2.5:
                break
        self._set_params(self.torso_mass, self.foot_friction)

    def _set_params(self, mass, friction):
        self.env.model.body_mass[1] = mass
        self.env.model.geom_friction[4][0] = friction

if __name__ == "__main__":
    seeds = [100, 200, 300, 400, 500] 
    for seed in seeds:
        
        train_env_mass = 6
        train_env_friction = 2 
        train_class = "Rand" #SetParams, Rand, AutoRand

        set_random_seed(seed, using_cuda=True)


        policy_kwargs = dict(log_std_init=-2,
                            ortho_init=False,
                            activation_fn=nn.ReLU,
                            net_arch=dict(pi=[128, 64], vf=[128, 64])
                            )

        if train_class == "SetParams":
            env = make_vec_env('Hopper-v4', n_envs=16, seed=seed, vec_env_cls=DummyVecEnv, wrapper_class=RandomizedEnvWrapper)
            set_params_fn = env.get_attr('_set_params')
            for i in range(len(set_params_fn)): #for each env it sets the mass to train_env_mass and the friction to train_env_friction
                set_params_fn[i](train_env_mass, train_env_friction)
        elif train_class == "Rand":
            env = make_vec_env('Hopper-v4', n_envs=16, seed=seed, vec_env_cls=DummyVecEnv, wrapper_class=RandomizedEnvWrapper)
            randomize_fn = env.get_attr('randomize')
            for i in range(len(randomize_fn)):
                randomize_fn[i]()
        elif train_class == "AutoRand":
            env = make_vec_env('Hopper-v4', n_envs=16, seed=seed, vec_env_cls=DummyVecEnv, wrapper_class=AutoRandomizedEnvWrapper)
        
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        print(env.get_attr('model')[0].body_mass)
        print(env.get_attr('model')[0].geom_friction[4][0])

        #"""
        model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, n_steps=128, batch_size=512, n_epochs=3, gamma=0.995, ent_coef=0.0, tensorboard_log="tensorboard/", sde_sample_freq=4, use_sde=True)
        model.learn(total_timesteps=3_000_000, tb_log_name='Hopper-v4') 
        model.save('Hopper-v4_'+str(train_env_mass)+'_'+str(train_env_friction)+'_'+str(train_class)+'_'+str(seed)+'.ckpt')
        env.save('Hopper-v4_vecnormalize_'+str(train_env_mass)+'_'+str(train_env_friction)+'_'+str(train_class)+'_'+str(seed)+'.pkl')
        #"""
