import os
import gymnasium as gym
import numpy as np
import copy

import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

class DR_Wrapper(gym.Wrapper):
    """
    A wrapper class to randomize the dynamics of the environment.

    Randomization can be additive (random sampled values are added to the default parameter values in
    the environment), multiplicative (random sampled values are multiplied to the default parameter values in
    the environment), or direct assignment (random sampled values are directly assigned to the parameter values,
    overrwiting the default values).

    "clip_range" is optionally used to specify the range of the parameter to be randomized. This is especially useful
    for gaussian randomizations.
    
    
    Arguments:
    env: gym.Env
        The environment to be wrapped.
    randomization_specs: dict
        A dictionary containing the specifications for randomizing the dynamics of the environment. 
        The keys are the attributed to be randomized (entried of the Mujoco mjModel data structure, written as Python
        code strings) and the values are dictionaries containing the following keys:
            - "gaussian": list
                A list containing the mean and standard deviation of the gaussian distribution to sample from.
                Default is [0,1] if an empty list is provided.
            - "uniform": list
                A list containing the lower and upper bounds of the uniform distribution to sample from.
            - "clip_range": list
                A list containing the lower and upper bounds of the range of the parameter to be randomized.
            - "type": str
                A string specifying the type of randomization to be performed. The possible values are:
                    - "+": Additive randomization.
                    - "*": Multiplicative randomization.
                    - "=": Direct assignment of the value.
                    Default is "=" (replace parameter)

    For example:
    {
        "model.body('torso').mass": {
            "gaussian": [0, 1], 
            "clip_range" : [0.5, 9], 
            "type" : ["*"]
        }
    }

    If 'gaussian' is included, a Gaussian distribution is used for randomization. In that case, 'clip_range' specifies
    hard limits for the values of the parameters. Gaussian random values are sampled until a value within the clip range
    is obtained.

    Exactly one of 'gaussian' or 'uniform' must be provided. If both are included, 'gaussian' overrides 'uniform'.

    """
    def __init__(self, env, randomization_specs={}):
        super().__init__(env)
        self.randomization_specs = randomization_specs

    def sample_task(self):
        """Sample random dynamics parameters based on the specified distribution."""
        task = {}
        for key in self.randomization_specs.keys():
            default_param_value = eval("self.env.model." + key)

            type = self.randomization_specs[key].get('type', '=')

            # is a clip range provided?
            clip_range = [-np.inf, np.inf]
            if "clip_range" in self.randomization_specs[key]:
                clip_range = [self.randomization_specs[key]['clip_range'][0], self.randomization_specs[key]['clip_range'][1]]

            if 'gaussian' in self.randomization_specs[key]: # gaussian perturbation
                gaussian_params = self.randomization_specs[key]['gaussian']
                if len(gaussian_params) == 0:
                    mean = 0
                    std = 1
                else:
                    mean = gaussian_params[0]
                    std = gaussian_params[1]

                while True:
                    # additive gasussian
                    sampled_perturbation = np.random.normal()*std + mean
                    if type == '=':
                        value = sampled_perturbation
                    elif type == '+':
                        value = default_param_value + sampled_perturbation
                    # multiplicative gaussian
                    elif type == '*':
                        value = default_param_value * sampled_perturbation
                    # check if value is within the range
                    if value >= clip_range[0] and value <= clip_range[1]:
                        task[key] = value
                        break

            elif 'uniform' in self.randomization_specs[key]: # uniform perturbation
                uniform_params = self.randomization_specs[key]['uniform']
                while True:
                    sampled_perturbation = np.random.uniform(uniform_params[0], uniform_params[1])
                    if type == '=':
                        value = sampled_perturbation
                    elif type == '+':
                        value = default_param_value + sampled_perturbation
                    elif type == '*':
                        value = default_param_value * sampled_perturbation

                    # check if value is within the range
                    if value >= clip_range[0] and value <= clip_range[1]:
                        task[key] = value
                        break


                task[key] = value

            else:
                print('ERROR: dr_specs key `', key, '` does not have a valid distribution specified.')

        return task
    
    def get_task(self):
        return self._task
    
    def set_task(self, task):
        """
        A task is a specific instantiation of the domain randomization specifications.
        It is a dictionary with the same keys as the randomization_specs dictionary, but with the values
        being the sampled values for the parameters.
        """
        self._task = copy.deepcopy(task)
        for key in task.keys():
            exec("self.env.model." + key + " = task[key]")

class Auto_DR_Wrapper(DR_Wrapper):
    def __init__(self, env, randomization_specs={}):
        super().__init__(env, randomization_specs=randomization_specs)

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        task = self.sample_task()
        self.set_task(task)
        #self.randomize()
        return obs