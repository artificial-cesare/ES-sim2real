import os
import gymnasium as gym
import numpy as np
import copy

import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

dr_quarks_hopper = {"body('torso').mass": {"gaussian": [], "range": [0.35, 9.75], "type" : "+"},
                    "body('thigh').mass": {"gaussian": [], "range": [0.35, 9.75], "type" : "+"},
                    "body('leg').mass": {"gaussian": [], "range": [0.35, 9.75], "type" : "+"},
                    "body('foot').mass": {"gaussian": [], "range": [0.35, 9.75], "type" : "+"},
                    "joint('foot_joint').damping" : {"range": [0.17, 2.93], "type" : "+"},
                    "joint('leg_joint').damping" : {"range": [0.17, 2.93], "type" : "+"},
                    "joint('thigh_joint').damping" : {"range": [0.17, 2.93], "type" : "+"},
                    "geom('floor').friction[0]": {"gaussian": [], "range": [0.17, 2.93], "type" : "+"}}

dr_randomization_specs_halfcheetah = {"body('torso').mass": {"gaussian": [], "range": [0.32, 12.4], "type" : "+"}, 
                         "body('bthigh').mass": {"gaussian": [], "range": [0.08, 2.99], "type" : "+"},
                         "body('bshin').mass": {"gaussian": [], "range": [0.08, 3.08], "type" : "+"},
                         "body('bfoot').mass": {"gaussian": [], "range": [0.05, 2.08], "type" : "+"},
                         "body('fthigh').mass": {"gaussian": [], "range": [0.07, 2.78], "type" : "+"},
                         "body('fshin').mass": {"gaussian": [], "range": [0.06, 2.30], "type" : "+"},
                         "body('ffoot').mass": {"gaussian": [], "range": [0.04, 1.66], "type" : "+"},
                         "geom('floor').friction[0]": {"gaussian": [], "range": [0.02, 0.78], "type" : "+"}}

dr_randomization_specs_ant = {"body('torso').mass": {"gaussian": [], "type" : "+"},
                 "body('front_left_leg').mass": {"gaussian": [], "type" : "+"},
                 "body('front_right_leg').mass": {"gaussian": [], "type" : "+"},
                 "body('back_leg').mass": {"gaussian": [], "type" : "+"},
                 "body('right_back_leg').mass": {"gaussian": [], "type" : "+"},
                 "joint('hip_1').damping": {"gaussian": [], "type" : "+"},
                 "joint('hip_2').damping": {"gaussian": [], "type" : "+"},
                 "joint('hip_3').damping": {"gaussian": [], "type" : "+"},
                 "joint('hip_4').damping": {"gaussian": [], "type" : "+"},
                 "geom('floor').friction[0]": {"gaussian": [], "type" : "+"}}

dr_randomization_specs_humanoid = {"body('torso').mass": {"gaussian": [], "type" : "+"},
                      "moodel.body('right_lower_arm').mass": {"gaussian": [], "type" : "+"},
                      "body('left_lower_arm').mass": {"gaussian": [], "type" : "+"},
                      "body('left_foot').mass": {"gaussian": [], "type" : "+"},
                      "body('right_foot').mass": {"gaussian": [], "type" : "+"},
                      "joint('left_knee').damping": {"gaussian": [], "type" : "+"},
                      "joint('right_knee').damping": {"gaussian": [], "type" : "+"},
                      "joint('left_elbow').damping": {"gaussian": [], "type" : "+"},
                      "joint('right_elbow').damping": {"gaussian": [], "type" : "+"},
                      "geom('floor').friction[0]": {"gaussian": [], "type" : "+"}} 

## if gaussian no need for '=' type, just have to provide mean and std
## if no range is provided, it will be calculated based on the default value
## type + is for additive, * is for multiplicative, = is for direct assignment

# examples: dr_randomization_specs_humanoid = {"model.body('torso').mass": {"gaussian": [], "range : [], "type" : ["+"]}}

class DR_Wrapper(gym.Wrapper):
    def __init__(self, env, randomization_specs): # non unwrappare 
        super().__init__(env)
        self.randomization_specs = randomization_specs
    
    def sample_task(self):
        """Sample random dynamics parameters based on the specified distribution."""
        task = {}
        for key in self.randomization_specs.keys():

            default = eval("self.env.model." + key)

            # set the range of the parameter 
            if "range" in self.randomization_specs[key]:
                hi = self.randomization_specs[key]['range'][1]
                lo = self.randomization_specs[key]['range'][0]

            # if no range is provided, calculate it based on the default value
            else:
                hi = (abs(default)*2) + ((default/100)*10)
                lo = ((default/100)*10)

            type = self.randomization_specs[key]['type']
            
            if 'gaussian' in self.randomization_specs[key]:
                if len(self.randomization_specs[key]['gaussian']) == 0:
                    mean = default
                    std = ((hi - lo)-mean)/2
                    while True:
                        # additive gasussian
                        if type == '+':
                            value = np.random.normal()*std + mean
                        # multiplicative gaussian
                        elif type == '*':
                            value = np.random.normal()*std * mean
                        # check if value is within the range
                        if value >= lo and value <= hi:
                            task[key] = task.get(key, 0) + value
                            break
                # direct assignment from gaussian  
                else: # no need for '=' type, just have to provide mean and std
                    mean = self.randomization_specs[key]['gaussian'][0]
                    std = self.randomization_specs[key]['gaussian'][1]
                    while True:
                        value = np.random.normal(loc=mean, scale=std)
                        if value >= lo and value <= hi:
                            task[key] = task.get(key, 0) + value
                            break
            else: # uniform distribution
                if type == '+':
                    low_bound = lo - default
                    up_bound = hi - default
                    value = default + (low_bound + (up_bound-low_bound) * np.random.random_sample())
                elif type == '*':
                    low_bound = lo/default
                    up_bound = hi/default
                    value = default * (low_bound + (up_bound-low_bound) * np.random.random_sample())
                elif type == '=':
                    value = np.random.uniform(lo, hi)
                task[key] = task.get(key, 0) + value
        return task
    
    def get_task(self):
        return self._task
    
    # to set a task manually create a dictionary with the keys as the parameters and the values as the new values
    def set_task(self, task):
        self._task = copy.deepcopy(task) # probably not needed since task is always a single float 
        for key in task.keys():
            exec("self.env.model." + key + " = task[key]")

class Auto_DR_Wrapper(DR_Wrapper):
    def __init__(self, env, randomization_specs):
        super().__init__(env, randomization_specs)

    def reset(self, seed=None):
        print("RESETTING")
        obs = self.env.reset(seed=seed)
        task = self.sample_task()
        self.set_task(task)
        #self.randomize()
        return obs