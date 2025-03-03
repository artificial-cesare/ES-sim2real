import os
import gymnasium as gym
import numpy as np
import copy

import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed

## TODO? make rand and autorand work with half ceetah

"""
make wrapper 'make metarlenv' that adds the methods   sample_task and set_task

    set_mass_fn = env.get_attr('_set_mass')

    for i in range(len(set_mass_fn)):
        set_mass_fn[i](train_env_mass)
    print(env.get_attr('model')[0].body_mass)

    or    env.env_method('method name', args...)
            env.env_method('_set_mass', train_env_mass)

"""
"""
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
"""

"""
env_method(method_name, *method_args, indices=None, **method_kwargs)[source]
Call instance methods of vectorized environments.

Parameters:	
method_name – (str) The name of the environment method to invoke.
indices – (list,int) Indices of envs whose method to call
method_args – (tuple) Any positional arguments to provide in the call
method_kwargs – (dict) Any keyword arguments to provide in the call
Returns:	
(list) List of items returned by the environment’s method call

get_attr(attr_name, indices=None)[source]
Return attribute from vectorized environment.

Parameters:	
attr_name – (str) The name of the attribute whose value to return
indices – (list,int) Indices of envs to get attribute from
Returns:	
(list) List of values of ‘attr_name’ in all environments
"""

dr_quarks_hopper = {'mass': [{'mean': 6.0, 'std': 1.5, 'lo': 3.0, 'hi': 9.0}], 'friction': [{'mean': 2.0, 'std': 0.25, 'lo': 1.5, 'hi': 2.5}], 'damping': [{'mean' : None, 'std' : None, 'hi' : 2.93, 'lo': 0.17}]}

# save def params for each env and pass them on to wrap as mean of dist 

# additive vs multiplicative noise

## NEEDED: domrand parameters for each env

class DR_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        name = env.envs[0].unwrapped.spec.id #have to change this since it's not vec_env 
        self.name == name
        self.masses = copy.deepcopy(self.env.model.body_mass[1:])
        self.frictions = copy.deepcopy(self.env.model.geom_friction[4][0]) # 0 is slide, 1 is spin, 2 is roll: 
        self.dampings = copy.deepcopy(self.env.model.dof_damping[3:])
        
    def sample_task(self, dr_distr, 
                    kwargs):
        """Sample random dynamics parameters"""
        task = {}
        if dr_distr == 'mult_uniform':
            for i in range(len(self.masses)): #for each body part you can have custom params 
                min_mass, max_mass = kwargs['mass'][i]['lo'], kwargs['mass'][i]['hi']
                mass = np.random.random()*(max_mass-min_mass) + min_mass
                task['mass'] = task.get('mass', []) + [mass]

            for i in range(len(self.frictions)):
                min_friction, max_friction = kwargs['friction'][i]['lo'], kwargs['friction'][i]['hi']
                friction = np.random.random()*(max_friction-min_friction) + min_friction
                task['friction'] = task.get('friction', []) + [friction]

            for i in range(len(self.dampings)):
                min_damping, max_damping = kwargs['damping'][i]['lo'], kwargs['damping'][i]['hi']
                damping = np.random.random()*(max_damping-min_damping) + min_damping
                task['damping'] = task.get('damping', []) + [damping]

        elif dr_distr == 'mult_gaussian':
            for i in range(len(self.masses)):
                mean_mass, std_mass = kwargs['mass'][i]['mean'], kwargs['mass'][i]['std']
                mass = np.random.randn()*std_mass + mean_mass
                task['mass'] = task.get('mass', []) + [mass]

            for i in range(len(self.frictions)):
                mean_friction, std_friction = kwargs['friction'][i]['mean'], kwargs['friction'][i]['std']
                friction = np.random.randn()*std_friction + mean_friction
                task['friction'] = task.get('friction', []) + [friction]

            for i in range(len(self.dampings)): 
                mean_damping, std_damping = kwargs['damping'][i]['mean'], kwargs['damping'][i]['std']
                damping = np.random.randn()*std_damping + mean_damping
                task['damping'] = task.get('damping', []) + [damping]

        return task
    

    dr_quarks_hopper = {
    'mass': [{'mean': 6.0, 'std': 1.5, 'lo': 3.0, 'hi': 9.0}],
    'friction': [{'mean': 2.0, 'std': 0.25, 'lo': 1.5, 'hi': 2.5}],
    'damping': [{'mean': None, 'std': None, 'lo': 0.17, 'hi': 2.93}]
}

class DR_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assumption: The environment provides spec id for each unwrapped env.
        self.name = env.envs[0].unwrapped.spec.id
        self.masses = copy.deepcopy(self.env.model.body_mass[1:])
        self.frictions = copy.deepcopy(self.env.model.geom_friction[:, 0])  # Consider all friction types
        self.dampings = copy.deepcopy(self.env.model.dof_damping[3:])
    
    def sample_task(self, dr_distr, kwargs):
        """Sample random dynamics parameters based on the specified distribution."""
        task = {}
        if dr_distr == 'mult_uniform':
            self._sample_uniform(task, kwargs)
        elif dr_distr == 'mult_gaussian':
            self._sample_gaussian(task, kwargs)
        return task
    
    def _sample_uniform(self, task, kwargs):
        """Sample parameters using a uniform distribution."""
        for key in ['mass', 'friction', 'damping']:
            for i in range(len(kwargs[key])):
                lo = kwargs[key][i]['lo']
                hi = kwargs[key][i]['hi']
                value = np.random.uniform(lo, hi)
                task[key] = task.get(key, []) + [value]

    def _sample_gaussian(self, task, kwargs):
        """Sample parameters using a Gaussian distribution."""
        for key in ['mass', 'friction', 'damping']:
            for i in range(len(kwargs[key])):
                mean = kwargs[key][i]['mean']
                std = kwargs[key][i]['std']
                if mean is not None and std is not None:
                    value = np.random.normal(mean, std)
                    task[key] = task.get(key, []) + [value]
    

    
    def get_task(self):
        return self._task

    def set_task(self, task):
        self._task = copy.copy(task)
        mass, friction = task['mass'], task['friction']
        self.env.model.body_mass[1] = mass #body mass shape: (5,) the second element is the torso mass ()
        self.env.model.geom_friction[4][0] = friction



    def get_task(self):
        """Get current dynamics parameters"""
        raise NotImplementedError

    def set_task(self, *task):
        """Set dynamics parameters to <task>"""
        raise NotImplementedError
    
    def get_task_search_bounds(self):
        dim_task = len(self.get_task())
        min_task = np.empty(dim_task)
        max_task = np.empty(dim_task)
        for i in range(dim_task):
            b = self.get_search_bounds_mean(i)
            min_task[i], max_task[i] = b[0], b[1]
        return min_task, max_task

    def sample_task(self):
        """Sample random dynamics parameters"""
        if self.sampling == 'uniform':
            return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

        elif self.sampling == 'truncnorm':
            a,b = -2, 2
            sample = []

            for i, (mean, std) in enumerate(zip(self.mean_task, self.stdev_task)):
                lower_bound = self.get_task_lower_bound(i) if hasattr(self, 'get_task_lower_bound') else -np.inf
                upper_bound = self.get_task_upper_bound(i) if hasattr(self, 'get_task_upper_bound') else np.inf

                attempts = 0
                obs = truncnorm.rvs(a, b, loc=mean, scale=std)
                while ( (obs < lower_bound) or (obs > upper_bound) ):
                    if attempts > 5:
                        obs = lower_bound if obs < lower_bound else upper_bound  # Clip value to its corresponding bound

                    obs = truncnorm.rvs(a, b, loc=mean, scale=std)
                    attempts += 1

                sample.append( obs )

            return np.array(sample)

        elif self.sampling == 'gaussian':
            sample = []

            for mean, std in zip(self.mean_task, self.stdev_task):
                lower_bound = self.get_task_lower_bound(i) if hasattr(self, 'get_task_lower_bound') else -np.inf
                upper_bound = self.get_task_upper_bound(i) if hasattr(self, 'get_task_upper_bound') else np.inf

                attempts = 0
                obs = np.random.randn()*std + mean
                while ( (obs < lower_bound) or (obs > upper_bound) ):
                    if attempts > 5:
                        obs = lower_bound if obs < lower_bound else upper_bound  # Clip value to its corresponding bound
                    obs = np.random.randn()*std + mean

                    attempts += 1

                sample.append( obs )

            return np.array(sample)

        elif self.sampling == 'fullgaussian':
            # Assumes that mean_task and cov_task are work in a normalized space [0, 4]
            sample = np.random.multivariate_normal(self.mean_task, self.cov_task)
            sample = np.clip(sample, 0, 4)

            sample = self.denormalize_parameters(sample)
            return sample

        elif self.sampling == 'multivariateGaussian':
            # Assumes the sampled parameters have been linearly normalized with slope
            # from [self.min_task, self.max_task] to [0, 1]
            ndims = self.mean_task.shape[0]

            valid = False
            while not valid:
                sample = np.random.multivariate_normal(self.mean_task, self.cov_task)

                sample = self._denormalize_parameters_multivariateGaussian(sample) # denormalize before checking boundaries

                # Check whether all values are within the search space (truncated multivariate gaussian)
                valid = np.all( np.concatenate([np.greater(sample, self.distr_low_bound),np.less(sample, self.distr_high_bound)]) )
            
            return sample


        elif self.sampling == 'beta':
            sample = []
            for i in range(len(self.distr)):
                m, M = self.distr[i]['m'], self.distr[i]['M']
                value = self.to_distr[i].sample()*(M - m) + m
                sample.append(value.item())

            return np.array(sample)
        else:
            raise ValueError('sampling value of random env needs to be set before using sample_task() or set_random_task(). Set it by uploading a DR distr.')

        return 
    

class MLAutoRandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        print("RESETTING")
        obs = self.env.reset(seed=seed)
        task = self.sample_task()
        self.set_task(task)
        #self.randomize()
        return obs
    
    """
    def randomize(self):
        ## TMP/HACKY, only for Hopper
        #For random samples from the normal distribution with mean mu and standard deviation sigma, use:
        #sigma * np.random.randn(...) + mu

        while True: 
            self.torso_mass = np.random.randn()*1.5 + 6
            if self.torso_mass >= 0.5 and self.torso_mass <= 9.5:
                self.foot_friction = np.random.randn()*0.25 + 2
                if self.foot_friction >= 1.5 and self.foot_friction <= 2.5:
                    break
        self._set_params(self.torso_mass, self.foot_friction)

    def _set_params(self, mass, friction):
        self.env.model.body_mass[1] = mass
        self.env.model.geom_friction[4][0] = friction
    """

    def sample_task(self):
        while True: 
            self.torso_mass = np.random.randn()*1.5 + 6
            if self.torso_mass >= 0.5 and self.torso_mass <= 9.5:
                self.foot_friction = np.random.randn()*0.25 + 2
                if self.foot_friction >= 1.5 and self.foot_friction <= 2.5:
                    break
        task = {'mass': self.torso_mass, 'friction': self.foot_friction}
        return task
    
    def get_task(self):
        return self._task

    def set_task(self, task):
        self._task = copy.copy(task)
        mass, friction = task['mass'], task['friction']
        self.env.model.body_mass[1] = mass #body mass shape: (5,) the second element is the torso mass ()
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
