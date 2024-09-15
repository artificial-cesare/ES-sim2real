import gymnasium as gym
import numpy as np


"""
make wrapper 'make metarlenv' that adds the methods   sample_task and set_task

    set_mass_fn = env.get_attr('_set_mass')

    for i in range(len(set_mass_fn)):
        set_mass_fn[i](train_env_mass)
    print(env.get_attr('model')[0].body_mass)

    or    env.env_method('method name', args...)
            env.env_method('_set_mass', train_env_mass)


class ManuallyRandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def randomize(self):
        min_mass, max_mass = (1,9)

        self.torso_mass = self.env.np_random.random()*(max_mass-min_mass) + min_mass
        self._set_mass(self.torso_mass)

    def _set_mass(self, mass):
        self.env.model.body_mass[1] = mass
        # env.model.geom_friction[4][0]=2

class AutoRandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.randomize()
        return obs

    def randomize(self):
        ## TMP/HACKY, only for Hopper

        self.torso_mass = self.env.np_random.standard_normal()*1.5 + 6
        self.torso_mass = np.clip(self.torso_mass, 0.5, 9.5)
        self._set_mass(self.torso_mass)

    def _set_mass(self, mass):
        self.env.model.body_mass[1] = mass
"""



