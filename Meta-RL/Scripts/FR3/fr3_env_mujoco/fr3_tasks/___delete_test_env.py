## Simple test to create a franka-fr3 env, render it, and then train with sb3 SAC

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fr3_robot')))

from fr3_env import FrankaFR3Robot

env = FrankaFR3Robot(render_mode='human')
env.reset()

import time
st = time.time()
i = 0
action = env.action_space.sample()
while time.time()-st < 10:
    if i%12==0:
        action = env.action_space.sample()
    env.step(action)
    i+=1