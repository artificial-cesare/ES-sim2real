# ___delete_test_env.py

import sys
import os

# Adjust the path to include the directory containing frankafr3robot.py and goal_mujoco_env.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fr3_robot')))

from fr3_test import FrankaFR3Robot

# Instantiate the environment with default parameters and render_mode set to 'human'
env = FrankaFR3Robot(render_mode='human')

# Reset the environment
observation, info = env.reset()

import time

st = time.time()
i = 0
action = env.action_space.sample()
while time.time() - st < 10:
    if i % 12 == 0:
        action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    i += 1

env.close()
