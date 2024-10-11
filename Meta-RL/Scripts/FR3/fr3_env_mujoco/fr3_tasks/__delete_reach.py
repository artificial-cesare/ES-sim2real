#test env for reach task with fr3 robot

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fr3_robot')))

from fr3_reach import FR3Reach

env = FR3Reach(render_mode='human')
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