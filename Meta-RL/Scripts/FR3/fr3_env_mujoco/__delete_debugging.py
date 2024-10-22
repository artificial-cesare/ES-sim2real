from os import path
import numpy as np
from fr3_robot.fr3_env import FrankaFR3Robot

env = FrankaFR3Robot(render_mode='human')
import time
st = time.time()
while time.time()-st < 5.0:
    env.step(env.action_space.sample())
    ee_pos = env.data.body("right_finger").xpos.copy()  # Gets the EE position 3, 
    ee_quat = env.data.body("right_finger").xquat.copy()  # Gets the EE orientation as a quaternion 4, 
    print("EE pos: ", ee_pos)
    print("EE quat: ", ee_quat)
    env.render()
env.close()