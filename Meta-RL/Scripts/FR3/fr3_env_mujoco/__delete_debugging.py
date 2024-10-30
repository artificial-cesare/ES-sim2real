"""
from os import path
import numpy as np
from fr3_robot.fr3_env import FrankaFR3Robot

env = FrankaFR3Robot(render_mode='human')
import time
st = time.time()
while time.time()-st < 5.0:
    env.step(env.action_space.sample())
    
    joints = env.model.name_actuatoradr
    actuators = env.model.name_jntadr
    print(joints)
    print(actuators)
    
    ee_pos = env.data.body("right_finger").xpos.copy()  # Gets the EE position 3, 
    ee_quat = env.data.body("right_finger").xquat.copy()  # Gets the EE orientation as a quaternion 4, 
    print("EE pos: ", ee_pos)
    print("EE quat: ", ee_quat)
    
    env.render()
env.close()
"""

from os import path
import numpy as np
from fr3_robot.fr3_env import FrankaFR3Robot
import time
import mujoco

def get_joint_and_actuator_names(duration=5.0):
    # Initialize the robot environment with rendering enabled
    env = FrankaFR3Robot(render_mode='human')
    
    # Lists to store joint and actuator names
    joint_names = []
    actuator_names = []
    
    # Start timing
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Take a random action in the environment
            env.step(env.action_space.sample())
            
            # Access the model from the environment
            model = env.model
            
            # Retrieve joint names using MuJoCo's API
            for j in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if name and name not in joint_names:
                    joint_names.append(name)
            
            # Retrieve actuator names using MuJoCo's API
            for a in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                if name and name not in actuator_names:
                    actuator_names.append(name)
            
            # Render the environment
            env.render()
            
            # Optional: Add a small sleep to reduce CPU usage
            time.sleep(0.01)
    
    finally:
        # Ensure the environment is closed properly
        env.close()
    
    return joint_names, actuator_names

# Example usage
if __name__ == "__main__":
    joints, actuators = get_joint_and_actuator_names(duration=5.0)
    print("Joints:", joints)
    print("Actuators:", actuators)