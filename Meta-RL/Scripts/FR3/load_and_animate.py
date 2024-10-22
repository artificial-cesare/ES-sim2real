import os 
import sys 
from os import path
import mujoco
import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

DEFAULT_SIZE = 480 

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "fr3_env_mujoco", "fr3_robot")))
XML_FILE = path.join(
    os.path.dirname(__file__), 
    "fr3_env_mujoco", 
    "fr3_robot", 
    "fr3_w_hand.xml")

class Animator(gym.Env):
    def __init__(self, 
                 XML_FILE, 
                 width = DEFAULT_SIZE, 
                 height = DEFAULT_SIZE,
                 frame_skip = 12
                 ):
        self.XML_FILE = XML_FILE
    # model description, i.e., all quantities which do not change over time
        self.model = mujoco.MjModel.from_xml_path(self.XML_FILE) # for reference attributes of model: https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h

        try:
            self.model.geom()
        except KeyError as e:
            print(e) # error can be used to print actual attribute names for geoms of model

    # state and quantities that depend on it. The state is made up of time, generalized positions and generalized velocities
        self.data = mujoco.MjData(self.model) # only changes once it's propagated through the simulation
        print("time:", self.data.time)
        print("qpos:", self.data.qpos)
        print("qvel:", self.data.qvel)

    # print x,y,z position of the first body
        print("pos of first body {}: {}".format(self.data.body(0).name, self.data.xpos[0]))

    # args for rendering
        self.width = width
        self.height = height

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            self.width,
            self.height,
        )


if __name__ == "__main__":
    # print the filename, if it exists
    if not os.path.exists(XML_FILE):
        print("{} File does not exist".format(XML_FILE))

    load_n_animate(XML_FILE)


