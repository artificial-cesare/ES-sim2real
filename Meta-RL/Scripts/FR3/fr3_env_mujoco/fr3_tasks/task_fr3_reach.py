# adapted from reach.py in fetch envs from gym-robotics

import os
import sys 
import mujoco 

from gymnasium import spaces
from gymnasium_robotics.core import GoalEnv

import numpy as np

# append parent directory to import the robot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fr3_robot.fr3_env import FrankaFR3Robot

TASK_XML_PATH = "task_fr3_reach.xml"



BONUS_THRESH = 0.3

class FR3Reach(FrankaFR3Robot, GoalEnv):
    def __init__(
        self,
        **kwargs,
    ):
        self.robot_env = FrankaFR3Robot(
            model_path= TASK_XML_PATH,
            **kwargs,
        )

        self.model = self.robot_env.model
        self.data = self.robot_env.data
        self.render_mode = self.robot_env.render_mode

        self.goal = [0,0,0] # placeholder for the goal, randomizes at reset

        #self.object_noise_ratio = (
        #    object_noise_ratio  # stochastic noise added to the object observations
        #)

        obs = self._get_obs()

        self.action_space = self.robot_env.action_space
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                                -np.inf,
                                np.inf,
                                shape=(3,),
                                dtype="float64",
                            ),
                achieved_goal=spaces.Box(
                                -np.inf,
                                np.inf,
                                shape=(3,),
                                dtype="float64",
                            ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info,
    ):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        complete = distance < BONUS_THRESH
        if complete:
            return 0.0
        return -distance

    def _get_obs(self):
        robot_obs = self.robot_env._get_obs()

        """
        obj_qpos = self.data.qpos[9:].copy()
        obj_qvel = self.data.qvel[9:].copy()

        # Simulate observation noise
        obj_qpos += (
            self.object_noise_ratio
            * self.robot_env.robot_pos_noise_amp[8:]
            * self.robot_env.np_random.uniform(low=-1.0, high=1.0, size=obj_qpos.shape)
        )
        obj_qvel += (
            self.object_noise_ratio
            * self.robot_env.robot_vel_noise_amp[9:]
            * self.robot_env.np_random.uniform(low=-1.0, high=1.0, size=obj_qvel.shape)
        )
        

        # For a 'reaching' task, the achieved goal is the position of the end effector
        achieved_goal = self.data.body("hand").xpos.copy()[:3] # really the franka wrist, but this is just a proof-of-concept task env
        """
        # try with fingers 
        achieved_goal = self.data.body("right_finger").xpos.copy()[:3]

        obs = {
            "observation": robot_obs, #np.concatenate((robot_obs, task_obs)),
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal,
        }

        return obs

    def step(self, action):
        robot_obs, _, terminated, truncated, info = self.robot_env.step(action)
        obs = self._get_obs()

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        if reward >= - BONUS_THRESH:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        distance = self.np_random.uniform(low=0.2, high=0.7)
        angle = self.np_random.uniform(low=-0.8*np.pi, high=0.8*np.pi)
        # shouldn't we be able to set the goal manually? 
        self.goal[0] = distance * np.cos(angle)
        self.goal[1] = distance * np.sin(angle)
        self.goal[2] = self.np_random.uniform(low=0.1, high=0.4)

        robot_obs, _ = self.robot_env.reset(seed=seed)
        obs = self._get_obs()

        info = {}

        # only for rendering purposes
        self.model.body("target").pos = self.goal.copy()
        mujoco.mj_forward(self.model, self.data)

        return obs, info

    def render(self):
        return self.robot_env.render()

    def close(self):
        self.robot_env.close()