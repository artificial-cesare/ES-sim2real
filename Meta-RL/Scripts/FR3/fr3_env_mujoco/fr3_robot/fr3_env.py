"""
Modified from gymnasium-robotics FrankaRobotEnv to use the Franka FR3 robot instead of the older Panda robot.
"""

# TODO: add argument in constructor for cartesian vs joint control; 
# and in any case, add xyz position and orientation of EE using forward kinematics (possibly also with noise)

from os import path
import os
import sys 

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from robotics_scripts.mujoco_utils import MujocoModelNames, robot_get_obs

from robotics_scripts.__goal_mujoco_env import GoalMujocoEnv

MAX_CARTESIAN_DISPLACEMENT = 0.2
MAX_ROTATION_DISPLACEMENT = 0.5

DEFAULT_CAMERA_CONFIG = {
    "distance": 2,
    "azimuth": 90.0, 
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.4]), #np.array([-0.2, 0.5, 2.0]),
}

class FrankaFR3Robot(GoalMujocoEnv):

    def __init__(
        self,
        model_path="fr3_w_hand.xml",
        frame_skip=40,
        robot_noise_ratio: float = 0.01,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(__file__),
            model_path,
        )

        super().__init__(
            xml_file_path,
            frame_skip,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        """
        initialize goalmujocoenv: 
        model_path: str,
        frame_skip: int,
        render_mode: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        max_geom: int = 1000,
        visual_options: Dict[int, bool] = {},
        """

        #self.init_qpos = self.data.qpos
        #self.init_qvel = self.data.qvel

        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float64) # this is the observation space in fetch reach
        #self.model_names = MujocoModelNames(self.model) # test with the imported fn as well
        
        self.robot_noise_ratio = robot_noise_ratio
        self.robot_pos_bound = np.zeros([len(self.joint_names), 2], dtype=float)
        self.robot_vel_bound = np.ones([len(self.joint_names), 2], dtype=float)
        self.robot_pos_noise_amp = 0.1 * np.ones(len(self.joint_names), dtype=float)
        self.robot_vel_noise_amp = 0.1 * np.ones(len(self.joint_names), dtype=float)

        self.robot_vel_bound[:, 0] = -10.0
        self.robot_vel_bound[:, 1] = 10.0

        for i in range(len(self.joint_names)):
            self.robot_pos_bound[i, 0] = self.model.joint(name=self.joint_names[i]).range[0]
            self.robot_pos_bound[i, 1] = self.model.joint(name=self.joint_names[i]).range[1]

        self.act_mid = np.mean(self.robot_pos_bound, axis=1)
        self.act_rng = 0.5 * (self.robot_pos_bound[:, 1] - self.robot_pos_bound[:, 0])


    """
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    """

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Denormalize the input action from [-1, 1] range to the each actuators control range
        action = self.act_mid + action * self.act_rng

        # enforce velocity limits
        ctrl_feasible = self._ctrl_velocity_limits(action)
        # enforce position limits
        ctrl_feasible = self._ctrl_position_limits(ctrl_feasible)

        self.do_simulation(ctrl_feasible, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()

        return obs, 0.0, False, False, {}

    def _get_obs(self):
        # Gather simulated observation
        if self.data.qpos is not None and self.joint_names:
            robot_qpos = np.squeeze(np.array([self.data.joint(name).qpos for name in self.joint_names]))
            robot_qvel = np.squeeze(np.array([self.data.joint(name).qvel for name in self.joint_names]))
        else:
            robot_qpos = np.zeros(8) # find info at https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#data
            robot_qvel = np.zeros(8)

        # Simulate observation noise
        robot_qpos += (
            self.robot_noise_ratio
            * self.robot_pos_noise_amp
            * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qpos.shape)
        )
        robot_qvel += (
            self.robot_noise_ratio
            * self.robot_vel_noise_amp
            * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qvel.shape)
        )

        self._last_robot_qpos = robot_qpos

        return np.concatenate((robot_qpos.copy(), robot_qvel.copy()))

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        obs = self._get_obs()

        return obs

    def _ctrl_velocity_limits(self, ctrl_velocity: np.ndarray):
        """Enforce velocity limits and estimate joint position control input (to achieve the desired joint velocity).

        ALERT: This depends on previous observation. This is not ideal as it breaks MDP assumptions. This is the original
        implementation from the D4RL environment: https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/kitchen/adept_envs/franka/robot/franka_robot.py#L259.

        Args:
            ctrl_velocity (np.ndarray): environment action with space: Box(low=-1.0, high=1.0, shape=(9,))

        Returns:
            ctrl_position (np.ndarray): input joint position given to the MuJoCo simulation actuators.
        """
        ctrl_feasible_vel = np.clip(
            ctrl_velocity, self.robot_vel_bound[:9, 0], self.robot_vel_bound[:9, 1]
        )
        ctrl_feasible_position = self._last_robot_qpos + ctrl_feasible_vel * self.dt
        return ctrl_feasible_position

    def _ctrl_position_limits(self, ctrl_position: np.ndarray):
        """Enforce joint position limits.

        Args:
            ctrl_position (np.ndarray): unbounded joint position control input .

        Returns:
            ctrl_feasible_position (np.ndarray): clipped joint position control input.
        """
        ctrl_feasible_position = np.clip(
            ctrl_position, self.robot_pos_bound[:9, 0], self.robot_pos_bound[:9, 1]
        )
        return ctrl_feasible_position