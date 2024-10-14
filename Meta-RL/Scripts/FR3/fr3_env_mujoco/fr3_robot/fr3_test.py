# goal_mujoco_env.py

from os import path
import os
import sys

from typing import Optional, Dict, Union
import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray

# Adjust the system path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from robotics_scripts.core import GoalEnv

# mujoco_env.py

from os import path
import os
import sys 

from typing import Optional, Dict, Union, Tuple
import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space


try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e


DEFAULT_SIZE = 480


def expand_model_path(model_path: str) -> str:
    """Expands the `model_path` to a full path if it starts with '~' or '.' or '/'."""
    if model_path.startswith(".") or model_path.startswith("/"):
        fullpath = model_path
    elif model_path.startswith("~"):
        fullpath = path.expanduser(model_path)
    else:
        fullpath = path.join(path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise OSError(f"File {fullpath} does not exist")

    return fullpath


from os import path
import os
import sys 

from typing import Optional, Dict, Union, Tuple
import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space


try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e


DEFAULT_SIZE = 480


def expand_model_path(model_path: str) -> str:
    """Expands the `model_path` to a full path if it starts with '~' or '.' or '/'."""
    if model_path.startswith(".") or model_path.startswith("/"):
        fullpath = model_path
    elif model_path.startswith("~"):
        fullpath = path.expanduser(model_path)
    else:
        fullpath = path.join(path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise OSError(f"File {fullpath} does not exist")

    return fullpath


class MujocoEnv(gym.Env):
    """Superclass for MuJoCo based environments."""

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Optional[Space],
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        max_geom: int = 1000,
        visual_options: Dict[int, bool] = {},
    ):
        """Base abstract class for mujoco based environments.

        Args:
            model_path: Path to the MuJoCo Model.
            frame_skip: Number of MuJoCo simulation steps per gym `step()`.
            observation_space: The observation space of the environment.
            render_mode: The `render_mode` used.
            width: The width of the render window.
            height: The height of the render window.
            camera_id: The camera ID used.
            camera_name: The name of the camera used (can not be used in conjunction with `camera_id`).
            default_camera_config: configuration for rendering camera.
            max_geom: max number of rendered geometries.
            visual_options: render flag options.

        Raises:
            OSError: when the `model_path` does not exist.
            error.DependencyNotInstalled: When `mujoco` is not installed.
        """
        # Directly set the fullpath to the XML model to avoid path expansion issues
        self.fullpath = expand_model_path(model_path)

        self.width = width
        self.height = height
        # Initialize simulation
        self.model, self.data = self._initialize_simulation()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        # Ensure metadata is set correctly
        assert hasattr(self, 'metadata'), "Subclass must define 'metadata' before calling MujocoEnv.__init__."
        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'
        if observation_space is not None:
            self.observation_space = observation_space
        self._set_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        # Initialize MujocoRenderer with keyword arguments to match expected signature
        self.mujoco_renderer = MujocoRenderer(
            model=self.model,
            data=self.data,
            camera_config=default_camera_config,  # Pass as keyword
            width=self.width,                      # Pass as keyword
            height=self.height,                    # Pass as keyword
            max_geom=max_geom,                     # Pass as keyword
            camera_id=camera_id,                   # Pass as keyword
            camera_name=camera_name,               # Pass as keyword
            visual_options=visual_options,         # Pass as keyword
        )

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _initialize_simulation(
        self,
    ) -> Tuple["mujoco.MjModel", "mujoco.MjData"]:
        """
        Initialize MuJoCo simulation data structures `MjModel` and `MjData`.
        """
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        # Copy visual settings
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def set_state(self, qpos, qvel):
        """Set the joints position qpos and velocity qvel of the model.

        Note: `qpos` and `qvel` is not the full physics state for all mujoco models/environments see https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulation.
        """
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        """Close rendering contexts processes."""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name):
        """Return the cartesian position of a body frame."""
        return self.data.body(body_name).xpos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

    def state_vector(self) -> NDArray[np.float64]:
        """Return the position and velocity joint states of the model.

        Note: `qpos` and `qvel` does not constitute the full physics state for all `mujoco` environments see https://mujoco.readthedocs.io/en/stable/computation/index.html#the-state.
        """
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    # methods to override:
    # ----------------------------
    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        raise NotImplementedError

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each environment subclass.
        """
        raise NotImplementedError

    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

from os import path
import os
import sys

from typing import Optional, Dict, Union
import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray

# Adjust the system path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from robotics_scripts.core import GoalEnv

class GoalMujocoEnv(GoalEnv, MujocoEnv):
    """
    A goal-based environment that uses MuJoCo for simulation.
    """
    
    # Define metadata as a class attribute
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 12,  # Example FPS; adjust as needed
    }
    
    def __init__(
        self,
        seed: int = 42,
        model_path: str = "fr3_w_hand.xml",
        frame_skip: int = 40,
        has_object: bool = True,
        n_substeps: int = 10,
        observation_space: Optional[Space] = None,
        render_mode: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        max_geom: int = 1000,
        visual_options: Dict[int, bool] = {},
        robot_noise_ratio: float = 0.01,  # Additional parameter
    ):
        """
        Initialize the GoalMujocoEnv.
        """
        # Initialize MujocoEnv using keyword arguments only
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_id=camera_id,
            camera_name=camera_name,
            default_camera_config=default_camera_config,
            max_geom=max_geom,
            visual_options=visual_options,
        )
        
        # Initialize GoalEnv
        GoalEnv.__init__(self)
        
        self.seed = seed
        self.has_object = has_object
        self.n_substeps = n_substeps
        self.robot_noise_ratio = robot_noise_ratio  # Store as instance variable
        self.np_random = np.random.RandomState(seed=seed)
        
        # Initialize goal (to be set appropriately in subclasses)
        self.goal = np.zeros(0)
        
        # If observation_space is still None after MujocoEnv initialization, define it
        if self.observation_space is None:
            # Define a default observation space or raise an error
            raise ValueError("Observation space must be provided by the subclass.")

    # Implement abstract methods from GoalEnv
    def compute_reward(self, achieved_goal: NDArray[np.float64], desired_goal: NDArray[np.float64], info: Dict[str, float]) -> float:
        """
        Compute the step reward. Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def compute_terminated(self, achieved_goal: NDArray[np.float64], desired_goal: NDArray[np.float64], info: Dict[str, float]) -> bool:
        """
        Compute whether the episode should terminate. Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def compute_truncated(self, achieved_goal: NDArray[np.float64], desired_goal: NDArray[np.float64], info: Dict[str, float]) -> bool:
        """
        Compute whether the episode should be truncated. Must be implemented by subclasses.
        """
        raise NotImplementedError

# fr3_test.py

from typing import Optional, Dict, Union, Tuple, Any
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Space

from robotics_scripts import rotations  # Ensure this module is correctly accessible

DEFAULT_CAMERA_CONFIG = {
    "distance": 2,
    "azimuth": 90.0, 
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.4]),  # Adjust as needed
}

class FrankaFR3Robot(GoalMujocoEnv):
    """
    Modified from gymnasium-robotics FrankaRobotEnv to use the Franka FR3 robot instead of the older Panda robot.
    """
    
    def __init__(
        self,
        seed: int = 42,
        model_path: str = "fr3_w_hand.xml",
        frame_skip: int = 40,
        has_object: bool = True,
        n_substeps: int = 10,
        observation_space: Optional[Space] = None,
        render_mode: Optional[str] = 'human',  # Default render_mode set to 'human'
        width: int = 640,
        height: int = 480,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = DEFAULT_CAMERA_CONFIG,
        max_geom: int = 1000,
        visual_options: Dict[int, bool] = {},
        robot_noise_ratio: float = 0.01,
    ):
        # Define joint names before calling super().__init__
        self.joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3',
            'fr3_joint4', 'fr3_joint5', 'fr3_joint6',
            'fr3_joint7', 'finger_joint1'
        ]
        
        # If observation_space is not provided, define it based on expected observations
        if observation_space is None:
            # Example: Define observation space as a Dict with 'observation', 'achieved_goal', 'desired_goal'
            # Adjust the shapes and bounds according to your specific environment
            # Here, we assume:
            # - observation: concatenated sensor data (adjust dimensions as needed)
            # - achieved_goal: position (3D)
            # - desired_goal: position (3D)
            observation_dim = 8 + 8  # Example dimensions; adjust as needed
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float64),
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            })
        
        # Initialize the GoalMujocoEnv superclass with keyword arguments
        super().__init__(
            seed=seed,
            model_path=model_path,
            frame_skip=frame_skip,
            has_object=has_object,
            n_substeps=n_substeps,
            observation_space=self.observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_id=camera_id,
            camera_name=camera_name,
            default_camera_config=default_camera_config,
            max_geom=max_geom,
            visual_options=visual_options,
            robot_noise_ratio=robot_noise_ratio,
        )
        
        # Initialize robot-specific parameters
        self.robot_vel_bound = np.ones((len(self.joint_names), 2), dtype=float)
        self.robot_pos_noise_amp = {
            'grip_pos': np.array([0.01, 0.01, 0.01]),
            'object_pos': np.array([0.01, 0.01, 0.01]),
            'object_rel_pos': np.array([0.005, 0.005, 0.005]),
            'gripper_state': np.array([0.02, 0.02]),
            'object_rot': np.array([0.05, 0.05, 0.05]),
            'object_velp': np.array([0.02, 0.02, 0.02]),
            'object_velr': np.array([0.05, 0.05, 0.05]),
            'grip_velp': np.array([0.02, 0.02, 0.02]),
            'gripper_vel': np.array([0.02, 0.02]),
        }
        self.robot_vel_bound[:, 0] = -10.0
        self.robot_vel_bound[:, 1] = 10.0
    
        # Retrieve joint position limits from the model
        self.robot_pos_bound = np.zeros((len(self.joint_names), 2), dtype=float)
        for i, joint_name in enumerate(self.joint_names):
            joint = self.model.joint(name=joint_name)
            if joint is not None and hasattr(joint, 'range') and len(joint.range) == 2:
                self.robot_pos_bound[i, 0] = joint.range[0]
                self.robot_pos_bound[i, 1] = joint.range[1]
            else:
                # Set default limits if not defined
                self.robot_pos_bound[i, 0] = -np.pi
                self.robot_pos_bound[i, 1] = np.pi
    
        # Calculate action scaling parameters based on actuator control range
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]) / 2.0
    
        # Initialize additional attributes if needed
        self._last_robot_qpos = self.init_qpos.copy()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Apply action scaling, enforce control limits, and perform a simulation step.
        """
        # Clip action to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
    
        # Denormalize the input action from [-1, 1] to the actuator control range
        action = self.act_mid + action * self.act_rng
    
        # Enforce velocity limits
        ctrl_feasible = self._ctrl_velocity_limits(action)
    
        # Enforce position limits
        ctrl_feasible = self._ctrl_position_limits(ctrl_feasible)
    
        # Pass the feasible control to the superclass step
        obs, reward, terminated, truncated, info = super().step(ctrl_feasible)
    
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Generate observations with per-component noise.
        """
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()
    
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
    
        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )
    
        # Define noise amplitudes per observation component
        noise_amp = np.concatenate([
            self.robot_pos_noise_amp['grip_pos'],
            self.robot_pos_noise_amp['object_pos'],
            self.robot_pos_noise_amp['object_rel_pos'],
            self.robot_pos_noise_amp['gripper_state'],
            self.robot_pos_noise_amp['object_rot'],
            self.robot_pos_noise_amp['object_velp'],
            self.robot_pos_noise_amp['object_velr'],
            self.robot_pos_noise_amp['grip_velp'],
            self.robot_pos_noise_amp['gripper_vel'],
        ])
    
        # Generate noise
        noise = self.robot_noise_ratio * noise_amp * self.np_random.uniform(
            low=-1.0, high=1.0, size=obs.shape
        )
    
        # Add noise to observations
        obs_noisy = obs + noise
    
        return {
            "observation": obs_noisy.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
    
    def reset_model(self) -> Dict[str, np.ndarray]:
        """
        Reset the robot's state and return the initial observation.
        """
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs
    
    def _ctrl_velocity_limits(self, ctrl_velocity: np.ndarray) -> np.ndarray:
        """
        Enforce velocity limits and estimate joint position control input.
        """
        ctrl_feasible_vel = np.clip(
            ctrl_velocity, self.robot_vel_bound[:, 0], self.robot_vel_bound[:, 1]
        )
        ctrl_feasible_position = self._last_robot_qpos + ctrl_feasible_vel * self.dt
        self._last_robot_qpos = ctrl_feasible_position.copy()
        return ctrl_feasible_position
    
    def _ctrl_position_limits(self, ctrl_position: np.ndarray) -> np.ndarray:
        """
        Enforce joint position limits.
        """
        ctrl_feasible_position = np.clip(
            ctrl_position, self.robot_pos_bound[:, 0], self.robot_pos_bound[:, 1]
        )
        return ctrl_feasible_position
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> float:
        """
        Compute the step reward based on the negative Euclidean distance between achieved and desired goals.
        """
        return -np.linalg.norm(achieved_goal - desired_goal)
    
    def compute_terminated(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> bool:
        """
        Determine if the episode should terminate based on the proximity to the desired goal.
        """
        return np.linalg.norm(achieved_goal - desired_goal) < 0.05
    
    def compute_truncated(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> bool:
        """
        Determine if the episode should be truncated. Here, it's always False, but can be modified as needed.
        """
        return False  # Modify if you have time-based or other truncation conditions
    
    def _get_reset_info(self) -> Dict[str, Any]:
        """
        Provide additional information upon resetting.
        """
        return {}
    
    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """
        Determine if the achieved goal matches the desired goal within a threshold.
        """
        return np.linalg.norm(achieved_goal - desired_goal) < 0.05
    
    def generate_mujoco_observations(self):
        """
        Generate observations from the MuJoCo simulation.
        
        Returns:
            Tuple containing various observation components.
        """
        # Example implementation; adjust based on your model's specifics
        grip_site_name = "hand_c"
        grip_pos = self.get_body_com(grip_site_name)
        
        # Example object site
        object_site_name = "object0"
        object_pos = self.get_body_com(object_site_name)
        
        # Relative position
        object_rel_pos = object_pos - grip_pos
        
        # Example velocities; adjust based on your simulation data
        grip_velp = self.data.site_xvelp[self.model.site_name2id(grip_site_name)]
        object_velp = self.data.site_xvelp[self.model.site_name2id(object_site_name)]
        object_velr = self.data.site_xvelr[self.model.site_name2id(object_site_name)]
        
        # Rotations; convert from rotation matrix to Euler angles
        object_rot = rotations.mat2euler(
            self.data.site_xmat[self.model.site_name2id(object_site_name)].reshape(3, 3)
        )
        
        # Gripper state (e.g., joint positions and velocities)
        gripper_state = self.data.qpos[-2:]
        gripper_vel = self.data.qvel[-2:]
        
        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )
