## new code for goal-based environments for the fr3 robot

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space



from mujoco_env import MujocoEnv
from .core import GoalEnv
from robotics_scripts import rotations
from typing import Optional, Dict, Union, Tuple, Any
import numpy as np

class GoalMujocoEnv(GoalEnv, MujocoEnv):
    """
    A goal-based environment that uses MuJoCo for simulation.

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
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 12,
    }

    def __init__(
        self,
        seed,
        model_path: str,
        frame_skip: int,
        has_object: bool,
        n_substeps: int, #n_substeps (integer): number of MuJoCo simulation timesteps per Gymnasium step.
        observation_space: Optional[Space],
        render_mode: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        max_geom: int = 1000,
        visual_options: Dict[int, bool] = {},
        robot_noise_ratio: float = 0.01, #robot_noise_ratio (float): ratio of noise to add to robot observations.
    ):
        # Initialize MujocoEnv
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=Optional[Space],  # Will be set later
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

        self.goal = np.zeros(0)

        self.has_object = has_object
        self.joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7', 'finger_joint1']
        self.n_substeps = n_substeps
        self.robot_noise_ratio = robot_noise_ratio
        self.np_random = np.random.RandomState(seed=seed)

        ## might need to change as defined in franka_config.xml
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
        

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float64)
        obs = self._get_obs()

        # Define the observation space as required by GoalEnv
        self._set_observation_space(obs)

    def _get_obs(self):
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

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
    
    def generate_mujoco_observations(self):
    # Updated from fetch to fr3, the grip site name from "robot0:grip" to "hand_c"
        grip_site_name = "hand_c"
        grip_pos = self._utils.get_site_xpos(self.model, self.data, grip_site_name)

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = self._utils.get_site_xvelp(self.model, self.data, grip_site_name) * dt

        # Update joint names to match the new fr3 model
        # Include all relevant joints, especially the finger joints
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self.joint_names
        )

        if self.has_object:
            # Ensure the object site name matches the new model
            # If the object site has a different name, update it accordingly
            object_site_name = "object0"
            object_pos = self._utils.get_site_xpos(self.model, self.data, object_site_name)
            
            # Rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, object_site_name)
            )
            
            # Velocities
            object_velp = self._utils.get_site_xvelp(self.model, self.data, object_site_name) * dt
            object_velr = self._utils.get_site_xvelr(self.model, self.data, object_site_name) * dt
            
            # Gripper state relative to the object
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        # Assuming the last two joints are the finger joints
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # Optionally, average if symmetric

        # **Add Noise to Observations**

        # Define a helper function to add noise
        def add_noise(obs):
            if obs.size > 0:
                noise = self.robot_noise_ratio * self.robot_pos_noise_amp * \
                        self.np_random.uniform(low=-1.0, high=1.0, size=obs.shape)
                return obs + noise
            return obs

        # Apply noise to each observation component
        grip_pos_noisy = add_noise(grip_pos)
        object_pos_noisy = add_noise(object_pos)
        object_rel_pos_noisy = add_noise(object_rel_pos)
        gripper_state_noisy = add_noise(gripper_state)
        object_rot_noisy = add_noise(object_rot)
        object_velp_noisy = add_noise(object_velp)
        object_velr_noisy = add_noise(object_velr)
        grip_velp_noisy = add_noise(grip_velp)
        gripper_vel_noisy = add_noise(gripper_vel)

        return (
            grip_pos_noisy,
            object_pos_noisy,
            object_rel_pos_noisy,
            gripper_state_noisy,
            object_rot_noisy,
            object_velp_noisy,
            object_velr_noisy,
            grip_velp_noisy,
            gripper_vel_noisy,
        )

    def _set_observation_space(self, obs):
        """Set up the observation space according to GoalEnv specifications."""
        # Example observation space, modify according to your environment's requirements
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the environment."""
        super().reset(seed=seed)
        # Enforce the observation space structure
        if not isinstance(self.observation_space, spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error(
                    f'GoalEnv requires the "{key}" key to be part of the observation dictionary.'
                )

        # Reset Mujoco simulation
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    def reset_model(self) -> Dict[str, np.ndarray]:
        """Reset the MuJoCo simulation and return the initial observation."""
        # Implement environment-specific reset logic here
        # For example:
        self.set_state(self.init_qpos, self.init_qvel)
        observation = self._get_obs()
        return observation

    def _get_achieved_goal(self) -> np.ndarray:
        """Get the current achieved goal."""
        # Implement logic to obtain the achieved goal
        return self.data.qpos.flat.copy()

    def _get_desired_goal(self) -> np.ndarray:
        """Get the desired goal."""
        # Implement logic to obtain the desired goal
        return self.goal.copy()

    @property
    def goal(self):
        """Define the desired goal."""
        # Replace with actual goal logic
        return self.init_qpos.copy()

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (dictionary): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (integer): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state. This is calculated by :meth:`compute_terminated` of `GoalEnv`.
            truncated (boolean): Whether the truncation condition outside the scope of the MDP is satisfied. Timically, due to a timelimit, but
            it is also calculated in :meth:`compute_truncated` of `GoalEnv`.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). In this case there is a single
            key `is_success` with a boolean value, True if the `achieved_goal` is the same as the `desired_goal`.
        """
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action, from mujoco env
        self.do_simulation(action, self.frame_skip)
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        
        # Compute termination conditions
        terminated = self.compute_terminated(obs['achieved_goal'], obs['desired_goal'], {})
        truncated = self.compute_truncated(obs['achieved_goal'], obs['desired_goal'], {})
        
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes it dependent on a desired goal and the one that was achieved.

        If you wish to include additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        # Implement the reward function
        # Example: negative distance between achieved and desired goal
        return -np.linalg.norm(achieved_goal - desired_goal)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Compute the step termination. Allows to customize the termination states depending on the desired and the achieved goal.

        If you wish to determine termination states independent of the goal, you can include necessary values to derive it in 'info'
        and compute it accordingly. The envirtonment reaches a termination state when this state leads to an episode ending in an episodic
        task thus breaking .

        More information can be found in: https://farama.org/New-Step-API#theory

        Termination states are

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            bool: The termination state that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert terminated == env.compute_terminated(ob['achieved_goal'], ob['desired_goal'], info)
        """
        # Implement termination logic
        # Example: terminate when the goal is achieved within a threshold
        return np.linalg.norm(achieved_goal - desired_goal) < 0.05

    def compute_truncated(self, achieved_goal, desired_goal, info):
        """Compute the step truncation. Allows to customize the truncated states depending on the desired and the achieved goal.

        If you wish to determine truncated states independent of the goal, you can include necessary values to derive it in 'info'
        and compute it accordingly. Truncated states are those that are out of the scope of the Markov Decision Process (MDP) such
        as time constraints in a continuing task.

        More information can be found in: https://farama.org/New-Step-API#theory

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            bool: The truncated state that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert truncated == env.compute_truncated(ob['achieved_goal'], ob['desired_goal'], info)
        """
        # Implement truncation logic
        # Example: truncate after a fixed number of steps
        return False  # Or implement time-based truncation

    def _get_reset_info(self) -> Dict[str, Any]:
        """Provide additional information upon resetting."""
        return {}

    def render(self):
        """Render the environment."""
        return MujocoEnv.render(self)

    def close(self):
        """Close the environment."""
        MujocoEnv.close(self)

