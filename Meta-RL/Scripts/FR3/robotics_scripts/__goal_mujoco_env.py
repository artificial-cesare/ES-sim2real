## new code for goal-based environments for the fr3 robot

import gymnasium as gym
from gymnasium import spaces, error
import mujoco
from .mujoco_env import MujocoEnv
from .core import GoalEnv
from typing import Optional, Dict, Union, Tuple, Any
import numpy as np

class GoalMujocoEnv(GoalEnv, MujocoEnv):
    """A goal-based environment that uses MuJoCo for simulation."""

    def __init__(
        self,
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
    ):
        # Initialize MujocoEnv
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=None,  # Will be set later
            render_mode=render_mode,
            width=width,
            height=height,
            camera_id=camera_id,
            camera_name=camera_name,
            default_camera_config=default_camera_config,
            max_geom=max_geom,
            visual_options=visual_options,
        )

        self._initialize_simulation()

        self.goal = np.zeros(0)
        obs = self._get_obs()

        # Define the observation space as required by GoalEnv
        self._set_observation_space(obs)

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
    
    def _initialize_simulation(self):
        self.model = self._mujoco.MjModel.from_xml_path(self.model_path)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

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

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        # Replace with actual observation logic
        observation = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        achieved_goal = self._get_achieved_goal()
        desired_goal = self._get_desired_goal()
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }

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
        """Perform a simulation step."""
        # Apply action
        self.do_simulation(action, self.frame_skip)
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        
        # Compute termination conditions
        terminated = self.compute_terminated(obs['achieved_goal'], obs['desired_goal'], {})
        truncated = self.compute_truncated(obs['achieved_goal'], obs['desired_goal'], {})
        
        info = {}
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
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

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

def _get_gripper_xpos(self):
    body_id = self._model_names.body_name2id["robot0:gripper_link"]
    return self.data.xpos[body_id]