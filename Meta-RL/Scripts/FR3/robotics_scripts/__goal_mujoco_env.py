## new code for goal-based environments for the fr3 robot

import gym
from gym import spaces, error
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

        # Define the observation space as required by GoalEnv
        self._set_observation_space()

    def _set_observation_space(self):
        """Set up the observation space according to GoalEnv specifications."""
        # Example observation space, modify according to your environment's requirements
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32) # nq: n of position coordinates, nv: n of degrees of freedom
        goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'observation': obs_space,
            'achieved_goal': goal_space,
            'desired_goal': goal_space,
        })

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
        """Compute the reward for the current step."""
        # Implement the reward function
        # Example: negative distance between achieved and desired goal
        return -np.linalg.norm(achieved_goal - desired_goal)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Determine whether the episode has terminated."""
        # Implement termination logic
        # Example: terminate when the goal is achieved within a threshold
        return np.linalg.norm(achieved_goal - desired_goal) < 0.05

    def compute_truncated(self, achieved_goal, desired_goal, info):
        """Determine whether the episode has been truncated."""
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
