# adapted from reach.py in fetch envs from gym-robotics

import os

from gymnasium.utils.ezpickle import EzPickle
import numpy as np

import sys 
# append parent directory to import the robot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fr3_robot.fr3_env import FrankaFR3Robot

MODEL_XML_PATH = r"C:\Users\cesar\OneDrive\Documenti\GitHub\ES-sim2real\Meta-RL\Scripts\FR3\fr3_env_mujoco\fr3_robot\fr3_reach.xml"

OBS_ELEMENT_GOALS = {
    "reach": np.array([-0.88, -0.01]),
    "push": np.array([-0.92, -0.01]),
    "slide": np.array([-0.69, -0.05]),
    "pick_and_place": np.array([0.37]),
}

BONUS_THRESH = 0.3

## TODO:
# change goal
# change reward

class FR3Reach(FrankaFR3Robot, EzPickle): #GoalEnv
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
        tasks_to_complete: "list[str]" = list(OBS_ELEMENT_GOALS.keys()),
        terminate_on_tasks_completed: bool = True,
        remove_task_when_completed: bool = True,
        object_noise_ratio: float = 0.0005,
        **kwargs,
    ):
        self.robot_env = FrankaFR3Robot(
            # to change with fr3_reach.xml 
            model_path= MODEL_XML_PATH,
            **kwargs,
        )

        self.robot_env.init_qpos = np.array(
            [
                1.48388023e-01,
                -1.76848573e00,
                1.84390296e00,
                -2.47685760e00,
                2.60252026e-01,
                7.12533105e-01,
                1.59515394e00,
                4.79267505e-02,
                3.71350919e-02,
                -2.66279850e-04,
                -5.18043486e-05,
                3.12877220e-05,
                -4.51199853e-05,
                -3.90842156e-06,
                -4.22629655e-05,
                6.28065475e-05,
                4.04984708e-05,
                4.62730939e-04,
                -2.26906415e-04,
                -4.65501369e-04,
                -6.44129196e-03,
                -1.77048263e-03,
                1.08009684e-03,
                -2.69397440e-01,
                3.50383255e-01,
                1.61944683e00,
                1.00618764e00,
                4.06395120e-03,
                -6.62095997e-03,
                -2.68278933e-04,
            ]
        )

        self.model = self.robot_env.model
        self.data = self.robot_env.data
        self.render_mode = self.robot_env.render_mode

        self.terminate_on_tasks_completed = terminate_on_tasks_completed
        self.remove_task_when_completed = remove_task_when_completed

        self.goal = {}
        self.tasks_to_complete = set(tasks_to_complete)
        # Validate list of tasks to complete
        for task in tasks_to_complete:
            if task not in OBS_ELEMENT_GOALS.keys():
                raise ValueError(
                    f"The task {task} cannot be found the the list of possible goals: {OBS_ELEMENT_GOALS.keys()}"
                )
            else:
                self.goal[task] = OBS_ELEMENT_GOALS[task]

        self.step_task_completions = (
            []
        )  # Tasks completed in the current environment step
        self.episode_task_completions = (
            []
        )  # Tasks completed that have been completed in the current episode
        self.object_noise_ratio = (
            object_noise_ratio  # stochastic noise added to the object observations
        )

        robot_obs = self.robot_env._get_obs()
        obs = self._get_obs(robot_obs)

        assert (
            int(np.round(1.0 / self.robot_env.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.robot_env.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.action_space = self.robot_env.action_space
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Dict(
                    {
                        task: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=goal.shape,
                            dtype="float64",
                        )
                        for task, goal in obs["achieved_goal"].items()
                    }
                ),
                achieved_goal=spaces.Dict(
                    {
                        task: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=goal.shape,
                            dtype="float64",
                        )
                        for task, goal in obs["achieved_goal"].items()
                    }
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

        EzPickle.__init__(
            self,
            tasks_to_complete,
            terminate_on_tasks_completed,
            remove_task_when_completed,
            object_noise_ratio,
            **kwargs,
        )

    def compute_reward(
        self,
        achieved_goal: "dict[str, np.ndarray]",
        desired_goal: "dict[str, np.ndarray]",
        info: "dict[str, Any]",
    ):
        self.step_task_completions.clear()
        for task in self.tasks_to_complete:
            distance = np.linalg.norm(achieved_goal[task] - desired_goal[task])
            complete = distance < BONUS_THRESH
            if complete:
                self.step_task_completions.append(task)

        return float(len(self.step_task_completions))

    def _get_obs(self, robot_obs):
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

        achieved_goal = {
            task: self.data.qpos[OBS_ELEMENT_INDICES[task]] for task in self.goal.keys()
        }

        obs = {
            "observation": np.concatenate((robot_obs, obj_qpos, obj_qvel)),
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal,
        }

        return obs

    def step(self, action):
        robot_obs, _, terminated, truncated, info = self.robot_env.step(action)
        obs = self._get_obs(robot_obs)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        if self.remove_task_when_completed:
            # When the task is accomplished remove from the list of tasks to be completed
            [
                self.tasks_to_complete.remove(element)
                for element in self.step_task_completions
            ]

        info = {"tasks_to_complete": list(self.tasks_to_complete)}
        info["step_task_completions"] = self.step_task_completions.copy()

        for task in self.step_task_completions:
            if task not in self.episode_task_completions:
                self.episode_task_completions.append(task)
        info["episode_task_completions"] = self.episode_task_completions
        if self.terminate_on_tasks_completed:
            # terminate if there are no more tasks to complete
            terminated = len(self.episode_task_completions) == len(self.goal.keys())

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.episode_task_completions.clear()
        robot_obs, _ = self.robot_env.reset(seed=seed)
        obs = self._get_obs(robot_obs)
        self.tasks_to_complete = set(self.goal.keys())
        info = {
            "tasks_to_complete": list(self.tasks_to_complete),
            "episode_task_completions": [],
            "step_task_completions": [],
        }

        return obs, info

    def render(self):
        return self.robot_env.render()

    def close(self):
        self.robot_env.close()
